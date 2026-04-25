import json
import os
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

from openpi.models import model as _model
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _pad_or_trim_last_dim(array: np.ndarray, target_dim: int) -> np.ndarray:
    """Pads or trims the last dimension to match the model dimension."""
    if array.shape[-1] == target_dim:
        return array
    if array.shape[-1] > target_dim:
        return array[..., :target_dim]

    pad_width = [(0, 0)] * array.ndim
    pad_width[-1] = (0, target_dim - array.shape[-1])
    return np.pad(array, pad_width, mode="constant")


def _ensure_action_horizon(actions: np.ndarray, action_horizon: int) -> np.ndarray:
    """Pads or trims the time dimension of actions."""
    if actions.shape[0] == action_horizon:
        return actions
    if actions.shape[0] > action_horizon:
        return actions[:action_horizon]

    if actions.shape[0] == 0:
        raise ValueError("actions cannot be empty")

    pad = np.repeat(actions[-1:], action_horizon - actions.shape[0], axis=0)
    return np.concatenate([actions, pad], axis=0)


def _ensure_hwc_uint8_image(image: Any) -> np.ndarray:
    """Converts a LeRobot image-like object to uint8 HWC numpy."""
    if hasattr(image, "numpy"):
        image = image.numpy()
    elif hasattr(image, "__array__"):
        image = np.asarray(image)

    if not isinstance(image, np.ndarray):
        try:
            image = np.asarray(image)
        except Exception as exc:  # pragma: no cover - defensive branch
            raise TypeError(f"Unsupported image type: {type(image)}") from exc

    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {image.shape}")

    # Convert CHW -> HWC when needed.
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            if image.max() <= 1.0:
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    return image


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class SubtaskJsonlDataset:
    """Simple dataset for pi05 subtask-generation training from flattened JSONL samples.

    This remains useful as a debug path, but the preferred path for your current setup is
    `LeRobotSubtaskDataset`, which consumes LeRobot episodes plus `meta/episodes.jsonl`.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        action_horizon: int,
        action_dim: int,
        max_token_len: int = 200,
        image_keys: tuple[str, ...] = _model.IMAGE_KEYS,
    ):
        self._manifest_path = Path(manifest_path)
        self._root_dir = self._manifest_path.parent
        self._image_keys = image_keys
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._tokenizer = PaligemmaTokenizer(max_len=max_token_len)
        self._records = _load_jsonl(self._manifest_path)

        if not self._records:
            raise ValueError(f"No records found in {self._manifest_path}")

    def __len__(self) -> int:
        return len(self._records)

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self._root_dir / path

    def _load_image(self, path_str: str) -> np.ndarray:
        path = self._resolve_path(path_str)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return image.astype(np.uint8)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._records[index]

        high_level_prompt = record["high_level_prompt"]
        low_level_prompt = record["low_level_prompt"]
        tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask = (
            self._tokenizer.tokenize_high_low_prompt(high_level_prompt, low_level_prompt)
        )

        image_paths = record["images"]
        image_mask_record = record.get("image_mask", {})
        images = {key: self._load_image(image_paths[key]) for key in self._image_keys}
        image_masks = {
            key: np.asarray(bool(image_mask_record.get(key, True)), dtype=np.bool_) for key in self._image_keys
        }

        state = np.asarray(record["state"], dtype=np.float32)
        if state.ndim != 1:
            raise ValueError(f"state must be 1D, got shape {state.shape}")
        state = _pad_or_trim_last_dim(state, self._action_dim)

        actions = np.asarray(record["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"actions must be 2D [horizon, dim], got shape {actions.shape}")
        actions = _ensure_action_horizon(actions, self._action_horizon)
        actions = _pad_or_trim_last_dim(actions, self._action_dim)

        return {
            "image": images,
            "image_mask": image_masks,
            "state": state,
            "tokenized_prompt": tokenized_prompt.astype(np.int32),
            "tokenized_prompt_mask": tokenized_prompt_mask.astype(np.bool_),
            "token_ar_mask": token_ar_mask.astype(np.int32),
            "token_loss_mask": token_loss_mask.astype(np.bool_),
            "actions": actions,
        }


class LeRobotSubtaskDataset:
    """Segment-level subtask dataset built on top of a LeRobot dataset.

    It expands each episode into one sample per `action_config` segment in `meta/episodes.jsonl`:
    - high-level prompt comes from `tasks`
    - low-level prompt comes from `action_config[*].action_text`
    - observation comes from the segment `start_frame`
    - actions come from `start_frame : start_frame + action_horizon`

    This is designed to match the new pi05 subtask-generation training objective.
    """

    def __init__(
        self,
        repo_id_or_path: str | Path,
        *,
        action_horizon: int,
        action_dim: int,
        max_token_len: int = 200,
        image_key_mapping: dict[str, str] | None = None,
        state_key: str = "state",
        action_key: str = "actions",
        tasks_index: int = 0,
        annotation_root: str | None = None,
    ):
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
            raise ModuleNotFoundError(
                "LeRobot is not installed in this environment. Install the LeRobot dependency "
                "or use the flattened JSONL debug loader instead."
            ) from exc

        self._repo_id_or_path = repo_id_or_path
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._state_key = state_key
        self._action_key = action_key
        self._tasks_index = tasks_index
        self._tokenizer = PaligemmaTokenizer(max_len=max_token_len)
        self._annotation_root = self._resolve_annotation_root(annotation_root)

        self._image_key_mapping = image_key_mapping or {
            "base_0_rgb": "face_view",
            "left_wrist_0_rgb": "left_wrist_view",
            "right_wrist_0_rgb": "right_wrist_view",
        }

        self._dataset = LeRobotDataset(repo_id_or_path, local_files_only=True)
        self._dataset_root = self._resolve_dataset_root(repo_id_or_path)
        self._annotation_episode_index = self._build_annotation_episode_index(self._annotation_root)
        self._episode_records = _load_jsonl(self._dataset_root / "meta" / "episodes.jsonl")
        self._episode_lengths = [int(record["length"]) for record in self._episode_records]
        self._episode_offsets = np.cumsum([0, *self._episode_lengths[:-1]]).tolist()
        self._segment_records = self._build_segment_index()

    def _resolve_dataset_root(self, repo_id_or_path: str | Path) -> Path:
        path = Path(repo_id_or_path)
        if path.exists():
            return path

        root_candidates = []
        for env_key in ("HF_LEROBOT_HOME", "LEROBOT_HOME"):
            env_value = os.environ.get(env_key)
            if env_value:
                root_candidates.append(Path(env_value) / str(repo_id_or_path))

        for candidate in root_candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not resolve local LeRobot dataset root for {repo_id_or_path!r}. "
            "Pass a dataset path directly or set HF_LEROBOT_HOME/LEROBOT_HOME."
        )

    def _build_segment_index(self) -> list[dict[str, Any]]:
        segment_records: list[dict[str, Any]] = []
        for episode_record in self._episode_records:
            episode_index = int(episode_record["episode_index"])
            tasks = episode_record.get("tasks", [])
            if isinstance(tasks, str):
                tasks = [tasks]
            if not tasks:
                continue
            high_level_prompt = tasks[min(self._tasks_index, len(tasks) - 1)]

            segments = self._load_episode_segments(episode_record)
            for segment in segments:
                action_text = segment.get("action_text") or segment.get("label") or segment.get("subtask") or ""
                if not action_text:
                    continue
                start_frame = int(segment["start_frame"])
                end_frame = int(segment["end_frame"])
                if end_frame <= start_frame:
                    continue
                segment_records.append(
                    {
                        "episode_index": episode_index,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "high_level_prompt": high_level_prompt,
                        "low_level_prompt": action_text,
                    }
                )

        if not segment_records:
            raise ValueError("No usable action_config segments found in meta/episodes.jsonl")

        return segment_records

    def _resolve_annotation_root(self, annotation_root: str | None) -> Path | None:
        candidate = annotation_root or os.environ.get("MANIPARENA_SUBTASK_ROOT")
        if not candidate:
            return None
        path = Path(candidate)
        return path if path.exists() else None

    def _build_annotation_episode_index(self, annotation_root: Path | None) -> dict[int, list[Path]]:
        if annotation_root is None:
            return {}

        indexed_paths: dict[int, list[Path]] = {}
        for path in annotation_root.rglob("episode_*.json"):
            episode_indices = self._extract_episode_indices(path)
            if not episode_indices:
                continue
            for episode_index in episode_indices:
                indexed_paths.setdefault(episode_index, []).append(path)
        return indexed_paths

    def _read_annotation_payload(self, path: Path) -> Any | None:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _extract_episode_indices(self, path: Path) -> set[int]:
        episode_indices: set[int] = set()
        for name in (path.stem, path.parent.name):
            match = re.search(r"episode_(\d+)", name)
            if match is not None:
                episode_indices.add(int(match.group(1)))
        return episode_indices

    def _annotation_match_score(self, path: Path) -> int:
        dataset_parts = {part.lower() for part in self._dataset_root.parts}
        candidate_parts = {part.lower() for part in path.parts}
        return len(dataset_parts & candidate_parts)

    def _annotation_record_match_score(self, payload: Any, episode_record: dict[str, Any], path: Path) -> int:
        score = self._annotation_match_score(path)
        if not isinstance(payload, dict):
            return score

        task_path = str(payload.get("task_path", "")).strip().lower().replace("\\", "/")
        if task_path:
            dataset_path = str(self._dataset_root).lower().replace("\\", "/")
            if task_path in dataset_path:
                score += 100

        episode_id = str(payload.get("episode_id", "")).strip().lower().replace("\\", "/")
        if episode_id:
            dataset_episode_path = (
                str(self._dataset_root / f"episode_{int(episode_record['episode_index']):06d}")
                .lower()
                .replace("\\", "/")
            )
            if episode_id in dataset_episode_path or dataset_episode_path.endswith(episode_id):
                score += 1000

        return score

    def _resolve_annotation_file(self, episode_record: dict[str, Any]) -> Path | None:
        episode_index = int(episode_record["episode_index"])
        candidates = self._annotation_episode_index.get(episode_index, [])
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda candidate: self._annotation_record_match_score(
                self._read_annotation_payload(candidate), episode_record, candidate
            ),
        )

    def _normalize_annotation_segments(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            for key in ("segments", "subtasks", "labels", "actions"):
                value = payload.get(key)
                if isinstance(value, list):
                    payload = value
                    break
            else:
                payload = [payload]

        if not isinstance(payload, list):
            return []

        normalized_segments: list[dict[str, Any]] = []
        for raw_segment in payload:
            if not isinstance(raw_segment, dict):
                continue
            action_text = raw_segment.get("label") or raw_segment.get("subtask") or raw_segment.get("action_text")
            start_frame = raw_segment.get("start")
            if start_frame is None:
                start_frame = raw_segment.get("start_frame")
            end_frame = raw_segment.get("end")
            if end_frame is None:
                end_frame = raw_segment.get("end_frame")
            if action_text is None or start_frame is None or end_frame is None:
                continue
            normalized_segments.append(
                {
                    "action_text": str(action_text),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                }
            )
        return normalized_segments

    def _load_episode_segments(self, episode_record: dict[str, Any]) -> list[dict[str, Any]]:
        annotation_file = self._resolve_annotation_file(episode_record)
        if annotation_file is not None:
            payload = self._read_annotation_payload(annotation_file)
            if payload is not None:
                normalized_segments = self._normalize_annotation_segments(payload)
                if normalized_segments:
                    return sorted(normalized_segments, key=lambda segment: int(segment["start_frame"]))

        return sorted(episode_record.get("action_config", []), key=lambda segment: int(segment["start_frame"]))

    def __len__(self) -> int:
        return len(self._segment_records)

    def _global_frame_index(self, episode_index: int, frame_index: int) -> int:
        return self._episode_offsets[episode_index] + frame_index

    def _get_frame(self, global_frame_index: int) -> dict[str, Any]:
        return self._dataset[global_frame_index]

    def _extract_images(self, frame: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.ndarray] = {}
        for model_key, source_key in self._image_key_mapping.items():
            if source_key in frame and frame[source_key] is not None:
                images[model_key] = _ensure_hwc_uint8_image(frame[source_key])
                image_masks[model_key] = np.asarray(True, dtype=np.bool_)
            else:
                if "base_0_rgb" not in images:
                    raise KeyError(
                        f"Missing required base camera {source_key!r} in frame keys {list(frame)}. "
                        "Pass an explicit image_key_mapping if your LeRobot feature names differ."
                    )
                images[model_key] = np.zeros_like(images["base_0_rgb"])
                image_masks[model_key] = np.asarray(False, dtype=np.bool_)
        return images, image_masks

    def _extract_actions(self, episode_index: int, start_frame: int) -> np.ndarray:
        actions = []
        episode_length = self._episode_lengths[episode_index]
        for offset in range(self._action_horizon):
            frame_index = min(start_frame + offset, episode_length - 1)
            frame = self._get_frame(self._global_frame_index(episode_index, frame_index))
            if self._action_key not in frame:
                raise KeyError(f"Action key {self._action_key!r} not found in frame keys {list(frame)}")
            actions.append(np.asarray(frame[self._action_key], dtype=np.float32))

        stacked = np.stack(actions, axis=0)
        return _pad_or_trim_last_dim(stacked, self._action_dim)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._segment_records[index]
        episode_index = record["episode_index"]
        start_frame = record["start_frame"]

        global_start = self._global_frame_index(episode_index, start_frame)
        frame = self._get_frame(global_start)

        images, image_masks = self._extract_images(frame)

        if self._state_key not in frame:
            raise KeyError(f"State key {self._state_key!r} not found in frame keys {list(frame)}")
        state = np.asarray(frame[self._state_key], dtype=np.float32)
        if state.ndim != 1:
            state = state.reshape(-1)
        state = _pad_or_trim_last_dim(state, self._action_dim)

        actions = self._extract_actions(episode_index, start_frame)

        tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask = (
            self._tokenizer.tokenize_high_low_prompt(record["high_level_prompt"], record["low_level_prompt"])
        )

        return {
            "image": images,
            "image_mask": image_masks,
            "state": state,
            "tokenized_prompt": tokenized_prompt.astype(np.int32),
            "tokenized_prompt_mask": tokenized_prompt_mask.astype(np.bool_),
            "token_ar_mask": token_ar_mask.astype(np.int32),
            "token_loss_mask": token_loss_mask.astype(np.bool_),
            "actions": actions.astype(np.float32),
        }


def create_subtask_jsonl_data_loader(
    manifest_path: str | Path,
    *,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    real_action_dim: int | None = None,
    max_token_len: int = 200,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    sharding=None,
):
    """Creates the flattened JSONL debug loader compatible with the openpi training loop."""
    dataset = SubtaskJsonlDataset(
        manifest_path,
        action_horizon=action_horizon,
        action_dim=action_dim,
        max_token_len=max_token_len,
    )
    torch_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework="jax",
    )
    data_config = _config.DataConfig(
        repo_id=str(manifest_path),
        asset_id=None,
        norm_stats=None,
        real_action_dim=real_action_dim or action_dim,
    )
    return _data_loader.DataLoaderImpl(data_config=data_config, data_loader=torch_loader)


def create_lerobot_subtask_data_loader(
    repo_id_or_path: str | Path,
    *,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    real_action_dim: int | None = None,
    max_token_len: int = 200,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    sharding=None,
    image_key_mapping: dict[str, str] | None = None,
    state_key: str = "state",
    action_key: str = "actions",
    tasks_index: int = 0,
    annotation_root: str | None = None,
):
    """Creates a segment-level LeRobot subtask loader compatible with the openpi training loop."""
    dataset = LeRobotSubtaskDataset(
        repo_id_or_path,
        action_horizon=action_horizon,
        action_dim=action_dim,
        max_token_len=max_token_len,
        image_key_mapping=image_key_mapping,
        state_key=state_key,
        action_key=action_key,
        tasks_index=tasks_index,
        annotation_root=annotation_root,
    )
    torch_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework="jax",
    )
    data_config = _config.DataConfig(
        repo_id=str(repo_id_or_path),
        asset_id=None,
        norm_stats=None,
        real_action_dim=real_action_dim or action_dim,
    )
    return _data_loader.DataLoaderImpl(data_config=data_config, data_loader=torch_loader)
