from collections.abc import Mapping
import dataclasses
import json
import pathlib

import cv2
from lerobot.common.datasets.utils import get_delta_indices, load_json, load_jsonlines
import numpy as np
import pandas as pd
import torch


def _stack_series(series: pd.Series) -> np.ndarray:
    values = series.tolist()
    first = np.asarray(values[0])
    if first.ndim == 0:
        return np.asarray(values)
    return np.stack([np.asarray(value) for value in values], axis=0)


def _read_video_frame(video_path: pathlib.Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success or frame is None:
            raise ValueError(f"Failed to decode frame {frame_index} from {video_path}")
    finally:
        capture.release()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.transpose(frame, (2, 0, 1)).astype(np.float32) / 255.0


def _path_match_keys(path: str | pathlib.Path) -> tuple[str, ...]:
    path = pathlib.Path(str(path))
    keys = [str(path), str(path.expanduser().resolve(strict=False))]
    parts = path.parts
    for marker in ("real", "sim"):
        if marker in parts:
            keys.append(str(pathlib.Path(*parts[parts.index(marker):])))
    if "data" in parts:
        idx = parts.index("data")
        if idx >= 1:
            keys.append(str(pathlib.Path(*parts[idx - 1:])))
    return tuple(dict.fromkeys(keys))


def _annotation_match_keys(root: pathlib.Path, path: pathlib.Path, item: Mapping) -> tuple[str, ...]:
    keys: list[str] = []
    parts = path.parts
    chunk_idx = next((idx for idx, part in enumerate(parts) if part.startswith("chunk-")), None)
    if chunk_idx is None:
        return tuple(dict.fromkeys(keys))

    parquet_name = f"{path.stem}.parquet"
    chunk = parts[chunk_idx]
    task_path = str(item.get("task_path") or "").strip()
    if task_path:
        keys.extend(_path_match_keys(pathlib.Path(task_path) / "data" / chunk / parquet_name))

    for marker in ("real", "sim"):
        if marker in parts and parts.index(marker) < chunk_idx:
            keys.extend(_path_match_keys(pathlib.Path(*parts[parts.index(marker):chunk_idx]) / "data" / chunk / parquet_name))

    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        relative_parts = ()
    if relative_parts and relative_parts[0].startswith("chunk-"):
        keys.extend(_path_match_keys(pathlib.Path("data") / relative_parts[0] / parquet_name))

    if chunk_idx >= 1:
        keys.extend(_path_match_keys(pathlib.Path(parts[chunk_idx - 1]) / "data" / chunk / parquet_name))

    return tuple(dict.fromkeys(keys))


def _load_subtask_spans(annotations_dir: str | pathlib.Path | None) -> dict[str, list[tuple[int, int, str]]]:
    if not annotations_dir:
        return {}

    root = pathlib.Path(annotations_dir).expanduser()
    paths = [root] if root.is_file() else sorted(root.rglob("*.json"))
    cache: dict[str, list[tuple[int, int, str]]] = {}
    for path in paths:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not (isinstance(item, dict) and isinstance(item.get("subtasks"), list)):
            continue

        spans: list[tuple[int, int, str]] = []
        for subtask in item["subtasks"]:
            label = str(subtask.get("label", "")).strip()
            if not label:
                continue
            try:
                start = int(subtask["start_frame"])
                end = int(subtask["end_frame"])
            except (KeyError, TypeError, ValueError):
                continue
            if end >= start:
                spans.append((start, end, label))
        if not spans:
            continue

        spans.sort(key=lambda span: (span[0], span[1]))
        for key in _annotation_match_keys(root, path, item):
            cache[key] = spans
    return cache


def _spans_for_parquet(
    spans_by_parquet: dict[str, list[tuple[int, int, str]]],
    parquet_path: pathlib.Path,
) -> list[tuple[int, int, str]]:
    for key in _path_match_keys(parquet_path):
        spans = spans_by_parquet.get(key)
        if spans is not None:
            return spans
    return []


def _find_subtask_span_in_spans(
    spans: list[tuple[int, int, str]],
    frame_idx: int,
) -> tuple[int, int, str] | None:
    for start, end, label in spans:
        if start <= frame_idx <= end:
            return start, end, label
    return None


def _find_subtask_span(
    spans_by_parquet: dict[str, list[tuple[int, int, str]]],
    parquet_path: pathlib.Path,
    frame_idx: int,
) -> tuple[int, int, str] | None:
    return _find_subtask_span_in_spans(_spans_for_parquet(spans_by_parquet, parquet_path), frame_idx)


def _sample_indices_for_subtask_spans(
    frame_indices: np.ndarray,
    spans: list[tuple[int, int, str]],
) -> list[int]:
    if not spans:
        return []

    frame_indices = np.asarray(frame_indices).reshape(-1)
    sample_indices: list[int] = []
    seen: set[int] = set()
    for start, end, _ in spans:
        left = int(np.searchsorted(frame_indices, start, side="left"))
        right = int(np.searchsorted(frame_indices, end, side="right"))
        for local_idx in range(max(0, left), min(len(frame_indices), right)):
            if local_idx not in seen:
                sample_indices.append(local_idx)
                seen.add(local_idx)
    return sample_indices


@dataclasses.dataclass(frozen=True)
class LocalLeRobotMetadata:
    repo_id: str
    root: pathlib.Path
    info: Mapping
    tasks: dict[int, str]
    episodes: list[int]

    @property
    def fps(self) -> float:
        return float(self.info["fps"])

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.info["features"].items() if ft["dtype"] == "video"]

    def get_data_file_path(self, episode_index: int) -> pathlib.Path:
        episode_chunk = episode_index // self.info["chunks_size"]
        return self.root / self.info["data_path"].format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )

    def get_video_file_path(self, episode_index: int, video_key: str) -> pathlib.Path:
        episode_chunk = episode_index // self.info["chunks_size"]
        return self.root / self.info["video_path"].format(
            episode_chunk=episode_chunk,
            video_key=video_key,
            episode_index=episode_index,
        )


def load_local_lerobot_metadata(repo_id: str, repo_root: str | pathlib.Path) -> LocalLeRobotMetadata:
    root = pathlib.Path(repo_root)
    info = load_json(root / "meta" / "info.json")
    tasks = {
        int(task["task_index"]): task["task"]
        for task in load_jsonlines(root / "meta" / "tasks.jsonl")
    }
    episodes = [int(episode["episode_index"]) for episode in load_jsonlines(root / "meta" / "episodes.jsonl")]
    return LocalLeRobotMetadata(repo_id=repo_id, root=root, info=info, tasks=tasks, episodes=episodes)


class LocalLeRobotDataset(torch.utils.data.Dataset):
    """Minimal local LeRobot loader for offline OpenPI training/inference."""

    def __init__(
        self,
        metadata: LocalLeRobotMetadata,
        delta_timestamps: dict[str, list[float]] | None = None,
        *,
        load_videos: bool = True,
        subtask_annotations_dir: str | None = None,
        sample_only_subtask_frames: bool = False,
        clamp_action_sequences_to_subtask: bool = False,
    ):
        self.meta = metadata
        self.delta_indices = None if delta_timestamps is None else get_delta_indices(delta_timestamps, self.meta.fps)
        self._load_videos = load_videos
        self._sample_only_subtask_frames = sample_only_subtask_frames
        self._clamp_action_sequences_to_subtask = clamp_action_sequences_to_subtask
        self._subtask_spans_by_parquet = _load_subtask_spans(subtask_annotations_dir)
        self._subtask_spans_by_episode: dict[int, list[tuple[int, int, str]]] = {}
        self._episode_frames = {}
        self._global_index = []

        for episode_index in self.meta.episodes:
            parquet_path = self.meta.get_data_file_path(episode_index)
            dataframe = pd.read_parquet(parquet_path)
            frame_data = {column: _stack_series(dataframe[column]) for column in dataframe.columns}
            self._episode_frames[episode_index] = frame_data
            subtask_spans = _spans_for_parquet(self._subtask_spans_by_parquet, parquet_path)
            self._subtask_spans_by_episode[episode_index] = subtask_spans
            if not sample_only_subtask_frames:
                self._global_index.extend((episode_index, local_idx) for local_idx in range(len(dataframe)))
                continue
            self._global_index.extend(
                (episode_index, local_idx)
                for local_idx in _sample_indices_for_subtask_spans(frame_data["frame_index"], subtask_spans)
            )

    def __len__(self) -> int:
        return len(self._global_index)

    def __getitem__(self, index: int) -> dict:
        episode_index, local_idx = self._global_index[index]
        episode_frames = self._episode_frames[episode_index]
        item = {key: np.asarray(values[local_idx]) for key, values in episode_frames.items()}
        item["_episode_index"] = np.asarray(episode_index, dtype=np.int64)
        item["_local_index"] = np.asarray(local_idx, dtype=np.int64)
        parquet_path = self.meta.get_data_file_path(episode_index)
        item["_parquet_path"] = str(parquet_path)
        current_frame = int(np.asarray(item["frame_index"]).item())
        subtask_span = _find_subtask_span_in_spans(
            self._subtask_spans_by_episode.get(episode_index, []),
            current_frame,
        )
        if subtask_span is not None:
            _, subtask_end_frame, subtask_label = subtask_span
            item["_subtask_end_frame"] = np.asarray(subtask_end_frame, dtype=np.int64)
            item["_subtask_label"] = subtask_label

        if self.delta_indices is not None:
            episode_length = len(next(iter(episode_frames.values())))
            frame_indices = np.asarray(episode_frames["frame_index"]).reshape(-1)
            for key, deltas in self.delta_indices.items():
                query_indices = []
                is_pad = []
                for delta in deltas:
                    target_idx = max(0, min(episode_length - 1, local_idx + delta))
                    padded = (local_idx + delta < 0) or (local_idx + delta >= episode_length)
                    if (
                        self._clamp_action_sequences_to_subtask
                        and subtask_span is not None
                    ):
                        target_frame = int(np.asarray(frame_indices[target_idx]).item())
                        if target_frame > subtask_end_frame:
                            target_idx = int(np.searchsorted(frame_indices, subtask_end_frame, side="right") - 1)
                            target_idx = max(0, min(episode_length - 1, target_idx))
                            padded = True
                    query_indices.append(target_idx)
                    is_pad.append(padded)
                item[key] = np.asarray(episode_frames[key][query_indices])
                item[f"{key}_is_pad"] = np.asarray(is_pad, dtype=bool)

        if self._load_videos:
            for video_key in self.meta.video_keys:
                item[video_key] = _read_video_frame(self.meta.get_video_file_path(episode_index, video_key), current_frame)

        task_index = int(np.asarray(item["task_index"]).item())
        item["task"] = self.meta.tasks[task_index]
        return item
