#!/usr/bin/env python3
"""Cascaded ManipArena policy: Qwen3-VL subtask planner + small_vlm action policy.

The policy consumes the ManipArena server observation format:
    {
        "state": {"follow1_pos": [...], "follow2_pos": [...]},
        "views": {
            "camera_front": "<base64 JPEG>",
            "camera_left": "<base64 JPEG>",
            "camera_right": "<base64 JPEG>",
        },
        "instruction": "...",
    }

It first predicts a high-level subtask with the Qwen3-VL LoRA adapter under
``qwen3_vl_subtask``.  The predicted subtask is then used as the text prompt for
the action model from ``small_vlm``.
"""

from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
import importlib
import json
import logging
import os
from pathlib import Path
import re
import socket
import sys
from typing import Any, Dict, Sequence

import cv2
import numpy as np
from PIL import Image

from maniparena.debug import format_action_payload_for_debug
from maniparena.policy import ModelPolicy

logger = logging.getLogger(__name__)


def _reset_socket_default_timeout(context: str) -> None:
    timeout = socket.getdefaulttimeout()
    if timeout is not None:
        logger.warning(
            "Resetting global socket default timeout from %s to None after %s",
            timeout,
            context,
        )
        socket.setdefaulttimeout(None)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
QWEN_SUBTASK_DIR = WORKSPACE_ROOT / "qwen3_vl_subtask"
SMALL_VLM_ROOT = WORKSPACE_ROOT / "small_vlm"

QWEN_BASE_MODEL = os.environ.get(
    "SUBTASK_QWEN_BASE_MODEL",
    "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
)
QWEN_LORA_DIR = os.environ.get(
    "SUBTASK_QWEN_LORA_DIR",
    str(QWEN_SUBTASK_DIR / "outputs" / "vla_subtask_qwen3vl8b_structured_4mode" / "checkpoint-240"),
)
QWEN_MAX_IMAGE_SIDE = int(os.environ.get("SUBTASK_QWEN_MAX_IMAGE_SIDE", "512"))
QWEN_MAX_NEW_TOKENS = int(os.environ.get("SUBTASK_QWEN_MAX_NEW_TOKENS", "256"))
QWEN_PLAN_MAX_NEW_TOKENS = int(os.environ.get("SUBTASK_QWEN_PLAN_MAX_NEW_TOKENS", "256"))


def _parse_history_offsets(value: str) -> tuple[int, ...]:
    offsets: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        offset = int(item)
        if offset <= 0:
            raise ValueError(f"SUBTASK_QWEN_HISTORY_FRAME_OFFSETS must contain positive integers: {offset}")
        offsets.append(offset)
    return tuple(offsets)


QWEN_HISTORY_FRAME_OFFSETS = _parse_history_offsets(
    os.environ.get("SUBTASK_QWEN_HISTORY_FRAME_OFFSETS", "64,32")
)
QWEN_OBSERVATION_FRAME_STRIDE = int(os.environ.get("SUBTASK_QWEN_OBSERVATION_FRAME_STRIDE", "32"))
if QWEN_OBSERVATION_FRAME_STRIDE <= 0:
    raise ValueError("SUBTASK_QWEN_OBSERVATION_FRAME_STRIDE must be positive")


def _history_offset_to_observation_steps(offset: int) -> int:
    # The ManipArena client calls this policy every N simulator frames, not every frame.
    # With the default N=32, prompt offsets 64/32/current map to observations t-2/t-1/t.
    return max(1, int(round(offset / float(QWEN_OBSERVATION_FRAME_STRIDE))))


QWEN_IMAGE_HISTORY_MAXLEN = max(
    (_history_offset_to_observation_steps(offset) for offset in QWEN_HISTORY_FRAME_OFFSETS),
    default=0,
) + 1
QWEN_SUBTASK_LABELS_JSON = os.environ.get("SUBTASK_QWEN_SUBTASK_LABELS_JSON", "").strip()

SMALL_VLM_CONFIG_NAME = os.environ.get("SMALL_VLM_CONFIG_NAME", "pi05_maniparena_ee")
SMALL_VLM_ACTION_SERVER_HOST = os.environ.get("SMALL_VLM_ACTION_SERVER_HOST", "127.0.0.1").strip()
SMALL_VLM_ACTION_SERVER_PORT = int(os.environ.get("SMALL_VLM_ACTION_SERVER_PORT", "18081"))
SMALL_VLM_USE_REMOTE_ACTION_SERVER = os.environ.get(
    "SMALL_VLM_USE_REMOTE_ACTION_SERVER",
    "1",
).strip().lower() not in ("0", "false", "no", "off")
SMALL_VLM_PROMPT_TEMPLATE = os.environ.get(
    "SMALL_VLM_PROMPT_TEMPLATE",
    "{subtask}",
)
SUBTASK_SWITCH_STABLE_STEPS = int(os.environ.get("SUBTASK_SWITCH_STABLE_STEPS", "3"))
MAX_COMPLETED_SUBTASKS = int(os.environ.get("MAX_COMPLETED_SUBTASKS", "8"))

ACTION_END_RATIO = 0.8
ACTION_START_STEP = 0
ACTION_END_STEP = 32
ACTION_OUTPUT_STEPS = 32
DEBUG_OUTPUT_STEPS = 4
Z_DOWN_BIAS_M = 0.02
Z_BIAS_MAX_HEIGHT_M = -0.2
STALL_TOLERANCE_M = 0.01
POSITION_STALL_TRIGGER_ACTIONS = 20
GRIPPER_OPEN_SWITCH_THRESHOLD = 1.0
GRIPPER_CLOSE_SWITCH_THRESHOLD = 3.0
GRIPPER_FORCE_OPEN_THRESHOLD = 4.0
GRIPPER_FORCE_CLOSE_THRESHOLD = 0.5
GRIPPER_TREND_STEPS = 5
GRIPPER_MIN_DELTA_PER_STEP = 0.01
GRIPPER_OPEN_VALUE = 5.0
GRIPPER_CLOSE_VALUE = 0.0
GRIPPER_FORWARD_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float32)
EPS = 1e-6

_CAM_MAP = {
    "camera_front": "observation.images.faceImg",
    "camera_left": "observation.images.leftImg",
    "camera_right": "observation.images.rightImg",
}
_CAMERA_NAMES = ["faceImg", "leftImg", "rightImg"]
_RESET_FLAG_KEYS = ("reset", "episode_reset", "new_episode", "episode_start", "is_first", "first_step")
_EPISODE_MARKER_KEYS = ("episode_id", "episode_index", "episode", "reset_id", "env_reset_id")
_FRAME_MARKER_KEYS = ("frame_index", "frame", "step", "step_count", "timestep", "time_step")


@dataclass
class _CascadeModels:
    subtask_model: Any
    subtask_tokenizer: Any
    action_policy: Any


class _RemoteActionPolicyClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = int(port)
        self.uri = f"ws://{host}:{port}"
        self._connection = None
        self._metadata: dict[str, Any] = {}

    def connect(self) -> None:
        import msgpack
        import msgpack_numpy
        from websockets.sync.client import connect as ws_connect

        msgpack_numpy.patch()
        self._connection = ws_connect(self.uri, max_size=None)
        metadata_bytes = self._connection.recv()
        self._metadata = msgpack.unpackb(metadata_bytes, raw=False)
        logger.info("Connected to remote small_vlm action server: %s metadata=%s", self.uri, self._metadata)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        import msgpack
        import msgpack_numpy

        msgpack_numpy.patch()

        if self._connection is None:
            self.connect()
        self._connection.send(msgpack.packb(observation, use_bin_type=True))
        response_bytes = self._connection.recv()
        if isinstance(response_bytes, str):
            raise RuntimeError(f"Remote small_vlm action server error: {response_bytes}")
        return msgpack.unpackb(response_bytes, raw=False)

    def reset(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
        self._connection = None


def _decode_image(v: Any) -> np.ndarray:
    """base64 JPEG string or numpy array -> RGB uint8 ndarray."""
    if isinstance(v, np.ndarray):
        arr = np.asarray(v)
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        return arr
    raw = base64.b64decode(v) if isinstance(v, str) else bytes(v)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize_pil(image: Image.Image, max_image_side: int) -> Image.Image:
    if max_image_side <= 0:
        return image
    width, height = image.size
    scale = max(width, height) / float(max_image_side)
    if scale <= 1.0:
        return image
    new_size = (int(round(width / scale)), int(round(height / scale)))
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    return image.resize(new_size, resampling)


def _to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")


_STRUCTURED_SUBTASK_FIELDS = (
    "SUBTASK_AT_CURRENT_PLUS_3_FRAME",
    "ANSWER",
    "CURRENT_STEP",
)


def _extract_structured_field(text: str, field: str) -> str:
    pattern = re.compile(rf"(?im)^\s*{re.escape(field)}\s*:\s*(.+?)\s*$")
    match = pattern.search(text or "")
    return match.group(1).strip().strip("\"'") if match else ""


def _extract_structured_section(text: str, section: str) -> str:
    pattern = re.compile(
        rf"(?ims)^\s*{re.escape(section)}\s*:\s*(.*?)(?=^\s*[A-Z0-9_]+\s*:|\Z)"
    )
    match = pattern.search(text or "")
    return match.group(1).strip() if match else ""


def _clean_completion(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            for key in ("next_subtask", "subtask", "label", "action", "answer"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except json.JSONDecodeError:
            pass

    for field in _STRUCTURED_SUBTASK_FIELDS:
        value = _extract_structured_field(text, field)
        if value:
            text = value
            break

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    for prefix in ("next subtask:", "subtask:", "answer:", "output:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
            break
    return text.strip().strip("\"'")


def _normalize_subtask(text: str) -> str:
    text = _clean_completion(text).lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _format_completed_subtasks(completed_subtasks: Sequence[str]) -> str:
    if not completed_subtasks:
        return "No completed subtasks yet."
    return "\n".join(
        f"{idx}. {subtask}" for idx, subtask in enumerate(completed_subtasks, start=1)
    )


def _format_plan_progress_hint(
    completed_subtasks: Sequence[str],
    active_subtask: str | None,
    plan_subtasks: Sequence[str],
) -> str:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

    for subtask in completed_subtasks:
        norm = _normalize_subtask(subtask)
        if norm and norm not in seen:
            rows.append((subtask, "done"))
            seen.add(norm)

    active_norm = _normalize_subtask(active_subtask or "")
    if active_subtask and active_norm and active_norm not in seen:
        rows.append((active_subtask, "current"))
        seen.add(active_norm)

    for subtask in plan_subtasks:
        norm = _normalize_subtask(subtask)
        if norm and norm not in seen:
            rows.append((subtask, "pending"))
            seen.add(norm)

    if not rows:
        return "No prior PLAN_PROGRESS estimate yet."
    return "\n".join(
        f"{idx}. {subtask}: {status}"
        for idx, (subtask, status) in enumerate(rows, start=1)
    )


def _inject_completed_subtask_history(
    user_prompt: str,
    completed_subtasks: Sequence[str],
) -> str:
    history_block = (
        "Completed subtasks inferred earlier in this episode "
        "(use as weak progress history and verify against the images):\n"
        f"{_format_completed_subtasks(completed_subtasks)}\n\n"
    )
    marker = "Image order:\n"
    if marker in user_prompt:
        return user_prompt.replace(marker, history_block + marker, 1)
    return user_prompt.rstrip() + "\n\n" + history_block.rstrip()


def _inject_plan_progress_hint(
    user_prompt: str,
    completed_subtasks: Sequence[str],
    active_subtask: str | None,
    plan_subtasks: Sequence[str],
) -> str:
    plan_progress = _format_plan_progress_hint(
        completed_subtasks,
        active_subtask,
        plan_subtasks,
    )
    plan_block = (
        "Estimated PLAN_PROGRESS from previous policy predictions "
        "(weak hint; may be incomplete or wrong, verify against the images):\n"
        "PLAN_PROGRESS:\n"
        f"{plan_progress}\n\n"
    )
    marker = "Image order:\n"
    if marker in user_prompt:
        return user_prompt.replace(marker, plan_block + marker, 1)
    return user_prompt.rstrip() + "\n\n" + plan_block.rstrip()


def _lookup_observation_value(obs: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in obs:
            return obs[key]
    metadata = obs.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            if key in metadata:
                return metadata[key]
    return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        return bool(np.asarray(value).reshape(-1)[0])
    return bool(value)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            value = np.asarray(value).reshape(-1)[0]
        return int(value)
    except (TypeError, ValueError):
        return None


def _ensure_qwen_subtask_import_path() -> None:
    if str(QWEN_SUBTASK_DIR) not in sys.path:
        sys.path.insert(0, str(QWEN_SUBTASK_DIR))


def _qwen_prompt_formatters() -> tuple[Any, Any]:
    _ensure_qwen_subtask_import_path()
    from train_vla_subtask import _format_system_prompt
    from train_vla_subtask import _format_user_prompt

    return _format_system_prompt, _format_user_prompt


def _qwen_plan_prompt_formatters() -> tuple[Any, Any]:
    _ensure_qwen_subtask_import_path()
    from train_vla_subtask import _format_plan_prediction_system_prompt
    from train_vla_subtask import _format_plan_prediction_user_prompt

    return _format_plan_prediction_system_prompt, _format_plan_prediction_user_prompt


def _parse_plan_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*]\s*)?(?:\d+[\).]\s*)?", "", line).strip()
        line = line.strip("\"'")
        if line:
            lines.append(line)
    return lines


def _snap_plan_to_candidates(
    plan_lines: Sequence[str],
    candidate_subtasks: Sequence[str],
) -> list[str]:
    plan: list[str] = []
    seen: set[str] = set()
    for line in plan_lines:
        subtask = _snap_subtask_to_candidates(_clean_completion(line), candidate_subtasks)
        norm = _normalize_subtask(subtask)
        if subtask and norm and norm not in seen:
            plan.append(subtask)
            seen.add(norm)
    return plan


def _snap_subtask_to_candidates(subtask: str, candidate_subtasks: Sequence[str]) -> str:
    if not subtask or not candidate_subtasks:
        return subtask

    try:
        _ensure_qwen_subtask_import_path()
        from semantic_subtask_matcher import snap_to_allowed_sentence

        snapped = snap_to_allowed_sentence(subtask, candidate_subtasks)
    except Exception as exc:
        logger.warning("Semantic subtask matching failed; using raw Qwen output. error=%s", exc)
        return subtask

    if snapped and snapped != subtask:
        logger.info("Semantic snapped Qwen subtask: raw=%r snapped=%r", subtask, snapped)
    return snapped or subtask


def _find_subtask_labels_path() -> Path | None:
    if QWEN_SUBTASK_LABELS_JSON:
        return Path(QWEN_SUBTASK_LABELS_JSON)

    lora_path = Path(QWEN_LORA_DIR)
    for candidate in (
        lora_path / "subtask_labels.json",
        lora_path.parent / "subtask_labels.json",
    ):
        if candidate.exists():
            return candidate
    return None


def _load_subtask_label_map() -> tuple[str | None, dict[str, list[str]]]:
    labels_path = _find_subtask_labels_path()
    if labels_path is None:
        logger.warning("No Qwen subtask label map found; prompts will omit candidate subtasks.")
        return None, {}
    if not labels_path.exists():
        raise FileNotFoundError(f"Qwen subtask label map does not exist: {labels_path}")

    try:
        _ensure_qwen_subtask_import_path()
        from train_vla_subtask import _load_subtask_label_map as load_label_map

        key_type, labels_by_key = load_label_map(labels_path)
    except Exception:
        with labels_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        key_type = str(payload.get("key_type", "")).strip() or None
        raw_labels = payload.get("labels_by_key", {})
        if key_type not in ("task", "instruction") or not isinstance(raw_labels, dict):
            raise
        labels_by_key = {
            str(key): [str(label) for label in labels if str(label).strip()]
            for key, labels in raw_labels.items()
            if isinstance(labels, list)
        }

    logger.info(
        "Loaded Qwen candidate subtask labels from %s (%d %s groups)",
        labels_path,
        len(labels_by_key),
        key_type,
    )
    return key_type, labels_by_key


def _normalize_label_key(text: str) -> str:
    text = (text or "").lower().replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _lookup_candidate_subtasks(
    instruction: str,
    *,
    key_type: str | None,
    labels_by_key: dict[str, list[str]],
) -> tuple[str, ...]:
    if not labels_by_key:
        return ()

    # Current qwen3_vl_subtask training artifacts use key_type="instruction".
    # Keep an exact lookup first, then a normalized fallback for minor formatting drift.
    labels = labels_by_key.get(instruction)
    if labels is not None:
        return tuple(labels)

    normalized_instruction = _normalize_label_key(instruction)
    for key, value in labels_by_key.items():
        if _normalize_label_key(key) == normalized_instruction:
            return tuple(value)

    logger.warning("No candidate subtasks matched instruction=%r with label key_type=%r", instruction, key_type)
    return ()


def _format_subtask_messages(
    instruction: str,
    completed_subtasks: Sequence[str],
    active_subtask: str | None,
    candidate_subtasks: Sequence[str],
    plan_subtasks: Sequence[str],
    history_frame_offsets: Sequence[int],
    images: Sequence[Image.Image],
) -> list[dict[str, Any]]:
    format_system_prompt, format_user_prompt = _qwen_prompt_formatters()
    system_prompt = format_system_prompt(
        instruction=instruction,
        candidate_subtasks=candidate_subtasks,
        has_plan=False,
    )
    user_prompt = format_user_prompt(
        instruction=instruction,
        completed_subtasks=completed_subtasks,
        camera_names=_CAMERA_NAMES,
        include_initial_frames=True,
        history_frame_offsets=history_frame_offsets,
        candidate_subtasks=candidate_subtasks,
        full_plan=(),
    )
    user_prompt = _inject_plan_progress_hint(
        user_prompt,
        completed_subtasks,
        active_subtask,
        plan_subtasks,
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in images),
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def _override_gripper_commands(
    actions_7d: np.ndarray,
    *,
    initially_open: bool,
    raw_history: deque[float],
) -> tuple[np.ndarray, bool, np.ndarray]:
    out = np.asarray(actions_7d, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] < 7:
        return out, initially_open, np.zeros(0, dtype=np.float32)

    is_open = bool(initially_open)
    raw_gripper = out[:, 6].copy()
    for i in range(out.shape[0]):
        gripper_val = float(raw_gripper[i])
        raw_history.append(gripper_val)
        if gripper_val > GRIPPER_FORCE_OPEN_THRESHOLD:
            is_open = True
        elif gripper_val < GRIPPER_FORCE_CLOSE_THRESHOLD:
            is_open = False
        else:
            recent = list(raw_history)
            deltas = np.diff(
                np.asarray(recent[-(GRIPPER_TREND_STEPS + 1):], dtype=np.float32)
            )
            has_open_trend = len(deltas) == GRIPPER_TREND_STEPS and bool(
                np.all(deltas > GRIPPER_MIN_DELTA_PER_STEP)
            )
            has_close_trend = len(deltas) == GRIPPER_TREND_STEPS and bool(
                np.all(deltas < -GRIPPER_MIN_DELTA_PER_STEP)
            )
            if is_open:
                if gripper_val < GRIPPER_CLOSE_SWITCH_THRESHOLD and has_close_trend:
                    is_open = False
            elif gripper_val > GRIPPER_OPEN_SWITCH_THRESHOLD and has_open_trend:
                is_open = True
        out[i, 6] = GRIPPER_OPEN_VALUE if is_open else GRIPPER_CLOSE_VALUE
    return out, is_open, raw_gripper


def _format_debug_action_rows(
    actions_7d: np.ndarray,
    raw_gripper: np.ndarray,
) -> list[list[Any]]:
    rows = np.asarray(actions_7d, dtype=np.float32).tolist()
    if len(raw_gripper) != len(rows):
        return rows
    for step, raw in zip(rows, raw_gripper.tolist()):
        if len(step) >= 7:
            step[6] = f"{float(step[6]):.3f} ({float(raw):.3f})"
    return rows


def _update_position_stall_state(
    current_pos: np.ndarray,
    *,
    anchor_pos: np.ndarray | None,
    stable_action_count: int,
    last_served_action_count: int,
) -> tuple[np.ndarray, int, bool]:
    pos = np.asarray(current_pos, dtype=np.float32).reshape(-1)[:3].copy()
    if pos[2] <= Z_BIAS_MAX_HEIGHT_M:
        return pos, 0, False
    if anchor_pos is None or np.any(np.abs(pos - anchor_pos) > STALL_TOLERANCE_M):
        return pos, 0, False
    next_count = stable_action_count + max(last_served_action_count, 0)
    if next_count >= POSITION_STALL_TRIGGER_ACTIONS:
        return pos, 0, True
    return anchor_pos.copy(), next_count, False


def _euler_xyz_to_quat_xyzw(euler_xyz: np.ndarray) -> np.ndarray:
    euler = np.asarray(euler_xyz, dtype=np.float32)
    if euler.ndim == 1:
        euler = euler.reshape(1, -1)
    half = euler.astype(np.float64) * 0.5
    cx, cy, cz = np.cos(half[:, 0]), np.cos(half[:, 1]), np.cos(half[:, 2])
    sx, sy, sz = np.sin(half[:, 0]), np.sin(half[:, 1]), np.sin(half[:, 2])
    return np.stack(
        [
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        ],
        axis=1,
    ).astype(np.float32)


def _rotate_vectors_by_quat_xyzw(vectors: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    q_vec = quat_xyzw[:, :3]
    q_w = quat_xyzw[:, 3:4]
    t = 2.0 * np.cross(q_vec, vectors)
    return vectors + q_w * t + np.cross(q_vec, t)


def _gripper_forward_vectors(actions_7d: np.ndarray) -> np.ndarray:
    quat = _euler_xyz_to_quat_xyzw(actions_7d[:, 3:6])
    axis = np.broadcast_to(GRIPPER_FORWARD_AXIS, (actions_7d.shape[0], 3)).astype(
        np.float32
    )
    direction = _rotate_vectors_by_quat_xyzw(axis, quat).astype(np.float32)
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    return direction / np.maximum(norm, EPS)


def _apply_downward_z_bias(actions_7d: np.ndarray, *, apply_bias: bool) -> np.ndarray:
    out = np.asarray(actions_7d, dtype=np.float32).copy()
    if not apply_bias or out.ndim != 2 or out.shape[1] < 6:
        return out
    offsets = _gripper_forward_vectors(out) * Z_DOWN_BIAS_M
    scales = np.ones((out.shape[0],), dtype=np.float32)
    z_delta = offsets[:, 2]
    would_cross_floor = (z_delta < -EPS) & (
        out[:, 2] + z_delta < Z_BIAS_MAX_HEIGHT_M
    )
    scales[would_cross_floor] = np.clip(
        (Z_BIAS_MAX_HEIGHT_M - out[would_cross_floor, 2])
        / z_delta[would_cross_floor],
        0.0,
        1.0,
    ).astype(np.float32)
    out[:, :3] += offsets * scales[:, None]
    return out


class SubtaskPolicy(ModelPolicy):
    """ManipArena policy that plans a subtask before action inference."""

    def __init__(
        self,
        checkpoint_path: str,
        control_mode: str,
        action_horizon: int,
        device: str = "cuda:0",
    ):
        if ACTION_END_STEP > 0:
            assert ACTION_END_STEP <= action_horizon
        elif ACTION_OUTPUT_STEPS > 0:
            assert ACTION_START_STEP + ACTION_OUTPUT_STEPS <= action_horizon
        self._subtask_label_key_type, self._subtask_labels_by_key = _load_subtask_label_map()
        self._initial_images: dict[str, np.ndarray] | None = None
        self._recent_image_frames: deque[dict[str, np.ndarray]] = deque(
            maxlen=max(1, QWEN_IMAGE_HISTORY_MAXLEN)
        )
        self._completed_subtasks: list[str] = []
        self._predicted_plan: list[str] = []
        self._plan_prediction_attempted = False
        self._last_subtask: str | None = None
        self._last_subtask_norm: str | None = None
        self._pending_subtask: str | None = None
        self._pending_subtask_norm: str | None = None
        self._pending_subtask_count = 0
        self._last_instruction: str | None = None
        self._last_episode_marker: str | None = None
        self._last_frame_marker: int | None = None
        self._left_gripper_open = False
        self._right_gripper_open = False
        self._left_gripper_history = deque(maxlen=GRIPPER_TREND_STEPS + 1)
        self._right_gripper_history = deque(maxlen=GRIPPER_TREND_STEPS + 1)
        self._left_stall_anchor: np.ndarray | None = None
        self._right_stall_anchor: np.ndarray | None = None
        self._left_stall_action_count = 0
        self._right_stall_action_count = 0
        self._left_apply_down_bias = False
        self._right_apply_down_bias = False
        self._last_served_action_count = 0
        self._last_plan_progress_hint = "No prior PLAN_PROGRESS estimate yet."
        super().__init__(
            checkpoint_path=checkpoint_path,
            control_mode=control_mode,
            action_horizon=action_horizon,
            device=device,
        )

    def load_model(self, checkpoint_path: str, device: str) -> _CascadeModels:
        qwen_model, qwen_tokenizer = self._load_subtask_model(device)
        if (
            SMALL_VLM_USE_REMOTE_ACTION_SERVER
            and SMALL_VLM_ACTION_SERVER_HOST
            and SMALL_VLM_ACTION_SERVER_PORT > 0
        ):
            action_policy = _RemoteActionPolicyClient(
                SMALL_VLM_ACTION_SERVER_HOST,
                SMALL_VLM_ACTION_SERVER_PORT,
            )
            logger.info(
                "Using remote small_vlm action server: ws://%s:%d",
                SMALL_VLM_ACTION_SERVER_HOST,
                SMALL_VLM_ACTION_SERVER_PORT,
            )
        else:
            action_policy = self._load_action_policy(checkpoint_path, device)
        _reset_socket_default_timeout("subtask policy model loading")
        return _CascadeModels(qwen_model, qwen_tokenizer, action_policy)

    def _load_subtask_model(self, device: str) -> tuple[Any, Any]:
        _ensure_qwen_subtask_import_path()
        from peft import PeftModel
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            QWEN_BASE_MODEL,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        model = PeftModel.from_pretrained(model, QWEN_LORA_DIR)
        FastVisionModel.for_inference(model)
        model.eval()
        logger.info("Subtask model loaded: base=%s lora=%s device=%s", QWEN_BASE_MODEL, QWEN_LORA_DIR, device)
        return model, tokenizer

    def _load_action_policy(self, checkpoint_path: str, device: str) -> Any:
        small_vlm_src = str(SMALL_VLM_ROOT / "src")
        if small_vlm_src not in sys.path:
            sys.path.insert(0, small_vlm_src)

        importlib.invalidate_caches()
        from openpi.policies import policy_config as pc
        from openpi.training import config as train_config

        cfg = train_config.get_config(SMALL_VLM_CONFIG_NAME)
        policy = pc.create_trained_policy(
            cfg,
            checkpoint_path,
            default_prompt=None,
            pytorch_device=device,
        )
        logger.info("small_vlm action policy loaded: config=%s", SMALL_VLM_CONFIG_NAME)
        return policy

    @property
    def metadata(self) -> Dict[str, Any]:
        # The stock server does not call policy.reset() on a new websocket
        # connection. It always reads metadata before serving observations, so
        # use that hook to clear per-episode Qwen/action history on reconnect.
        self.reset()
        return super().metadata

    def _reset_episode_state(self, reason: str, *, reset_action_policy: bool) -> None:
        self._initial_images = None
        self._recent_image_frames.clear()
        self._completed_subtasks.clear()
        self._predicted_plan.clear()
        self._plan_prediction_attempted = False
        self._last_subtask = None
        self._last_subtask_norm = None
        self._pending_subtask = None
        self._pending_subtask_norm = None
        self._pending_subtask_count = 0
        self._left_gripper_open = False
        self._right_gripper_open = False
        self._left_gripper_history.clear()
        self._right_gripper_history.clear()
        self._left_stall_anchor = None
        self._right_stall_anchor = None
        self._left_stall_action_count = 0
        self._right_stall_action_count = 0
        self._left_apply_down_bias = False
        self._right_apply_down_bias = False
        self._last_served_action_count = 0
        self._last_plan_progress_hint = "No prior PLAN_PROGRESS estimate yet."
        logger.info("SubtaskPolicy episode state reset (%s)", reason)
        action_policy = getattr(getattr(self, "model", None), "action_policy", None)
        if reset_action_policy and hasattr(action_policy, "reset"):
            action_policy.reset()

    def _maybe_reset_episode_state(
        self,
        obs: Dict[str, Any],
        *,
        instruction: str,
    ) -> None:
        reason: str | None = None

        reset_flag = _lookup_observation_value(obs, _RESET_FLAG_KEYS)
        if reset_flag is not None and _coerce_bool(reset_flag):
            reason = "observation reset flag"

        episode_marker_raw = _lookup_observation_value(obs, _EPISODE_MARKER_KEYS)
        episode_marker = None if episode_marker_raw is None else str(episode_marker_raw)
        if (
            reason is None
            and episode_marker is not None
            and self._last_episode_marker is not None
            and episode_marker != self._last_episode_marker
        ):
            reason = f"episode marker changed {self._last_episode_marker!r}->{episode_marker!r}"

        frame_marker = _coerce_int(_lookup_observation_value(obs, _FRAME_MARKER_KEYS))
        if (
            reason is None
            and frame_marker is not None
            and self._last_frame_marker is not None
            and frame_marker < self._last_frame_marker
        ):
            reason = f"frame marker rewound {self._last_frame_marker}->{frame_marker}"

        if (
            reason is None
            and self._last_instruction is not None
            and instruction
            and instruction != self._last_instruction
        ):
            reason = f"instruction changed {self._last_instruction!r}->{instruction!r}"

        if reason is not None:
            self._reset_episode_state(reason, reset_action_policy=True)

        self._last_instruction = instruction
        if episode_marker is not None:
            self._last_episode_marker = episode_marker
        if frame_marker is not None:
            self._last_frame_marker = frame_marker

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        state_dict = obs.get("state", {})
        if self.control_mode == "joints":
            f1 = np.asarray(
                state_dict.get("follow1_joints", state_dict.get("follow1_pos", np.zeros(7))),
                dtype=np.float32,
            )[:7]
            f2 = np.asarray(
                state_dict.get("follow2_joints", state_dict.get("follow2_pos", np.zeros(7))),
                dtype=np.float32,
            )[:7]
        else:
            f1 = np.asarray(state_dict.get("follow1_pos", np.zeros(7)), dtype=np.float32)[:7]
            f2 = np.asarray(state_dict.get("follow2_pos", np.zeros(7)), dtype=np.float32)[:7]

        instruction = obs.get("instruction", "") or ""
        state = np.concatenate([f1[:6], f1[6:7], f2[:6], f2[6:7]]).astype(np.float32)
        self._maybe_reset_episode_state(obs, instruction=instruction)

        if self.control_mode != "joints":
            self._left_stall_anchor, self._left_stall_action_count, self._left_apply_down_bias = _update_position_stall_state(
                f1[:3],
                anchor_pos=self._left_stall_anchor,
                stable_action_count=self._left_stall_action_count,
                last_served_action_count=self._last_served_action_count,
            )
            self._right_stall_anchor, self._right_stall_action_count, self._right_apply_down_bias = _update_position_stall_state(
                f2[:3],
                anchor_pos=self._right_stall_anchor,
                stable_action_count=self._right_stall_action_count,
                last_served_action_count=self._last_served_action_count,
            )

        views = obs.get("views", {})
        missing_views = [client_key for client_key in _CAM_MAP if client_key not in views]
        if missing_views:
            raise KeyError(f"Missing ManipArena camera views: {missing_views}")
        current_images = {
            key: _decode_image(views[client_key])
            for client_key, key in _CAM_MAP.items()
        }
        self._recent_image_frames.append({key: value.copy() for key, value in current_images.items()})
        if self._initial_images is None:
            self._initial_images = {key: value.copy() for key, value in current_images.items()}

        return {
            "instruction": instruction,
            "state": state,
            "current_images": current_images,
        }

    def _predict_episode_plan(
        self,
        instruction: str,
        candidate_subtasks: Sequence[str],
    ) -> list[str]:
        if self._initial_images is None:
            return []

        import torch

        format_system_prompt, format_user_prompt = _qwen_plan_prompt_formatters()
        images = [
            _resize_pil(_to_pil(self._initial_images[key]), QWEN_MAX_IMAGE_SIDE)
            for key in _CAM_MAP.values()
        ]
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": format_system_prompt()}],
            },
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in images),
                    {
                        "type": "text",
                        "text": format_user_prompt(
                            instruction=instruction,
                            camera_names=_CAMERA_NAMES,
                            candidate_subtasks=candidate_subtasks,
                        ),
                    },
                ],
            },
        ]
        input_text = self.model.subtask_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = self.model.subtask_tokenizer(
            text=input_text,
            images=images,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.subtask_model.generate(
                **inputs,
                max_new_tokens=QWEN_PLAN_MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        prompt_len = inputs["input_ids"].shape[1]
        completion = self.model.subtask_tokenizer.batch_decode(
            outputs[:, prompt_len:],
            skip_special_tokens=True,
        )[0]
        plan = _snap_plan_to_candidates(
            _parse_plan_lines(completion),
            candidate_subtasks,
        )
        logger.info("qwen_predicted_plan raw=%r parsed=%s", completion, plan)
        return plan

    def _predict_subtask(self, model_input: Dict[str, Any]) -> str:
        import torch

        history_frames = list(self._recent_image_frames)
        if not history_frames:
            history_frames = [model_input["current_images"]]

        def frame_for_offset(offset: int) -> dict[str, np.ndarray]:
            current_idx = len(history_frames) - 1
            target_idx = current_idx - _history_offset_to_observation_steps(offset)
            if target_idx >= 0:
                return history_frames[target_idx]

            available_offsets = [
                smaller_offset
                for smaller_offset in QWEN_HISTORY_FRAME_OFFSETS
                if (
                    smaller_offset < offset
                    and current_idx - _history_offset_to_observation_steps(smaller_offset) >= 0
                )
            ]
            if available_offsets:
                fallback_offset = max(available_offsets)
                return history_frames[current_idx - _history_offset_to_observation_steps(fallback_offset)]
            return history_frames[current_idx]

        temporal_frames = [frame_for_offset(offset) for offset in QWEN_HISTORY_FRAME_OFFSETS]
        temporal_frames.append(history_frames[-1])

        initial_images = self._initial_images or history_frames[0]
        images = [
            _resize_pil(_to_pil(initial_images[key]), QWEN_MAX_IMAGE_SIDE)
            for key in _CAM_MAP.values()
        ]
        for frame_images in temporal_frames:
            images.extend(
                _resize_pil(_to_pil(frame_images[key]), QWEN_MAX_IMAGE_SIDE)
                for key in _CAM_MAP.values()
            )

        candidate_subtasks = _lookup_candidate_subtasks(
            model_input["instruction"],
            key_type=self._subtask_label_key_type,
            labels_by_key=self._subtask_labels_by_key,
        )
        if not self._plan_prediction_attempted:
            self._predicted_plan = self._predict_episode_plan(
                model_input["instruction"],
                candidate_subtasks,
            )
            self._plan_prediction_attempted = True
        plan_subtasks = self._predicted_plan or list(candidate_subtasks)
        self._last_plan_progress_hint = _format_plan_progress_hint(
            self._completed_subtasks,
            self._last_subtask,
            plan_subtasks,
        )
        logger.info("qwen_input_plan_progress:\nPLAN_PROGRESS:\n%s", self._last_plan_progress_hint)
        messages = _format_subtask_messages(
            instruction=model_input["instruction"],
            completed_subtasks=self._completed_subtasks,
            active_subtask=self._last_subtask,
            candidate_subtasks=candidate_subtasks,
            plan_subtasks=plan_subtasks,
            history_frame_offsets=QWEN_HISTORY_FRAME_OFFSETS,
            images=images,
        )
        input_text = self.model.subtask_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = self.model.subtask_tokenizer(
            text=input_text,
            images=images,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.subtask_model.generate(
                **inputs,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = outputs[:, prompt_len:]
        completion = self.model.subtask_tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
        )[0]
        history_evidence = _extract_structured_section(completion, "HISTORY_EVIDENCE")
        visual_evidence = _extract_structured_section(completion, "VISUAL_EVIDENCE")
        logger.info(
            "qwen_evidence:\nHISTORY_EVIDENCE:\n%s\nVISUAL_EVIDENCE:\n%s",
            history_evidence or "(missing)",
            visual_evidence or "(missing)",
        )
        cleaned_subtask = _clean_completion(completion)
        subtask = _snap_subtask_to_candidates(
            cleaned_subtask,
            candidate_subtasks,
        )
        logger.info(
            "qwen_subtask raw=%r cleaned=%r predicted=%r",
            completion,
            cleaned_subtask,
            subtask,
        )

        if subtask:
            subtask_norm = _normalize_subtask(subtask)
            if not self._last_subtask:
                self._last_subtask = subtask
                self._last_subtask_norm = subtask_norm
                self._pending_subtask = None
                self._pending_subtask_norm = None
                self._pending_subtask_count = 0
            elif subtask_norm == self._last_subtask_norm:
                self._pending_subtask = None
                self._pending_subtask_norm = None
                self._pending_subtask_count = 0
            elif subtask_norm:
                if subtask_norm == self._pending_subtask_norm:
                    self._pending_subtask_count += 1
                else:
                    self._pending_subtask = subtask
                    self._pending_subtask_norm = subtask_norm
                    self._pending_subtask_count = 1

                if self._pending_subtask_count >= max(1, SUBTASK_SWITCH_STABLE_STEPS):
                    if (
                        self._last_subtask
                        and (
                            not self._completed_subtasks
                            or _normalize_subtask(self._completed_subtasks[-1]) != self._last_subtask_norm
                        )
                    ):
                        self._completed_subtasks.append(self._last_subtask)
                        if MAX_COMPLETED_SUBTASKS > 0:
                            self._completed_subtasks = self._completed_subtasks[-MAX_COMPLETED_SUBTASKS:]
                    self._last_subtask = self._pending_subtask
                    self._last_subtask_norm = self._pending_subtask_norm
                    self._pending_subtask = None
                    self._pending_subtask_norm = None
                    self._pending_subtask_count = 0
        stable_subtask = self._last_subtask or subtask or model_input["instruction"]
        logger.info(
            "qwen_stable_subtask to_expert=%r pending=%r pending_count=%d",
            stable_subtask,
            self._pending_subtask,
            self._pending_subtask_count,
        )
        return stable_subtask

    def _make_action_input(self, model_input: Dict[str, Any], subtask: str) -> Dict[str, Any]:
        prompt = SMALL_VLM_PROMPT_TEMPLATE.format(
            instruction=model_input["instruction"],
            subtask=subtask,
        )
        action_input: Dict[str, Any] = {
            "observation.state": model_input["state"],
            "prompt": prompt,
            "subtask": subtask,
            "instruction": model_input["instruction"],
        }
        for key, image in model_input["current_images"].items():
            action_input[key] = image
        return action_input

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        subtask = self._predict_subtask(model_input)
        action_input = self._make_action_input(model_input, subtask)
        result = self.model.action_policy.infer(action_input)
        actions = np.asarray(result["actions"], dtype=np.float32)
        logger.info(
            "subtask=%r predicted_plan=%s action_shape=%s",
            subtask,
            self._predicted_plan,
            actions.shape,
        )
        return {"actions": actions, "subtask": subtask}

    def reset(self):
        self._last_instruction = None
        self._last_episode_marker = None
        self._last_frame_marker = None
        self._reset_episode_state("external reset", reset_action_policy=True)

    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        actions = np.asarray(model_output["actions"], dtype=np.float32)
        start_idx = max(0, ACTION_START_STEP)
        if ACTION_END_STEP > 0:
            end_idx = min(ACTION_END_STEP, actions.shape[0])
        elif ACTION_OUTPUT_STEPS > 0:
            end_idx = min(start_idx + ACTION_OUTPUT_STEPS, actions.shape[0])
        else:
            end_idx = min(actions.shape[0], start_idx + max(2, int(ACTION_END_RATIO * actions.shape[0])))
        if end_idx <= start_idx:
            start_idx = 0
            end_idx = min(ACTION_OUTPUT_STEPS, actions.shape[0]) if ACTION_OUTPUT_STEPS > 0 else max(2, int(ACTION_END_RATIO * actions.shape[0]))
        actions = actions[start_idx:end_idx]
        self._last_served_action_count = int(actions.shape[0])

        left = np.asarray(actions[:, :7], dtype=np.float32)
        right = np.asarray(actions[:, 7:14], dtype=np.float32)
        if self.control_mode != "joints":
            left = _apply_downward_z_bias(left, apply_bias=self._left_apply_down_bias)
            right = _apply_downward_z_bias(right, apply_bias=self._right_apply_down_bias)
        self._left_apply_down_bias = False
        self._right_apply_down_bias = False
        left, self._left_gripper_open, left_raw_gripper = _override_gripper_commands(
            left,
            initially_open=self._left_gripper_open,
            raw_history=self._left_gripper_history,
        )
        right, self._right_gripper_open, right_raw_gripper = _override_gripper_commands(
            right,
            initially_open=self._right_gripper_open,
            raw_history=self._right_gripper_history,
        )
        debug_left = _format_debug_action_rows(left, left_raw_gripper)
        debug_right = _format_debug_action_rows(right, right_raw_gripper)

        if self.control_mode == "joints":
            payload = {
                "follow1_joints": left.tolist(),
                "follow2_joints": right.tolist(),
                "follow1_pos": left.tolist(),
                "follow2_pos": right.tolist(),
                "subtask": model_output["subtask"],
            }
            debug_payload = {
                "follow1_joints": debug_left,
                "follow2_joints": debug_right,
                "follow1_pos": debug_left,
                "follow2_pos": debug_right,
            }
        else:
            payload = {
                "follow1_pos": left.tolist(),
                "follow2_pos": right.tolist(),
                "subtask": model_output["subtask"],
            }
            debug_payload = {"follow1_pos": debug_left, "follow2_pos": debug_right}

        logger.debug(
            "SubtaskPolicy payload subtask=%r:%s",
            model_output["subtask"],
            format_action_payload_for_debug(debug_payload, max_steps=DEBUG_OUTPUT_STEPS),
        )
        return payload


MyPolicy = SubtaskPolicy
