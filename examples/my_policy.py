
#!/usr/bin/env python3
"""OpenPI policy example — Bimanual 14D (end-effector).

Prerequisites:
    pip install openpi    # or install from the openpi repo

Usage:
    python serve.py \
        --checkpoint /path/to/openpi/checkpoints/step_10000 \
        --control-mode end_pose \
        --port 8000

    Replace `MyPolicy` import in serve.py or copy this file to examples/my_policy.py.

Set OPENPI_CONFIG_NAME below to match your trained OpenPI config.
"""

from __future__ import annotations

import base64
from collections import deque
from datetime import datetime
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

from maniparena.debug import format_action_payload_for_debug
from maniparena.policy import ModelPolicy

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────

OPENPI_CONFIG_NAME = "pi05_maniparena_ee"  # set this to match your trained OpenPI config
DEFAULT_PROMPT = None
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
VLM_API_DIR = WORKSPACE_ROOT / "vlm-api"
VLM_TRACKER_PROVIDER = os.environ.get("VLM_TRACKER_PROVIDER", "google")
VLM_TRACKER_MODEL = os.environ.get("VLM_TRACKER_MODEL", "gemini-3-flash-preview")
VLM_TRACKER_SKILLS_JSON = os.environ.get(
    "VLM_TRACKER_SKILLS_JSON",
    "/mnt/data/haoliang/maniparena/skills/robot_ai_skills.json",
)
VLM_TRACKER_SAVE_DIR = os.environ.get("VLM_TRACKER_SAVE_DIR", "").strip()
VLM_TRACKER_ENABLED = os.environ.get(
    "VLM_TRACKER_ENABLED",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
VLM_TRACKER_TARGET_TASKS = {
    "real/semantic_reasoning/classify_items_as_shape",
    "real/semantic_reasoning/press_button_in_order",
}
VLM_POLICY_SUBTASK_LOG_ENABLED = os.environ.get(
    "VLM_POLICY_SUBTASK_LOG_ENABLED",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
VLM_POLICY_SUBTASK_LOG_DIR = os.environ.get(
    "VLM_POLICY_SUBTASK_LOG_DIR",
    str(WORKSPACE_ROOT / "vlm-api" / "policy_subtask_outputs"),
).strip()
ACTION_END_RATIO = 0.8  # keep first 80% of predicted actions
ACTION_START_STEP = 0
ACTION_END_STEP = 20  # exclusive; sends actions in [0, 20)
ACTION_OUTPUT_STEPS = 20  # hard cap on transmitted action steps
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

assert ACTION_START_STEP >= 0
assert ACTION_END_STEP == 0 or ACTION_END_STEP > ACTION_START_STEP
assert (
    ACTION_OUTPUT_STEPS <= 0
    or ACTION_END_STEP <= 0
    or ACTION_END_STEP - ACTION_START_STEP == ACTION_OUTPUT_STEPS
)


# ── OpenPI Camera key mapping ────────────────────────────────────

_CAM_MAP = {
    "camera_front": "observation.images.faceImg",
    "camera_left": "observation.images.leftImg",
    "camera_right": "observation.images.rightImg",
}

_TASK_ALIASES = {
    "real/semantic_reasoning/pick_fruits_into_basket": (
        "pick fruits into basket",
        "put the fruits into the basket",
        "fruits basket",
        "pick_fruits_into_basket",
    ),
    "real/semantic_reasoning/press_button_in_order": (
        "press the buttons in order",
        "press buttons in order",
        "press_button_in_order",
        "button order",
        "buttons order",
    ),
    "real/execution_reasoning/put_blocks_to_color": (
        "move the block onto the designated colored square",
        "put blocks to color",
        "put_blocks_to_color",
        "colored square",
        "color block",
    ),
    "real/execution_reasoning/put_ring_onto_rod": (
        "put the ring onto the vertical pole",
        "put ring onto rod",
        "put_ring_onto_rod",
        "vertical pole",
        "vertical rod",
    ),
    "real/execution_reasoning/put_spoon_to_bowl": (
        "put the spoon into the bowl",
        "put spoon to bowl",
        "put_spoon_to_bowl",
        "spoon bowl",
    ),
    "real/semantic_reasoning/classify_items_as_shape": (
        "classify by object shape",
        "classify items as shape",
        "classify_items_as_shape",
        "object shape",
        "sphere cylinder cube",
    ),
}

_CANONICAL_INSTRUCTIONS = {
    "real/semantic_reasoning/pick_fruits_into_basket": "Put the fruits into the basket.",
    "real/semantic_reasoning/press_button_in_order": "Press the buttons in order",
    "real/execution_reasoning/put_blocks_to_color": "Move the block onto the designated colored square",
    "real/execution_reasoning/put_ring_onto_rod": "Put the ring onto the vertical pole",
    "real/execution_reasoning/put_spoon_to_bowl": "Put the spoon into the bowl",
    "real/semantic_reasoning/classify_items_as_shape": "Classify by object shape",
}

_RESET_FLAG_KEYS = (
    "reset",
    "episode_reset",
    "new_episode",
    "episode_start",
    "is_first",
    "first_step",
)
_EPISODE_MARKER_KEYS = (
    "episode_id",
    "episode_index",
    "episode",
    "reset_id",
    "env_reset_id",
)
_FRAME_MARKER_KEYS = (
    "frame_index",
    "frame",
    "step",
    "step_count",
    "timestep",
    "time_step",
)


def _decode_image(v: Any) -> np.ndarray:
    """base64 JPEG string or numpy array → RGB uint8 ndarray."""
    if isinstance(v, np.ndarray):
        return v.astype(np.uint8) if v.dtype != np.uint8 else v
    raw = base64.b64decode(v) if isinstance(v, str) else bytes(v)
    if cv2 is not None:
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Install either opencv-python or pillow to decode remote camera images."
        ) from exc
    return np.asarray(Image.open(BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _normalize_task_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _safe_path_component(value: Any, default: str = "unknown") -> str:
    text = str(value if value is not None else default).strip() or default
    text = text.replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return text.strip("_") or default


def _episode_log_component(value: Any) -> str:
    if value is None:
        return "episode_unknown"
    try:
        return f"episode_{int(value):06d}"
    except (TypeError, ValueError):
        return _safe_path_component(f"episode_{value}")


def _image_debug_summary(image: np.ndarray | None) -> dict[str, Any] | None:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.size == 0:
        return {"shape": list(arr.shape), "dtype": str(arr.dtype), "empty": True}
    return {
        "shape": [int(item) for item in arr.shape],
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _extract_text_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        return str(value.flat[0]) if value.size else ""
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        for key in ("task", "instruction", "name", "text"):
            text = _extract_text_field(value.get(key))
            if text:
                return text
        return ""
    return str(value).strip()


def _extract_task_text(obs: Dict[str, Any]) -> str:
    for key in (
        "task",
        "task_name",
        "task_instruction",
        "instruction",
        "prompt",
        "INSTRUCTION",
        "PROMPT",
    ):
        text = _extract_text_field(obs.get(key))
        if text:
            return text
    meta = obs.get("meta") or obs.get("metadata")
    if isinstance(meta, dict):
        for key in ("task", "task_name", "instruction"):
            text = _extract_text_field(meta.get(key))
            if text:
                return text
    return ""


def _match_task_relpath(task_text: str) -> str | None:
    normalized = _normalize_task_text(task_text)
    if not normalized:
        return None
    best_relpath = None
    best_score = 0
    for relpath, aliases in _TASK_ALIASES.items():
        rel_norm = _normalize_task_text(relpath)
        candidates = [_normalize_task_text(alias) for alias in aliases] + [rel_norm]
        score = 0
        for candidate in candidates:
            if not candidate:
                continue
            if candidate == normalized:
                score = max(score, 1000 + len(candidate))
            elif candidate in normalized or normalized in candidate:
                score = max(score, 500 + min(len(candidate), len(normalized)))
            else:
                candidate_tokens = set(candidate.split())
                normalized_tokens = set(normalized.split())
                overlap = len(candidate_tokens & normalized_tokens)
                if overlap:
                    score = max(score, overlap * 10)
        if score > best_score:
            best_score = score
            best_relpath = relpath
    return best_relpath if best_score >= 20 else None


def _extract_episode_marker(obs: Dict[str, Any]) -> Any:
    for key in _EPISODE_MARKER_KEYS:
        if key in obs:
            return obs.get(key)
    meta = obs.get("meta") or obs.get("metadata")
    if isinstance(meta, dict):
        for key in _EPISODE_MARKER_KEYS:
            if key in meta:
                return meta.get(key)
    return None


def _extract_frame_index(obs: Dict[str, Any]) -> int | None:
    for key in _FRAME_MARKER_KEYS:
        if key in obs:
            try:
                return int(obs.get(key))
            except (TypeError, ValueError):
                pass
    meta = obs.get("meta") or obs.get("metadata")
    if isinstance(meta, dict):
        for key in _FRAME_MARKER_KEYS:
            if key in meta:
                try:
                    return int(meta.get(key))
                except (TypeError, ValueError):
                    pass
    return None


def _obs_requests_reset(obs: Dict[str, Any]) -> bool:
    for key in _RESET_FLAG_KEYS:
        if key in obs and bool(obs.get(key)):
            return True
    meta = obs.get("meta") or obs.get("metadata")
    if isinstance(meta, dict):
        return any(bool(meta.get(key)) for key in _RESET_FLAG_KEYS)
    return False


def _as_float_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, dict) and "data" in value and "shape" in value:
        arr = np.asarray(value["data"], dtype=np.float32).reshape(tuple(value["shape"]))
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=np.float32)
    else:
        return None
    try:
        return np.asarray(arr, dtype=np.float32).reshape(-1)
    except Exception:
        return None


def _first_float_vector(*values: Any) -> np.ndarray | None:
    for value in values:
        arr = _as_float_vector(value)
        if arr is not None and arr.size:
            return arr
    return None


def _extract_full_robot_state(obs: Dict[str, Any], state_dict: Dict[str, Any]) -> np.ndarray | None:
    raw_state = obs.get("state")
    candidates = [
        raw_state if not isinstance(raw_state, dict) else None,
        obs.get("observation.state"),
        obs.get("state_vector"),
        obs.get("robot_state"),
        state_dict.get("observation.state"),
        state_dict.get("state_vector"),
        state_dict.get("robot_state"),
        state_dict.get("full_state"),
        state_dict.get("raw_state"),
    ]
    arr = _first_float_vector(*candidates)
    if arr is not None and arr.size >= 14:
        return arr
    return None


def _xyz_or_none(values: np.ndarray, start: int) -> list[float] | None:
    if values.size < start + 3:
        return None
    return [float(item) for item in values[start : start + 3]]


def _build_tracker_gripper_state(
    obs: Dict[str, Any],
    state_dict: Dict[str, Any],
    f1: np.ndarray,
    f2: np.ndarray,
    *,
    frame_index: int | None,
) -> Dict[str, Any] | None:
    full_state = _extract_full_robot_state(obs, state_dict)
    if full_state is not None and full_state.size >= 56:
        return {
            "frame_index": frame_index,
            "id6": float(full_state[6]),
            "id13": float(full_state[13]),
            "id34": float(full_state[34]),
            "id55": float(full_state[55]),
            "eef_xyz_id012": _xyz_or_none(full_state, 0),
            "eef_xyz_id789": _xyz_or_none(full_state, 7),
            "source": "full_observation.state",
            "state_length": int(full_state.size),
        }

    state: Dict[str, Any] = {
        "frame_index": frame_index,
        "source": "maniparena_policy_obs",
    }
    if f1.size >= 7:
        state["eef_xyz_id012"] = _xyz_or_none(f1, 0)
        state["id6"] = float(f1[6])
    if f2.size >= 7:
        state["eef_xyz_id789"] = _xyz_or_none(f2, 0)
        state["id13"] = float(f2[6])

    if full_state is not None:
        if full_state.size > 34:
            state["id34"] = float(full_state[34])
        if full_state.size > 55:
            state["id55"] = float(full_state[55])

    return state if any(key in state for key in ("id6", "id13", "id34", "id55")) else None


def _override_gripper_commands(
    actions_7d: np.ndarray,
    *,
    initially_open: bool,
    raw_history: deque[float],
) -> tuple[np.ndarray, bool, np.ndarray]:
    """Apply stateful gripper switching using threshold + trend checks."""
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
                np.asarray(
                    recent[-(GRIPPER_TREND_STEPS + 1):],
                    dtype=np.float32,
                )
            )
            has_open_trend = (
                len(deltas) == GRIPPER_TREND_STEPS
                and bool(
                    np.all(deltas > GRIPPER_MIN_DELTA_PER_STEP)
                )
            )
            has_close_trend = (
                len(deltas) == GRIPPER_TREND_STEPS
                and bool(
                    np.all(deltas < -GRIPPER_MIN_DELTA_PER_STEP)
                )
            )
            if is_open:
                if (
                    gripper_val < GRIPPER_CLOSE_SWITCH_THRESHOLD
                    and has_close_trend
                ):
                    is_open = False
            else:
                if (
                    gripper_val > GRIPPER_OPEN_SWITCH_THRESHOLD
                    and has_open_trend
                ):
                    is_open = True
        out[i, 6] = (
            GRIPPER_OPEN_VALUE if is_open
            else GRIPPER_CLOSE_VALUE
        )
    return out, is_open, raw_gripper


def _format_debug_action_rows(
    actions_7d: np.ndarray,
    raw_gripper: np.ndarray,
) -> list[list[Any]]:
    """Format actions for debug logs, preserving sent and raw gripper values."""
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
    """Track whether the end-effector has hovered in a small xyz box."""
    pos = np.asarray(current_pos, dtype=np.float32).reshape(-1)[:3].copy()
    if pos[2] <= Z_BIAS_MAX_HEIGHT_M:
        return pos, 0, False
    if anchor_pos is None or np.any(
        np.abs(pos - anchor_pos) > STALL_TOLERANCE_M
    ):
        return pos, 0, False

    next_count = stable_action_count + max(
        last_served_action_count, 0,
    )
    if next_count >= POSITION_STALL_TRIGGER_ACTIONS:
        return pos, 0, True
    return anchor_pos.copy(), next_count, False


def _euler_xyz_to_quat_xyzw(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles to XYZW quaternions."""
    euler = np.asarray(euler_xyz, dtype=np.float32)
    if euler.ndim == 1:
        euler = euler.reshape(1, -1)
    if euler.shape[-1] != 3:
        raise ValueError(
            f"Expected euler shape (*, 3), got {euler.shape}"
        )

    half = euler.astype(np.float64) * 0.5
    cx = np.cos(half[:, 0])
    cy = np.cos(half[:, 1])
    cz = np.cos(half[:, 2])
    sx = np.sin(half[:, 0])
    sy = np.sin(half[:, 1])
    sz = np.sin(half[:, 2])

    return np.stack(
        [
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        ],
        axis=1,
    ).astype(np.float32)


def _rotate_vectors_by_quat_xyzw(
    vectors: np.ndarray,
    quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Rotate vectors by XYZW quaternions."""
    q_vec = quat_xyzw[:, :3]
    q_w = quat_xyzw[:, 3:4]
    t = 2.0 * np.cross(q_vec, vectors)
    return vectors + q_w * t + np.cross(q_vec, t)


def _gripper_forward_vectors(actions_7d: np.ndarray) -> np.ndarray:
    """Return each action's gripper-forward direction in action xyz space."""
    quat = _euler_xyz_to_quat_xyzw(actions_7d[:, 3:6])
    axis = np.broadcast_to(
        GRIPPER_FORWARD_AXIS,
        (actions_7d.shape[0], 3),
    ).astype(np.float32)
    direction = _rotate_vectors_by_quat_xyzw(axis, quat).astype(np.float32)
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    return direction / np.maximum(norm, EPS)


def _apply_downward_z_bias(
    actions_7d: np.ndarray,
    *,
    apply_bias: bool,
) -> np.ndarray:
    """Nudge along gripper direction, clipping the z drop to the floor."""
    out = np.asarray(actions_7d, dtype=np.float32).copy()
    if (
        not apply_bias
        or out.ndim != 2
        or out.shape[1] < 6
    ):
        return out

    offsets = _gripper_forward_vectors(out) * Z_DOWN_BIAS_M
    scales = np.ones((out.shape[0],), dtype=np.float32)
    z_delta = offsets[:, 2]
    would_cross_floor = (
        (z_delta < -EPS)
        & (out[:, 2] + z_delta < Z_BIAS_MAX_HEIGHT_M)
    )
    scales[would_cross_floor] = np.clip(
        (
            Z_BIAS_MAX_HEIGHT_M
            - out[would_cross_floor, 2]
        )
        / z_delta[would_cross_floor],
        0.0,
        1.0,
    ).astype(np.float32)
    out[:, :3] += offsets * scales[:, None]
    return out


# ── Policy ────────────────────────────────────────────────────────


class MyPolicy(ModelPolicy):
    """OpenPI policy adapter for ManipArena bimanual (14D EE)."""

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
        self._left_gripper_open = False
        self._right_gripper_open = False
        self._left_gripper_history = deque(
            maxlen=GRIPPER_TREND_STEPS + 1,
        )
        self._right_gripper_history = deque(
            maxlen=GRIPPER_TREND_STEPS + 1,
        )
        self._left_stall_anchor: np.ndarray | None = None
        self._right_stall_anchor: np.ndarray | None = None
        self._left_stall_action_count = 0
        self._right_stall_action_count = 0
        self._left_apply_down_bias = False
        self._right_apply_down_bias = False
        self._last_served_action_count = 0
        self._vlm_tracker: Any | None = None
        self._vlm_tracker_task_relpath: str | None = None
        self._vlm_tracker_episode_marker: Any = None
        self._vlm_tracker_last_frame_index: int | None = None
        self._vlm_tracker_prev_image: np.ndarray | None = None
        self._vlm_tracker_prev_gripper_state: Any | None = None
        self._vlm_tracker_observation_count = 0
        self._vlm_tracker_last_subtask: str | None = None
        self._vlm_tracker_last_status: str | None = None
        self._vlm_tracker_last_error: Any | None = None
        self._vlm_tracker_last_result_current_subtask: str | None = None
        self._vlm_tracker_last_result_plan: Any | None = None
        self._vlm_tracker_initial_plan_captured = False
        self._vlm_tracker_initial_plan: list[str] | None = None
        self._vlm_tracker_initial_current_subtask: str | None = None
        self._vlm_tracker_initial_frame_index: int | None = None
        self._vlm_tracker_initial_visible_instruction_text: str | None = None
        self._vlm_tracker_initial_mode: str | None = None
        self._vlm_policy_subtask_log_run_dir: Path | None = None
        self._vlm_policy_subtask_log_path: Path | None = None
        self._vlm_policy_subtask_log_count = 0
        self._vlm_policy_subtask_log_announced = False
        super().__init__(
            checkpoint_path=checkpoint_path,
            control_mode=control_mode,
            action_horizon=action_horizon,
            device=device,
        )

    def load_model(self, checkpoint_path: str, device: str) -> Any:
        from openpi.policies import policy_config as pc
        from openpi.training import config as train_config

        cfg = train_config.get_config(OPENPI_CONFIG_NAME)
        policy = pc.create_trained_policy(
            cfg, checkpoint_path,
            default_prompt=DEFAULT_PROMPT,
            pytorch_device=device,
        )
        logger.info(f"OpenPI model loaded: config={OPENPI_CONFIG_NAME}")
        return policy

    def _reset_vlm_tracker(self) -> None:
        self._vlm_tracker = None
        self._vlm_tracker_task_relpath = None
        self._vlm_tracker_episode_marker = None
        self._vlm_tracker_last_frame_index = None
        self._vlm_tracker_prev_image = None
        self._vlm_tracker_prev_gripper_state = None
        self._vlm_tracker_observation_count = 0
        self._vlm_tracker_last_subtask = None
        self._vlm_tracker_last_status = None
        self._vlm_tracker_last_error = None
        self._vlm_tracker_last_result_current_subtask = None
        self._vlm_tracker_last_result_plan = None
        self._vlm_tracker_initial_plan_captured = False
        self._vlm_tracker_initial_plan = None
        self._vlm_tracker_initial_current_subtask = None
        self._vlm_tracker_initial_frame_index = None
        self._vlm_tracker_initial_visible_instruction_text = None
        self._vlm_tracker_initial_mode = None
        self._reset_vlm_policy_subtask_log()

    def _reset_vlm_policy_subtask_log(self) -> None:
        self._vlm_policy_subtask_log_run_dir = None
        self._vlm_policy_subtask_log_path = None
        self._vlm_policy_subtask_log_count = 0
        self._vlm_policy_subtask_log_announced = False

    def _load_vlm_tracker_class(self) -> Any:
        if str(VLM_API_DIR) not in sys.path:
            sys.path.insert(0, str(VLM_API_DIR))
        from episode_subtask_tracker import EpisodeSubtaskTracker

        return EpisodeSubtaskTracker

    def _new_vlm_tracker(self, task_relpath: str, instruction: str) -> Any:
        tracker_cls = self._load_vlm_tracker_class()
        save_dir = None
        if VLM_TRACKER_SAVE_DIR:
            safe_task = task_relpath.replace("/", "__")
            marker = (
                str(self._vlm_tracker_episode_marker)
                if self._vlm_tracker_episode_marker is not None
                else "unknown_episode"
            )
            save_dir = Path(VLM_TRACKER_SAVE_DIR).expanduser() / safe_task / marker
        return tracker_cls(
            instruction=instruction,
            task_relpath=task_relpath,
            skills_json=VLM_TRACKER_SKILLS_JSON,
            provider=VLM_TRACKER_PROVIDER,
            model=VLM_TRACKER_MODEL,
            save_dir=save_dir,
        )

    def _capture_vlm_initial_plan(
        self,
        result: Dict[str, Any],
        *,
        frame_index: int | None,
    ) -> None:
        if self._vlm_tracker_initial_plan_captured:
            return
        mode = str(result.get("mode") or "")
        if mode and not mode.startswith("initialize"):
            return

        raw_plan = result.get("plan")
        self._vlm_tracker_initial_plan = (
            [str(item) for item in raw_plan if isinstance(item, str)]
            if isinstance(raw_plan, list)
            else None
        )
        current_subtask = str(result.get("current_subtask") or "").strip()
        self._vlm_tracker_initial_current_subtask = (
            current_subtask
            if current_subtask and current_subtask.lower() not in {"unknown", "none", "null"}
            else None
        )
        visible_instruction_text = result.get("visible_instruction_text")
        self._vlm_tracker_initial_visible_instruction_text = (
            str(visible_instruction_text)
            if visible_instruction_text is not None
            else None
        )
        self._vlm_tracker_initial_frame_index = frame_index
        self._vlm_tracker_initial_mode = mode or None
        self._vlm_tracker_initial_plan_captured = True

    def _vlm_initial_plan_record(self) -> dict[str, Any]:
        return {
            "captured": self._vlm_tracker_initial_plan_captured,
            "mode": self._vlm_tracker_initial_mode,
            "frame_index": self._vlm_tracker_initial_frame_index,
            "plan": self._vlm_tracker_initial_plan,
            "initial_current_subtask": self._vlm_tracker_initial_current_subtask,
            "visible_instruction_text": self._vlm_tracker_initial_visible_instruction_text,
        }

    def _ensure_vlm_policy_subtask_log(
        self,
        *,
        task_relpath: str | None,
        episode_marker: Any,
    ) -> Path | None:
        if (
            not VLM_POLICY_SUBTASK_LOG_ENABLED
            or not VLM_POLICY_SUBTASK_LOG_DIR
            or task_relpath not in VLM_TRACKER_TARGET_TASKS
        ):
            return None
        if self._vlm_policy_subtask_log_path is not None:
            return self._vlm_policy_subtask_log_path

        root = Path(VLM_POLICY_SUBTASK_LOG_DIR).expanduser()
        safe_task = _safe_path_component(task_relpath)
        episode_name = _episode_log_component(episode_marker)
        run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._vlm_policy_subtask_log_run_dir = root / safe_task / episode_name / run_name
        self._vlm_policy_subtask_log_run_dir.mkdir(parents=True, exist_ok=True)
        self._vlm_policy_subtask_log_path = (
            self._vlm_policy_subtask_log_run_dir / "pi05_subtasks.jsonl"
        )
        if not self._vlm_policy_subtask_log_announced:
            logger.info("Writing PI05 subtask log to %s", self._vlm_policy_subtask_log_path)
            self._vlm_policy_subtask_log_announced = True
        return self._vlm_policy_subtask_log_path

    def _write_vlm_policy_subtask_log(
        self,
        *,
        task_relpath: str | None,
        task_text: str,
        frame_index: int | None,
        prompt: str | None,
        subtask_prompt: str | None,
        used_vlm_subtask: bool,
        front_image: np.ndarray | None = None,
    ) -> None:
        try:
            episode_marker = self._vlm_tracker_episode_marker
            log_path = self._ensure_vlm_policy_subtask_log(
                task_relpath=task_relpath,
                episode_marker=episode_marker,
            )
            if log_path is None:
                return

            front_image_path = None
            if (
                front_image is not None
                and self._vlm_policy_subtask_log_run_dir is not None
                and self._vlm_policy_subtask_log_count == 0
            ):
                front_image_path = (
                    self._vlm_policy_subtask_log_run_dir
                    / f"front_frame_{frame_index or 0:06d}.png"
                )
                try:
                    from PIL import Image

                    Image.fromarray(np.asarray(front_image, dtype=np.uint8)).save(front_image_path)
                    front_image_path = str(front_image_path)
                except Exception as exc:
                    front_image_path = f"failed_to_save_image: {exc}"

            record = {
                "round": self._vlm_policy_subtask_log_count,
                "frame_index": frame_index,
                "episode": episode_marker,
                "task": task_relpath,
                "task_text": task_text,
                "matched_skill_instruction": _CANONICAL_INSTRUCTIONS.get(task_relpath),
                "initial_plan": self._vlm_initial_plan_record(),
                "current_subtask_for_pi05": subtask_prompt if used_vlm_subtask else prompt,
                "front_image": _image_debug_summary(front_image),
                "saved_front_image": front_image_path,
                "pi05_instruction": prompt,
                "vlm_current_subtask": subtask_prompt,
                "used_vlm_subtask": used_vlm_subtask,
                "vlm_status": self._vlm_tracker_last_status,
                "vlm_result_current_subtask": self._vlm_tracker_last_result_current_subtask,
                "vlm_result_plan": self._vlm_tracker_last_result_plan,
                "vlm_error": self._vlm_tracker_last_error,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if self._vlm_policy_subtask_log_run_dir is not None:
                latest_path = self._vlm_policy_subtask_log_run_dir / "latest_subtask.json"
                latest_path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                initial_plan_path = self._vlm_policy_subtask_log_run_dir / "initial_plan.json"
                initial_plan_path.write_text(
                    json.dumps(record["initial_plan"], ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            self._vlm_policy_subtask_log_count += 1
        except Exception as exc:
            logger.warning("Failed to write PI05 subtask log: %s", exc)

    def _maybe_reset_vlm_tracker_for_obs(
        self,
        *,
        obs: Dict[str, Any],
        task_relpath: str | None,
        frame_index: int | None,
    ) -> None:
        episode_marker = _extract_episode_marker(obs)
        should_reset = False
        if _obs_requests_reset(obs):
            should_reset = True
        if task_relpath != self._vlm_tracker_task_relpath:
            should_reset = True
        if (
            episode_marker is not None
            and self._vlm_tracker_episode_marker is not None
            and episode_marker != self._vlm_tracker_episode_marker
        ):
            should_reset = True
        if (
            frame_index is not None
            and self._vlm_tracker_last_frame_index is not None
            and frame_index < self._vlm_tracker_last_frame_index
        ):
            should_reset = True

        if should_reset:
            self._reset_vlm_tracker()

        if episode_marker is not None:
            self._vlm_tracker_episode_marker = episode_marker

    def _tracker_images_and_state(
        self,
        current_image: np.ndarray,
        current_gripper_state: Any,
    ) -> tuple[list[np.ndarray], Any]:
        if self._vlm_tracker_prev_image is None:
            return [current_image], current_gripper_state

        previous_state = self._vlm_tracker_prev_gripper_state
        state_sequence = []
        if isinstance(previous_state, dict):
            prev = dict(previous_state)
            prev["role"] = "previous_observation"
            state_sequence.append(prev)
        elif previous_state is not None:
            state_sequence.append(previous_state)

        if isinstance(current_gripper_state, dict):
            cur = dict(current_gripper_state)
            cur["role"] = "current_observation"
            state_sequence.append(cur)
        elif current_gripper_state is not None:
            state_sequence.append(current_gripper_state)

        gripper_state = {"sequence": state_sequence} if state_sequence else current_gripper_state
        return [self._vlm_tracker_prev_image, current_image], gripper_state

    def _current_subtask_prompt(
        self,
        *,
        obs: Dict[str, Any],
        task_relpath: str | None,
        task_text: str,
        front_image: np.ndarray,
        gripper_state: Any,
        frame_index: int | None,
    ) -> str | None:
        if (
            not VLM_TRACKER_ENABLED
            or task_relpath not in VLM_TRACKER_TARGET_TASKS
        ):
            self._vlm_tracker_last_status = "disabled_or_non_target_task"
            self._vlm_tracker_last_error = None
            self._vlm_tracker_last_result_current_subtask = None
            self._vlm_tracker_last_result_plan = None
            return None

        self._maybe_reset_vlm_tracker_for_obs(
            obs=obs,
            task_relpath=task_relpath,
            frame_index=frame_index,
        )

        try:
            if self._vlm_tracker is None:
                instruction = _CANONICAL_INSTRUCTIONS.get(task_relpath, "")
                self._vlm_tracker = self._new_vlm_tracker(task_relpath, instruction)
                self._vlm_tracker_task_relpath = task_relpath

            images, tracker_gripper_state = self._tracker_images_and_state(
                front_image,
                gripper_state,
            )
            result = self._vlm_tracker.observe(
                images,
                frame_index=frame_index,
                gripper_state=tracker_gripper_state,
            )
            self._capture_vlm_initial_plan(result, frame_index=frame_index)
            subtask = str(result.get("current_subtask") or "").strip()
            self._vlm_tracker_last_result_current_subtask = subtask or None
            self._vlm_tracker_last_result_plan = result.get("plan")
            self._vlm_tracker_last_error = result.get("vlm_error")
            if subtask and subtask.lower() not in {"unknown", "none", "null"}:
                self._vlm_tracker_last_subtask = subtask
                self._vlm_tracker_last_status = "used_current_subtask"
                return subtask
            if self._vlm_tracker_last_subtask:
                self._vlm_tracker_last_status = "reused_last_subtask"
                return self._vlm_tracker_last_subtask
            self._vlm_tracker_last_status = (
                "vlm_error_no_subtask" if result.get("vlm_error") else "no_current_subtask"
            )
        except Exception as exc:
            self._vlm_tracker_last_status = "exception"
            self._vlm_tracker_last_error = str(exc)
            self._vlm_tracker_last_result_current_subtask = None
            self._vlm_tracker_last_result_plan = None
            logger.warning(
                "VLM subtask tracker failed for task=%s frame=%s; using original instruction. error=%s",
                task_relpath,
                frame_index,
                exc,
            )
            return self._vlm_tracker_last_subtask
        finally:
            self._vlm_tracker_prev_image = np.asarray(front_image, dtype=np.uint8).copy()
            self._vlm_tracker_prev_gripper_state = gripper_state
            self._vlm_tracker_last_frame_index = frame_index
            self._vlm_tracker_observation_count += 1

        return None

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """ManipArena observation → OpenPI observation dict."""
        raw_state = obs.get("state", {})
        state_dict = raw_state if isinstance(raw_state, dict) else {}
        raw_state_vector = _as_float_vector(raw_state)
        default_f1 = (
            raw_state_vector[:7]
            if raw_state_vector is not None and raw_state_vector.size >= 7
            else np.zeros(7)
        )
        default_f2 = (
            raw_state_vector[7:14]
            if raw_state_vector is not None and raw_state_vector.size >= 14
            else np.zeros(7)
        )
        if self.control_mode == "joints":
            f1 = np.asarray(
                state_dict.get(
                    "follow1_joints",
                    state_dict.get("follow1_pos", default_f1),
                ),
                dtype=np.float32,
            )[:7]
            f2 = np.asarray(
                state_dict.get(
                    "follow2_joints",
                    state_dict.get("follow2_pos", default_f2),
                ),
                dtype=np.float32,
            )[:7]
        else:
            f1 = np.asarray(state_dict.get("follow1_pos", default_f1), dtype=np.float32)[:7]
            f2 = np.asarray(state_dict.get("follow2_pos", default_f2), dtype=np.float32)[:7]
        state = np.concatenate([f1[:6], f1[6:7], f2[:6], f2[6:7]]).astype(np.float32)
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

        views = obs.get("views") or {}
        decoded_views: Dict[str, np.ndarray] = {}
        for client_key, openpi_key in _CAM_MAP.items():
            raw = views.get(client_key)
            if raw is not None:
                decoded_views[client_key] = _decode_image(raw)
            else:
                decoded_views[client_key] = np.zeros((480, 640, 3), dtype=np.uint8)

        task_text = _extract_task_text(obs)
        task_relpath = _match_task_relpath(task_text)
        frame_index = _extract_frame_index(obs)
        if frame_index is None:
            frame_index = self._vlm_tracker_observation_count
        gripper_state = _build_tracker_gripper_state(
            obs,
            state_dict,
            f1,
            f2,
            frame_index=frame_index,
        )
        subtask_prompt = self._current_subtask_prompt(
            obs=obs,
            task_relpath=task_relpath,
            task_text=task_text,
            front_image=decoded_views["camera_front"],
            gripper_state=gripper_state,
            frame_index=frame_index,
        )
        prompt = subtask_prompt or obs.get("instruction", DEFAULT_PROMPT) or DEFAULT_PROMPT
        if subtask_prompt:
            logger.info(
                "Using VLM current_subtask as PI05 instruction: task=%s frame=%s subtask=%r",
                task_relpath,
                frame_index,
                subtask_prompt,
            )
        self._write_vlm_policy_subtask_log(
            task_relpath=task_relpath,
            task_text=task_text,
            frame_index=frame_index,
            prompt=prompt,
            subtask_prompt=subtask_prompt,
            used_vlm_subtask=bool(subtask_prompt),
            front_image=decoded_views["camera_front"],
        )

        openpi_obs: Dict[str, Any] = {
            "observation.state": state,
            "prompt": prompt,
        }

        for client_key, openpi_key in _CAM_MAP.items():
            openpi_obs[openpi_key] = decoded_views[client_key]

        return openpi_obs

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        result = self.model.infer(model_input)
        return np.asarray(result["actions"], dtype=np.float32)

    def reset(self):
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
        self._reset_vlm_tracker()
        super().reset()

    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        """OpenPI actions (T, 14) → ManipArena response dict.

        NOTE: values must be Python lists (.tolist()), not numpy arrays.
        """
        actions = model_output
        start_idx = max(0, ACTION_START_STEP)
        if ACTION_END_STEP > 0:
            end_idx = min(ACTION_END_STEP, actions.shape[0])
        elif ACTION_OUTPUT_STEPS > 0:
            end_idx = min(
                start_idx + ACTION_OUTPUT_STEPS,
                actions.shape[0],
            )
        else:
            end_idx = min(
                actions.shape[0],
                start_idx + max(
                    2,
                    int(ACTION_END_RATIO * actions.shape[0]),
                ),
            )
        if end_idx <= start_idx:
            start_idx = 0
            if ACTION_OUTPUT_STEPS > 0:
                end_idx = min(ACTION_OUTPUT_STEPS, actions.shape[0])
            else:
                end_idx = max(
                    2,
                    int(ACTION_END_RATIO * actions.shape[0]),
                )
        actions = actions[start_idx:end_idx]
        self._last_served_action_count = int(actions.shape[0])

        left = np.asarray(actions[:, :7], dtype=np.float32)
        right = np.asarray(actions[:, 7:14], dtype=np.float32)
        if self.control_mode != "joints":
            left = _apply_downward_z_bias(
                left,
                apply_bias=self._left_apply_down_bias,
            )
            right = _apply_downward_z_bias(
                right,
                apply_bias=self._right_apply_down_bias,
            )
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
        #left = actions[:, :7]
        #right = actions[:, 7:14]
        final_actions = np.concatenate([left, right], axis=1)
        payload: Dict[str, Any]
        debug_left = _format_debug_action_rows(
            left, left_raw_gripper,
        )
        debug_right = _format_debug_action_rows(
            right, right_raw_gripper,
        )
        if self.control_mode == "joints":
            payload = {
                "follow1_joints": left.tolist(),
                "follow2_joints": right.tolist(),
                "follow1_pos": left.tolist(),
                "follow2_pos": right.tolist(),
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
            }
            debug_payload = {
                "follow1_pos": debug_left,
                "follow2_pos": debug_right,
            }

        logger.debug(
            "OpenPI convert_output payload shape=%s:%s",
            final_actions.shape,
            format_action_payload_for_debug(
                debug_payload, max_steps=DEBUG_OUTPUT_STEPS,
            ),
        )
        return payload
