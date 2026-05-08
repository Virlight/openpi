
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
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict

import cv2
import numpy as np

from maniparena.debug import format_action_payload_for_debug
from maniparena.policy import ModelPolicy

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────

OPENPI_CONFIG_NAME = "pi05_maniparena_ee"  # set this to match your trained OpenPI config
DEFAULT_PROMPT = None
DA3_POINT_MEAN_ENABLED = os.environ.get("DA3_POINT_MEAN_ENABLED", "1") != "0"
DA3_REPO_SRC = os.environ.get("DA3_REPO_SRC", "/mnt/models/haoliang/3DCV/Depth-Anything-3/src")
DA3_MODEL_DIR = os.environ.get("DA3_MODEL_DIR", "depth-anything/DA3NESTED-GIANT-LARGE")
DA3_DEVICE = os.environ.get("DA3_DEVICE")
DA3_CAMERA_KEYS_ENV = os.environ.get("DA3_CAMERA_KEYS", "camera_left,camera_right")
# Use a higher-than-native processing resolution so DA3 runs on an upscaled view.
DA3_PROCESS_RES = int(os.environ.get("DA3_PROCESS_RES", "896"))
# Match the higher processing resolution with a denser debug point sampling budget.
DA3_SAMPLES_PER_VIEW = int(os.environ.get("DA3_SAMPLES_PER_VIEW", "2048"))
DA3_DEBUG_PLY_DIR = Path(os.environ.get("DA3_DEBUG_PLY_DIR", "debug/da3_point_mean"))
DA3_SAVE_DEBUG_PLY = os.environ.get("DA3_SAVE_DEBUG_PLY", "1") != "0"
ACTION_END_RATIO = 0.8  # keep first 80% of predicted actions
ACTION_START_STEP = 0
ACTION_END_STEP = 30  # exclusive; with horizon=40 this selects the middle 20 steps
ACTION_OUTPUT_STEPS = 30  # hard cap on transmitted action steps
DEBUG_OUTPUT_STEPS = 4
Z_DOWN_BIAS_M = 0.02
Z_BIAS_MAX_HEIGHT_M = -0.2
STALL_TOLERANCE_M = 0.01
POSITION_STALL_TRIGGER_ACTIONS = 20
GRIPPER_OPEN_SWITCH_THRESHOLD = 1.0
GRIPPER_CLOSE_SWITCH_THRESHOLD = 3.0
GRIPPER_FORCE_OPEN_THRESHOLD = 3.3
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


def _parse_da3_camera_keys(raw_value: str) -> tuple[str, ...]:
    keys = tuple(key.strip() for key in raw_value.split(",") if key.strip())
    if not keys:
        raise ValueError("DA3_CAMERA_KEYS must contain at least one camera key.")
    invalid = [key for key in keys if key not in _CAM_MAP]
    if invalid:
        raise ValueError(
            f"Unsupported DA3 camera keys: {invalid}. Available keys: {tuple(_CAM_MAP)}"
        )
    return keys


DA3_CAMERA_KEYS = _parse_da3_camera_keys(DA3_CAMERA_KEYS_ENV)


def _decode_image(v: Any) -> np.ndarray:
    """base64 JPEG string or numpy array → RGB uint8 ndarray."""
    if isinstance(v, np.ndarray):
        return v.astype(np.uint8) if v.dtype != np.uint8 else v
    raw = base64.b64decode(v) if isinstance(v, str) else bytes(v)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _decode_mask(v: Any, image_shape: tuple[int, int, int]) -> np.ndarray:
    """Decode a mask payload to HxW uint8; nonzero pixels are treated as foreground."""
    if isinstance(v, np.ndarray):
        mask = v
    else:
        raw = base64.b64decode(v) if isinstance(v, str) else bytes(v)
        mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError("cv2.imdecode failed for mask")

    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.max(axis=-1)
    elif mask.ndim != 2:
        raise ValueError(f"Expected mask shape HxW or HxWxC, got {mask.shape}")

    h, w = image_shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.uint8)


def _mask_payload_for_view(obs: Dict[str, Any], client_key: str) -> Any:
    masks = obs.get("masks", {})
    views = obs.get("views", {})
    candidates = (
        masks.get(client_key) if isinstance(masks, dict) else None,
        masks.get(f"{client_key}_mask") if isinstance(masks, dict) else None,
        views.get(f"{client_key}_mask") if isinstance(views, dict) else None,
        obs.get(f"{client_key}_mask"),
    )
    for value in candidates:
        if value is not None:
            return value
    return None


def _ensure_da3_import_path() -> None:
    if DA3_REPO_SRC and Path(DA3_REPO_SRC).exists() and DA3_REPO_SRC not in sys.path:
        sys.path.insert(0, DA3_REPO_SRC)


def _da3_samples_per_view() -> int | None:
    return None if DA3_SAMPLES_PER_VIEW <= 0 else DA3_SAMPLES_PER_VIEW


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
        self._da3_point_mean = None
        self._last_da3_point_mean_result = None
        self._da3_debug_ply_count = 0
        self._da3_warned_missing_masks = False
        super().__init__(
            checkpoint_path=checkpoint_path,
            control_mode=control_mode,
            action_horizon=action_horizon,
            device=device,
        )
        self._da3_point_mean = self._load_da3_point_mean_estimator(device)

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

    def _load_da3_point_mean_estimator(self, device: str) -> Any:
        if not DA3_POINT_MEAN_ENABLED:
            logger.info("DA3 masked point mean disabled")
            return None

        _ensure_da3_import_path()
        from depth_anything_3.masked_point_mean import MaskedPointMeanEstimator

        da3_device = DA3_DEVICE or device
        estimator = MaskedPointMeanEstimator(
            model_dir=DA3_MODEL_DIR,
            device=da3_device,
            process_res=DA3_PROCESS_RES,
            expected_num_views=len(DA3_CAMERA_KEYS),
            samples_per_view=_da3_samples_per_view(),
            use_confidence=True,
        )
        logger.info(
            "DA3 masked point mean loaded: model=%s device=%s cameras=%s process_res=%s samples_per_view=%s",
            DA3_MODEL_DIR,
            da3_device,
            DA3_CAMERA_KEYS,
            DA3_PROCESS_RES,
            _da3_samples_per_view(),
        )
        return estimator

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """ManipArena observation → OpenPI observation dict."""
        state_dict = obs.get("state", {})
        if self.control_mode == "joints":
            f1 = np.asarray(state_dict.get("follow1_joints", state_dict.get("follow1_pos", np.zeros(7))), dtype=np.float32)[:7]
            f2 = np.asarray(state_dict.get("follow2_joints", state_dict.get("follow2_pos", np.zeros(7))), dtype=np.float32)[:7]
        else:
            f1 = np.asarray(state_dict.get("follow1_pos", np.zeros(7)), dtype=np.float32)[:7]
            f2 = np.asarray(state_dict.get("follow2_pos", np.zeros(7)), dtype=np.float32)[:7]
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

        openpi_obs: Dict[str, Any] = {
            "observation.state": state,
            "prompt": obs.get("instruction", DEFAULT_PROMPT) or DEFAULT_PROMPT,
        }

        views = obs.get("views", {})
        da3_images: list[np.ndarray] = []
        da3_masks: list[np.ndarray] = []
        for client_key, openpi_key in _CAM_MAP.items():
            raw = views.get(client_key)
            if raw is not None:
                image = _decode_image(raw)
            else:
                image = np.zeros((480, 640, 3), dtype=np.uint8)
            openpi_obs[openpi_key] = image

            mask_payload = _mask_payload_for_view(obs, client_key)
            if mask_payload is None:
                if not self._da3_warned_missing_masks:
                    logger.warning(
                        "DA3 mask payloads are missing; using full-image masks for debug PLY"
                    )
                    self._da3_warned_missing_masks = True
                mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            else:
                mask = _decode_mask(mask_payload, image.shape)
            if client_key in DA3_CAMERA_KEYS:
                da3_images.append(image)
                da3_masks.append(mask)

        openpi_obs["_da3_images"] = da3_images
        openpi_obs["_da3_masks"] = da3_masks

        return openpi_obs

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        da3_images = model_input.pop("_da3_images", None)
        da3_masks = model_input.pop("_da3_masks", None)
        self._run_da3_point_mean_debug(da3_images, da3_masks)
        result = self.model.infer(model_input)
        return np.asarray(result["actions"], dtype=np.float32)

    def _run_da3_point_mean_debug(
        self,
        images: list[np.ndarray] | None,
        masks: list[np.ndarray] | None,
    ) -> None:
        if self._da3_point_mean is None or images is None or masks is None:
            return
        expected_views = len(DA3_CAMERA_KEYS)
        if len(images) != expected_views or len(masks) != expected_views:
            logger.warning(
                "Skipping DA3 point mean: expected %s images and %s masks, got %s/%s",
                expected_views,
                expected_views,
                len(images) if images is not None else None,
                len(masks) if masks is not None else None,
            )
            return

        debug_ply_path = None
        if DA3_SAVE_DEBUG_PLY:
            DA3_DEBUG_PLY_DIR.mkdir(parents=True, exist_ok=True)
            debug_ply_path = DA3_DEBUG_PLY_DIR / f"obs_{self._da3_debug_ply_count:06d}.ply"
            self._da3_debug_ply_count += 1

        try:
            result = self._da3_point_mean.estimate(
                images=images,
                masks=masks,
                samples_per_view=_da3_samples_per_view(),
                debug_ply_path=debug_ply_path,
            )
        except Exception:
            logger.exception("DA3 masked point mean failed")
            return

        self._last_da3_point_mean_result = result
        logger.info(
            "DA3 masked point mean=%s points=%s/%s ply=%s",
            np.array2string(result.mean, precision=4),
            result.num_points_used,
            result.num_valid_points,
            result.debug_ply_path,
        )

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
        self._last_da3_point_mean_result = None
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
