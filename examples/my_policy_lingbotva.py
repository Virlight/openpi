"""Embedded Lingbot policy for ManipArena.

This variant does not connect to a remote Lingbot websocket server.
Instead it instantiates ``VA_Server`` in-process and adapts its inputs/outputs
to the ManipArena policy protocol.

Start the server with:
    python serve.py --policy examples.my_policynew:MyPolicy --checkpoint /path/to/model --port 8000

Optional environment variables:
    LINGBOT_SAVE_ROOT  Directory for Lingbot debug outputs.
                       Default: ``./visualization_embedded``
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
from scipy.spatial.transform import Rotation

from maniparena.policy import ModelPolicy
from maniparena.utils import (
    convert_model_output_to_action,
    convert_observation_to_model_input,
)

logger = logging.getLogger(__name__)

WAN_VA_ROOT = Path(r"/mnt/models/haoliang/CVPR2026-Workshop/lingbot-va")
if str(WAN_VA_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_VA_ROOT))

from wan_va.configs import VA_CONFIGS  # noqa: E402
from wan_va.wan_va_server_new import VA_Server  # noqa: E402


def _resolve_config_name() -> str:
    return "robotwin"


def _resolve_model_path(checkpoint_path: str) -> str:
    value = (checkpoint_path or "").strip()
    if value and Path(value).exists():
        return value

    config_name = _resolve_config_name()
    default_path = str("/mnt/models/haoliang/CVPR2026-Workshop/lingbot-va/lingbot-va-posttrain-robotwin")
    return default_path


def _build_embedded_config(checkpoint_path: str, device: str):
    config_name = _resolve_config_name()
    base_config = VA_CONFIGS[config_name]
    config = copy.deepcopy(base_config)

    model_path = _resolve_model_path(checkpoint_path)
    config.wan22_pretrained_model_name_or_path = model_path
    config.infer_mode = "server"
    config.save_root = os.environ.get("LINGBOT_SAVE_ROOT", "./Lingbot_visualization_embedded")
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1

    if device.startswith("cuda"):
        try:
            config.local_rank = int(device.split(":")[1])
        except (IndexError, ValueError):
            config.local_rank = 0
    else:
        config.local_rank = 0

    os.makedirs(config.save_root, exist_ok=True)
    return config


def _relative_quat16_to_action14(action16: np.ndarray) -> np.ndarray:
    if action16.ndim != 2 or action16.shape[1] != 16:
        raise ValueError(f"Expected shape (T, 16), got {action16.shape}")

    left_rot = Rotation.from_quat(action16[:, 3:7]).as_euler("xyz").astype(np.float32)
    right_rot = Rotation.from_quat(action16[:, 11:15]).as_euler("xyz").astype(np.float32)

    return np.concatenate(
        [
            action16[:, 0:3],
            left_rot,
            action16[:, 7:8],
            action16[:, 8:11],
            right_rot,
            action16[:, 15:16],
        ],
        axis=1,
    ).astype(np.float32)


class MyPolicy(ModelPolicy):

    def load_model(self, checkpoint_path: str, device: str) -> Any:
        config = _build_embedded_config(checkpoint_path, device)
        model_path = config.wan22_pretrained_model_name_or_path

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Lingbot model path does not exist: {model_path}. "
                f"Pass a real model directory via --checkpoint or update the robotwin config."
            )

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(config.local_rank)
        elif device.startswith("cuda"):
            logger.warning("Requested CUDA device %s but CUDA is unavailable.", device)

        if dist.is_initialized(): # 判断分布式运行是否已经初始化
            logger.info("Reusing existing torch.distributed process group.")
        else:
            logger.info(
                "Instantiating embedded Lingbot with config=%s model=%s",
                _resolve_config_name(),
                model_path,
            )

        self._episode_started = False    # 判断episode开始没有
        self._last_instruction: str | None = None      # 换没换新任务
        self._last_lingbot_action: np.ndarray | None = None      #上一次Linbot原始返回的action
        self._request_count = 0 # 这个policy目前处理了多少请求
        self._embedded_config = config
        self._last_instruction = None
        self._embedded_config = config
        self._client_chunk_length = int(os.environ.get("LINGBOT_CLIENT_CHUNK_LENGTH", "4"))
        self._pending_actions14 = None
        self._pending_raw_action = None
        self._pending_action_cursor = 0
        self._pending_cache_frames = []
        self._serving_initial_rollout = False
        return VA_Server(config)

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        frame = self._build_lingbot_frame(model_input)
        prompt = frame["task"]
        is_new_episode = (not self._episode_started) or (prompt != self._last_instruction)

        if is_new_episode:
            logger.info("Starting embedded Lingbot episode %r", prompt)
            self.model.infer({"reset": True, "prompt": prompt})
            self._episode_started = True
            self._last_instruction = prompt
            self._last_lingbot_action = None
            self._clear_pending_cycle()
            return self._start_new_rollout(frame, prompt)

        if self._pending_actions14 is not None:
            self._pending_cache_frames.append(frame)
            if self._pending_action_cursor < self._pending_actions14.shape[0]:
                return self._serve_pending_chunk()
            self._flush_pending_cycle()

        return self._start_new_rollout(frame, prompt)

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return convert_observation_to_model_input(obs, self.control_mode, decode_images=True)

    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        return convert_model_output_to_action(
            model_output,
            self.control_mode,
            self.action_horizon,
        )
    
    def reset(self) -> None:
        self._episode_started = False
        self._last_instruction = None
        self._last_lingbot_action = None
        self._request_count = 0
        self._clear_pending_cycle()

    def _build_lingbot_frame(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        front = model_input.get("front")
        left = model_input.get("left")
        right = model_input.get("right")
        state = np.asarray(model_input.get("state", np.zeros(14)), dtype=np.float32).reshape(-1)
        prompt = str(model_input.get("instruction", "") or "")

        if front is None:
            front = np.zeros((480, 640, 3), dtype=np.uint8)
        if left is None:
            left = np.zeros_like(front)
        if right is None:
            right = np.zeros_like(front)

        return {
            "observation.images.faceImg": np.asarray(front, dtype=np.uint8),
            "observation.images.leftImg": np.asarray(left, dtype=np.uint8),
            "observation.images.rightImg": np.asarray(right, dtype=np.uint8),
            "observation.state": state.astype(np.float32),
            "task": prompt,
        }

    def _extract_actions(self, result: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(result, dict) or "action" not in result:
            raise ValueError(f"Lingbot response missing action: {type(result)}")

        raw_action = np.asarray(result["action"], dtype=np.float32)
        if raw_action.ndim != 3:
            raise ValueError(f"Expected Lingbot action shape (C, F, N), got {raw_action.shape}")

        steps = raw_action.transpose(1, 2, 0).reshape(-1, raw_action.shape[0])
        if steps.shape[1] == 14:
            return steps.astype(np.float32), raw_action
        if steps.shape[1] == 16:
            return _relative_quat16_to_action14(steps), raw_action
        raise ValueError(f"Unsupported Lingbot action dim {steps.shape[1]} from {raw_action.shape}")

    def _clear_pending_cycle(self) -> None:
        self._pending_actions14 = None
        self._pending_raw_action = None
        self._pending_action_cursor = 0
        self._pending_cache_frames = []
        self._serving_initial_rollout = False

    def _start_new_rollout(self, frame: Dict[str, Any], prompt: str) -> np.ndarray:
        result = self.model.infer({"obs": frame, "prompt": prompt})
        actions14, raw_action = self._extract_actions(result)
        self._last_lingbot_action = raw_action
        self._serving_initial_rollout = (self._request_count == 0)
        if self._serving_initial_rollout and actions14.shape[0] >= 2:
            # First rollout: execute only the second half of the initial chunk
            # (previously 32-step -> last 16; now 8-step -> last 4).
            half = actions14.shape[0] // 2
            actions14 = actions14[half:]
        self._pending_actions14 = actions14.astype(np.float32)
        self._pending_raw_action = raw_action
        self._pending_action_cursor = 0
        self._pending_cache_frames = []
        self._request_count += 1
        return self._serve_pending_chunk()

    def _serve_pending_chunk(self) -> np.ndarray:
        if self._pending_actions14 is None:
            raise RuntimeError("No pending Lingbot rollout to serve.")

        start = self._pending_action_cursor
        end = min(start + self._client_chunk_length, self._pending_actions14.shape[0])
        chunk = self._pending_actions14[start:end]
        self._pending_action_cursor = end

        if chunk.shape[0] == 0:
            chunk = np.zeros((self._client_chunk_length, 14), dtype=np.float32)

        return self._pad_for_client(chunk)

    def _pad_for_client(self, actions: np.ndarray) -> np.ndarray:
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"Expected action shape (T, 14), got {actions.shape}")
        if actions.shape[0] >= self.action_horizon:
            return actions[: self.action_horizon].astype(np.float32)

        pad = np.zeros((self.action_horizon - actions.shape[0], 14), dtype=np.float32)
        return np.concatenate([actions.astype(np.float32), pad], axis=0)

    def _flush_pending_cycle(self) -> None:
        if self._pending_raw_action is None or not self._pending_cache_frames:
            self._clear_pending_cycle()
            return

        self.model.infer(
            {
                "obs": self._pending_cache_frames,
                "compute_kv_cache": True,
                "imagine": False,
                "state": self._pending_raw_action,
            }
        )
        self._clear_pending_cycle()

    def _fit_action_horizon(self, actions: np.ndarray) -> np.ndarray:
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"Expected action shape (T, 14), got {actions.shape}")
        if actions.shape[0] == self.action_horizon:
            return actions.astype(np.float32)
        if actions.shape[0] > self.action_horizon:
            return actions[: self.action_horizon].astype(np.float32)
        if actions.shape[0] == 0:
            return np.zeros((self.action_horizon, 14), dtype=np.float32)

        pad = np.repeat(actions[-1:, :], self.action_horizon - actions.shape[0], axis=0)
        return np.concatenate([actions, pad], axis=0).astype(np.float32)
    
