#!/usr/bin/env python3
"""Standalone OpenPI action server for cascaded Qwen -> stage2 OpenPI inference.

Run this file in the OpenPI environment.  It accepts already
converted observations from examples/subtask_policy.py:

    {
        "observation.state": np.ndarray[14],
        "observation.images.faceImg": np.ndarray[H,W,3],
        "observation.images.leftImg": np.ndarray[H,W,3],
        "observation.images.rightImg": np.ndarray[H,W,3],
        "prompt": "<subtask>",
    }

and returns:

    {"actions": np.ndarray[T, 14]}
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

MANIPARENA_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(MANIPARENA_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MANIPARENA_REPO_ROOT))

from maniparena.server import WebSocketModelServer

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OPENPI_ROOT = WORKSPACE_ROOT / "openpi"


class SmallVLMActionPolicy:
    def __init__(
        self,
        checkpoint_path: str,
        config_name: str,
        config_module: str,
        openpi_root: str,
        device: str,
        action_horizon: int,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.config_name = config_name
        self.config_module = config_module
        self.openpi_root = Path(openpi_root).expanduser().resolve()
        self.device = device
        self.action_horizon = int(action_horizon)
        self.policy = self._load_policy()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "control_mode": "internal_action_server",
            "action_horizon": self.action_horizon,
            "state_dim": 14,
            "state_dim_per_arm": 7,
            "protocol_version": "2.0",
            "config_name": self.config_name,
            "config_module": self.config_module,
            "openpi_root": str(self.openpi_root),
        }

    def _load_policy(self) -> Any:
        openpi_src = str(self.openpi_root / "src")
        if openpi_src not in sys.path:
            sys.path.insert(0, openpi_src)

        importlib.invalidate_caches()
        from openpi.policies import policy_config as pc

        if self.config_module == "stage2":
            from openpi.training_stage2 import config as train_config
        else:
            from openpi.training import config as train_config

        cfg = train_config.get_config(self.config_name)
        policy = pc.create_trained_policy(
            cfg,
            self.checkpoint_path,
            default_prompt=None,
            pytorch_device=self.device,
        )
        logger.info(
            "Loaded OpenPI action policy: config=%s config_module=%s checkpoint=%s device=%s openpi_root=%s",
            self.config_name,
            self.config_module,
            self.checkpoint_path,
            self.device,
            self.openpi_root,
        )
        return policy

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        obs = _restore_msgpack_numpy_arrays(obs)
        result = self.policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        return {"actions": actions.tolist()}

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()


def _restore_msgpack_numpy_arrays(value: Any) -> Any:
    """Restore numpy arrays if msgpack-numpy arrived as plain dicts.

    This keeps the internal small_vlm server tolerant to environments where
    msgpack_numpy.patch() was not active on one side of the websocket.
    """
    if isinstance(value, dict):
        if set(value.keys()) >= {"data", "dtype", "shape"}:
            try:
                arr = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
                return arr.reshape(tuple(value["shape"]))
            except Exception:
                pass
        if set(value.keys()) >= {"data", "type", "shape"}:
            try:
                arr = np.frombuffer(value["data"], dtype=np.dtype(value["type"]))
                return arr.reshape(tuple(value["shape"]))
            except Exception:
                pass
        if set(value.keys()) >= {b"data", b"dtype", b"shape"}:
            try:
                arr = np.frombuffer(value[b"data"], dtype=np.dtype(value[b"dtype"].decode()))
                return arr.reshape(tuple(value[b"shape"]))
            except Exception:
                pass
        if set(value.keys()) >= {b"data", b"type", b"shape"}:
            try:
                dtype = value[b"type"]
                if isinstance(dtype, bytes):
                    dtype = dtype.decode()
                arr = np.frombuffer(value[b"data"], dtype=np.dtype(dtype))
                return arr.reshape(tuple(value[b"shape"]))
            except Exception:
                pass
        return {key: _restore_msgpack_numpy_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_msgpack_numpy_arrays(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenPI action WebSocket server for subtask_policy.py",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-name", default="pi05_maniparena_ee_stage2")
    parser.add_argument(
        "--config-module",
        choices=("stage2", "base"),
        default="stage2",
        help="'stage2' loads openpi.training_stage2.config; 'base' loads openpi.training.config.",
    )
    parser.add_argument("--openpi-root", default=str(OPENPI_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    policy = SmallVLMActionPolicy(
        checkpoint_path=args.checkpoint,
        config_name=args.config_name,
        config_module=args.config_module,
        openpi_root=args.openpi_root,
        device=args.device,
        action_horizon=args.action_horizon,
    )
    WebSocketModelServer(
        policy=policy,
        host=args.host,
        port=args.port,
    ).serve_forever()


if __name__ == "__main__":
    main()
