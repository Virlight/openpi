"""Cosmos Policy adapter for ManipArena bimanual (14D end-effector).

Serves a Cosmos Policy model over the ManipArena WebSocket protocol.

Prerequisites (install inside the cosmos-policy Docker container):
    pip install -r requirements.txt   # maniparena-repo deps

Usage:
    # Copy this file to examples/my_policy.py (so launch.py can find it),
    # or symlink it:
    #   ln -sf cosmos_policy_example.py examples/my_policy.py

    # Then launch the server:
    python serve.py \
        --checkpoint nvidia/Cosmos-Policy-ALOHA-Predict2-2B \
        --control-mode end_pose \
        --action-horizon 50 \
        --port 8000

    # Open-loop eval:
    python scripts/eval_openloop.py \
        --server ws://localhost:8000 \
        --dataset ./data/maniparena/sim/press_button_in_order \
        --episode 0 \
        --action-chunk 50 \
        --save-dir openloop_plots

Environment variables:
    COSMOS_POLICY_DIR   Path to the cosmos-policy repo root.
                        Default: ../cosmos-policy  (sibling of maniparena-repo)
    MANIPARENA_STATS    Path to dataset_statistics.json for action/proprio scaling.
                        Default: <dataset>/dataset_statistics.json  (or ALOHA stats)
    COSMOS_T5_CACHE     Path to pre-computed T5 embeddings .pkl (optional, speeds up).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: add cosmos-policy to sys.path and inject the maniparena platform
# flag so that constants.py picks up ACTION_DIM=14, PROPRIO_DIM=14.
# ---------------------------------------------------------------------------
if not any("maniparena" in a.lower() for a in sys.argv):
    sys.argv.append("--maniparena")

_COSMOS_DIR = os.environ.get(
    "COSMOS_POLICY_DIR",
    str(Path(__file__).resolve().parents[1].parent / "cosmos-policy"),
)
if _COSMOS_DIR not in sys.path:
    sys.path.insert(0, _COSMOS_DIR)

from maniparena.policy import ModelPolicy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cosmos Policy config — mirrors the ALOHA deploy configuration.
# ---------------------------------------------------------------------------
_ALOHA_INFERENCE_CONFIG = (
    "cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_"
    "foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80"
    "__inference_only"
)

DEFAULT_PROMPT = "complete the manipulation task"


def _decode_image(v: Any) -> np.ndarray:
    """base64 JPEG / raw bytes / numpy passthrough -> RGB uint8 ndarray."""
    if isinstance(v, np.ndarray):
        return v.astype(np.uint8) if v.dtype != np.uint8 else v
    raw = base64.b64decode(v) if isinstance(v, str) else bytes(v)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _compute_stats_from_episode(parquet_path: str) -> dict:
    """Fallback: compute min/max stats from a single parquet episode."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    actions = np.stack(df["action"].tolist()).astype(np.float32)[:, :14]
    states = np.stack(df["observation.state"].tolist()).astype(np.float32)[:, :14]
    return {
        "actions_min": actions.min(axis=0).tolist(),
        "actions_max": actions.max(axis=0).tolist(),
        "proprio_min": states.min(axis=0).tolist(),
        "proprio_max": states.max(axis=0).tolist(),
    }


# ══════════════════════════════════════════════════════════════════
#  Policy
# ══════════════════════════════════════════════════════════════════


class MyPolicy(ModelPolicy):
    """Cosmos Policy adapter for ManipArena bimanual tasks."""

    def load_model(self, checkpoint_path: str, device: str) -> Any:
        import torch
        from cosmos_policy.experiments.robot.libero.run_libero_eval import (
            PolicyEvalConfig,
        )
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_model,
            init_t5_text_embeddings_cache,
            load_dataset_stats,
        )

        # ----- Dataset stats -----
        stats_env = os.environ.get("MANIPARENA_STATS", "")
        if stats_env and os.path.isfile(stats_env):
            stats_path = stats_env
        else:
            stats_path = ""
            logger.warning(
                "MANIPARENA_STATS not set or missing — "
                "action un-normalization will use ALOHA defaults. "
                "Set MANIPARENA_STATS=/path/to/dataset_statistics.json for correct scaling."
            )

        # ----- T5 embedding cache -----
        t5_path = os.environ.get("COSMOS_T5_CACHE", "")
        init_t5_text_embeddings_cache(t5_path if t5_path else None)

        # ----- Config -----
        # chunk_size must be 50 to match the ALOHA checkpoint's training config
        # (the latent-to-action extraction depends on this). We trim the output
        # to self.action_horizon in convert_output.
        self._cfg = PolicyEvalConfig(
            suite="maniparena",
            config=_ALOHA_INFERENCE_CONFIG,
            ckpt_path=checkpoint_path,
            config_file="cosmos_policy/config/config.py",
            dataset_stats_path=stats_path or f"{checkpoint_path}/aloha_dataset_statistics.json",
            t5_text_embeddings_path=t5_path,
            use_third_person_image=True,
            num_third_person_images=1,
            use_wrist_image=True,
            num_wrist_images=2,
            use_proprio=True,
            normalize_proprio=True,
            unnormalize_actions=True,
            trained_with_image_aug=True,
            chunk_size=50,
            num_open_loop_steps=50,
            flip_images=False,
            use_jpeg_compression=False,
            num_denoising_steps_action=10,
            num_denoising_steps_future_state=1,
            num_denoising_steps_value=1,
            deterministic=True,
            seed=195,
        )

        self._dataset_stats = load_dataset_stats(self._cfg.dataset_stats_path)

        model, self._cosmos_config = get_model(self._cfg)
        logger.info("Cosmos Policy loaded on %s", device)
        if torch.cuda.is_available():
            logger.info("GPU mem: %.1f GB", torch.cuda.memory_allocated() / 1e9)
        return model

    # ── convert_input ─────────────────────────────────────────────

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """ManipArena WebSocket observation -> Cosmos Policy observation dict."""
        state_dict = obs.get("state", {})
        f1 = np.asarray(state_dict.get("follow1_pos", np.zeros(7)), dtype=np.float32)[:7]
        f2 = np.asarray(state_dict.get("follow2_pos", np.zeros(7)), dtype=np.float32)[:7]
        proprio = np.concatenate([f1, f2]).astype(np.float32)

        views = obs.get("views", {})

        def _img(cam_key: str) -> np.ndarray:
            raw = views.get(cam_key)
            if raw is not None:
                return _decode_image(raw)
            return np.zeros((224, 224, 3), dtype=np.uint8)

        instruction = ""
        for k in ("instruction", "INSTRUCTION", "prompt", "PROMPT"):
            val = obs.get(k)
            if val is not None:
                instruction = str(val)
                break

        return {
            "left_wrist_image": _img("camera_left"),
            "right_wrist_image": _img("camera_right"),
            "primary_image": _img("camera_front"),
            "proprio": proprio,
            "instruction": instruction or DEFAULT_PROMPT,
        }

    # ── run_inference ─────────────────────────────────────────────

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        from cosmos_policy.experiments.robot.cosmos_utils import get_action

        instruction = model_input.pop("instruction", DEFAULT_PROMPT)

        observation = {
            "left_wrist_image": model_input["left_wrist_image"],
            "right_wrist_image": model_input["right_wrist_image"],
            "primary_image": model_input["primary_image"],
            "proprio": model_input["proprio"],
        }

        action_return_dict = get_action(
            self._cfg,
            self.model,
            self._dataset_stats,
            observation,
            instruction,
            num_denoising_steps_action=self._cfg.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=False,
        )

        actions = action_return_dict["actions"]
        return np.array(actions, dtype=np.float32)

    # ── convert_output ────────────────────────────────────────────

    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        """(T, 14) actions -> ManipArena response with follow1_pos / follow2_pos."""
        actions = np.asarray(model_output, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)

        T = min(actions.shape[0], self.action_horizon)
        actions = actions[:T]

        left = actions[:, :7]
        right = actions[:, 7:14]

        return {
            "follow1_pos": left.tolist(),
            "follow2_pos": right.tolist(),
        }
