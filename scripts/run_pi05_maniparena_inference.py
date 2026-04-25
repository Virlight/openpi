import argparse
import dataclasses
import json
import logging
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.policies import maniparena_policy
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local inference for the custom Pi05 ManipArena path. "
            "This script loads a trained checkpoint, builds a ManipArena-style "
            "observation, generates the current low-level subtask, and samples "
            "an action chunk."
        )
    )
    parser.add_argument("--config-name", default="pi05_maniparena_ee", help="Named training config to load.")
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help=(
            "Checkpoint directory. You can pass either a concrete step directory "
            "(containing params/assets) or the experiment root; in the latter case "
            "the latest numeric step subdirectory will be used."
        ),
    )
    parser.add_argument("--face-image", required=True, help="Path to ManipArena front camera image.")
    parser.add_argument("--left-image", required=True, help="Path to ManipArena left wrist image.")
    parser.add_argument("--right-image", required=True, help="Path to ManipArena right wrist image.")
    parser.add_argument(
        "--state-npy",
        help="Optional .npy file containing the raw ManipArena state vector.",
    )
    parser.add_argument(
        "--state-json",
        help="Optional .json file containing the raw ManipArena state vector as a list.",
    )
    parser.add_argument(
        "--state-values",
        nargs="+",
        type=float,
        help="Optional raw state values passed directly on the command line.",
    )
    parser.add_argument(
        "--prompt",
        help=(
            "Exact prompt string to feed into the tokenizer. "
            "If omitted, --task/--history will be used to format "
            "the history-aware prompt."
        ),
    )
    parser.add_argument("--task", help="High-level task text. Used when --prompt is not provided.")
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        help="Past subtasks. Can be passed multiple times. Used when --prompt is not provided.",
    )
    parser.add_argument(
        "--low-level-prompt",
        default="",
        help=(
            "Optional current low-level prompt. Leave empty to let sample_low_level_task "
            "generate the current subtask autoregressively."
        ),
    )
    parser.add_argument("--num-steps", type=int, default=10, help="Number of denoising steps for action sampling.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output-npy",
        help="Optional path to save the final unnormalized action chunk as a .npy file.",
    )
    return parser.parse_args()


def _resolve_checkpoint_dir(checkpoint_dir: str | pathlib.Path) -> pathlib.Path:
    checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()
    if (checkpoint_dir / "params").exists():
        return checkpoint_dir

    numeric_children = [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.isdigit() and (p / "params").exists()]
    if not numeric_children:
        raise FileNotFoundError(
            f"Could not find a usable checkpoint under {checkpoint_dir}. "
            "Pass either a concrete step directory containing 'params' or an experiment root with numeric step folders."
        )
    return max(numeric_children, key=lambda p: int(p.name))


def _load_image(path: str | pathlib.Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _load_state(args: argparse.Namespace) -> np.ndarray:
    if args.state_values is not None:
        return np.asarray(args.state_values, dtype=np.float32)
    if args.state_npy is not None:
        return np.asarray(np.load(args.state_npy), dtype=np.float32).reshape(-1)
    if args.state_json is not None:
        with open(args.state_json, "r", encoding="utf-8") as f:
            return np.asarray(json.load(f), dtype=np.float32).reshape(-1)
    raise ValueError("One of --state-values, --state-npy, or --state-json must be provided.")


def _format_history_prompt(task: str, history: list[str]) -> str:
    task_text = str(task).strip().replace("_", " ")
    prompt_lines = [f"Task: {task_text}", "History:"]
    if history:
        for idx, history_item in enumerate(history, start=1):
            history_text = str(history_item).strip().replace("_", " ")
            prompt_lines.append(f"{idx}. {history_text}")
    else:
        prompt_lines.append("None")
    return "\n".join(prompt_lines)


def _build_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if not args.task:
        raise ValueError("Either --prompt or --task must be provided.")
    return _format_history_prompt(args.task, list(args.history))


def _find_state_source(train_config: _config.TrainConfig) -> str:
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    for transform in data_config.data_transforms.inputs:
        if isinstance(transform, maniparena_policy.ManipArenaInputs):
            return transform.state_source
    return "ee"


def _build_inference_transforms(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
) -> tuple[_transforms.DataTransformFn, _transforms.DataTransformFn]:
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("asset_id is required to load norm stats for inference.")

    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    input_transform = _transforms.compose(
        [
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    output_transform = _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )
    return input_transform, output_transform


def _decode_subtask(tokens: np.ndarray, max_token_len: int) -> str:
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=max_token_len)
    text = tokenizer.detokenize(tokens.astype(np.int32))
    if "Action:" in text:
        text = text.split("Action:", 1)[0]
    return text.strip()


def _prepare_raw_observation(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "observation.state": _load_state(args),
        "observation.images.faceImg": _load_image(args.face_image),
        "observation.images.leftImg": _load_image(args.left_image),
        "observation.images.rightImg": _load_image(args.right_image),
        "prompt": _build_prompt(args),
        "low_level_prompt": args.low_level_prompt,
    }


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint_dir)
    train_config = dataclasses.replace(_config.get_config(args.config_name))
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    input_transform, output_transform = _build_inference_transforms(train_config, checkpoint_dir)

    raw_obs = _prepare_raw_observation(args)
    processed_inputs = input_transform(raw_obs)
    batched_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], processed_inputs)
    observation = _model.Observation.from_dict(batched_inputs)

    rng = jax.random.key(args.seed)
    action_output = model.sample_actions(rng, observation, num_steps=args.num_steps)
    if not isinstance(action_output, tuple):
        raise TypeError("Expected Pi05 sample_actions() to return (actions, output_tokens).")

    actions, output_tokens = action_output
    outputs = {
        "state": np.asarray(batched_inputs["state"][0]),
        "actions": np.asarray(actions[0]),
    }
    outputs = output_transform(outputs)

    generated_subtask = _decode_subtask(np.asarray(output_tokens[0]), train_config.model.max_token_len)
    state_source = _find_state_source(train_config)

    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"config_name: {args.config_name}")
    print(f"state_source: {state_source}")
    print("generated_subtask:")
    print(generated_subtask)
    print("predicted_actions_shape:")
    print(outputs["actions"].shape)
    print("predicted_actions:")
    print(np.asarray(outputs["actions"], dtype=np.float32))

    if args.output_npy:
        output_path = pathlib.Path(args.output_npy).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, np.asarray(outputs["actions"], dtype=np.float32))
        print(f"saved_actions: {output_path}")


if __name__ == "__main__":
    main()
