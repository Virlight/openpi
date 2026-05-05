"""Small wandb logging helpers for training sanity checks."""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any

import numpy as np
import wandb

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

_MAX_PROMPT_SAMPLES = 5
_MAX_TEXT_CHARS = 4000
_TOKEN_PREVIEW_LEN = 80
_STATE_PREVIEW_LEN = 24

_SAMPLE_COLUMNS = [
    "sample_index",
    "episode_index",
    "frame_index",
    "subtask_end_frame",
    "prompt_source",
    "task_prompt",
    "llm_input",
    "state_dim",
    "normalized_state_preview",
    "discretized_state_preview",
    "action_shape",
    "padded_action_steps",
    "prompt_token_count",
    "prompt_truncated",
]

_COLUMN_DESCRIPTIONS = {
    "sample_index": "Index sampled from the training dataset after any dataset-level filtering.",
    "episode_index": "Original LeRobot episode index when available.",
    "frame_index": "Original frame index for the sampled timestep when available.",
    "subtask_end_frame": "End frame of the active stage2 subtask span when subtask annotations are used.",
    "prompt_source": "Where the prompt came from, e.g. stage2 subtask annotation, dataset prompt, or injected default.",
    "task_prompt": "Task/subtask text used before tokenizer formatting.",
    "llm_input": "Exact text preview constructed for the language tokenizer before token-id encoding.",
    "state_dim": "Number of normalized state values included in the discrete language input.",
    "normalized_state_preview": "First normalized continuous state values before 256-bin discretization.",
    "discretized_state_preview": "First discrete state tokens that appear in the PI05/FAST language input.",
    "action_shape": "Shape of the transformed action array available at tokenization time.",
    "padded_action_steps": "Number of future action steps padded by stage2 subtask clamping, when available.",
    "prompt_token_count": "Number of non-padding language tokens after tokenization.",
    "prompt_truncated": "Whether the tokenizer input reached max_token_len, indicating possible truncation.",
}


def log_prompt_samples(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    *,
    step: int = 0,
    num_samples: int = _MAX_PROMPT_SAMPLES,
) -> None:
    """Log a few prompt/tokenizer inputs to wandb without adding strings to the training batch."""
    if num_samples <= 0:
        return
    if data_config.rlds_data_dir is not None:
        logging.info("Skipping LLM prompt sample logging for RLDS data loaders.")
        return

    try:
        records = collect_prompt_samples(config, data_config, num_samples=num_samples)
    except Exception:
        logging.exception("Failed to collect LLM prompt samples for wandb.")
        return

    if not records:
        logging.info("No LLM prompt samples found to log to wandb.")
        return

    table = wandb.Table(columns=_SAMPLE_COLUMNS)
    for record in records:
        table.add_data(*(record.get(column, "") for column in _SAMPLE_COLUMNS))

    schema_table = wandb.Table(columns=["column", "meaning"])
    for column in _SAMPLE_COLUMNS:
        schema_table.add_data(column, _COLUMN_DESCRIPTIONS[column])

    wandb.log(
        {
            "llm_input_samples": table,
            "llm_input_sample_columns": schema_table,
        },
        step=step,
    )


def collect_prompt_samples(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    *,
    num_samples: int = _MAX_PROMPT_SAMPLES,
) -> list[dict[str, Any]]:
    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon=config.model.action_horizon,
        model_config=config.model,
    )
    transforms: list[_transforms.DataTransformFn] = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]

    records: list[dict[str, Any]] = []
    for sample_index in range(min(num_samples, len(dataset))):
        record = _collect_prompt_sample(dataset[sample_index], transforms, sample_index=sample_index)
        if record is not None:
            records.append(record)
    return records


def _collect_prompt_sample(
    data: _transforms.DataDict,
    transforms: Sequence[_transforms.DataTransformFn],
    *,
    sample_index: int,
) -> dict[str, Any] | None:
    source_data = data
    for transform in transforms:
        if isinstance(transform, _transforms.TokenizePrompt):
            record = _record_paligemma_prompt(source_data, data, transform, sample_index=sample_index)
            tokenized = transform(data)
            return {**record, **_token_stats(tokenized, record.get("max_token_len"))}
        if isinstance(transform, _transforms.TokenizeFASTInputs):
            record = _record_fast_prompt(source_data, data, transform, sample_index=sample_index)
            tokenized = transform(data)
            return {**record, **_token_stats(tokenized, record.get("max_token_len"))}
        data = transform(data)
    return None


def _record_paligemma_prompt(
    source_data: _transforms.DataDict,
    data: _transforms.DataDict,
    transform: _transforms.TokenizePrompt,
    *,
    sample_index: int,
) -> dict[str, Any]:
    if "prompt" not in data:
        raise ValueError("Cannot log LLM prompt sample because the transformed data has no 'prompt' key.")

    raw_prompt = _as_text(data["prompt"])
    cleaned_prompt = raw_prompt.strip().replace("_", " ").replace("\n", " ")
    state = data.get("state") if transform.discrete_state_input else None
    if state is None:
        llm_input = f"{cleaned_prompt}\n"
        prompt_template = "{prompt}\\n"
        note = "No explicit system prompt is configured. Pi0 tokenizes the user prompt followed by a newline."
    else:
        state_str = _discretized_state_text(state)
        llm_input = f"Task: {cleaned_prompt}, State: {state_str};\nAction: "
        prompt_template = "Task: {prompt}, State: {discretized_state};\\nAction: "
        note = (
            "No explicit system prompt is configured. Pi05 tokenizes the user prompt with the discretized state prefix."
        )

    return {
        **_source_metadata(source_data, data),
        "sample_index": sample_index,
        "prompt_source": _prompt_source(source_data),
        "tokenizer": type(transform.tokenizer).__name__,
        "max_token_len": getattr(transform.tokenizer, "_max_len", ""),
        "chat_system_prompt": "<none>",
        "raw_prompt": _truncate(raw_prompt),
        "cleaned_prompt": _truncate(cleaned_prompt),
        "task_prompt": _truncate(cleaned_prompt),
        "prompt_template": prompt_template,
        "llm_input": _truncate(llm_input),
        **_state_metadata(state),
        **_action_metadata(source_data, data),
        "note": note,
    }


def _record_fast_prompt(
    source_data: _transforms.DataDict,
    data: _transforms.DataDict,
    transform: _transforms.TokenizeFASTInputs,
    *,
    sample_index: int,
) -> dict[str, Any]:
    if "prompt" not in data:
        raise ValueError("Cannot log LLM prompt sample because the transformed data has no 'prompt' key.")
    if "state" not in data:
        raise ValueError("Cannot log FAST prompt sample because the transformed data has no 'state' key.")

    raw_prompt = _as_text(data["prompt"])
    cleaned_prompt = raw_prompt.lower().strip().replace("_", " ")
    state_str = _discretized_state_text(data["state"])
    llm_input = f"Task: {cleaned_prompt}, State: {state_str};\n"
    prompt_template = "Task: {prompt}, State: {discretized_state};\\n"
    if data.get("actions") is not None:
        llm_input = f"{llm_input}Action: <FAST action tokens>|"
        prompt_template = f"{prompt_template}Action: {{FAST action tokens}}|"

    return {
        **_source_metadata(source_data, data),
        "sample_index": sample_index,
        "prompt_source": _prompt_source(source_data),
        "tokenizer": type(transform.tokenizer).__name__,
        "max_token_len": getattr(transform.tokenizer, "_max_len", ""),
        "chat_system_prompt": "<none>",
        "raw_prompt": _truncate(raw_prompt),
        "cleaned_prompt": _truncate(cleaned_prompt),
        "task_prompt": _truncate(cleaned_prompt),
        "prompt_template": prompt_template,
        "llm_input": _truncate(llm_input),
        **_state_metadata(data["state"]),
        **_action_metadata(source_data, data),
        "note": (
            "No explicit system prompt is configured. FAST action target tokens are summarized instead of expanded."
        ),
    }


def _token_stats(data: _transforms.DataDict, max_token_len: Any) -> dict[str, Any]:
    tokens = np.asarray(data.get("tokenized_prompt", []))
    token_mask = np.asarray(data.get("tokenized_prompt_mask", []), dtype=bool)
    valid_count = int(token_mask.sum()) if token_mask.size else int(tokens.size)
    valid_tokens = tokens[:valid_count]
    preview_tokens = valid_tokens[:_TOKEN_PREVIEW_LEN]
    preview = " ".join(str(int(token)) for token in preview_tokens)
    if valid_tokens.size > _TOKEN_PREVIEW_LEN:
        preview += f" ... (+{valid_tokens.size - _TOKEN_PREVIEW_LEN} more)"

    loss_mask = data.get("token_loss_mask")
    target_count = int(np.asarray(loss_mask, dtype=bool).sum()) if loss_mask is not None else 0
    max_len = int(max_token_len) if max_token_len != "" else 0

    return {
        "valid_token_count": valid_count,
        "prompt_token_count": valid_count,
        "target_token_count": target_count,
        "input_truncated": bool(max_len and valid_count >= max_len),
        "prompt_truncated": bool(max_len and valid_count >= max_len),
        "token_ids_preview": preview,
    }


def _source_metadata(source_data: _transforms.DataDict, data: _transforms.DataDict) -> dict[str, Any]:
    del data
    return {
        "episode_index": _optional_scalar(source_data.get("_episode_index")),
        "frame_index": _optional_scalar(source_data.get("frame_index")),
        "subtask_end_frame": _optional_scalar(source_data.get("_subtask_end_frame")),
    }


def _prompt_source(source_data: _transforms.DataDict) -> str:
    if "_subtask_label" in source_data:
        return "stage2_subtask_annotation"
    if "prompt" in source_data:
        return "dataset_prompt"
    if "task" in source_data or "task_index" in source_data:
        return "lerobot_task_or_default"
    return "injected_default_or_unknown"


def _state_metadata(state: Any) -> dict[str, Any]:
    if state is None:
        return {
            "state_dim": 0,
            "normalized_state_preview": "",
            "discretized_state_preview": "",
        }

    state_array = np.asarray(state).reshape(-1)
    discretized_state = _discretized_state(state_array)
    return {
        "state_dim": int(state_array.size),
        "normalized_state_preview": _array_preview(state_array, precision=4),
        "discretized_state_preview": _array_preview(discretized_state, precision=0),
    }


def _action_metadata(source_data: _transforms.DataDict, data: _transforms.DataDict) -> dict[str, Any]:
    actions = data.get("actions")
    action_is_pad = source_data.get("action_is_pad")
    return {
        "action_shape": _shape_text(actions),
        "action_padding_count": _padding_count(action_is_pad),
        "padded_action_steps": _padding_count(action_is_pad),
    }


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value

    array = np.asarray(value)
    if array.shape == ():
        item = array.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    return str(value)


def _optional_scalar(value: Any) -> Any:
    if value is None:
        return ""
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return _array_preview(array.reshape(-1), precision=0)


def _shape_text(value: Any) -> str:
    if value is None:
        return ""
    return "x".join(str(dim) for dim in np.asarray(value).shape)


def _padding_count(value: Any) -> Any:
    if value is None:
        return ""
    return int(np.asarray(value, dtype=bool).sum())


def _discretized_state_text(state: Any) -> str:
    state_array = np.asarray(state).reshape(-1)
    discretized_state = _discretized_state(state_array)
    return " ".join(str(int(value)) for value in discretized_state)


def _discretized_state(state_array: np.ndarray) -> np.ndarray:
    return np.digitize(state_array, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1


def _array_preview(array: np.ndarray, *, precision: int) -> str:
    flat = np.asarray(array).reshape(-1)
    values = flat[:_STATE_PREVIEW_LEN]
    if precision == 0:
        text = " ".join(str(int(value)) for value in values)
    else:
        text = " ".join(f"{float(value):.{precision}f}" for value in values)
    if flat.size > _STATE_PREVIEW_LEN:
        text = f"{text} ... (+{flat.size - _STATE_PREVIEW_LEN} more)"
    return text


def _truncate(text: str, max_chars: int = _MAX_TEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n...[truncated {omitted} chars]"
