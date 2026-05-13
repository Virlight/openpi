import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _select_state_slice(state: np.ndarray, state_source: str) -> np.ndarray:
    if state_source == "ee":
        return state[..., :14]
    if state_source == "joints":
        return state[..., 14:28]
    raise ValueError(f"Unsupported ManipArena state source: {state_source}")


_EE_DIM = 14
_TABLETOP_STATE_DIM = 56
_FULL_STATE_DIM = 62
_MOBILE_STATE_START = 56
_MOBILE_DIM = 6
_MOBILE_OBS_DIM = 3
_COMPACT_MOBILE_STATE_DIM = _EE_DIM + _MOBILE_OBS_DIM
_ALL_OBS_ACTION_DIM = _EE_DIM + _MOBILE_DIM
_EE_DELTA_MASK = np.asarray(transforms.make_bool_mask(6, -1, 6, -1), dtype=bool)
_MOBILE_DELTA_MASK = np.asarray(
    transforms.make_bool_mask(_MOBILE_OBS_DIM, -(_MOBILE_DIM - _MOBILE_OBS_DIM)),
    dtype=bool,
)


def _pad_state_to_full(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[-1] == _TABLETOP_STATE_DIM:
        return np.pad(state, (0, _FULL_STATE_DIM - _TABLETOP_STATE_DIM))
    if state.shape[-1] >= _FULL_STATE_DIM:
        return state[..., :_FULL_STATE_DIM]
    raise ValueError(f"ManipArena full-state input expects 56D or 62D state, got {state.shape[-1]}D")


def _select_all_obs_actions(actions: np.ndarray, *, has_mobile: bool) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if has_mobile and actions.shape[-1] >= _FULL_STATE_DIM:
        return np.concatenate(
            [actions[..., :_EE_DIM], actions[..., _MOBILE_STATE_START:_FULL_STATE_DIM]],
            axis=-1,
        )
    if has_mobile and actions.shape[-1] >= _ALL_OBS_ACTION_DIM:
        return actions[..., :_ALL_OBS_ACTION_DIM]
    if not has_mobile and actions.shape[-1] >= _EE_DIM:
        return actions[..., :_EE_DIM]
    expected = (
        f"{_ALL_OBS_ACTION_DIM}D or {_FULL_STATE_DIM}D" if has_mobile else f"{_EE_DIM}D or {_TABLETOP_STATE_DIM}D"
    )
    raise ValueError(f"ManipArena all-obs action input expects {expected} actions, got {actions.shape[-1]}D")


def _all_obs_action_reference(state: np.ndarray, *, has_mobile: bool) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if not has_mobile:
        return state[..., :_EE_DIM]
    if state.shape[-1] >= _FULL_STATE_DIM:
        return np.concatenate(
            [state[..., :_EE_DIM], state[..., _MOBILE_STATE_START:_FULL_STATE_DIM]],
            axis=-1,
        )
    if _COMPACT_MOBILE_STATE_DIM <= state.shape[-1] < _TABLETOP_STATE_DIM:
        mobile_padding = np.zeros((*state.shape[:-1], _MOBILE_DIM - _MOBILE_OBS_DIM), dtype=state.dtype)
        return np.concatenate([state[..., :_COMPACT_MOBILE_STATE_DIM], mobile_padding], axis=-1)
    if state.shape[-1] < _FULL_STATE_DIM:
        state = _pad_state_to_full(state)
    return np.concatenate([state[..., :_EE_DIM], state[..., _MOBILE_STATE_START:_FULL_STATE_DIM]], axis=-1)


def _all_obs_compact_state(state: np.ndarray, *, has_mobile: bool) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if not has_mobile and _EE_DIM <= state.shape[-1] < _TABLETOP_STATE_DIM:
        return state[..., :_EE_DIM]
    if has_mobile and _COMPACT_MOBILE_STATE_DIM <= state.shape[-1] < _TABLETOP_STATE_DIM:
        return state[..., :_COMPACT_MOBILE_STATE_DIM]
    state = _pad_state_to_full(state)
    if not has_mobile:
        return state[..., :_EE_DIM]
    return np.concatenate(
        [state[..., :_EE_DIM], state[..., _MOBILE_STATE_START : _MOBILE_STATE_START + _MOBILE_OBS_DIM]],
        axis=-1,
    )


def _all_obs_action_delta_mask(*, has_mobile: bool) -> np.ndarray:
    if not has_mobile:
        return _EE_DELTA_MASK
    return np.concatenate([_EE_DELTA_MASK, _MOBILE_DELTA_MASK], axis=0)


def _has_mobile(data: dict) -> bool:
    return bool(np.asarray(data.get("_maniparena_has_mobile", True)).item())


@dataclasses.dataclass(frozen=True)
class ManipArenaInputs(transforms.DataTransformFn):
    """Adapt ManipArena observations/actions to the OpenPI policy interface."""

    model_type: _model.ModelType
    state_source: str = "ee"
    include_images: bool = True

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation.state"], dtype=np.float32).reshape(-1)
        state = _select_state_slice(state, self.state_source)
        inputs = {"state": state}

        if self.include_images:
            front = _parse_image(data["observation.images.faceImg"])
            left = _parse_image(data["observation.images.leftImg"])
            right = _parse_image(data["observation.images.rightImg"])
            inputs["image"] = {
                "base_0_rgb": front,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            }
            inputs["image_mask"] = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            inputs["actions"] = _select_state_slice(actions, self.state_source)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class ManipArenaOutputs(transforms.DataTransformFn):
    """Trim model outputs back to ManipArena's 14D action space."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14], dtype=np.float32)}


@dataclasses.dataclass(frozen=True)
class ManipArenaAllObsInputs(transforms.DataTransformFn):
    """Use compact ManipArena state and compact action targets."""

    model_type: _model.ModelType
    include_images: bool = True

    def __call__(self, data: dict) -> dict:
        raw_state = np.asarray(data["observation.state"], dtype=np.float32).reshape(-1)
        has_mobile = raw_state.shape[-1] >= _FULL_STATE_DIM
        inputs = {
            "state": _all_obs_compact_state(raw_state, has_mobile=has_mobile),
            "_maniparena_has_mobile": np.asarray(has_mobile, dtype=bool),
        }

        if self.include_images:
            front = _parse_image(data["observation.images.faceImg"])
            left = _parse_image(data["observation.images.leftImg"])
            right = _parse_image(data["observation.images.rightImg"])
            inputs["image"] = {
                "base_0_rgb": front,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            }
            inputs["image_mask"] = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            }

        if "actions" in data:
            inputs["actions"] = _select_all_obs_actions(data["actions"], has_mobile=has_mobile)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class ManipArenaAllObsDeltaActions(transforms.DataTransformFn):
    """Convert compact all-obs actions to deltas using the matching state indices."""

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data

        actions = np.asarray(data["actions"], dtype=np.float32).copy()
        has_mobile = _has_mobile(data)
        reference = _all_obs_action_reference(data["state"], has_mobile=has_mobile)
        mask = _all_obs_action_delta_mask(has_mobile=has_mobile)
        actions[..., : mask.shape[-1]] -= np.expand_dims(np.where(mask, reference, 0), axis=-2)
        return {**data, "actions": actions}


@dataclasses.dataclass(frozen=True)
class ManipArenaAllObsAbsoluteActions(transforms.DataTransformFn):
    """Invert ManipArenaAllObsDeltaActions for policy outputs."""

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data

        actions = np.asarray(data["actions"], dtype=np.float32).copy()
        has_mobile = _has_mobile(data)
        reference = _all_obs_action_reference(data["state"], has_mobile=has_mobile)
        mask = _all_obs_action_delta_mask(has_mobile=has_mobile)
        actions[..., : mask.shape[-1]] += np.expand_dims(np.where(mask, reference, 0), axis=-2)
        if not has_mobile:
            actions[..., _EE_DIM : _EE_DIM + _MOBILE_DIM] = 0.0
        return {**data, "actions": actions}


@dataclasses.dataclass(frozen=True)
class ManipArenaAllObsOutputs(transforms.DataTransformFn):
    """Return 14D EE actions for tabletop samples and 20D EE+mobile actions otherwise."""

    def __call__(self, data: dict) -> dict:
        action_dim = _ALL_OBS_ACTION_DIM if _has_mobile(data) else _EE_DIM
        return {"actions": np.asarray(data["actions"][:, :action_dim], dtype=np.float32)}
