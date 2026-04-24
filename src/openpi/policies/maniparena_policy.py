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
