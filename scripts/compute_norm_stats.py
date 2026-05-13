"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.policies.maniparena_policy as maniparena_policy
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

_ALL_OBS_EE_STATE_DIM = 14
_ALL_OBS_MOBILE_STATE_DIM = 3
_ALL_OBS_COMPACT_STATE_DIM = _ALL_OBS_EE_STATE_DIM + _ALL_OBS_MOBILE_STATE_DIM
_ALL_OBS_EE_ACTION_DIM = 14
_ALL_OBS_FULL_ACTION_DIM = 20
_ALL_OBS_MOBILE_MASK_KEY = "_maniparena_has_mobile"


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class KeepNormStatKeys(transforms.DataTransformFn):
    """Keep only the tensors needed for norm stats computation.

    This avoids batching irrelevant fields such as images, whose shapes may differ
    across datasets that otherwise share compatible state/action spaces. The
    ManipArena all-obs mobile mask is retained so padded mobile-only dimensions
    can be excluded from norm-stat updates.
    """

    def __call__(self, x: dict) -> dict:
        return {key: x[key] for key in ("state", "actions", _ALL_OBS_MOBILE_MASK_KEY) if key in x}


class PadManipArenaAllObsForNormStats(transforms.DataTransformFn):
    """Pad compact all-obs tensors to their max width for norm-stat batching."""

    def __call__(self, x: dict) -> dict:
        if _ALL_OBS_MOBILE_MASK_KEY not in x:
            return x
        output = dict(x)
        if "state" in output:
            output["state"] = transforms.pad_to_dim(output["state"], _ALL_OBS_COMPACT_STATE_DIM, axis=-1)
        if "actions" in output:
            output["actions"] = transforms.pad_to_dim(output["actions"], _ALL_OBS_FULL_ACTION_DIM, axis=-1)
        return output


def _concat_norm_stats(*stats: normalize.NormStats) -> normalize.NormStats:
    return normalize.NormStats(
        mean=np.concatenate([stat.mean for stat in stats], axis=-1),
        std=np.concatenate([stat.std for stat in stats], axis=-1),
        q01=np.concatenate([stat.q01 for stat in stats], axis=-1),
        q99=np.concatenate([stat.q99 for stat in stats], axis=-1),
    )


class ManipArenaAllObsNormStats:
    """Compute stats without letting tabletop padding affect mobile-only dimensions."""

    def __init__(self):
        self._state_ee = normalize.RunningStats()
        self._state_mobile = normalize.RunningStats()
        self._actions_ee = normalize.RunningStats()
        self._actions_mobile = normalize.RunningStats()

    def update(self, batch: dict) -> None:
        if _ALL_OBS_MOBILE_MASK_KEY not in batch:
            raise KeyError(f"Missing {_ALL_OBS_MOBILE_MASK_KEY!r} needed for masked all-obs norm stats.")

        state = np.asarray(batch["state"])
        actions = np.asarray(batch["actions"])
        has_mobile = np.asarray(batch[_ALL_OBS_MOBILE_MASK_KEY], dtype=bool).reshape(-1)

        if state.shape[-1] != _ALL_OBS_COMPACT_STATE_DIM:
            raise ValueError(f"Expected all-obs state dim {_ALL_OBS_COMPACT_STATE_DIM}, got {state.shape[-1]}.")
        if actions.shape[-1] != _ALL_OBS_FULL_ACTION_DIM:
            raise ValueError(f"Expected all-obs action dim {_ALL_OBS_FULL_ACTION_DIM}, got {actions.shape[-1]}.")
        if state.shape[0] != has_mobile.shape[0] or actions.shape[0] != has_mobile.shape[0]:
            raise ValueError("ManipArena mobile mask batch dimension does not match state/actions.")

        self._state_ee.update(state[..., :_ALL_OBS_EE_STATE_DIM])
        self._actions_ee.update(actions[..., :_ALL_OBS_EE_ACTION_DIM])

        if np.any(has_mobile):
            self._state_mobile.update(state[has_mobile, _ALL_OBS_EE_STATE_DIM:_ALL_OBS_COMPACT_STATE_DIM])
            self._actions_mobile.update(actions[has_mobile, ..., _ALL_OBS_EE_ACTION_DIM:_ALL_OBS_FULL_ACTION_DIM])

    def get_statistics(self) -> dict[str, normalize.NormStats]:
        return {
            "state": _concat_norm_stats(self._state_ee.get_statistics(), self._state_mobile.get_statistics()),
            "actions": _concat_norm_stats(
                self._actions_ee.get_statistics(),
                self._actions_mobile.get_statistics(),
            ),
        }


def _uses_maniparena_all_obs_inputs(data_config: _config.DataConfig) -> bool:
    return any(
        isinstance(transform, maniparena_policy.ManipArenaAllObsInputs)
        for transform in data_config.data_transforms.inputs
    )


def _optimize_data_config_for_norm_stats(
    data_config: _config.DataConfig,
) -> tuple[_config.DataConfig, bool]:
    """Trim image-heavy ManipArena transforms that are irrelevant for norm stats."""

    optimized_repack = list(data_config.repack_transforms.inputs)
    optimized_inputs = list(data_config.data_transforms.inputs)
    should_skip_videos = False

    maniparena_input_types = (
        maniparena_policy.ManipArenaInputs,
        maniparena_policy.ManipArenaAllObsInputs,
    )
    has_all_obs_inputs = any(
        isinstance(transform, maniparena_policy.ManipArenaAllObsInputs) for transform in optimized_inputs
    )
    if any(isinstance(transform, maniparena_input_types) for transform in optimized_inputs):
        optimized_repack = [
            transforms.RepackTransform(
                {
                    "observation.state": "observation.state",
                    "actions": "action",
                }
            )
            if isinstance(transform, transforms.RepackTransform)
            else transform
            for transform in optimized_repack
        ]
        optimized_inputs = [
            dataclasses.replace(transform, include_images=False)
            if isinstance(transform, maniparena_input_types)
            else transform
            for transform in optimized_inputs
        ]
        if has_all_obs_inputs:
            optimized_inputs.append(PadManipArenaAllObsForNormStats())
        should_skip_videos = True

    if not should_skip_videos:
        return data_config, False

    return (
        dataclasses.replace(
            data_config,
            repack_transforms=transforms.Group(
                inputs=tuple(optimized_repack),
                outputs=data_config.repack_transforms.outputs,
            ),
            data_transforms=transforms.Group(
                inputs=tuple(optimized_inputs),
                outputs=data_config.data_transforms.outputs,
            ),
        ),
        True,
    )


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    data_config, skip_videos = _optimize_data_config_for_norm_stats(data_config)
    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon,
        model_config,
        load_videos=not skip_videos,
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
            # Norm stats are only computed for state/actions, so drop images and other
            # dataset-specific fields before batching.
            KeepNormStatKeys(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
            # Norm stats are only computed for state/actions, so drop images and other
            # dataset-specific fields before batching.
            KeepNormStatKeys(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    batch_size = config.batch_size if batch_size is None else batch_size
    num_workers = config.num_workers if num_workers is None else num_workers

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            batch_size,
            config.model,
            num_workers,
            max_frames,
        )

    if _uses_maniparena_all_obs_inputs(data_config):
        stats = ManipArenaAllObsNormStats()
    else:
        stats = {key: normalize.RunningStats() for key in ("state", "actions")}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        if isinstance(stats, ManipArenaAllObsNormStats):
            stats.update(batch)
        else:
            for key, running_stats in stats.items():
                running_stats.update(np.asarray(batch[key]))

    if isinstance(stats, ManipArenaAllObsNormStats):
        norm_stats = stats.get_statistics()
    else:
        norm_stats = {key: running_stats.get_statistics() for key, running_stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
