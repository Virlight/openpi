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


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class KeepNormStatKeys(transforms.DataTransformFn):
    """Keep only the tensors needed for norm stats computation.

    This avoids batching irrelevant fields such as images, whose shapes may differ
    across datasets that otherwise share compatible state/action spaces.
    """

    def __call__(self, x: dict) -> dict:
        return {key: x[key] for key in ("state", "actions") if key in x}


def _optimize_data_config_for_norm_stats(
    data_config: _config.DataConfig,
) -> tuple[_config.DataConfig, bool]:
    """Trim image-heavy ManipArena transforms that are irrelevant for norm stats."""

    optimized_repack = list(data_config.repack_transforms.inputs)
    optimized_inputs = list(data_config.data_transforms.inputs)
    should_skip_videos = False

    if any(isinstance(transform, maniparena_policy.ManipArenaInputs) for transform in optimized_inputs):
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
            if isinstance(transform, maniparena_policy.ManipArenaInputs)
            else transform
            for transform in optimized_inputs
        ]
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

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
