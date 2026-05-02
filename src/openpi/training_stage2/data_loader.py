"""Stage2 dataset construction."""

import logging
import os

import torch

import openpi.models.model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _base_data_loader
from openpi.training import local_lerobot_dataset as _local_lerobot_dataset
from openpi.training import local_lerobot_metadata as _local_lerobot_metadata
from openpi.training_stage2.local_lerobot_dataset import Stage2LocalLeRobotDataset
from openpi.training_stage2.transforms import PromptFromSubtaskAnnotations
import openpi.transforms as _transforms


def is_stage2_data_config(data_config: _config.DataConfig) -> bool:
    return any(
        hasattr(data_config, name)
        for name in (
            "subtask_annotations_dir",
            "extra_subtask_annotation_repos",
            "sample_only_subtask_frames",
            "clamp_action_sequences_to_subtask",
        )
    )


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    load_videos: bool = True,
) -> _base_data_loader.Dataset:
    """Create the stage2 local LeRobot dataset with subtask prompt annotations."""
    del model_config

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create stage2 dataset.")

    if data_config.extra_repos:
        sub_datasets = [
            _create_single_local_dataset(
                sub_repo_id,
                sub_repo_root,
                action_horizon,
                data_config.action_sequence_keys,
                prompt_from_task=data_config.prompt_from_task,
                subtask_annotations_dir=_subtask_annotations_dir_for_repo(data_config, sub_repo_id),
                sample_only_subtask_frames=getattr(data_config, "sample_only_subtask_frames", False),
                clamp_action_sequences_to_subtask=getattr(
                    data_config,
                    "clamp_action_sequences_to_subtask",
                    False,
                ),
                load_videos=load_videos,
            )
            for sub_repo_id, sub_repo_root in data_config.extra_repos
        ]
        logging.info(
            "Stage2 multi-dataset mode: concatenating %d datasets (%s)",
            len(sub_datasets),
            ", ".join(r for r, _ in data_config.extra_repos),
        )
        return torch.utils.data.ConcatDataset(sub_datasets)

    if data_config.repo_root is None:
        raise ValueError("Stage2 datasets require local LeRobot data via repo_root or extra_repos.")

    return _create_single_local_dataset(
        repo_id,
        data_config.repo_root,
        action_horizon,
        data_config.action_sequence_keys,
        prompt_from_task=data_config.prompt_from_task,
        subtask_annotations_dir=getattr(data_config, "subtask_annotations_dir", None),
        sample_only_subtask_frames=getattr(data_config, "sample_only_subtask_frames", False),
        clamp_action_sequences_to_subtask=getattr(data_config, "clamp_action_sequences_to_subtask", False),
        load_videos=load_videos,
    )


def _create_single_local_dataset(
    repo_id: str,
    repo_root: str,
    action_horizon: int,
    action_sequence_keys: _config.Sequence[str],
    *,
    prompt_from_task: bool,
    subtask_annotations_dir: str | None = None,
    sample_only_subtask_frames: bool = False,
    clamp_action_sequences_to_subtask: bool = False,
    load_videos: bool = True,
) -> _base_data_loader.Dataset:
    _local_lerobot_metadata.ensure_local_episodes_stats(repo_id, repo_root)
    dataset_meta = _local_lerobot_dataset.load_local_lerobot_metadata(repo_id, repo_root)
    dataset: _base_data_loader.Dataset = Stage2LocalLeRobotDataset(
        dataset_meta,
        delta_timestamps={key: [t / dataset_meta.fps for t in range(action_horizon)] for key in action_sequence_keys},
        load_videos=load_videos,
        subtask_annotations_dir=subtask_annotations_dir,
        sample_only_subtask_frames=sample_only_subtask_frames,
        clamp_action_sequences_to_subtask=clamp_action_sequences_to_subtask,
    )
    if prompt_from_task:
        dataset = _base_data_loader.TransformedDataset(
            dataset,
            [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
        )
    if subtask_annotations_dir:
        dataset = _base_data_loader.TransformedDataset(
            dataset,
            [PromptFromSubtaskAnnotations(subtask_annotations_dir)],
        )
    return dataset


def _subtask_annotations_dir_for_repo(data_config: _config.DataConfig, repo_id: str) -> str | None:
    fallback = getattr(data_config, "subtask_annotations_dir", None)
    for annotation_repo_id, annotation_root in getattr(data_config, "extra_subtask_annotation_repos", ()):
        if annotation_repo_id == repo_id:
            if not annotation_root or os.path.exists(os.path.expanduser(annotation_root)):
                return annotation_root
            logging.warning(
                "Subtask annotation root for repo %s does not exist: %s. Falling back to %s.",
                repo_id,
                annotation_root,
                fallback,
            )
            return fallback
    return fallback
