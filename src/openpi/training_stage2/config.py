"""Stage2 training configs.

The default training configs in ``openpi.training.config`` remain the single-stage
entry points. This module contains the stage2 variants and any extra data
configuration fields they need.
"""

import dataclasses
import difflib
import os
import pathlib
from typing import TypeAlias

import etils.epath as epath
from typing_extensions import override
import tyro

import openpi.methods.text_cfg as text_cfg
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.maniparena_policy as maniparena_policy
import openpi.training.config as _base_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

AssetsConfig: TypeAlias = _base_config.AssetsConfig
DataConfigFactory: TypeAlias = _base_config.DataConfigFactory
GroupFactory: TypeAlias = _base_config.GroupFactory
ModelTransformFactory: TypeAlias = _base_config.ModelTransformFactory
TrainConfig: TypeAlias = _base_config.TrainConfig

WORKSPACE_ROOT = _base_config.WORKSPACE_ROOT
OPENPI_ROOT = _base_config.OPENPI_ROOT
DEFAULT_MANIPARENA_ROOT = _base_config.DEFAULT_MANIPARENA_ROOT
DEFAULT_MANIPARENA_SUBTASK_ANNOTATIONS_ROOT = os.environ.get(
    "MANIPARENA_SUBTASK_ANNOTATIONS_ROOT",
    os.path.join(DEFAULT_MANIPARENA_ROOT, "annotations"),
)


@dataclasses.dataclass(frozen=True)
class Stage2DataConfig(_base_config.DataConfig):
    # If set, override the prompt with the ManipArena subtask label whose
    # [start_frame, end_frame] span contains the current frame_index.
    subtask_annotations_dir: str | None = None
    # Multi-dataset mode: list of (repo_id, annotation_root) pairs. This mirrors
    # extra_repos so each task only scans its own annotation JSON tree.
    extra_subtask_annotation_repos: tuple[tuple[str, str], ...] = ()
    # If true, only frames covered by an annotation subtask span become training samples.
    sample_only_subtask_frames: bool = False
    # If true, future action sequences are clamped to the current subtask end frame;
    # later steps repeat the end-frame action and are marked as padding.
    clamp_action_sequences_to_subtask: bool = False


@dataclasses.dataclass(frozen=True)
class EmptyGroupFactory:
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        return _transforms.Group()


@dataclasses.dataclass(frozen=True)
class Stage2MultiSimpleDataConfig(_base_config.DataConfigFactory):
    """Train on multiple local LeRobot datasets using stage2 subtask annotations."""

    # Used only as the norm-stats asset_id; not a real HuggingFace repo.
    repo_id: str = tyro.MISSING
    # All task datasets: (task_repo_id, local_repo_root)
    all_repos: tyro.conf.Suppress[tuple[tuple[str, str], ...]] = ()
    # All annotation datasets: (task_repo_id, local_annotation_root)
    all_annotation_repos: tyro.conf.Suppress[tuple[tuple[str, str], ...]] = ()
    # Factory for the data transforms (applied identically to every sub-dataset).
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=EmptyGroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)
    # Base stage2 config that will be updated by this factory.
    base_config: tyro.conf.Suppress[Stage2DataConfig | None] = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> Stage2DataConfig:
        base = dataclasses.replace(
            self.base_config or Stage2DataConfig(),
            repo_id=self.repo_id,
            repo_root=None,
            asset_id=self.repo_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), self.repo_id),
            use_quantile_norm=model_config.model_type != _base_config.ModelType.PI0,
        )
        return dataclasses.replace(
            base,
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            extra_repos=self.all_repos,
            extra_subtask_annotation_repos=self.all_annotation_repos,
        )


MANIPARENA_EE_REPOS = (
    (
        "classify_items_as_shape",
        os.path.join(DEFAULT_MANIPARENA_ROOT, "real/semantic_reasoning/classify_items_as_shape"),
    ),
    (
        "press_button_in_order",
        os.path.join(DEFAULT_MANIPARENA_ROOT, "real/semantic_reasoning/press_button_in_order"),
    ),
    (
        "put_blocks_to_color",
        os.path.join(DEFAULT_MANIPARENA_ROOT, "real/execution_reasoning/put_blocks_to_color"),
    ),
    (
        "put_ring_onto_rod",
        os.path.join(DEFAULT_MANIPARENA_ROOT, "real/execution_reasoning/put_ring_onto_rod"),
    ),
    (
        "put_spoon_to_bowl",
        os.path.join(DEFAULT_MANIPARENA_ROOT, "real/execution_reasoning/put_spoon_to_bowl"),
    ),
)

MANIPARENA_EE_ANNOTATION_REPOS = (
    (
        "classify_items_as_shape",
        os.path.join(
            DEFAULT_MANIPARENA_SUBTASK_ANNOTATIONS_ROOT,
            "real/semantic_reasoning/classify_items_as_shape",
        ),
    ),
    (
        "press_button_in_order",
        os.path.join(
            DEFAULT_MANIPARENA_SUBTASK_ANNOTATIONS_ROOT,
            "real/semantic_reasoning/press_button_in_order",
        ),
    ),
    (
        "put_blocks_to_color",
        os.path.join(
            DEFAULT_MANIPARENA_SUBTASK_ANNOTATIONS_ROOT,
            "real/execution_reasoning/put_blocks_to_color",
        ),
    ),
    (
        "put_ring_onto_rod",
        os.path.join(
            DEFAULT_MANIPARENA_SUBTASK_ANNOTATIONS_ROOT,
            "real/execution_reasoning/put_ring_onto_rod",
        ),
    ),
    (
        "put_spoon_to_bowl",
        os.path.join(
            DEFAULT_MANIPARENA_SUBTASK_ANNOTATIONS_ROOT,
            "real/execution_reasoning/put_spoon_to_bowl",
        ),
    ),
)


_CONFIGS = [
    TrainConfig(
        name="pi05_maniparena_ee_stage2",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_horizon=32,
        ),
        data=Stage2MultiSimpleDataConfig(
            # Stage2 keeps the same state/action normalization as stage1.
            assets=AssetsConfig(assets_dir=str(OPENPI_ROOT / "assets" / "pi05_maniparena_ee")),
            repo_id="maniparena_5_ee",
            all_repos=MANIPARENA_EE_REPOS,
            all_annotation_repos=MANIPARENA_EE_ANNOTATION_REPOS,
            data_transforms=lambda model: _transforms.Group(
                inputs=[maniparena_policy.ManipArenaInputs(model_type=model.model_type, state_source="ee")],
                outputs=[maniparena_policy.ManipArenaOutputs()],
            ).push(
                inputs=[_transforms.DeltaActions(_transforms.make_bool_mask(6, -1, 6, -1))],
                outputs=[_transforms.AbsoluteActions(_transforms.make_bool_mask(6, -1, 6, -1))],
            ),
            base_config=Stage2DataConfig(
                prompt_from_task=False,
                subtask_annotations_dir=os.environ.get(
                    "MANIPARENA_SUBTASK_ANNOTATIONS_DIR",
                    DEFAULT_MANIPARENA_SUBTASK_ANNOTATIONS_ROOT,
                ),
                sample_only_subtask_frames=True,
                clamp_action_sequences_to_subtask=True,
                action_sequence_keys=("action",),
                repack_transforms=_transforms.Group(
                    inputs=[
                        _transforms.RepackTransform(
                            {
                                "observation.state": "observation.state",
                                "observation.images.faceImg": "observation.images.faceImg",
                                "observation.images.leftImg": "observation.images.leftImg",
                                "observation.images.rightImg": "observation.images.rightImg",
                                "actions": "action",
                                "prompt": "prompt",
                            }
                        )
                    ]
                ),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.environ.get(
                "PI05_BASE_CHECKPOINT",
                "./checkpoints/pi05_base/params",
            )
        ),
        assets_base_dir=str(OPENPI_ROOT / "assets"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=7_000,
            decay_lr=5e-6,
        ),
        num_train_steps=7_000,
        save_interval=1_000,
        keep_period=1_000,
    ),
]

# 
# Add text-CFG variants after the base configs are defined. 
# ``with_text_cfg`` returns a new TrainConfig derived from the matched base config; it does not modify the original config already in ``_CONFIGS``.
# 
_CONFIGS.extend(
    [
        text_cfg.with_text_cfg(
            next(config for config in _CONFIGS if config.name == "pi05_maniparena_ee_stage2"),
            "pi05_maniparena_ee_stage2_text_cfg",
        ),
    ]
)

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Stage2 config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
