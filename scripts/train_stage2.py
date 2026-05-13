"""JAX training entry point for stage2 configs.

This wrapper keeps the default ``scripts/train.py`` entry point tied to
``openpi.training.config`` and uses ``openpi.training_stage2.config`` for stage2
training configs and ``openpi.training_stage2.data_loader`` for stage2 dataset handling.

Launch from the openpi repo root:

    cd /mnt/models/haoliang/CVPR2026-Workshop/openpi

    # Uses the existing stage-one norm stats:
    # assets/pi05_maniparena_ee/maniparena_5_ee/norm_stats.json

    # With memory preallocation (may cause OOM on some GPUs, adjust XLA_PYTHON_CLIENT_MEM_FRACTION as needed):
    CUDA_VISIBLE_DEVICES=1 \
    PI05_BASE_CHECKPOINT=/mnt/models/haoliang/CVPR2026-Workshop/openpi/checkpoints/pi05_maniparena_ee/run23/5000/params \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    python scripts/train_stage2.py pi05_maniparena_ee_stage2 \
        --exp-name subtask_run01 \
        --batch-size 1 \
        --overwrite

    # Without memory preallocation (may cause fragmentation and slower training, but more robust to OOM):
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    CUDA_VISIBLE_DEVICES=1 \
    PI05_BASE_CHECKPOINT=/mnt/models/haoliang/CVPR2026-Workshop/openpi/checkpoints/pi05_maniparena_ee/run23/5000/params \
    python scripts/train_stage2.py pi05_maniparena_ee_stage2 \
        --exp-name subtask_run01 \
        --batch-size 1 \
        --overwrite

    # Text-CFG variant:
    CUDA_VISIBLE_DEVICES=1 \
    PI05_BASE_CHECKPOINT=/mnt/models/haoliang/CVPR2026-Workshop/openpi/checkpoints/pi05_maniparena_ee/run23/5000/params \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    python scripts/train_stage2.py pi05_maniparena_ee_stage2_text_cfg \
        --exp-name subtask_text_cfg_run01 \
        --batch-size 1 \
        --overwrite
"""

import train as _train

import openpi.training.data_loader as _base_data_loader
import openpi.training_stage2.config as _config
import openpi.training_stage2.data_loader as _stage2_data_loader

_base_create_torch_dataset = _base_data_loader.create_torch_dataset


def _create_torch_dataset_with_stage2(data_config, action_horizon, model_config, *, load_videos=True):
    if _stage2_data_loader.is_stage2_data_config(data_config):
        return _stage2_data_loader.create_torch_dataset(
            data_config, action_horizon, model_config, load_videos=load_videos
        )
    return _base_create_torch_dataset(data_config, action_horizon, model_config, load_videos=load_videos)


_base_data_loader.create_torch_dataset = _create_torch_dataset_with_stage2


if __name__ == "__main__":
    _train.main(_config.cli())
