"""Generate missing LeRobot metadata for local datasets.

Examples:
    python scripts/prepare_lerobot_metadata.py --config-name pi05_maniparena_ee
    python scripts/prepare_lerobot_metadata.py --repo-id put_blocks_to_color --repo-root /path/to/dataset
"""

import tyro

from openpi.training import config as _config
from openpi.training import local_lerobot_metadata as _local_lerobot_metadata


def main(
    config_name: str | None = None,
    repo_id: str | None = None,
    repo_root: str | None = None,
):
    if config_name is not None:
        train_config = _config.get_config(config_name)
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        repo_id = data_config.repo_id
        repo_root = data_config.repo_root

    if repo_id is None or repo_root is None:
        raise ValueError("Either provide --config-name or both --repo-id and --repo-root.")

    path = _local_lerobot_metadata.ensure_local_episodes_stats(repo_id, repo_root)
    if path is None:
        print("No metadata generation needed.")
    else:
        print(f"episodes_stats ready: {path}")


if __name__ == "__main__":
    tyro.cli(main)
