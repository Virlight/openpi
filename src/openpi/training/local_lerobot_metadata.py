import logging
import pathlib

import cv2
from lerobot.common.datasets.compute_stats import get_feature_stats, sample_indices
from lerobot.common.datasets.utils import load_json, load_jsonlines, serialize_dict, write_jsonlines
import numpy as np
import packaging.version
import pandas as pd


def _stack_series(series: pd.Series) -> np.ndarray:
    values = series.tolist()
    first = np.asarray(values[0])
    if first.ndim == 0:
        return np.asarray(values)
    return np.stack([np.asarray(value) for value in values], axis=0)


def _compute_video_feature_stats(video_path: pathlib.Path, episode_length: int) -> dict[str, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    try:
        sampled = sample_indices(episode_length)
        frames = []
        for frame_idx in sampled:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            success, frame = capture.read()
            if not success or frame is None:
                raise ValueError(f"Failed to decode frame {frame_idx} from {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(np.transpose(frame, (2, 0, 1)))
    finally:
        capture.release()

    frame_array = np.asarray(frames, dtype=np.uint8)
    stats = get_feature_stats(frame_array, axis=(0, 2, 3), keepdims=True)
    return {
        key: value if key == "count" else np.squeeze(value / 255.0, axis=0)
        for key, value in stats.items()
    }


def _compute_episode_stats(dataset_root: pathlib.Path, info: dict, episode_index: int) -> dict[str, dict[str, np.ndarray]]:
    episode_chunk = episode_index // info["chunks_size"]
    parquet_path = dataset_root / info["data_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    dataframe = pd.read_parquet(parquet_path)
    episode_length = len(dataframe)

    episode_stats = {}
    for feature_key, feature_info in info["features"].items():
        dtype = feature_info["dtype"]
        if dtype == "video":
            video_path = dataset_root / info["video_path"].format(
                episode_chunk=episode_chunk,
                video_key=feature_key,
                episode_index=episode_index,
            )
            episode_stats[feature_key] = _compute_video_feature_stats(video_path, episode_length)
            continue

        if feature_key not in dataframe.columns:
            logging.debug("Skipping feature missing from parquet: %s", feature_key)
            continue

        feature_array = _stack_series(dataframe[feature_key])
        episode_stats[feature_key] = get_feature_stats(
            feature_array,
            axis=0,
            keepdims=feature_array.ndim == 1,
        )

    return episode_stats


def ensure_local_episodes_stats(repo_id: str, repo_root: str | pathlib.Path | None) -> pathlib.Path | None:
    """Generate missing LeRobot v2.1 episodes_stats metadata for a local dataset."""

    if repo_root is None:
        return None

    dataset_root = pathlib.Path(repo_root)
    episodes_stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    if episodes_stats_path.exists():
        return episodes_stats_path

    info = load_json(dataset_root / "meta" / "info.json")
    version = packaging.version.parse(info["codebase_version"])
    if version < packaging.version.parse("v2.1"):
        return None

    logging.info("Generating missing episodes_stats.jsonl for local dataset %s at %s", repo_id, dataset_root)
    episodes = load_jsonlines(dataset_root / "meta" / "episodes.jsonl")

    records = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        stats = _compute_episode_stats(dataset_root, info, episode_index)
        records.append({"episode_index": episode_index, "stats": serialize_dict(stats)})

    tmp_path = episodes_stats_path.with_suffix(".jsonl.tmp")
    write_jsonlines(records, tmp_path)
    tmp_path.replace(episodes_stats_path)
    return episodes_stats_path
