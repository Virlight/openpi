from collections.abc import Mapping
import dataclasses
import pathlib

import cv2
from lerobot.common.datasets.utils import get_delta_indices, load_json, load_jsonlines
import numpy as np
import pandas as pd
import torch


def _stack_series(series: pd.Series) -> np.ndarray:
    values = series.tolist()
    first = np.asarray(values[0])
    if first.ndim == 0:
        return np.asarray(values)
    return np.stack([np.asarray(value) for value in values], axis=0)


def _read_video_frame(video_path: pathlib.Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success or frame is None:
            raise ValueError(f"Failed to decode frame {frame_index} from {video_path}")
    finally:
        capture.release()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.transpose(frame, (2, 0, 1)).astype(np.float32) / 255.0


@dataclasses.dataclass(frozen=True)
class LocalLeRobotMetadata:
    repo_id: str
    root: pathlib.Path
    info: Mapping
    tasks: dict[int, str]
    episodes: list[int]

    @property
    def fps(self) -> float:
        return float(self.info["fps"])

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.info["features"].items() if ft["dtype"] == "video"]

    def get_data_file_path(self, episode_index: int) -> pathlib.Path:
        episode_chunk = episode_index // self.info["chunks_size"]
        return self.root / self.info["data_path"].format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )

    def get_video_file_path(self, episode_index: int, video_key: str) -> pathlib.Path:
        episode_chunk = episode_index // self.info["chunks_size"]
        return self.root / self.info["video_path"].format(
            episode_chunk=episode_chunk,
            video_key=video_key,
            episode_index=episode_index,
        )


def load_local_lerobot_metadata(repo_id: str, repo_root: str | pathlib.Path) -> LocalLeRobotMetadata:
    root = pathlib.Path(repo_root)
    info = load_json(root / "meta" / "info.json")
    tasks = {
        int(task["task_index"]): task["task"]
        for task in load_jsonlines(root / "meta" / "tasks.jsonl")
    }
    episodes = [int(episode["episode_index"]) for episode in load_jsonlines(root / "meta" / "episodes.jsonl")]
    return LocalLeRobotMetadata(repo_id=repo_id, root=root, info=info, tasks=tasks, episodes=episodes)


class LocalLeRobotDataset(torch.utils.data.Dataset):
    """Minimal local LeRobot loader for offline OpenPI training/inference."""

    def __init__(self, metadata: LocalLeRobotMetadata, delta_timestamps: dict[str, list[float]] | None = None):
        self.meta = metadata
        self.delta_indices = None if delta_timestamps is None else get_delta_indices(delta_timestamps, self.meta.fps)
        self._episode_frames = {}
        self._global_index = []

        for episode_index in self.meta.episodes:
            dataframe = pd.read_parquet(self.meta.get_data_file_path(episode_index))
            frame_data = {column: _stack_series(dataframe[column]) for column in dataframe.columns}
            self._episode_frames[episode_index] = frame_data
            self._global_index.extend((episode_index, local_idx) for local_idx in range(len(dataframe)))

    def __len__(self) -> int:
        return len(self._global_index)

    def __getitem__(self, index: int) -> dict:
        episode_index, local_idx = self._global_index[index]
        episode_frames = self._episode_frames[episode_index]
        item = {key: np.asarray(values[local_idx]) for key, values in episode_frames.items()}

        if self.delta_indices is not None:
            episode_length = len(next(iter(episode_frames.values())))
            for key, deltas in self.delta_indices.items():
                query_indices = [max(0, min(episode_length - 1, local_idx + delta)) for delta in deltas]
                item[key] = np.asarray(episode_frames[key][query_indices])
                item[f"{key}_is_pad"] = np.asarray(
                    [(local_idx + delta < 0) or (local_idx + delta >= episode_length) for delta in deltas],
                    dtype=bool,
                )

        current_frame = int(np.asarray(item["frame_index"]).item())
        for video_key in self.meta.video_keys:
            item[video_key] = _read_video_frame(self.meta.get_video_file_path(episode_index, video_key), current_frame)

        task_index = int(np.asarray(item["task_index"]).item())
        item["task"] = self.meta.tasks[task_index]
        return item
