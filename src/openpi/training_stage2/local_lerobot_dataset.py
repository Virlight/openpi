"""Stage2 local LeRobot dataset with subtask annotation support."""

import numpy as np

from openpi.training.local_lerobot_dataset import LocalLeRobotDataset
from openpi.training.local_lerobot_dataset import LocalLeRobotMetadata
from openpi.training_stage2 import subtask_annotations

_IndexedSubtaskSpan = subtask_annotations.SubtaskSpan | None


class Stage2LocalLeRobotDataset(LocalLeRobotDataset):
    """Local LeRobot dataset extended with subtask span filtering and clamping."""

    def __init__(
        self,
        metadata: LocalLeRobotMetadata,
        delta_timestamps: dict[str, list[float]] | None = None,
        *,
        load_videos: bool = True,
        subtask_annotations_dir: str | None = None,
        sample_only_subtask_frames: bool = False,
        clamp_action_sequences_to_subtask: bool = False,
    ):
        super().__init__(metadata, delta_timestamps, load_videos=load_videos)

        self._subtask_annotations_dir = subtask_annotations_dir
        self._clamp_action_sequences_to_subtask = clamp_action_sequences_to_subtask
        self._subtask_spans_by_parquet = subtask_annotations.load_subtask_spans(subtask_annotations_dir)
        self._subtask_spans_by_episode: dict[int, list[subtask_annotations.SubtaskSpan]] = {}
        self._indexed_subtask_spans: list[_IndexedSubtaskSpan] = [None] * len(self._global_index)

        for episode_index in metadata.episodes:
            parquet_path = metadata.get_data_file_path(episode_index)
            self._subtask_spans_by_episode[episode_index] = subtask_annotations.spans_for_parquet(
                self._subtask_spans_by_parquet, parquet_path
            )

        if sample_only_subtask_frames:
            new_global_index: list[tuple[int, int]] = []
            new_indexed_subtask_spans: list[_IndexedSubtaskSpan] = []
            for episode_index, frame_data in self._episode_frames.items():
                spans = self._subtask_spans_by_episode[episode_index]
                samples = subtask_annotations.sample_indices_for_subtask_spans(frame_data["frame_index"], spans)
                new_global_index.extend((episode_index, local_idx) for local_idx, _ in samples)
                new_indexed_subtask_spans.extend(subtask_span for _, subtask_span in samples)
            self._global_index = new_global_index
            self._indexed_subtask_spans = new_indexed_subtask_spans

    def _indexed_subtask_span(self, index: int) -> _IndexedSubtaskSpan:
        if index < len(self._indexed_subtask_spans):
            return self._indexed_subtask_spans[index]
        return None

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)

        episode_index, local_idx = self._global_index[index]
        episode_frames = self._episode_frames[episode_index]
        current_frame = int(np.asarray(item["frame_index"]).item())
        subtask_span = self._indexed_subtask_span(index) or subtask_annotations.find_subtask_span_in_spans(
            self._subtask_spans_by_episode.get(episode_index, []),
            current_frame,
        )

        if self._subtask_annotations_dir:
            item["_episode_index"] = np.asarray(episode_index, dtype=np.int64)
            item["_local_index"] = np.asarray(local_idx, dtype=np.int64)
            item["_parquet_path"] = str(self.meta.get_data_file_path(episode_index))

        if subtask_span is None:
            return item

        _, subtask_end_frame, subtask_label = subtask_span
        item["_subtask_end_frame"] = np.asarray(subtask_end_frame, dtype=np.int64)
        item["_subtask_label"] = subtask_label

        if self._clamp_action_sequences_to_subtask and self.delta_indices is not None:
            episode_length = len(next(iter(episode_frames.values())))
            frame_indices = np.asarray(episode_frames["frame_index"]).reshape(-1)
            for key, deltas in self.delta_indices.items():
                query_indices = []
                is_pad = []
                for delta in deltas:
                    target_idx = max(0, min(episode_length - 1, local_idx + delta))
                    padded = (local_idx + delta < 0) or (local_idx + delta >= episode_length)
                    target_frame = int(np.asarray(frame_indices[target_idx]).item())
                    if target_frame > subtask_end_frame:
                        target_idx = int(np.searchsorted(frame_indices, subtask_end_frame, side="right") - 1)
                        target_idx = max(0, min(episode_length - 1, target_idx))
                        padded = True
                    query_indices.append(target_idx)
                    is_pad.append(padded)
                item[key] = np.asarray(episode_frames[key][query_indices])
                item[f"{key}_is_pad"] = np.asarray(is_pad, dtype=bool)

        return item
