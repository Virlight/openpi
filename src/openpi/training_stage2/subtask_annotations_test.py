import json

import numpy as np

from openpi.training_stage2 import subtask_annotations


def test_load_subtask_spans_filters_idle_and_keeps_padding_inside_active_span(tmp_path):
    annotation_path = tmp_path / "real" / "task" / "chunk-000" / "episode_000000.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(
        json.dumps(
            {
                "task_path": "real/task",
                "active_start": 2,
                "active_end": 30,
                "subtasks": [
                    {"label": "idle", "start_frame": 0, "end_frame": 1},
                    {"label": "first", "start_frame": 2, "end_frame": 10},
                    {"label": "second", "start_frame": 11, "end_frame": 30},
                    {"label": "idle", "start_frame": 31, "end_frame": 40},
                ],
            }
        ),
        encoding="utf-8",
    )

    spans_by_key = subtask_annotations.load_subtask_spans(tmp_path)

    assert spans_by_key["real/task/data/chunk-000/episode_000000.parquet"] == [
        (2, 15, "first"),
        (6, 30, "second"),
    ]


def test_load_subtask_spans_does_not_pad_past_short_neighbor(tmp_path):
    annotation_path = tmp_path / "real" / "task" / "chunk-000" / "episode_000001.json"
    annotation_path.parent.mkdir(parents=True)
    annotation_path.write_text(
        json.dumps(
            {
                "task_path": "real/task",
                "subtasks": [
                    {"label": "first", "start_frame": 2, "end_frame": 10},
                    {"label": "tiny", "start_frame": 11, "end_frame": 13},
                    {"label": "third", "start_frame": 14, "end_frame": 30},
                ],
            }
        ),
        encoding="utf-8",
    )

    spans_by_key = subtask_annotations.load_subtask_spans(tmp_path)

    assert spans_by_key["real/task/data/chunk-000/episode_000001.parquet"] == [
        (2, 13, "first"),
        (6, 18, "tiny"),
        (11, 30, "third"),
    ]


def test_sample_indices_for_subtask_spans_keeps_overlap_per_subtask():
    frame_indices = np.arange(8)
    spans = [(0, 4, "first"), (3, 6, "second")]

    samples = subtask_annotations.sample_indices_for_subtask_spans(frame_indices, spans)

    assert samples == [
        (0, (0, 4, "first")),
        (1, (0, 4, "first")),
        (2, (0, 4, "first")),
        (3, (0, 4, "first")),
        (4, (0, 4, "first")),
        (3, (3, 6, "second")),
        (4, (3, 6, "second")),
        (5, (3, 6, "second")),
        (6, (3, 6, "second")),
    ]
