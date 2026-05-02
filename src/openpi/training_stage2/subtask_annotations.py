"""Utilities for ManipArena stage2 subtask annotations."""

from collections.abc import Mapping
import json
import pathlib

import numpy as np

SUBTASK_FRAME_PADDING = 5
IDLE_LABEL = "idle"
SubtaskSpan = tuple[int, int, str]


def path_match_keys(path: str | pathlib.Path) -> tuple[str, ...]:
    """Return equivalent path keys used to match annotation JSONs to parquet episodes."""
    path = pathlib.Path(str(path))
    keys = [str(path), str(path.expanduser().resolve(strict=False))]
    parts = path.parts
    keys.extend(str(pathlib.Path(*parts[parts.index(marker) :])) for marker in ("real", "sim") if marker in parts)
    if "data" in parts:
        idx = parts.index("data")
        if idx >= 1:
            keys.append(str(pathlib.Path(*parts[idx - 1 :])))
    return tuple(dict.fromkeys(keys))


def annotation_match_keys(root: pathlib.Path, path: pathlib.Path, item: Mapping) -> tuple[str, ...]:
    """Return parquet path keys implied by one annotation JSON."""
    keys: list[str] = []

    parquet_path = str(item.get("parquet") or item.get("parquet_path") or item.get("data_path") or "").strip()
    if parquet_path:
        keys.extend(path_match_keys(parquet_path))

    parts = path.parts
    chunk_idx = next((idx for idx, part in enumerate(parts) if part.startswith("chunk-")), None)
    if chunk_idx is None:
        return tuple(dict.fromkeys(keys))

    parquet_name = f"{path.stem}.parquet"
    chunk = parts[chunk_idx]
    task_path = str(item.get("task_path") or "").strip()
    if task_path:
        keys.extend(path_match_keys(pathlib.Path(task_path) / "data" / chunk / parquet_name))

    for marker in ("real", "sim"):
        if marker in parts and parts.index(marker) < chunk_idx:
            keys.extend(
                path_match_keys(pathlib.Path(*parts[parts.index(marker) : chunk_idx]) / "data" / chunk / parquet_name)
            )

    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        relative_parts = ()
    if relative_parts and relative_parts[0].startswith("chunk-"):
        keys.extend(path_match_keys(pathlib.Path("data") / relative_parts[0] / parquet_name))

    if chunk_idx >= 1:
        keys.extend(path_match_keys(pathlib.Path(parts[chunk_idx - 1]) / "data" / chunk / parquet_name))

    return tuple(dict.fromkeys(keys))


def _optional_int(item: Mapping, key: str) -> int | None:
    try:
        return int(item[key])
    except (KeyError, TypeError, ValueError):
        return None


def load_subtask_spans(annotations_dir: str | pathlib.Path | None) -> dict[str, list[SubtaskSpan]]:
    """Load annotation JSONs into a parquet-path-key to subtask-span mapping."""
    if not annotations_dir:
        return {}

    root = pathlib.Path(annotations_dir).expanduser()
    if not root.exists():
        return {}

    paths = [root] if root.is_file() else sorted(root.rglob("*.json"))
    cache: dict[str, list[SubtaskSpan]] = {}
    for path in paths:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not (isinstance(item, dict) and isinstance(item.get("subtasks"), list)):
            continue

        raw_spans: list[SubtaskSpan] = []
        for subtask in item["subtasks"]:
            label = str(subtask.get("label", "")).strip()
            if not label or label.casefold() == IDLE_LABEL:
                continue
            try:
                start = int(subtask["start_frame"])
                end = int(subtask["end_frame"])
            except (KeyError, TypeError, ValueError):
                continue
            if end >= start:
                raw_spans.append((start, end, label))
        if not raw_spans:
            continue

        raw_spans.sort(key=lambda span: (span[0], span[1]))
        raw_active_start = min(start for start, _, _ in raw_spans)
        raw_active_end = max(end for _, end, _ in raw_spans)
        active_start = _optional_int(item, "active_start")
        active_end = _optional_int(item, "active_end")
        if active_start is None or active_end is None or active_end < active_start:
            active_start = raw_active_start
            active_end = raw_active_end

        spans: list[SubtaskSpan] = []
        for idx, (start, end, label) in enumerate(raw_spans):
            previous_span = raw_spans[idx - 1] if idx > 0 else None
            next_span = raw_spans[idx + 1] if idx + 1 < len(raw_spans) else None

            left_bound = start
            if previous_span is not None and previous_span[1] + 1 >= start:
                left_bound = previous_span[0]

            right_bound = end
            if next_span is not None and next_span[0] <= end + 1:
                right_bound = next_span[1]

            padded_start = max(active_start, left_bound, start - SUBTASK_FRAME_PADDING)
            padded_end = min(active_end, right_bound, end + SUBTASK_FRAME_PADDING)
            if padded_end >= padded_start:
                spans.append((padded_start, padded_end, label))
        if not spans:
            continue

        for key in annotation_match_keys(root, path, item):
            cache[key] = spans
    return cache


def spans_for_parquet(
    spans_by_parquet: dict[str, list[SubtaskSpan]],
    parquet_path: pathlib.Path,
) -> list[SubtaskSpan]:
    for key in path_match_keys(parquet_path):
        spans = spans_by_parquet.get(key)
        if spans is not None:
            return spans
    return []


def find_subtask_span_in_spans(spans: list[SubtaskSpan], frame_idx: int) -> SubtaskSpan | None:
    for start, end, label in spans:
        if start <= frame_idx <= end:
            return start, end, label
    return None


def sample_indices_for_subtask_spans(
    frame_indices: np.ndarray,
    spans: list[SubtaskSpan],
) -> list[tuple[int, SubtaskSpan]]:
    if not spans:
        return []

    frame_indices = np.asarray(frame_indices).reshape(-1)
    sample_indices: list[tuple[int, SubtaskSpan]] = []
    for start, end, label in spans:
        left = int(np.searchsorted(frame_indices, start, side="left"))
        right = int(np.searchsorted(frame_indices, end, side="right"))
        sample_indices.extend(
            (local_idx, (start, end, label))
            for local_idx in range(max(0, left), min(len(frame_indices), right))
        )
    return sample_indices
