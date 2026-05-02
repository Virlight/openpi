"""Stage2 data transforms."""

import dataclasses
import pathlib

import numpy as np

from openpi.training_stage2 import subtask_annotations
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class PromptFromSubtaskAnnotations(_transforms.DataTransformFn):
    """Override prompt with the subtask label covering the current frame."""

    annotations_dir: str
    _cache: dict[str, list[subtask_annotations.SubtaskSpan]] = dataclasses.field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _loaded: dict[str, bool] = dataclasses.field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def _load_cache(self) -> None:
        if self._loaded.get("done"):
            return

        self._cache.update(subtask_annotations.load_subtask_spans(self.annotations_dir))
        self._loaded["done"] = True

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if "_subtask_label" in data:
            label = data["_subtask_label"]
            if not isinstance(label, str):
                label = str(label.item() if hasattr(label, "item") else label)
            if label:
                return {**data, "prompt": label}

        if not self.annotations_dir:
            return data
        if "_parquet_path" not in data or "frame_index" not in data:
            return data

        self._load_cache()
        parquet_path = data["_parquet_path"]
        if not isinstance(parquet_path, str):
            parquet_path = str(parquet_path.item() if hasattr(parquet_path, "item") else parquet_path)

        spans = subtask_annotations.spans_for_parquet(self._cache, pathlib.Path(parquet_path))
        if not spans:
            return data

        frame_idx = int(np.asarray(data["frame_index"]).item())
        span = subtask_annotations.find_subtask_span_in_spans(spans, frame_idx)
        if span is None:
            return data
        return {**data, "prompt": span[2]}
