#!/usr/bin/env python3
"""Open-loop eval for small_vlm trained with subtask prompts.

This evaluates the action model directly, without the ManipArena websocket
server and without Qwen subtask prediction. For each LeRobot frame, it looks up
the ground-truth subtask label from annotation JSON and feeds that label as the
policy prompt.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SMALL_VLM_ROOT = Path(os.environ.get("SMALL_VLM_ROOT", WORKSPACE_ROOT / "small_vlm"))


def _load_episode(dataset_root: Path, episode_idx: int) -> pd.DataFrame:
    chunk = episode_idx // 1000
    path = dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_idx:06d}.parquet"
    return pd.read_parquet(path)


def _load_video_frames(dataset_root: Path, episode_idx: int, cam_key: str) -> list[np.ndarray]:
    chunk = episode_idx // 1000
    fname = f"episode_{episode_idx:06d}.mp4"
    chunk_dir = dataset_root / "videos" / f"chunk-{chunk:03d}"
    path = chunk_dir / cam_key / fname
    if not path.exists():
        path = chunk_dir / cam_key.rsplit(".", 1)[-1] / fname
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {cam_key} under {chunk_dir}")

    container = av.open(str(path))
    try:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    finally:
        container.close()


def _task_rel_path(dataset_root: Path) -> Path:
    parts = dataset_root.resolve().parts
    for marker in ("real", "sim"):
        if marker in parts:
            return Path(*parts[parts.index(marker):])
    return Path(dataset_root.name)


def _find_annotation_path(annotation_root: Path, dataset_root: Path, episode_idx: int) -> Path:
    chunk = episode_idx // 1000
    rel_task = _task_rel_path(dataset_root)
    rel_path = Path(f"chunk-{chunk:03d}") / f"episode_{episode_idx:06d}.json"
    candidates = [
        annotation_root / rel_task / rel_path,
        annotation_root / rel_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(annotation_root.rglob(f"episode_{episode_idx:06d}.json"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        rel_text = str(rel_task)
        for match in matches:
            if rel_text in str(match):
                return match
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find annotation JSON for episode {episode_idx}.\n"
        f"Tried:\n{tried}"
    )


def _load_subtask_spans(annotation_path: Path) -> list[tuple[int, int, str]]:
    item = json.loads(annotation_path.read_text(encoding="utf-8"))
    spans: list[tuple[int, int, str]] = []
    for subtask in item.get("subtasks", []):
        label = str(subtask.get("label", "")).strip()
        if not label:
            continue
        start = int(subtask["start_frame"])
        end = int(subtask["end_frame"])
        if end >= start:
            spans.append((start, end, label))
    spans.sort(key=lambda span: (span[0], span[1]))
    return spans


def _find_subtask(spans: list[tuple[int, int, str]], frame_idx: int) -> tuple[int, int, str] | None:
    for start, end, label in spans:
        if start <= frame_idx <= end:
            return start, end, label
    return None


def _load_policy(config_name: str, checkpoint: str, device: str) -> Any:
    small_vlm_src = str(SMALL_VLM_ROOT / "src")
    if small_vlm_src not in sys.path:
        sys.path.insert(0, small_vlm_src)

    from openpi.policies import policy_config
    from openpi.training import config as train_config

    cfg = train_config.get_config(config_name)
    return policy_config.create_trained_policy(
        cfg,
        checkpoint,
        default_prompt=None,
        pytorch_device=device,
    )


def _plot(gt: np.ndarray, pred: np.ndarray, save_path: Path, tag: str) -> None:
    dim = gt.shape[1]
    fig = plt.figure(figsize=(12, 3 * dim))
    for i in range(dim):
        ax = fig.add_subplot(dim, 1, i + 1)
        ax.plot(gt[:, i], label="Ground Truth", color="blue", linewidth=1.0)
        ax.plot(pred[:, i], label="Prediction", color="orange", linewidth=1.0)
        ax.set_title(f"{tag} - Dim {i + 1}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"[SAVED] {save_path}")


def run(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset)
    annotation_root = Path(args.annotation_root)

    print(f"[1/5] Loading episode {args.episode} from {dataset_root} ...")
    dataframe = _load_episode(dataset_root, args.episode)
    frame_indices = np.asarray(dataframe["frame_index"].tolist(), dtype=np.int64)
    annotation_path = _find_annotation_path(annotation_root, dataset_root, args.episode)
    spans = _load_subtask_spans(annotation_path)
    print(f"       Annotation: {annotation_path}")
    print(f"       Subtasks: {len(spans)}  Frames: {len(dataframe)}")

    print("[2/5] Loading videos ...")
    front = _load_video_frames(dataset_root, args.episode, "observation.images.faceImg")
    left = _load_video_frames(dataset_root, args.episode, "observation.images.leftImg")
    right = _load_video_frames(dataset_root, args.episode, "observation.images.rightImg")
    num_frames = min(len(dataframe), len(front), len(left), len(right))
    if args.max_steps > 0:
        num_frames = min(num_frames, args.max_steps)
    print(f"       Usable frames: {num_frames}")

    print(f"[3/5] Loading policy config={args.config_name} checkpoint={args.checkpoint} ...")
    policy = _load_policy(args.config_name, args.checkpoint, args.device)

    print(f"[4/5] Running subtask-conditioned open-loop (chunk={args.action_chunk}) ...")
    gt_all: list[np.ndarray] = []
    pred_all: list[np.ndarray] = []
    eval_frames: list[int] = []
    eval_subtasks: list[str] = []
    idx = 0
    infer_count = 0
    skipped = 0

    while idx < num_frames:
        frame_idx = int(frame_indices[idx])
        span = _find_subtask(spans, frame_idx)
        if span is None:
            skipped += 1
            idx += 1
            continue

        _, subtask_end, subtask = span
        state = np.asarray(dataframe["observation.state"].iloc[idx], dtype=np.float32)
        obs = {
            "observation.state": state[:14],
            "observation.images.faceImg": front[idx],
            "observation.images.leftImg": left[idx],
            "observation.images.rightImg": right[idx],
            "prompt": subtask,
        }
        result = policy.infer(obs)
        pred_chunk = np.asarray(result["actions"], dtype=np.float32)

        use_len = min(args.action_chunk, pred_chunk.shape[0], num_frames - idx)
        if not args.allow_cross_subtask:
            end_local = int(np.searchsorted(frame_indices, subtask_end, side="right"))
            use_len = min(use_len, max(1, end_local - idx))

        for k in range(use_len):
            gt_action = np.asarray(dataframe["action"].iloc[idx + k], dtype=np.float32)
            dim = min(pred_chunk.shape[1], gt_action.shape[0])
            gt_all.append(gt_action[:dim])
            pred_all.append(pred_chunk[k, :dim])
            eval_frames.append(int(frame_indices[idx + k]))
            eval_subtasks.append(subtask)

        infer_count += 1
        print(
            f"       infer #{infer_count}: idx={idx}, frame={frame_idx}, "
            f"chunk={use_len}, subtask={subtask!r}"
        )
        idx += use_len

    if not gt_all:
        raise RuntimeError("No frames were evaluated. Check annotation spans and episode frame_index.")

    gt_arr = np.asarray(gt_all, dtype=np.float32)
    pred_arr = np.asarray(pred_all, dtype=np.float32)

    print(f"[5/5] Saving results ({infer_count} inferences, {len(gt_arr)} steps, skipped={skipped}) ...")
    stem = Path(args.save_dir) / f"{args.tag}_ep{args.episode:03d}"
    _plot(gt_arr, pred_arr, stem.with_suffix(".jpg"), args.tag)
    np.savez(
        stem.with_suffix(".npz"),
        gt=gt_arr,
        pred=pred_arr,
        frame_index=np.asarray(eval_frames, dtype=np.int64),
        subtask=np.asarray(eval_subtasks, dtype=object),
        annotation_path=str(annotation_path),
        checkpoint=str(args.checkpoint),
        config_name=str(args.config_name),
    )
    print(f"[SAVED] {stem.with_suffix('.npz')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Subtask-conditioned open-loop eval for small_vlm")
    parser.add_argument("--checkpoint", required=True, help="Trained small_vlm checkpoint step directory")
    parser.add_argument("--dataset", required=True, help="LeRobot task dataset root")
    parser.add_argument("--annotation-root", required=True, help="Annotation root or per-task annotation root")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--config-name", default="pi05_maniparena_ee")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-dir", default="openloop_plots")
    parser.add_argument("--tag", default="subtask_openloop")
    parser.add_argument("--max-steps", type=int, default=0, help="0 = all frames")
    parser.add_argument("--action-chunk", type=int, default=32)
    parser.add_argument(
        "--allow-cross-subtask",
        action="store_true",
        help="Allow one predicted chunk to be compared beyond the current subtask end frame.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
