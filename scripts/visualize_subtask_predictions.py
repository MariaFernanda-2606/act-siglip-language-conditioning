#!/usr/bin/env python3
"""Render a dataset episode with temporal subtask predictions overlaid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import pyarrow.parquet as pq


def episode_metadata(root: Path, episode: int) -> dict:
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path, filters=[("episode_index", "=", episode)])
        if table.num_rows:
            return table.to_pylist()[0]
    raise ValueError(f"Episode {episode} was not found")


def active(atoms: list[dict], time_s: float) -> tuple[int, dict]:
    for index, atom in enumerate(atoms):
        if atom["start_timestamp"] <= time_s < atom["end_timestamp"] or index == len(atoms) - 1:
            return index, atom
    return len(atoms) - 1, atoms[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Original LeRobot v3 dataset")
    parser.add_argument("--predictions", type=Path, required=True, help="Hybrid/Qwen/kinematic predictions JSON")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--camera-key", default="observation.images.cam_low")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text())
    atoms = predictions["episodes"][str(args.episode)]["atoms"]
    if not atoms:
        raise ValueError(f"Episode {args.episode} has no predicted subtasks")
    info = json.loads((args.root / "meta" / "info.json").read_text())
    meta = episode_metadata(args.root, args.episode)
    prefix = f"videos/{args.camera_key}/"
    video = args.root / info["video_path"].format(
        video_key=args.camera_key,
        chunk_index=int(meta[f"{prefix}chunk_index"]),
        file_index=int(meta[f"{prefix}file_index"]),
    )
    video_start = float(meta[f"{prefix}from_timestamp"])
    video_end = float(meta[f"{prefix}to_timestamp"])
    container = av.open(str(video))
    stream = container.streams.video[0]
    frames = (frame for frame in container.decode(stream) if frame.time is not None and video_start <= float(frame.time) < video_end)
    try:
        first = next(frames)
    except StopIteration as error:
        container.close()
        raise RuntimeError(f"No frames in {video} for episode {args.episode}") from error
    image = first.to_ndarray(format="bgr24")
    height, width = image.shape[:2]
    fps = float(info["fps"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    duration = max(float(atoms[-1]["end_timestamp"]), 1e-6)

    def render(frame) -> None:
        local_time = float(frame.time) - video_start
        index, atom = active(atoms, local_time)
        image = frame.to_ndarray(format="bgr24")
        cv2.rectangle(image, (0, 0), (width, 94), (0, 0, 0), -1)
        cv2.putText(image, f"Hybrid subtask {index + 1}/{len(atoms)}", (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        cv2.putText(image, atom["content"][:92], (16, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(image, f"t={local_time:.2f}s | [{atom['start_timestamp']:.2f}, {atom['end_timestamp']:.2f})", (16, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
        y = height - 22
        cv2.rectangle(image, (16, y), (width - 16, y + 7), (70, 70, 70), -1)
        x = int(16 + (width - 32) * min(max(local_time / duration, 0), 1))
        cv2.circle(image, (x, y + 3), 6, (0, 220, 255), -1)
        writer.write(image)

    render(first)
    for frame in frames:
        render(frame)
    container.close()
    writer.release()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
