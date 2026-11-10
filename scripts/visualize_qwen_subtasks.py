#!/usr/bin/env python3
"""Render one annotated LeRobot v3 episode with Qwen subtask labels overlaid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import pyarrow.parquet as pq


def episode_metadata(root: Path, episode: int) -> dict:
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            if int(row["episode_index"]) == episode:
                return row
    raise ValueError(f"Episode {episode} was not found in {root / 'meta' / 'episodes'}")


def episode_rows(root: Path, episode: int) -> list[dict]:
    for path in sorted((root / "data").rglob("*.parquet")):
        rows = [row for row in pq.read_table(path).to_pylist() if int(row["episode_index"]) == episode]
        if rows:
            return rows
    raise ValueError(f"Episode {episode} has no data rows")


def subtasks(rows: list[dict]) -> list[dict]:
    language = rows[0].get("language_persistent")
    if not language:
        raise ValueError("No language_persistent annotations found. Run lerobot-annotate first.")
    spans = [
        {"text": row.get("content") or "", "start": float(row["timestamp"])}
        for row in language
        if row.get("style") == "subtask"
    ]
    spans.sort(key=lambda item: item["start"])
    if not spans:
        raise ValueError("The dataset has language annotations but no style='subtask' rows.")
    end = float(rows[-1]["timestamp"])
    for current, following in zip(spans, spans[1:]):
        current["end"] = following["start"]
    spans[-1]["end"] = end
    return spans


def active_span(spans: list[dict], timestamp: float) -> tuple[int, dict]:
    for index, span in enumerate(spans):
        if span["start"] <= timestamp < span["end"] or index == len(spans) - 1:
            return index, span
    return len(spans) - 1, spans[-1]


def draw_overlay(frame, span: dict, index: int, total: int, time_s: float) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 88), (0, 0, 0), -1)
    cv2.putText(frame, f"Episode subtask {index + 1}/{total}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    cv2.putText(frame, span["text"][:90], (16, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"t={time_s:.2f}s | segment [{span['start']:.2f}, {span['end']:.2f})", (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
    bar_y = height - 20
    cv2.rectangle(frame, (16, bar_y), (width - 16, bar_y + 7), (70, 70, 70), -1)
    duration = max(span["end"], 0.001)
    x = int(16 + (width - 32) * min(time_s / duration, 1.0))
    cv2.circle(frame, (x, bar_y + 3), 6, (0, 220, 255), -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Annotated LeRobot v3 dataset directory")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--camera-key", default="observation.images.cam_high")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    info = json.loads((args.root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    metadata = episode_metadata(args.root, args.episode)
    rows = episode_rows(args.root, args.episode)
    spans = subtasks(rows)

    prefix = f"videos/{args.camera_key}/"
    chunk = int(metadata[f"{prefix}chunk_index"])
    file_index = int(metadata[f"{prefix}file_index"])
    start = float(metadata[f"{prefix}from_timestamp"])
    end = float(metadata[f"{prefix}to_timestamp"])
    video_template = info["video_path"]
    video_path = args.root / video_template.format(video_key=args.camera_key, chunk_index=chunk, file_index=file_index)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    # The Trossen recordings are AV1. OpenCV often asks FFmpeg for a hardware
    # decoder, which is unavailable on many Linux workstations; PyAV decodes
    # the same stream in software and exposes BGR frames for OpenCV drawing.
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = (
        frame
        for frame in container.decode(stream)
        if frame.time is not None and start <= float(frame.time) < end
    )
    try:
        first_frame = next(frames)
    except StopIteration as error:
        container.close()
        raise RuntimeError(f"Could not decode {video_path} in [{start:.2f}, {end:.2f})s") from error

    first_image = first_frame.to_ndarray(format="bgr24")
    height, width = first_image.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    def render(frame) -> None:
        time_s = float(frame.time)
        index, span = active_span(spans, time_s)
        image = frame.to_ndarray(format="bgr24")
        draw_overlay(image, span, index, len(spans), time_s)
        writer.write(image)

    render(first_frame)
    for frame in frames:
        render(frame)

    container.close()
    writer.release()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
