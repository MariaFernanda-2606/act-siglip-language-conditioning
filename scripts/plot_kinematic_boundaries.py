#!/usr/bin/env python3
"""Plot arm/gripper motion energy against manual subtask boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


def episode_rows(root: Path, episode: int) -> list[dict]:
    for parquet in sorted((root / "data").rglob("*.parquet")):
        rows = [row for row in pq.read_table(parquet).to_pylist() if int(row["episode_index"]) == episode]
        if rows:
            return rows
    raise ValueError(f"Episode {episode} not found")


def smooth(values: np.ndarray, frames: int) -> np.ndarray:
    kernel = np.ones(frames, dtype=float) / frames
    return np.convolve(values, kernel, mode="same")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoothing-s", type=float, default=0.5)
    args = parser.parse_args()

    info = json.loads((args.root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    rows = episode_rows(args.root, args.episode)
    time = np.array([float(row["timestamp"]) for row in rows])
    state = np.asarray([row["observation.state"] for row in rows], dtype=float)
    velocity = np.gradient(state, 1 / fps, axis=0)
    window = max(1, round(args.smoothing_s * fps))

    # Joints 0:6 / 7:13 describe each arm. Joint 6 / 13 is its gripper.
    signals = {
        "left arm motion": smooth(np.linalg.norm(velocity[:, :6], axis=1), window),
        "right arm motion": smooth(np.linalg.norm(velocity[:, 7:13], axis=1), window),
        "left gripper motion": smooth(np.abs(velocity[:, 6]), window),
        "right gripper motion": smooth(np.abs(velocity[:, 13]), window),
    }
    annotations = json.loads(args.ground_truth.read_text())
    atoms = annotations["episodes"][str(args.episode)]["atoms"]
    boundaries = [atom for atom in atoms if atom.get("style") == "subtask"]

    width, panel_height, margin = 1500, 210, 90
    height = 90 + panel_height * len(signals) + 80
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((margin, 18), f"Episode {args.episode}: kinematic signals and manual boundaries", fill="black", font=font)
    duration = max(float(time[-1]), 1e-6)
    boundary_colors = [(220, 50, 47), (38, 139, 210), (133, 153, 0), (181, 137, 0), (108, 113, 196)]
    for panel, (name, values) in enumerate(signals.items()):
        top = 70 + panel * panel_height
        bottom = top + panel_height - 35
        draw.rectangle((margin, top, width - margin, bottom), outline=(180, 180, 180))
        draw.text((8, top + 5), name, fill="black", font=font)
        vmax = max(float(values.max()), 1e-9)
        points = []
        for t, value in zip(time, values, strict=True):
            x = margin + (width - 2 * margin) * float(t) / duration
            y = bottom - (bottom - top) * float(value) / vmax
            points.append((round(x), round(y)))
        if len(points) > 1:
            draw.line(points, fill=(30, 70, 160), width=2)
        for index, atom in enumerate(boundaries):
            x = margin + (width - 2 * margin) * float(atom["start_timestamp"]) / duration
            draw.line((round(x), top, round(x), bottom), fill=boundary_colors[index], width=2)
    legend_y = height - 55
    for index, atom in enumerate(boundaries):
        color = boundary_colors[index]
        x = margin + (index % 3) * 450
        y = legend_y + (index // 3) * 20
        draw.line((x, y + 5, x + 24, y + 5), fill=color, width=3)
        draw.text((x + 30, y), atom["content"], fill="black", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
