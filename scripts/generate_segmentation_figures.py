#!/usr/bin/env python3
"""Generate publication-ready segmentation figures without matplotlib.

Outputs PNG files for global metrics, boundary-tolerance curves, per-episode
comparisons and qualitative timelines.  The script uses only Pillow/Numpy,
which are already available in the LeRobot annotation environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PHASES = ("pick", "move", "open", "place", "close")
PALETTE = {"pick": (220, 70, 70), "move": (55, 130, 210), "open": (75, 160, 95), "place": (225, 160, 40), "close": (140, 100, 190), "unknown": (120, 120, 120)}
QWEN, KINEMATIC, HYBRID = (45, 110, 190), (60, 160, 100), (220, 130, 45)


def font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def phase(atom: dict) -> str:
    return atom.get("generic_phase") or atom.get("content", "unknown").lower().split("_", 1)[0]


def atoms(doc: dict, episode: str) -> list[dict]:
    result = [a for a in doc["episodes"][episode]["atoms"] if a.get("style", "subtask") == "subtask"]
    return sorted(result, key=lambda a: a["start_frame"])


def iou(a: dict, b: dict) -> float:
    overlap = max(0, min(a["end_frame"], b["end_frame"]) - max(a["start_frame"], b["start_frame"]))
    union = max(a["end_frame"], b["end_frame"]) - min(a["start_frame"], b["start_frame"])
    return overlap / union if union else 0.0


def metrics(gt: dict, pred: dict, episodes: list[str]) -> tuple[float, float, float, dict[str, tuple[float, float]]]:
    tiou, mae, f1, per_episode = [], [], [], {}
    for ep in episodes:
        truth, guess = atoms(gt, ep), atoms(pred, ep)
        truth_by = {phase(a): a for a in truth}
        guess_by = {phase(a): a for a in guess}
        values = [iou(truth_by[p], guess_by[p]) for p in PHASES if p in truth_by and p in guess_by]
        errors = [abs(guess[i]["start_timestamp"] - truth[i]["start_timestamp"]) for i in range(1, min(len(truth), len(guess), 5))]
        if values:
            tiou.extend(values)
        if errors:
            mae.extend(errors)
        expected, received = [a["content"].lower() for a in truth], [a["content"].lower() for a in guess]
        tp = sum(a == b for a, b in zip(expected, received))
        precision, recall = tp / len(received) if received else 0.0, tp / len(expected) if expected else 0.0
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        per_episode[ep] = (float(np.mean(values)) if values else 0.0, float(np.mean(errors)) if errors else 0.0)
    return float(np.mean(tiou)), float(np.mean(mae)), float(np.mean(f1)), per_episode


def canvas(width=1600, height=900, title=""):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((55, 28), title, fill="black", font=font(30))
    return image, draw


def bar_chart(path: Path, qwen, kinematic, hybrid):
    image, draw = canvas(title="Temporal subtask segmentation: held-out test set")
    panels = [("mean tIoU (higher is better)", [qwen[0], kinematic[0], hybrid[0]], 1.0), ("Boundary MAE in seconds (lower is better)", [qwen[1], kinematic[1], hybrid[1]], max(qwen[1], kinematic[1], hybrid[1]) * 1.2), ("Colour-aware action F1 (higher is better)", [qwen[2], None, hybrid[2]], 1.0)]
    names, colors = ("Qwen", "Kinematic", "Hybrid"), (QWEN, KINEMATIC, HYBRID)
    for p, (title, values, vmax) in enumerate(panels):
        left, top, right, bottom = 70 + p * 510, 130, 520 + p * 510, 730
        draw.text((left, 85), title, fill="black", font=font(18))
        draw.line((left, bottom, right, bottom), fill="black", width=2)
        draw.line((left, top, left, bottom), fill="black", width=2)
        for tick in range(5):
            value = vmax * tick / 4
            y = bottom - (bottom - top) * tick / 4
            draw.line((left - 5, y, right, y), fill=(225, 225, 225), width=1)
            draw.text((left - 53, y - 8), f"{value:.1f}", fill="black", font=font(13))
        for i, value in enumerate(values):
            x = left + 55 + i * 130
            if value is None:
                draw.text((x + 12, bottom - 40), "n/a", fill=(80, 80, 80), font=font(16))
            else:
                y = bottom - (bottom - top) * value / vmax
                draw.rectangle((x, y, x + 82, bottom), fill=colors[i])
                draw.text((x + 6, y - 25), f"{value:.3f}" if p != 1 else f"{value:.2f}s", fill="black", font=font(15))
            draw.text((x, bottom + 15), names[i], fill="black", font=font(15))
    image.save(path)


def tolerance_curve(path: Path, gt, qwen, hybrid, episodes):
    thresholds = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
    image, draw = canvas(title="Boundary tolerance curve")
    left, top, right, bottom = 120, 120, 1510, 740
    draw.text((left, 75), "Fraction of the four manual boundaries within each tolerance", fill="black", font=font(18))
    errors_by_method = []
    for pred in (qwen, hybrid):
        errors = []
        for ep in episodes:
            a, b = atoms(gt, ep), atoms(pred, ep)
            errors.extend(abs(b[i]["start_timestamp"] - a[i]["start_timestamp"]) for i in range(1, min(5, len(a), len(b))))
        errors_by_method.append(np.asarray(errors))
    for tick in range(6):
        y = bottom - (bottom - top) * tick / 5
        draw.line((left, y, right, y), fill=(225, 225, 225))
        draw.text((left - 55, y - 8), f"{tick / 5:.1f}", fill="black", font=font(14))
    points_x = np.linspace(left, right, len(thresholds))
    for threshold, x in zip(thresholds, points_x):
        draw.text((x - 13, bottom + 15), f"{threshold:g}", fill="black", font=font(14))
    for values, color, name in zip(errors_by_method, (QWEN, HYBRID), ("Qwen", "Hybrid")):
        points = [(int(x), int(bottom - (bottom - top) * np.mean(values <= t))) for t, x in zip(thresholds, points_x)]
        draw.line(points, fill=color, width=5)
        for x, y in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
        legend_x = 1160 if name == "Qwen" else 1330
        draw.rectangle((legend_x, 70, legend_x + 22, 86), fill=color)
        draw.text((legend_x + 30, 67), name, fill="black", font=font(16))
    draw.text((690, 805), "boundary tolerance (seconds)", fill="black", font=font(18))
    image.save(path)


def paired(path: Path, qwen_per, hybrid_per, episodes):
    image, draw = canvas(height=1050, title="Per-episode improvement from Qwen to hybrid")
    for panel, (name, index, vmax) in enumerate((("mean tIoU (higher is better)", 0, 1.0), ("Boundary MAE (seconds; lower is better)", 1, max(max(v[1] for v in qwen_per.values()), max(v[1] for v in hybrid_per.values())) * 1.1))):
        left, top, right, bottom = 100, 120 + panel * 430, 1510, 440 + panel * 430
        draw.text((left, top - 35), name, fill="black", font=font(18))
        draw.line((left, bottom, right, bottom), fill="black", width=2)
        for tick in range(5):
            y = bottom - (bottom - top) * tick / 4
            draw.line((left, y, right, y), fill=(230, 230, 230))
            draw.text((left - 58, y - 8), f"{vmax * tick / 4:.1f}", fill="black", font=font(13))
        for i, ep in enumerate(episodes):
            x = left + 18 + i * (right - left - 36) / (len(episodes) - 1)
            q, h = qwen_per[ep][index], hybrid_per[ep][index]
            qy, hy = bottom - (bottom - top) * q / vmax, bottom - (bottom - top) * h / vmax
            draw.line((x, qy, x, hy), fill=(160, 160, 160), width=2)
            draw.ellipse((x - 4, qy - 4, x + 4, qy + 4), fill=QWEN)
            draw.ellipse((x - 4, hy - 4, x + 4, hy + 4), fill=HYBRID)
            if i % 3 == 0:
                draw.text((x - 8, bottom + 9), ep, fill="black", font=font(11))
    draw.rectangle((1180, 42, 1200, 57), fill=QWEN); draw.text((1208, 39), "Qwen", fill="black", font=font(15))
    draw.rectangle((1320, 42, 1340, 57), fill=HYBRID); draw.text((1348, 39), "Hybrid", fill="black", font=font(15))
    image.save(path)


def timelines(path: Path, gt, qwen, hybrid, episodes):
    image, draw = canvas(height=250 + len(episodes) * 260, title="Qualitative timelines: manual vs. Qwen vs. hybrid")
    for eindex, ep in enumerate(episodes):
        top = 100 + eindex * 260
        groups = (("Manual", atoms(gt, ep)), ("Qwen", atoms(qwen, ep)), ("Hybrid", atoms(hybrid, ep)))
        duration = max(a["end_timestamp"] for _, group in groups for a in group)
        draw.text((55, top), f"Episode {ep}", fill="black", font=font(18))
        for row, (name, group) in enumerate(groups):
            y = top + 40 + row * 58
            draw.text((55, y + 12), name, fill="black", font=font(16))
            left, right = 180, 1510
            draw.rectangle((left, y, right, y + 30), outline=(100, 100, 100))
            for atom in group:
                x0 = left + (right - left) * atom["start_timestamp"] / duration
                x1 = left + (right - left) * atom["end_timestamp"] / duration
                p = phase(atom)
                draw.rectangle((x0, y, x1, y + 30), fill=PALETTE.get(p, PALETTE["unknown"]))
                if x1 - x0 > 78:
                    draw.text((x0 + 4, y + 8), p, fill="white", font=font(12))
        draw.text((180, top + 220), f"0 s{' ' * 85}{duration:.1f} s", fill=(70, 70, 70), font=font(13))
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--kinematic", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeline-episodes", nargs="*", type=int, default=[5, 55, 107])
    args = parser.parse_args()
    gt, qwen, kinematic, hybrid = (json.loads(p.read_text()) for p in (args.ground_truth, args.qwen, args.kinematic, args.hybrid))
    split = json.loads(args.split.read_text())
    episodes = [str(e) for e in split["test"]["episodes"] if str(e) in qwen["episodes"] and str(e) in hybrid["episodes"]]
    q = metrics(gt, qwen, episodes)
    k = metrics(gt, kinematic, episodes)
    h = metrics(gt, hybrid, episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bar_chart(args.output_dir / "01_global_metrics.png", q, k, h)
    tolerance_curve(args.output_dir / "02_boundary_tolerance.png", gt, qwen, hybrid, episodes)
    paired(args.output_dir / "03_per_episode_comparison.png", q[3], h[3], episodes)
    selected = [str(e) for e in args.timeline_episodes]
    timelines(args.output_dir / "04_qualitative_timelines.png", gt, qwen, hybrid, selected)
    print("Wrote:")
    for name in ("01_global_metrics.png", "02_boundary_tolerance.png", "03_per_episode_comparison.png", "04_qualitative_timelines.png"):
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
