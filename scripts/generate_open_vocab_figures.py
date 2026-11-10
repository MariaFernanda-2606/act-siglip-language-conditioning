#!/usr/bin/env python3
"""Generate static figures for open-vocabulary hybrid annotation evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PHASES = ("pick", "move", "open", "place", "close")
COLORS = {"blue": (55, 110, 180), "green": (65, 145, 90), "orange": (220, 135, 45), "gray": (150, 150, 150)}


def font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def canvas(title: str, width: int = 1500, height: int = 900):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((55, 32), title, fill="black", font=font(30))
    return image, draw


def bars(draw, left, top, right, bottom, labels, values, colors, ymax, value_format=".3f", ylabel=""):
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    draw.line((left, top, left, bottom), fill="black", width=2)
    for tick in range(5):
        value = ymax * tick / 4
        y = bottom - (bottom - top) * tick / 4
        draw.line((left, y, right, y), fill=(228, 228, 228))
        draw.text((left - 66, y - 8), format(value, ".2f"), fill="black", font=font(13))
    if ylabel:
        draw.text((left, top - 62), ylabel, fill="black", font=font(17))
    slot = (right - left) / len(labels)
    for index, (label, value, colour) in enumerate(zip(labels, values, colors)):
        x0 = left + slot * index + slot * 0.22
        x1 = left + slot * (index + 1) - slot * 0.22
        y = bottom - (bottom - top) * value / ymax
        draw.rectangle((x0, y, x1, bottom), fill=colour)
        draw.text(((x0 + x1) / 2 - 24, y - 28), format(value, value_format), fill="black", font=font(15))
        draw.text(((x0 + x1) / 2 - 35, bottom + 14), label, fill="black", font=font(15))


def summary_figure(path: Path, temporal: dict, semantic: dict):
    image, draw = canvas("Open-vocabulary hybrid annotation: held-out test metrics")
    bars(draw, 90, 160, 700, 710,
         ["tIoU", "Boundary MAE"], [temporal["mean_tIoU"], temporal["boundary_MAE_seconds"]],
         [COLORS["blue"], COLORS["orange"]], 2.0, ".3f", "Temporal quality (tIoU higher; MAE lower, seconds)")
    bars(draw, 820, 160, 1420, 710,
         ["Coverage", "Matched cosine", "Conservative cosine"],
         [semantic["semantic_coverage"], semantic["mean_matched_cosine"], semantic["mean_cosine_with_unmatched_as_zero"]],
         [COLORS["orange"], COLORS["green"], COLORS["blue"]], 1.0, ".3f", "Semantic quality (higher is better)")
    draw.text((90, 790), "Cosine uses SigLIP embeddings; conservative cosine assigns 0 to omitted manual phases.", fill=(70, 70, 70), font=font(17))
    image.save(path)


def phase_figure(path: Path, semantic: dict):
    counts = defaultdict(lambda: [0, 0, []])
    for entry in semantic["episodes"].values():
        for row in entry["alignment"]:
            if row["manual_index"] is None:
                continue
            phase = row["manual_text"].split()[0].lower()
            counts[phase][1] += 1
            if row["cosine"] is not None:
                counts[phase][0] += 1
                counts[phase][2].append(row["cosine"])
    image, draw = canvas("Open-vocabulary Qwen: semantic coverage and similarity by phase", height=860)
    labels = list(PHASES)
    coverage = [counts[p][0] / counts[p][1] if counts[p][1] else 0 for p in labels]
    cosine = [sum(counts[p][2]) / len(counts[p][2]) if counts[p][2] else 0 for p in labels]
    bars(draw, 90, 155, 700, 680, labels, coverage, [COLORS["orange"]] * 5, 1.0, ".2f", "Coverage: matched manual phases")
    bars(draw, 820, 155, 1420, 680, labels, cosine, [COLORS["green"]] * 5, 1.0, ".2f", "Mean SigLIP cosine among matches")
    for x, p in ((90, "Left: Qwen often omits move and open."), (820, "Right: when emitted, phrases remain semantically close to manual labels.")):
        draw.text((x, 760), p, fill=(70, 70, 70), font=font(16))
    image.save(path)


def provenance_figure(path: Path, manifest: Path):
    source_counts = Counter()
    subtask_counts = Counter()
    with manifest.open() as handle:
        for line in handle:
            record = json.loads(line)
            source_counts.update(segment["language_source"] for segment in record["segments"])
            subtask_counts[len(record["segments"])] += 1
    qwen = source_counts["qwen_open_vocab"]
    fallback = source_counts["kinematic_fallback"]
    total = qwen + fallback
    image, draw = canvas("Final ACT language manifest: annotation provenance", height=720)
    left, top, right, bottom = 100, 220, 1400, 390
    qwen_right = left + (right - left) * qwen / total
    draw.rectangle((left, top, qwen_right, bottom), fill=COLORS["green"])
    draw.rectangle((qwen_right, top, right, bottom), fill=COLORS["gray"])
    draw.text((left, 165), f"Qwen open vocabulary: {qwen}/{total} ({qwen / total:.1%})", fill="black", font=font(22))
    draw.text((qwen_right + 25, 420), f"Kinematic fallback: {fallback}/{total} ({fallback / total:.1%})", fill="black", font=font(22))
    draw.text((100, 530), "Every episode has five temporally defined subtasks; fallback is used only where Qwen supplied no aligned phrase.", fill=(70, 70, 70), font=font(18))
    draw.text((100, 580), f"Episodes: {sum(subtask_counts.values())}; segments: {total}; segments per episode: {dict(subtask_counts)}", fill=(70, 70, 70), font=font(16))
    image.save(path)


def granularity_figure(path: Path, semantic: dict):
    counts = Counter(len(entry["qwen_subtasks"]) for entry in semantic["episodes"].values())
    labels, values = [str(key) for key in sorted(counts)], [counts[key] for key in sorted(counts)]
    image, draw = canvas("Qwen open-vocabulary granularity on held-out test episodes", height=760)
    bars(draw, 110, 155, 1410, 600, labels, values, [COLORS["blue"]] * len(labels), max(values) * 1.15, ".0f", "Episodes")
    draw.text((110, 675), "The manual ontology has five phases per episode. Fewer Qwen subtasks indicate merged or omitted phases.", fill=(70, 70, 70), font=font(18))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-metrics", type=Path, required=True)
    parser.add_argument("--semantic-metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    temporal = json.loads(args.temporal_metrics.read_text())
    semantic = json.loads(args.semantic_metrics.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_figure(args.output_dir / "01_test_metrics.png", temporal, semantic["summary"])
    phase_figure(args.output_dir / "02_semantic_by_phase.png", semantic)
    provenance_figure(args.output_dir / "03_annotation_provenance.png", args.manifest)
    granularity_figure(args.output_dir / "04_qwen_granularity.png", semantic)
    print("Wrote figures to", args.output_dir)


if __name__ == "__main__":
    main()
