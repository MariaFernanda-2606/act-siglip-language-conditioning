#!/usr/bin/env python3
"""Generate presentation figures comparing every tested annotation method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


NAVY = (37, 67, 103)
BLUE = (64, 123, 190)
TEAL = (49, 145, 127)
ORANGE = (222, 139, 47)
PURPLE = (133, 94, 170)
GRAY = (136, 145, 154)
LIGHT = (242, 245, 248)
GRID = (219, 225, 232)
METHOD_COLORS = [ORANGE, PURPLE, TEAL, BLUE, NAVY]


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def canvas(title: str, subtitle: str = "", width: int = 1700, height: int = 980):
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 112), fill=NAVY)
    draw.text((62, 30), title, fill="white", font=font(32, True))
    if subtitle:
        draw.text((64, 75), subtitle, fill=(220, 232, 245), font=font(17))
    return image, draw


def horizontal_bars(draw, box, labels, values, colors, maximum, title, value_fmt, direction):
    left, top, right, bottom = box
    draw.text((left, top - 48), title, fill=NAVY, font=font(23, True))
    draw.text((left, top - 18), direction, fill=GRAY, font=font(15))
    row_h = (bottom - top) / len(labels)
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = top + index * row_h + 15
        draw.text((left, y + 11), label, fill=(35, 42, 50), font=font(18))
        x0, x1 = left + 225, right - 85
        draw.rounded_rectangle((x0, y, x1, y + 41), radius=12, fill=LIGHT)
        filled = x0 + (x1 - x0) * value / maximum
        draw.rounded_rectangle((x0, y, filled, y + 41), radius=12, fill=color)
        draw.text((right - 72, y + 9), format(value, value_fmt), fill=(35, 42, 50), font=font(18, True))


def temporal(path: Path, records: list[dict]):
    image, draw = canvas(
        "Temporal comparison of tested annotation methods",
        "Held-out test set: 30 manually annotated episodes. Higher tIoU and lower boundary MAE are better.",
    )
    labels = [record["name"] for record in records]
    colors = [record["color"] for record in records]
    horizontal_bars(draw, (70, 235, 810, 765), labels, [r["tiou"] for r in records], colors, 1.0,
                    "Mean temporal IoU", ".3f", "Higher is better")
    mae_max = max(record["mae"] for record in records) * 1.12
    horizontal_bars(draw, (900, 235, 1640, 765), labels, [r["mae"] for r in records], colors, mae_max,
                    "Boundary MAE (seconds)", ".2f", "Lower is better")
    draw.rounded_rectangle((70, 835, 1630, 910), radius=15, fill=(237, 245, 249))
    draw.text((95, 855), "Key result: both hybrid variants inherit kinematic temporal boundaries; their temporal scores therefore equal the kinematic baseline.", fill=NAVY, font=font(18, True))
    image.save(path)


def semantic(path: Path, closed: dict, open_semantic: dict):
    image, draw = canvas(
        "Semantic comparison: fixed labels versus open language",
        "The two panels use different, appropriate metrics; they should not be read as the same numerical measure.",
    )
    draw.rounded_rectangle((70, 190, 795, 830), radius=24, fill=(247, 249, 251), outline=GRID, width=2)
    draw.rounded_rectangle((905, 190, 1630, 830), radius=24, fill=(247, 249, 251), outline=GRID, width=2)
    draw.text((110, 235), "Fixed-vocabulary Qwen", fill=NAVY, font=font(25, True))
    draw.text((110, 275), "Exact colour-aware action F1", fill=GRAY, font=font(17))
    score = closed["colour_aware_action_F1"]
    draw.rounded_rectangle((110, 350, 750, 425), radius=18, fill=LIGHT)
    draw.rounded_rectangle((110, 350, 110 + 640 * score, 425), radius=18, fill=PURPLE)
    draw.text((110, 455), f"{score:.3f}", fill=PURPLE, font=font(58, True))
    draw.text((110, 550), "Measures exact agreement with the", fill=(35, 42, 50), font=font(18))
    draw.text((110, 580), "closed colour-specific label set.", fill=(35, 42, 50), font=font(18))

    draw.text((945, 235), "Open-vocabulary Qwen", fill=NAVY, font=font(25, True))
    draw.text((945, 275), "SigLIP semantic alignment against manual text", fill=GRAY, font=font(17))
    metrics = [
        ("Coverage", open_semantic["semantic_coverage"], ORANGE),
        ("Matched cosine", open_semantic["mean_matched_cosine"], TEAL),
        ("Conservative cosine", open_semantic["mean_cosine_with_unmatched_as_zero"], BLUE),
    ]
    for index, (label, value, color) in enumerate(metrics):
        y = 350 + index * 130
        draw.text((945, y), label, fill=(35, 42, 50), font=font(19, True))
        draw.rounded_rectangle((945, y + 35, 1560, y + 82), radius=12, fill=LIGHT)
        draw.rounded_rectangle((945, y + 35, 945 + 615 * value, y + 82), radius=12, fill=color)
        draw.text((1580, y + 43), f"{value:.3f}", fill=(35, 42, 50), font=font(18, True))
    draw.text((945, 755), "High cosine when Qwen emits a phrase;", fill=(35, 42, 50), font=font(17))
    draw.text((945, 784), "coverage penalizes omitted phases.", fill=(35, 42, 50), font=font(17))
    image.save(path)


def method_matrix(path: Path):
    image, draw = canvas(
        "What changes across methods?",
        "Each row identifies which component supplies the language semantics and temporal boundaries.",
        height=900,
    )
    columns = ["Method", "Language output", "Temporal boundaries", "SigLIP alignment", "Fallback"]
    widths = [285, 405, 385, 295, 215]
    x = 65
    top = 185
    for width, title in zip(widths, columns):
        draw.rounded_rectangle((x, top, x + width - 10, top + 65), radius=10, fill=NAVY)
        draw.text((x + 16, top + 20), title, fill="white", font=font(17, True))
        x += width
    rows = [
        ("Qwen pure\nclosed", "Fixed colour-specific\nlabels from Qwen", "Qwen timestamps", "No", "No"),
        ("Qwen pure\nopen", "Free Qwen phrases", "Qwen timestamps", "No", "No"),
        ("Kinematic", "Canonical phase\ntemplates", "Robot-state\nkinematics", "No", "Yes"),
        ("Hybrid\nclosed", "Qwen colour + fixed\nphase templates", "Robot-state\nkinematics", "No", "Yes"),
        ("Hybrid\nopen", "Free Qwen phrases\nwhen available", "Robot-state\nkinematics", "Yes", "Yes"),
    ]
    for row_index, row in enumerate(rows):
        y = top + 82 + row_index * 112
        x = 65
        for col_index, (width, text) in enumerate(zip(widths, row)):
            fill = (245, 248, 250) if row_index % 2 == 0 else (232, 239, 244)
            if row_index == 4:
                fill = (222, 239, 234)
            draw.rounded_rectangle((x, y, x + width - 10, y + 95), radius=10, fill=fill)
            lines = text.split("\n")
            for line_index, line in enumerate(lines):
                draw.text((x + 15, y + 19 + line_index * 28), line, fill=NAVY if col_index == 0 else (35, 42, 50), font=font(18, col_index == 0))
            x += width
    draw.rounded_rectangle((65, 775, 1630, 845), radius=14, fill=(237, 245, 249))
    draw.text((90, 797), "Open hybrid is the only method that retains natural Qwen language, replaces unreliable Qwen timing with kinematics, and uses SigLIP to align semantic variants.", fill=NAVY, font=font(18, True))
    image.save(path)


def provenance(path: Path, manifest: Path):
    counts = {"qwen_open_vocab": 0, "kinematic_fallback": 0}
    episodes = 0
    with manifest.open() as handle:
        for line in handle:
            record = json.loads(line)
            episodes += 1
            for segment in record["segments"]:
                counts[segment["language_source"]] += 1
    total = sum(counts.values())
    image, draw = canvas("Open-hybrid training annotations: provenance", "All usable episodes have five temporal segments; language source is recorded for every segment.", height=720)
    draw.text((85, 225), f"{episodes} episodes", fill=NAVY, font=font(42, True))
    draw.text((85, 280), f"{total} subtask segments", fill=GRAY, font=font(25))
    left, top, right, bottom = 535, 260, 1600, 420
    split = left + (right - left) * counts["qwen_open_vocab"] / total
    draw.rounded_rectangle((left, top, right, bottom), radius=22, fill=LIGHT)
    draw.rounded_rectangle((left, top, split, bottom), radius=22, fill=TEAL)
    draw.rectangle((split, top, right, bottom), fill=GRAY)
    qwen_share = counts["qwen_open_vocab"] / total
    draw.text((535, 465), f"Qwen free text  {counts['qwen_open_vocab']} ({qwen_share:.1%})", fill=TEAL, font=font(24, True))
    draw.text((1100, 465), f"Kinematic fallback  {counts['kinematic_fallback']} ({1-qwen_share:.1%})", fill=GRAY, font=font(24, True))
    draw.rounded_rectangle((85, 565, 1600, 635), radius=14, fill=(237, 245, 249))
    draw.text((110, 587), "Fallback does not alter the kinematic interval. It supplies a colour-conditioned instruction only when Qwen has no aligned open-language phrase.", fill=NAVY, font=font(18))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-closed", type=Path, required=True)
    parser.add_argument("--qwen-open", type=Path, required=True)
    parser.add_argument("--kinematic", type=Path, required=True)
    parser.add_argument("--hybrid-closed", type=Path, required=True)
    parser.add_argument("--hybrid-open", type=Path, required=True)
    parser.add_argument("--open-semantic", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    closed_qwen = json.loads(args.qwen_closed.read_text())
    open_qwen = json.loads(args.qwen_open.read_text())
    kinematic = json.loads(args.kinematic.read_text())
    closed_hybrid = json.loads(args.hybrid_closed.read_text())
    open_hybrid = json.loads(args.hybrid_open.read_text())
    open_semantic = json.loads(args.open_semantic.read_text())["summary"]
    records = [
        {"name": "Qwen pure (closed)", "tiou": closed_qwen["mean_tIoU"], "mae": closed_qwen["boundary_MAE_seconds"], "color": METHOD_COLORS[0]},
        {"name": "Qwen pure (open)", "tiou": open_qwen["mean_tIoU"], "mae": open_qwen["boundary_MAE_seconds"], "color": METHOD_COLORS[1]},
        {"name": "Kinematic", "tiou": kinematic["mean_tIoU"], "mae": kinematic["boundary_MAE_seconds"], "color": METHOD_COLORS[2]},
        {"name": "Hybrid (closed)", "tiou": closed_hybrid["mean_tIoU"], "mae": closed_hybrid["boundary_MAE_seconds"], "color": METHOD_COLORS[3]},
        {"name": "Hybrid (open)", "tiou": open_hybrid["mean_tIoU"], "mae": open_hybrid["boundary_MAE_seconds"], "color": METHOD_COLORS[4]},
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporal(args.output_dir / "01_temporal_comparison.png", records)
    semantic(args.output_dir / "02_semantic_comparison.png", closed_qwen, open_semantic)
    method_matrix(args.output_dir / "03_method_components.png")
    provenance(args.output_dir / "04_open_hybrid_provenance.png", args.manifest)
    (args.output_dir / "comparison_summary.json").write_text(json.dumps({"methods": records, "open_semantic": open_semantic}, indent=2) + "\n")
    print(f"Wrote comparison figures to {args.output_dir}")


if __name__ == "__main__":
    main()
