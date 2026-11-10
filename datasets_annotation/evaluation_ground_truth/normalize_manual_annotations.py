#!/usr/bin/env python3
"""Create a canonical, traceable ground-truth file from LeRobot annotations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parent.parent
SOURCE = ROOT / "place_the_cubes_in_the_drawers" / "meta" / "lerobot_annotations.json"
OUTPUT = Path(__file__).parent / "manual_ground_truth_45_episodes_canonical.json"
REPORT = Path(__file__).parent / "manual_ground_truth_45_episodes_correction_report.json"


COLORS = ("red", "yellow", "green")


def color_from_text(text: str) -> str:
    matches = [color for color in COLORS if color in text.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one color in subtask label: {text!r}")
    return matches[0]


def canonical_label(text: str, color: str) -> str:
    text = text.lower().strip()
    if text.startswith("pick"):
        return f"pick_{color}_cube"
    if text.startswith("move towards"):
        return f"move_to_{color}_drawer"
    if text.startswith("open"):
        return f"open_{color}_drawer"
    if text.startswith("place"):
        return f"place_{color}_cube_in_{color}_drawer"
    if text.startswith("close"):
        return f"close_{color}_drawer"
    raise ValueError(f"Unmapped subtask label: {text!r}")


def main() -> None:
    with SOURCE.open() as source_file:
        annotations = json.load(source_file)

    before = Counter()
    after = Counter()
    changed = Counter()

    for episode in annotations["episodes"].values():
        for atom in episode["atoms"]:
            if atom.get("style") != "subtask":
                continue
            original = atom["content"]
            color = color_from_text(original)
            normalized = canonical_label(original, color)
            before[original] += 1
            after[normalized] += 1
            if original != normalized:
                changed[original] += 1
            atom["original_content"] = original
            atom["content"] = normalized
            atom["cube_color"] = color
            atom["drawer_color"] = color

    with OUTPUT.open("w") as output_file:
        json.dump(annotations, output_file, indent=2)
        output_file.write("\n")

    report = {
        "source": str(SOURCE),
        "canonical_file": str(OUTPUT),
        "episodes": len(annotations["episodes"]),
        "label_counts_before": dict(sorted(before.items())),
        "label_counts_after": dict(sorted(after.items())),
        "rewritten_label_counts": dict(sorted(changed.items())),
        "canonical_taxonomy": sorted(after),
    }
    with REPORT.open("w") as report_file:
        json.dump(report, report_file, indent=2)
        report_file.write("\n")

    print(f"Wrote {OUTPUT}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
