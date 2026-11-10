#!/usr/bin/env python3
"""Attach a Qwen-inferred colour to kinematic temporal segments.

The Qwen output is used only for colour/semantic information.  Its timestamps
and action ordering are intentionally discarded, so this is a genuine hybrid
baseline rather than a post-processing of Qwen's boundaries.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


COLORS = ("red", "green", "yellow")


def vote_colour(atoms: list[dict]) -> tuple[str | None, dict[str, int]]:
    votes = {color: 0 for color in COLORS}
    for atom in atoms:
        words = re.findall(r"[a-z]+", atom.get("content", "").lower())
        for color in COLORS:
            votes[color] += words.count(color)
    best = max(votes, key=votes.get)
    return (best if votes[best] else None), votes


def label(phase: str, color: str) -> str:
    return {
        "pick": f"pick_{color}_cube",
        "move": f"move_to_{color}_drawer",
        "open": f"open_{color}_drawer",
        "place": f"place_{color}_cube_in_{color}_drawer",
        "close": f"close_{color}_drawer",
    }[phase]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinematic", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kinematic = json.loads(args.kinematic.read_text())
    qwen = json.loads(args.qwen.read_text())
    result = {"method": "qwen_colour_plus_kinematic_boundaries_v1", "episodes": {}}
    for episode, entry in sorted(kinematic["episodes"].items(), key=lambda item: int(item[0])):
        qwen_entry = qwen["episodes"].get(episode, {"atoms": []})
        color, votes = vote_colour(qwen_entry["atoms"])
        atoms = []
        for atom in entry["atoms"]:
            copied = dict(atom)
            copied["content"] = label(atom["generic_phase"], color) if color else atom["generic_phase"]
            copied["qwen_colour"] = color
            atoms.append(copied)
        result["episodes"][episode] = {"atoms": atoms, "qwen_colour_votes": votes, "qwen_colour": color}
        print(f"episode {episode}: Qwen colour={color or 'unknown'} votes={votes}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
