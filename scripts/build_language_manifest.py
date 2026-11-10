#!/usr/bin/env python3
"""Convert hybrid temporal annotations into an ACT language sidecar manifest.

The manifest is JSONL (one episode per line), so it is compact and does not
duplicate language for every frame. Frame intervals are half-open: [start,end).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def subtask_text(phase: str, color: str) -> str:
    templates = {
        "pick": "pick up the {color} cube",
        "move": "move the {color} cube to the {color} drawer",
        "open": "open the {color} drawer",
        "place": "place the {color} cube inside the {color} drawer",
        "close": "close the {color} drawer",
    }
    return templates[phase].format(color=color)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True, help="Hybrid predictions JSON")
    parser.add_argument("--output", type=Path, required=True, help="Episode JSONL manifest")
    parser.add_argument("--vocabulary-output", type=Path, required=True, help="Canonical text vocabulary JSON")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in (args.output, args.vocabulary_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite if intentional")

    predictions = json.loads(args.predictions.read_text())
    vocabulary: dict[str, str] = {}
    records = []
    for episode, entry in sorted(predictions["episodes"].items(), key=lambda item: int(item[0])):
        color = entry.get("qwen_colour")
        if color not in {"red", "green", "yellow"}:
            print(f"WARNING: episode {episode} skipped: no reliable Qwen colour")
            continue
        segments = []
        previous_end = 0
        for atom in entry["atoms"]:
            phase = atom["generic_phase"]
            start, end = int(atom["start_frame"]), int(atom["end_frame"])
            if phase not in {"pick", "move", "open", "place", "close"} or end <= start:
                raise ValueError(f"Episode {episode}: invalid segment {atom}")
            if start != previous_end:
                raise ValueError(f"Episode {episode}: non-contiguous segments at frame {start}")
            identifier = atom["content"]
            text = subtask_text(phase, color)
            vocabulary[identifier] = text
            segments.append({
                "start_frame": start,
                "end_frame": end,
                "start_timestamp": float(atom["start_timestamp"]),
                "end_timestamp": float(atom["end_timestamp"]),
                "subtask_id": identifier,
                "subtask_text": text,
            })
            previous_end = end
        records.append({
            "episode_index": int(episode),
            "task_text": f"pick up the {color} cube and place it in the {color} drawer",
            "cube_color": color,
            "drawer_color": color,
            "segments": segments,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    args.vocabulary_output.parent.mkdir(parents=True, exist_ok=True)
    args.vocabulary_output.write_text(json.dumps({
        "schema_version": "act_language_manifest_v1",
        "source_predictions": str(args.predictions),
        "frame_interval": "[start_frame, end_frame)",
        "subtask_vocabulary": dict(sorted(vocabulary.items())),
        "episodes": len(records),
    }, indent=2) + "\n")
    print(f"Wrote {len(records)} episode records to {args.output}")
    print(f"Wrote {len(vocabulary)} canonical subtask texts to {args.vocabulary_output}")


if __name__ == "__main__":
    main()
