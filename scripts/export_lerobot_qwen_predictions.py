#!/usr/bin/env python3
"""Export LeRobot language_persistent Qwen annotations to evaluation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def generic_phase(label: str) -> str:
    label = label.lower().strip()
    for phase in ("pick", "move", "open", "place", "close"):
        if label.startswith(phase):
            return phase
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Annotated LeRobot v3 copy")
    parser.add_argument("--episodes", nargs="*", type=int, help="Optional episode filter")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = set(args.episodes) if args.episodes else None
    grouped: dict[int, list[dict]] = {}
    for path in sorted((args.root / "data").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            episode = int(row["episode_index"])
            if selected is None or episode in selected:
                grouped.setdefault(episode, []).append(row)
    result = {"method": "qwen_raw_lerobot", "episodes": {}}
    for episode, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["frame_index"]))
        language = rows[0].get("language_persistent") or []
        spans = [entry for entry in language if entry.get("style") == "subtask"]
        spans.sort(key=lambda entry: float(entry.get("timestamp", 0)))
        atoms = []
        for index, span in enumerate(spans):
            start_time = float(span["timestamp"])
            end_time = float(spans[index + 1]["timestamp"]) if index + 1 < len(spans) else float(rows[-1]["timestamp"])
            start_frame = min(range(len(rows)), key=lambda i: abs(float(rows[i]["timestamp"]) - start_time))
            end_frame = len(rows) if index + 1 == len(spans) else min(range(len(rows)), key=lambda i: abs(float(rows[i]["timestamp"]) - end_time))
            if end_frame <= start_frame:
                continue
            atoms.append({
                "content": span.get("content", ""),
                "generic_phase": generic_phase(span.get("content", "")),
                "start_timestamp": start_time,
                "end_timestamp": end_time,
                "start_frame": start_frame,
                "end_frame": end_frame,
            })
        # The copied v3 dataset contains all episodes, while LeRobot only
        # annotates --only_episodes. Do not export unannotated episodes: they
        # are absent predictions, not Qwen predictions with zero subtasks.
        if not atoms:
            continue
        result["episodes"][str(episode)] = {"atoms": atoms}
        print(f"episode {episode}: exported {len(atoms)} Qwen subtasks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
