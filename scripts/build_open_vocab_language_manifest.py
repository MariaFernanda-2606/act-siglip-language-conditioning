#!/usr/bin/env python3
"""Build an ACT sidecar manifest from open-vocabulary hybrid annotations.

Each segment carries one active natural-language instruction.  Qwen text is
preserved verbatim when available; otherwise a colour-conditioned kinematic
fallback is used.  Dataset task metadata, rather than Qwen, supplies the
episode-level goal and colour.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


COLORS = ("red", "green", "yellow")


def colour_from_task(task: str) -> str:
    words = set(re.findall(r"[a-z]+", task.lower()))
    found = [colour for colour in COLORS if colour in words]
    if len(found) != 1:
        raise ValueError(f"Could not infer exactly one colour from task: {task!r}")
    return found[0]


def fallback_text(phase: str, colour: str) -> str:
    return {
        "pick": f"pick up the {colour} cube",
        "move": f"move the {colour} cube to the {colour} drawer",
        "open": f"open the {colour} drawer",
        "place": f"place the {colour} cube inside the {colour} drawer",
        "close": f"close the {colour} drawer",
    }[phase]


def identifier(prefix: str, text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return f"{prefix}:{normalized}"


def episode_task_indices(dataset_root: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for shard in sorted((dataset_root / "data").rglob("*.parquet")):
        table = pq.read_table(shard, columns=["episode_index", "task_index"])
        for row in table.to_pylist():
            mapping.setdefault(int(row["episode_index"]), int(row["task_index"]))
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocabulary-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in (args.output, args.vocabulary_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite if intentional")

    tasks = {int(row["task_index"]): str(row["task"])
             for row in pq.read_table(args.dataset_root / "meta" / "tasks.parquet").to_pylist()}
    task_indices = episode_task_indices(args.dataset_root)
    predictions = json.loads(args.predictions.read_text())
    vocabulary: dict[str, str] = {}
    source_counts: Counter[str] = Counter()
    records = []
    for episode, entry in sorted(predictions["episodes"].items(), key=lambda item: int(item[0])):
        ep = int(episode)
        if ep not in task_indices:
            raise ValueError(f"Episode {ep} is absent from the dataset")
        dataset_task = tasks[task_indices[ep]]
        colour = colour_from_task(dataset_task)
        goal = f"pick up the {colour} cube and place it inside the {colour} drawer, then close the drawer"
        previous_end = 0
        segments = []
        for atom in entry["atoms"]:
            phase = atom["generic_phase"]
            start, end = int(atom["start_frame"]), int(atom["end_frame"])
            if phase not in {"pick", "move", "open", "place", "close"} or end <= start:
                raise ValueError(f"Episode {ep}: invalid segment {atom}")
            if start != previous_end:
                raise ValueError(f"Episode {ep}: non-contiguous segments at frame {start}")
            qwen_text = atom.get("qwen_open_vocab_text")
            source = "qwen_open_vocab" if qwen_text else "kinematic_fallback"
            text = str(qwen_text).strip() if qwen_text else fallback_text(phase, colour)
            subtask_id = identifier("qwen" if qwen_text else f"fallback_{phase}_{colour}", text)
            vocabulary[subtask_id] = text
            source_counts[source] += 1
            segments.append({
                "start_frame": start,
                "end_frame": end,
                "start_timestamp": float(atom["start_timestamp"]),
                "end_timestamp": float(atom["end_timestamp"]),
                "generic_phase": phase,
                "subtask_id": subtask_id,
                "subtask_text": text,
                "language_source": source,
                "semantic_cosine": atom.get("semantic_cosine"),
                "canonical_fallback_text": fallback_text(phase, colour),
            })
            previous_end = end
        records.append({
            "episode_index": ep,
            "task_text": goal,
            "dataset_task_text": dataset_task,
            "cube_color": colour,
            "drawer_color": colour,
            "qwen_available": bool(entry.get("qwen_available", False)),
            "segments": segments,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    args.vocabulary_output.parent.mkdir(parents=True, exist_ok=True)
    args.vocabulary_output.write_text(json.dumps({
        "schema_version": "act_open_vocabulary_language_manifest_v1",
        "source_predictions": str(args.predictions),
        "frame_interval": "[start_frame, end_frame)",
        "episodes": len(records),
        "segment_language_source_counts": dict(source_counts),
        "subtask_vocabulary": dict(sorted(vocabulary.items())),
    }, indent=2) + "\n")
    print(f"Wrote {len(records)} episode records to {args.output}")
    print(f"Wrote {len(vocabulary)} unique subtask texts to {args.vocabulary_output}")
    print(f"Language sources: {dict(source_counts)}")


if __name__ == "__main__":
    main()
