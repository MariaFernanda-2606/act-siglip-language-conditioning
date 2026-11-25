#!/usr/bin/env python3
"""Build a per-frame index into the SigLIP text table.

For every dataset frame (ordered by the global `index` column) it stores which
row of the embedding table holds the active subtask, the generic phase, and the
episode-level task text.  Embedding vectors themselves are not repeated.

Output (.npz), all arrays of length total_frames:
    index            global dataset index
    episode_index    episode of the frame
    frame_index      frame within the episode
    subtask_row      row in embeddings table for the active subtask
    task_row         row in embeddings table for the episode task text
    phase            0..4 = pick, move, open, place, close
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PHASES = ("pick", "move", "open", "place", "close")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True, help="Output of encode_subtask_vocabulary.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --overwrite if intentional")

    table = np.load(args.embeddings)
    row_of = {str(text): row for row, text in enumerate(table["texts"])}

    shards = sorted((args.dataset_root / "data").rglob("*.parquet"))
    columns = ["index", "episode_index", "frame_index"]
    frames = {c: np.concatenate([pq.read_table(s, columns=[c]).column(c).to_numpy() for s in shards]) for c in columns}
    order = np.argsort(frames["index"], kind="stable")
    index, episode, frame = (frames[c][order].astype(np.int64) for c in columns)
    if not np.array_equal(index, np.arange(len(index))):
        raise ValueError("Global `index` column is not a contiguous 0..N-1 range")

    n = len(index)
    subtask_row = np.full(n, -1, dtype=np.int32)
    task_row = np.full(n, -1, dtype=np.int32)
    phase = np.full(n, -1, dtype=np.int8)

    starts = np.flatnonzero(np.r_[True, episode[1:] != episode[:-1]])
    episode_start = {int(episode[s]): int(s) for s in starts}
    episode_length = dict(zip(episode_start, np.diff(np.r_[starts, n]).tolist()))

    seen = set()
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        ep = int(record["episode_index"])
        if ep not in episode_start:
            raise ValueError(f"Episode {ep} is not in the dataset")
        base, length = episode_start[ep], episode_length[ep]
        if not np.array_equal(frame[base:base + length], np.arange(length)):
            raise ValueError(f"Episode {ep}: frame_index is not 0..{length - 1}")
        segments = record["segments"]
        if segments[0]["start_frame"] != 0 or segments[-1]["end_frame"] != length:
            raise ValueError(f"Episode {ep}: segments do not cover [0, {length})")
        task_row[base:base + length] = row_of[record["task_text"]]
        for segment in segments:
            s, e = base + segment["start_frame"], base + segment["end_frame"]
            subtask_row[s:e] = row_of[segment["subtask_text"]]
            phase[s:e] = PHASES.index(segment["generic_phase"])
        seen.add(ep)

    missing = sorted(set(episode_start) - seen)
    if missing or (subtask_row < 0).any() or (phase < 0).any():
        raise ValueError(f"Unlabelled frames remain; episodes missing from manifest: {missing[:20]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        index=index, episode_index=episode, frame_index=frame,
        subtask_row=subtask_row, task_row=task_row, phase=phase,
        phases=np.array(PHASES),
        source_manifest=np.array(args.manifest.name),
        source_embeddings=np.array(args.embeddings.name),
    )
    counts = np.bincount(phase, minlength=len(PHASES))
    print(f"Wrote index for {n} frames / {len(seen)} episodes to {args.output}")
    print("Frames per phase: " + ", ".join(f"{p}={c} ({c / n:.1%})" for p, c in zip(PHASES, counts)))
    print(f"Distinct subtask rows used: {len(np.unique(subtask_row))}, task rows: {len(np.unique(task_row))}")


if __name__ == "__main__":
    main()
