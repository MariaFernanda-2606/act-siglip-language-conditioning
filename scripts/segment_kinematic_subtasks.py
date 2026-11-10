#!/usr/bin/env python3
"""Segment cube-to-drawer episodes from robot kinematics.

This is deliberately a *hybrid* annotation component: the motion signals set
the temporal boundaries, while a vision/language model may provide the cube
colour.  It learns only from the manual development split and never changes a
LeRobot dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PHASES = ("pick", "move", "open", "place", "close")


def episode_rows(root: Path, episode: int) -> list[dict]:
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, filters=[("episode_index", "=", episode)])
        if table.num_rows:
            return table.to_pylist()
    raise ValueError(f"Episode {episode} was not found in {root}")


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.vstack([np.convolve(values[:, i], kernel, mode="same") for i in range(values.shape[1])]).T


def features(rows: list[dict], fps: float, smoothing_s: float) -> np.ndarray:
    """Return arm/gripper motion and position features, normalized per episode."""
    state = np.asarray([row["observation.state"] for row in rows], dtype=float)
    velocity = np.gradient(state, 1.0 / fps, axis=0)
    raw = np.column_stack((
        np.linalg.norm(velocity[:, :6], axis=1),
        np.linalg.norm(velocity[:, 7:13], axis=1),
        np.abs(velocity[:, 6]),
        np.abs(velocity[:, 13]),
        state[:, 6],
        state[:, 13],
    ))
    raw = moving_average(raw, max(1, round(smoothing_s * fps)))
    # Per-episode robust scaling makes the detector insensitive to joint units
    # and overall speed differences between recordings.
    median = np.median(raw, axis=0)
    scale = np.median(np.abs(raw - median), axis=0) * 1.4826
    return (raw - median) / np.maximum(scale, 1e-6)


def generic_phase(atom: dict) -> str:
    name = atom["content"].lower()
    for phase in PHASES:
        if name.startswith(phase + "_"):
            return phase
    raise ValueError(f"Unknown subtask label: {atom['content']!r}")


def atoms_for(gt: dict, episode: int) -> list[dict]:
    atoms = [a for a in gt["episodes"][str(episode)]["atoms"] if a.get("style") == "subtask"]
    atoms.sort(key=lambda a: a["start_frame"])
    got = tuple(generic_phase(a) for a in atoms)
    if got != PHASES:
        raise ValueError(f"Episode {episode}: expected {PHASES}, got {got}")
    return atoms


def phase_targets(n_frames: int, atoms: list[dict]) -> np.ndarray:
    target = np.empty(n_frames, dtype=int)
    for index, atom in enumerate(atoms):
        start = max(0, int(atom["start_frame"]))
        end = min(n_frames, int(atom["end_frame"]))
        target[start:end] = index
    return target


def fit_prototypes(root: Path, gt: dict, episodes: list[int], fps: float, smoothing_s: float) -> tuple[np.ndarray, np.ndarray]:
    by_phase: list[list[np.ndarray]] = [[] for _ in PHASES]
    for episode in episodes:
        x = features(episode_rows(root, episode), fps, smoothing_s)
        y = phase_targets(len(x), atoms_for(gt, episode))
        for phase in range(len(PHASES)):
            by_phase[phase].append(x[y == phase])
    stacked = [np.concatenate(items, axis=0) for items in by_phase]
    means = np.vstack([np.mean(values, axis=0) for values in stacked])
    # Diagonal variance is robust and needs no external ML package.
    variances = np.vstack([np.var(values, axis=0) + 0.15 for values in stacked])
    return means, variances


def segment(cost: np.ndarray, min_frames: int) -> list[int]:
    """Minimum-cost five consecutive segments; return [0,b1,...,b4,n]."""
    n, phases = cost.shape
    if n < phases * min_frames:
        raise ValueError("Episode is too short for the requested minimum phase duration")
    cumulative = np.vstack((np.zeros((1, phases)), np.cumsum(cost, axis=0)))
    previous = np.full((phases, n + 1), -1, dtype=int)
    dp = np.full((phases, n + 1), np.inf)
    for end in range(min_frames, n + 1):
        dp[0, end] = cumulative[end, 0] - cumulative[0, 0]
    for phase in range(1, phases):
        for end in range((phase + 1) * min_frames, n + 1):
            starts = np.arange(phase * min_frames, end - min_frames + 1)
            candidates = dp[phase - 1, starts] + cumulative[end, phase] - cumulative[starts, phase]
            best = int(np.argmin(candidates))
            dp[phase, end] = candidates[best]
            previous[phase, end] = int(starts[best])
    ends = [n]
    end = n
    for phase in range(phases - 1, 0, -1):
        end = previous[phase, end]
        ends.append(int(end))
    return [0] + list(reversed(ends))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="LeRobot v3 dataset root")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--subset", choices=("development", "test"), default="test")
    parser.add_argument("--episodes", nargs="*", type=int, help="Overrides --subset")
    parser.add_argument("--leave-one-out", action="store_true", help="For development diagnostics, fit each prediction without its own episode")
    parser.add_argument("--min-duration-s", type=float, default=1.0)
    parser.add_argument("--smoothing-s", type=float, default=0.5)
    parser.add_argument("--cube-color", choices=("red", "green", "yellow"), help="Optional colour supplied by the VLM/task instruction")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    info = json.loads((args.root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    gt = json.loads(args.ground_truth.read_text())
    split = json.loads(args.split.read_text())
    train_episodes = [int(e) for e in split["development"]["episodes"]]
    target_episodes = args.episodes or [int(e) for e in split[args.subset]["episodes"]]
    if args.leave_one_out and not set(target_episodes).issubset(train_episodes):
        raise ValueError("--leave-one-out requires target episodes from the development split")
    means, variances = fit_prototypes(args.root, gt, train_episodes, fps, args.smoothing_s)

    result = {
        "method": "kinematic_phase_prototypes_v1",
        "training_episodes": train_episodes,
        "fps": fps,
        "smoothing_s": args.smoothing_s,
        "min_duration_s": args.min_duration_s,
        "leave_one_out": args.leave_one_out,
        "episodes": {},
        "skipped_episodes": {},
    }
    for episode in target_episodes:
        if args.leave_one_out:
            means, variances = fit_prototypes(
                args.root, gt, [e for e in train_episodes if e != episode], fps, args.smoothing_s
            )
        rows = episode_rows(args.root, episode)
        if len(rows) < 2:
            reason = f"only {len(rows)} frame; velocity and temporal segmentation require at least 2"
            result["skipped_episodes"][str(episode)] = reason
            print(f"WARNING: episode {episode} skipped: {reason}")
            continue
        x = features(rows, fps, args.smoothing_s)
        costs = np.sum((x[:, None, :] - means[None, :, :]) ** 2 / variances[None, :, :], axis=2)
        bounds = segment(costs, max(1, round(args.min_duration_s * fps)))
        atoms = []
        for phase, start, end in zip(PHASES, bounds[:-1], bounds[1:], strict=True):
            label = f"{phase}_{args.cube_color}" if args.cube_color else phase
            atoms.append({
                "content": label,
                "generic_phase": phase,
                "start_frame": start,
                "end_frame": end,
                "start_timestamp": float(rows[start]["timestamp"]),
                "end_timestamp": float(rows[end - 1]["timestamp"]),
            })
        result["episodes"][str(episode)] = {"atoms": atoms}
        print(f"episode {episode}: " + ", ".join(f"{a['generic_phase']}={a['start_timestamp']:.2f}s" for a in atoms))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
