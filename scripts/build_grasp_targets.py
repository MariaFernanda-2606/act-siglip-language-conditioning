#!/usr/bin/env python3
"""Per-frame "where will the grasp be?" targets for the act_siglip auxiliary grasp head.

For every episode, the grasp pose is the robot state at the last frame of the pick phase (phase 0 in the
frame index), restricted to the arm that picks the cube (left arm joints 0..5 by default; the gripper is
left out). The cubes are shuffled between episodes and every episode starts from the same pose, so early
in the pick phase this pose can only be predicted by finding, in the images, the cube named in the text.

Targets are standardised with the mean / std over episodes. Every frame of an episode gets its episode's
target; the policy applies the loss to pick-phase frames only.

Output (.npz):
    index        (N,)    global dataset index (same order as the frame index)
    target       (N, D)  standardised grasp pose of the frame's episode
    mean, std    (D,)    to undo the standardisation
    state_dims   (D,)    which observation.state dimensions are used
    names        (D,)    their names
    episode_index, grasp_frame (E,)  episode and frame of each grasp

Example:
    python scripts/build_grasp_targets.py \
        --root ~/.cache/huggingface/lerobot/MariaFPG/place_the_cubes_in_the_drawers_v3_annotated
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--frame-index", type=Path, default=None,
                    help="default: <root>/annotations/language_conditioning/frame_subtask_index_v3.npz")
    ap.add_argument("--state-dims", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
                    help="observation.state dims of the grasp pose (default: left arm joints 0..5)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <root>/annotations/language_conditioning/grasp_targets_v3.npz")
    args = ap.parse_args()

    root = args.root.expanduser()
    lc = root / "annotations/language_conditioning"
    fi = np.load(args.frame_index or lc / "frame_subtask_index_v3.npz")
    out = args.out or lc / "grasp_targets_v3.npz"

    index_parts, state_parts = [], []
    for f in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        t = pq.read_table(f, columns=["index", "observation.state"])
        index_parts.append(t.column("index").to_numpy())
        state_parts.append(np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)))
    index = np.concatenate(index_parts)
    state = np.concatenate(state_parts).astype(np.float32)
    state = state[np.argsort(index)]
    if not np.array_equal(np.sort(index), fi["index"]):
        raise ValueError("dataset indices don't match the frame index")

    names = json.loads((root / "meta/info.json").read_text())["features"]["observation.state"]["names"]
    dims = np.array(args.state_dims)
    episode, phase = fi["episode_index"], fi["phase"]
    episodes = np.unique(episode)
    grasp_frame = np.empty(len(episodes), dtype=np.int64)
    grasp = np.empty((len(episodes), len(dims)), dtype=np.float32)
    row_of_episode = np.empty(episodes.max() + 1, dtype=np.int64)
    for i, e in enumerate(episodes):
        frames = np.flatnonzero(episode == e)
        pick = frames[phase[frames] == 0]
        if len(pick) == 0:
            raise ValueError(f"episode {e} has no pick phase")
        grasp_frame[i] = fi["frame_index"][pick[-1]]
        grasp[i] = state[pick[-1], dims]
        row_of_episode[e] = i

    mean, std = grasp.mean(0), grasp.std(0).clip(min=1e-3)
    target = ((grasp - mean) / std)[row_of_episode[episode]].astype(np.float32)
    np.savez_compressed(out, index=fi["index"], target=target, mean=mean, std=std, state_dims=dims,
                        names=np.array([names[d] for d in dims]), episode_index=episodes, grasp_frame=grasp_frame)
    np.set_printoptions(precision=3, suppress=True)
    print(f"{len(episodes)} episodes, dims {[names[d] for d in dims]}")
    print(f"grasp pose std over episodes: {std}")
    print(f"grasp frame: median {np.median(grasp_frame) / 30:.1f} s")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
