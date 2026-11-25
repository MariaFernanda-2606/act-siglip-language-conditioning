"""Offline (open-loop) evaluation of act_siglip checkpoints on held-out episodes.

For every sampled frame and every checkpoint it records:
  * action chunk L1 (normalised space, padded steps masked) when conditioning on
      - the frame's own training label text ("own"), and
      - each of the 5 canonical subtask instructions of the episode colour ("canon", shape [N, 5]);
    column `phase` of "canon" is the deployment condition (planner sends canonical text),
    the other columns are the swapped-instruction test;
  * done-head logits for [own text, canonical phase 0..4] (zero latent, as at inference);
  * colour-swap test: actions for the canonical instruction of the current phase in the 2 other colours
    ("l1_colour" [N, 2] and "colour_change" = mean |pred(other colour) - pred(own colour)|). With the cubes
    shuffled between episodes, only the colour in the text says which cube to take: a policy that uses the
    language must change its actions here, above all in the pick phase (gate before going to the robot).

Videos are decoded once and all checkpoints are evaluated on the same batch.
Outputs one npz per checkpoint plus a summary table (summary.json / printed).

Example:
  python scripts/evaluate_act_siglip_offline.py \
      --run-dir outputs/train/act_siglip_subtask_v3 --split development --stride 5 \
      --out outputs/eval/act_siglip_subtask_v3_dev
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()
from lerobot_policy_act_siglip import ACTSiglipPolicy  # noqa: E402
from lerobot_policy_act_siglip.modeling_act_siglip import OBS_LANGUAGE  # noqa: E402

DEFAULT_ROOT = ("/home/trossen-ai/.cache/huggingface/lerobot/MariaFPG/"
                "place_the_cubes_in_the_drawers_v3_annotated")
N_PHASES = 5
FIRST_STEPS = 10  # "short horizon" L1: the first actions actually executed
EARLY_FRAMES = 150  # first 5 s of each episode


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """ROC AUC via the rank statistic (P(score_pos > score_neg))."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort() + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_policy(path: Path, frame_index: str, device: str) -> ACTSiglipPolicy:
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.language_table_path = None          # table comes from the checkpoint state dict
    cfg.frame_index_path = frame_index      # frame -> label / phase / colour lookups
    cfg.device = device
    return ACTSiglipPolicy.from_pretrained(path, config=cfg).to(device).eval()


def masked_l1(pred: torch.Tensor, target: torch.Tensor, pad: torch.Tensor, steps: int | None = None) -> torch.Tensor:
    """Per-sample mean |pred - target| over valid (non-padded) steps and all action dims."""
    if steps is not None:
        pred, target, pad = pred[:, :steps], target[:, :steps], pad[:, :steps]
    valid = (~pad).float().unsqueeze(-1)
    err = ((pred - target).abs() * valid).sum((1, 2))
    return err / (valid.sum((1, 2)) * pred.shape[-1]).clamp_min(1)


def win_rate(err_own: np.ndarray, err_other: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of (frame, other colour) pairs where the right colour's error is lower (0.5 = chance)."""
    if not mask.any():
        return float("nan")
    return float((err_own[mask, None] < err_other[mask]).mean())


def summarise(r: dict[str, np.ndarray]) -> dict[str, float]:
    phase = r["phase"]
    n = len(phase)
    rows = np.arange(n)
    l1_correct = r["l1_canon"][rows, phase]
    l1_wrong = np.where(np.eye(N_PHASES, dtype=bool)[phase], np.nan, r["l1_canon"])
    # Done head: canonical instruction of the current phase -> not done; of an earlier phase -> done.
    logits = r["done_logits"]
    neg = logits[rows, 1 + phase]
    pos = np.concatenate([logits[phase == p, 1 + q] for p in range(1, N_PHASES) for q in range(p)])
    own = logits[:, 0]
    s = {
        "frames": n,
        "l1_own": float(r["l1_own"].mean()),
        "l1_canon": float(l1_correct.mean()),
        f"l1_canon_first{FIRST_STEPS}": float(r["l1_canon_first"][rows, phase].mean()),
        "l1_swapped": float(np.nanmean(l1_wrong)),
        "swap_ratio": float(np.nanmean(l1_wrong) / l1_correct.mean()),
        "done_acc": float(((neg < 0).sum() + (pos > 0).sum()) / (len(neg) + len(pos))),
        "done_auc": auc(pos, neg),
        "done_false_pos_own": float((own > 0).mean()),
        "colour_swap_ratio": float(r["l1_colour"].mean() / l1_correct.mean()),
        "colour_change": float(r["colour_change"].mean()),
        "colour_change_pick": float(r["colour_change"][phase == 0].mean()) if (phase == 0).any() else float("nan"),
        # Fraction of (frame, other colour) pairs where the right colour's actions are closer to the demo.
        # 0.5 = the text colour doesn't help; ~1 = the policy goes for the cube named in the text.
        "colour_win_rate": float((l1_correct[:, None] < r["l1_colour"]).mean()),
        "colour_win_rate_pick": win_rate(l1_correct, r["l1_colour"], phase == 0),
        # Start of the episode: arm still at the common start pose, only the text says which cube.
        "colour_win_rate_early": win_rate(l1_correct, r["l1_colour"], r["frame_index"] < EARLY_FRAMES),
    }
    # Grasp head (if the checkpoint has one), pick frames, standardised units: does the predicted grasp
    # move to the cube named in the text?
    pick = phase == 0
    if "grasp_err" in r and pick.any() and not np.isnan(r["grasp_err"][pick]).all():
        s["grasp_err_pick"] = float(r["grasp_err"][pick].mean())
        s["grasp_err_other_colour"] = float(r["grasp_err_colour"][pick].mean())
        s["grasp_win_rate_pick"] = win_rate(r["grasp_err"], r["grasp_err_colour"], pick)
        s["grasp_win_rate_early"] = win_rate(r["grasp_err"], r["grasp_err_colour"], pick & (r["frame_index"] < EARLY_FRAMES))
    else:
        s.update({k: float("nan") for k in ("grasp_err_pick", "grasp_err_other_colour", "grasp_win_rate_pick",
                                             "grasp_win_rate_early")})
    for p in range(N_PHASES):
        s[f"l1_canon_phase{p}"] = float(l1_correct[phase == p].mean()) if (phase == p).any() else float("nan")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--checkpoints", nargs="*", default=None, help="e.g. 050000 100000 (default: all numbered)")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--split", choices=["development", "test"], default="development")
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="evaluate these episodes instead of --split (e.g. training episodes, to check memorisation)")
    ap.add_argument("--stride", type=int, default=5, help="evaluate every n-th frame")
    ap.add_argument("--max-frames", type=int, default=None, help="debug: stop after this many frames")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--frame-index", default=None,
                    help="default: <root>/annotations/language_conditioning/frame_subtask_index_v3.npz")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summarise-only", action="store_true",
                    help="recompute summary.json from the ckpt_*.npz already in --out (no model, no GPU)")
    args = ap.parse_args()

    cols = ["l1_own", "l1_canon", f"l1_canon_first{FIRST_STEPS}", "swap_ratio", "colour_swap_ratio",
            "colour_change_pick", "colour_win_rate", "colour_win_rate_pick", "colour_win_rate_early",
            "grasp_err_pick", "grasp_win_rate_pick", "grasp_win_rate_early", "done_acc", "done_auc",
            "done_false_pos_own"] + [f"l1_canon_phase{p}" for p in range(N_PHASES)]
    if args.summarise_only:
        files = sorted(args.out.glob("ckpt_*.npz"))
        summary = {f.stem[5:]: summarise(dict(np.load(f))) for f in files}
        path = args.out / "summary.json"
        info = json.loads(path.read_text()) if path.exists() else {}
        path.write_text(json.dumps(info | {"checkpoints": summary}, indent=2))
        print_table(summary, cols)
        return

    root = Path(args.root)
    frame_index = args.frame_index or str(root / "annotations/language_conditioning/frame_subtask_index_v3.npz")
    split = json.loads((root / "annotations/evaluation/evaluation_split_48.json").read_text())
    episodes = args.episodes or split[args.split]["episodes"]

    ckpt_dir = args.run_dir / "checkpoints"
    names = args.checkpoints or sorted(p.name for p in ckpt_dir.iterdir() if p.name.isdigit())
    policies = {n: load_policy(ckpt_dir / n / "pretrained_model", frame_index, args.device) for n in names}
    first = policies[names[0]]
    chunk = first.config.chunk_size
    pre, _ = make_pre_post_processors(first.config, pretrained_path=str(ckpt_dir / names[0] / "pretrained_model"))
    print(f"checkpoints: {names}")

    fps = 30
    ds = LeRobotDataset("local", root=root, episodes=episodes,
                        delta_timestamps={"action": [i / fps for i in range(chunk)]})
    positions = list(range(0, len(ds), args.stride))
    if args.max_frames:
        positions = positions[: args.max_frames]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, positions), batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers)
    print(f"{args.split}: {len(episodes)} episodes, {len(positions)} frames (stride {args.stride})")

    keys = ["l1_own", "l1_own_first", "l1_canon", "l1_canon_first", "done_logits", "l1_colour", "colour_change",
            "grasp_err", "grasp_err_colour"]
    res = {n: {k: [] for k in keys} for n in names}
    meta = {"index": [], "episode_index": [], "frame_index": [], "phase": []}
    t0 = time.time()
    with torch.no_grad():
        for i, raw in enumerate(loader):
            b = pre(raw)
            idx = b["index"].long()
            target, pad = b["action"], b["action_is_pad"]
            phase = first.frame_phase[idx]
            meta["index"].append(idx.cpu()); meta["phase"].append(phase.cpu())
            meta["episode_index"].append(raw["episode_index"]); meta["frame_index"].append(raw["frame_index"])
            for n, pol in policies.items():
                bb = pol._prepare(b)  # own label text from the frame index
                colour = pol.frame_colour[idx]
                canon = [pol._centre(pol.language_table[pol.canonical_rows[colour, torch.full_like(idx, p)]])
                         for p in range(N_PHASES)]
                actions_own, _, logits = pol.model(bb, tuple(canon))
                has_grasp = getattr(pol, "_has_grasp_targets", False)
                rows = torch.arange(len(idx), device=idx.device)
                l1c, l1cf = [], []
                actions_by_phase, grasp_by_phase = [], []
                for p in range(N_PHASES):
                    actions_p = pol.model({**bb, OBS_LANGUAGE: canon[p]})[0]
                    actions_by_phase.append(actions_p)
                    if has_grasp:
                        grasp_by_phase.append(pol.model.last_grasp)
                    l1c.append(masked_l1(actions_p, target, pad)); l1cf.append(masked_l1(actions_p, target, pad, FIRST_STEPS))
                own_canon = torch.stack(actions_by_phase, 0)[phase, rows]
                nan = torch.full((len(idx),), float("nan"), device=idx.device)
                if has_grasp:
                    grasp_target = pol.frame_grasp[idx]
                    grasp_own = (torch.stack(grasp_by_phase, 0)[phase, rows] - grasp_target).abs().mean(-1)
                l1x, change, grasp_x = [], [], []
                for k in (1, 2):  # the two other colours, same phase
                    other = pol._centre(pol.language_table[pol.canonical_rows[(colour + k) % 3, phase]])
                    actions_x = pol.model({**bb, OBS_LANGUAGE: other})[0]
                    l1x.append(masked_l1(actions_x, target, pad))
                    change.append((actions_x - own_canon).abs().mean((1, 2)))
                    grasp_x.append((pol.model.last_grasp - grasp_target).abs().mean(-1) if has_grasp else nan)
                r = res[n]
                r["l1_own"].append(masked_l1(actions_own, target, pad).cpu())
                r["l1_own_first"].append(masked_l1(actions_own, target, pad, FIRST_STEPS).cpu())
                r["l1_canon"].append(torch.stack(l1c, 1).cpu()); r["l1_canon_first"].append(torch.stack(l1cf, 1).cpu())
                r["done_logits"].append(logits.T.cpu())
                r["l1_colour"].append(torch.stack(l1x, 1).cpu()); r["colour_change"].append(torch.stack(change, 1).mean(1).cpu())
                r["grasp_err"].append((grasp_own if has_grasp else nan).cpu())
                r["grasp_err_colour"].append(torch.stack(grasp_x, 1).cpu())
            if i % 20 == 0:
                done = min((i + 1) * args.batch_size, len(positions))
                el = time.time() - t0
                print(f"  {done}/{len(positions)} frames  {el / 60:.1f} min  (eta {el / done * (len(positions) - done) / 60:.1f} min)",
                      flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {k: torch.cat(v).numpy() for k, v in meta.items()}
    summary = {}
    for n in names:
        r = {k: torch.cat(v).numpy() for k, v in res[n].items()} | meta
        np.savez_compressed(args.out / f"ckpt_{n}.npz", **r)
        summary[n] = summarise(r)
    (args.out / "summary.json").write_text(json.dumps(
        {"run_dir": str(args.run_dir), "split": args.split, "stride": args.stride, "frame_index": frame_index,
         "checkpoints": summary}, indent=2))

    print_table(summary, cols)
    print(f"saved to {args.out}  ({(time.time() - t0) / 60:.1f} min)")


def print_table(summary: dict[str, dict[str, float]], cols: list[str]) -> None:
    print("\nckpt    " + " ".join(f"{c.replace('l1_canon_', ''):>13s}" for c in cols))
    for n, s in summary.items():
        print(f"{n:7s} " + " ".join(f"{s[c]:13.4f}" for c in cols))
    print("\nphase0..4 = pick, move, open, place, close")


if __name__ == "__main__":
    main()
