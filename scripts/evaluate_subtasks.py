#!/usr/bin/env python3
"""Evaluate temporal subtask annotations against canonical manual annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PHASES = ("pick", "move", "open", "place", "close")


def phase(atom: dict) -> str:
    return atom.get("generic_phase") or atom["content"].split("_", 1)[0]


def subtask_atoms(document: dict, episode: str) -> list[dict]:
    atoms = document["episodes"][episode]["atoms"]
    atoms = [a for a in atoms if a.get("style", "subtask") == "subtask"]
    return sorted(atoms, key=lambda a: a["start_frame"])


def iou(a: dict, b: dict) -> float:
    overlap = max(0, min(a["end_frame"], b["end_frame"]) - max(a["start_frame"], b["start_frame"]))
    union = max(a["end_frame"], b["end_frame"]) - min(a["start_frame"], b["start_frame"])
    return overlap / union if union else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--episodes", nargs="*", type=int, help="Optional manual-ground-truth episode filter")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gt, pred = json.loads(args.ground_truth.read_text()), json.loads(args.predictions.read_text())
    selected = {str(episode) for episode in args.episodes} if args.episodes else None
    evaluated = set(gt["episodes"]) & set(pred["episodes"])
    if selected is not None:
        evaluated &= selected
    rows = []
    for episode in sorted(evaluated, key=int):
        truth, predicted = subtask_atoms(gt, episode), subtask_atoms(pred, episode)
        truth_by_phase = {phase(a): a for a in truth}
        pred_by_phase = {phase(a): a for a in predicted}
        for name in PHASES:
            if name in truth_by_phase and name in pred_by_phase:
                rows.append({"episode": int(episode), "phase": name, "tIoU": iou(truth_by_phase[name], pred_by_phase[name])})
        for index in range(1, len(PHASES)):
            name = f"{PHASES[index - 1]}_to_{PHASES[index]}"
            if len(truth) > index and len(predicted) > index:
                error = abs(predicted[index]["start_timestamp"] - truth[index]["start_timestamp"])
                rows.append({"episode": int(episode), "boundary": name, "boundary_abs_error_s": error})
    tiou = [r["tIoU"] for r in rows if "tIoU" in r]
    boundary = [r["boundary_abs_error_s"] for r in rows if "boundary_abs_error_s" in r]
    # With exactly one ordered instance of each phase per episode, micro-F1 is
    # the fraction of phase labels that match (the current segmenter always has
    # five; retain this explicit calculation for future Qwen outputs).
    f1_values = []
    exact_f1_values = []
    for episode in sorted(evaluated, key=int):
        truth, predicted = subtask_atoms(gt, episode), subtask_atoms(pred, episode)
        truth_names, pred_names = [phase(a) for a in truth], [phase(a) for a in predicted]
        tp = sum(a == b for a, b in zip(truth_names, pred_names))
        precision = tp / len(pred_names) if pred_names else 0.0
        recall = tp / len(truth_names) if truth_names else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        truth_labels = [a["content"].lower().strip() for a in truth]
        pred_labels = [a["content"].lower().strip() for a in predicted]
        exact_tp = sum(a == b for a, b in zip(truth_labels, pred_labels))
        exact_precision = exact_tp / len(pred_labels) if pred_labels else 0.0
        exact_recall = exact_tp / len(truth_labels) if truth_labels else 0.0
        exact_f1_values.append(
            2 * exact_precision * exact_recall / (exact_precision + exact_recall)
            if exact_precision + exact_recall else 0.0
        )
    report = {
        "episodes_evaluated": len(set(r["episode"] for r in rows)),
        "mean_tIoU": float(np.mean(tiou)) if tiou else None,
        "boundary_MAE_seconds": float(np.mean(boundary)) if boundary else None,
        "action_classification_F1": float(np.mean(f1_values)) if f1_values else None,
        "colour_aware_action_F1": float(np.mean(exact_f1_values)) if exact_f1_values else None,
        "per_item": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_item"}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
