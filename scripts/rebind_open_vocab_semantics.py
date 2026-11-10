#!/usr/bin/env python3
"""Attach Qwen open-vocabulary text to kinematic segments by SigLIP cosine similarity.

Qwen supplies *what* each step is in free language.  The kinematic predictor
supplies *when* the five ordered phases occur.  A global sequence alignment
keeps their temporal order while allowing Qwen to emit extra or missing steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, SiglipModel


CANONICAL = (
    ("pick", "pick up the cube"),
    ("move", "move the cube to the drawer"),
    ("open", "open the drawer"),
    ("place", "place the cube in the drawer"),
    ("close", "close the drawer"),
)
GAP_QWEN = -0.10       # extra free-text Qwen step
GAP_CANONICAL = -0.20  # a canonical phase that Qwen omitted


def embed(texts: list[str], model: SiglipModel, processor: AutoProcessor, device: str) -> torch.Tensor:
    inputs = processor(text=texts, padding="max_length", return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        features = model.get_text_features(**inputs)
    # transformers releases differ: older SigLIP versions return the feature
    # tensor directly, while newer ones return a ModelOutput.
    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    return torch.nn.functional.normalize(features.float(), dim=1).cpu()


def align(similarity: list[list[float]]) -> list[tuple[int | None, int | None, float | None]]:
    """Needleman-Wunsch alignment of canonical rows with Qwen columns."""
    n = len(similarity)
    m = len(similarity[0]) if n else 0
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + GAP_CANONICAL
        trace[i][0] = "canonical_gap"
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + GAP_QWEN
        trace[0][j] = "qwen_gap"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = (
                (score[i - 1][j - 1] + similarity[i - 1][j - 1], "match"),
                (score[i - 1][j] + GAP_CANONICAL, "canonical_gap"),
                (score[i][j - 1] + GAP_QWEN, "qwen_gap"),
            )
            score[i][j], trace[i][j] = max(choices, key=lambda item: item[0])
    output: list[tuple[int | None, int | None, float | None]] = []
    i, j = n, m
    while i or j:
        step = trace[i][j]
        if step == "match":
            output.append((i - 1, j - 1, similarity[i - 1][j - 1]))
            i, j = i - 1, j - 1
        elif step == "canonical_gap":
            output.append((i - 1, None, None))
            i -= 1
        else:
            output.append((None, j - 1, None))
            j -= 1
    return list(reversed(output))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinematic", type=Path, required=True, help="Five-phase kinematic predictions JSON")
    parser.add_argument("--qwen", type=Path, required=True, help="Raw open-vocabulary Qwen export JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="google/siglip-base-patch16-224")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-unannotated-kinematic", action="store_true",
                        help="Keep kinematic episodes with no Qwen parse, using canonical fallback for all phases")
    args = parser.parse_args()

    kinematic = json.loads(args.kinematic.read_text())
    qwen = json.loads(args.qwen.read_text())
    print(f"Loading SigLIP text encoder: {args.model_id} on {args.device}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = SiglipModel.from_pretrained(args.model_id).to(args.device).eval()
    canonical_texts = [text for _, text in CANONICAL]

    result = {"method": "qwen_open_vocab_siglip_semantics_plus_kinematic_boundaries_v1", "episodes": {}}
    report = {"encoder": args.model_id, "episodes": {}, "mean_matched_cosine": None}
    all_cosines: list[float] = []
    # A partial Qwen run (e.g. --episodes '[0]') must not silently generate
    # canonical-only pseudo-labels for every other kinematic episode.
    selected_episodes = sorted(
        kinematic["episodes"] if args.include_unannotated_kinematic else qwen["episodes"], key=int
    )
    missing_kinematic = [episode for episode in selected_episodes if episode not in kinematic["episodes"]]
    if missing_kinematic:
        raise ValueError(f"No kinematic prediction for Qwen episode(s): {missing_kinematic}")
    for episode in selected_episodes:
        kin_entry = kinematic["episodes"][episode]
        qwen_atoms = qwen["episodes"].get(episode, {}).get("atoms", [])
        qwen_texts = [str(atom.get("content", "")).strip() for atom in qwen_atoms]
        matched: dict[int, tuple[int, float]] = {}
        dropped: list[str] = []
        if qwen_texts:
            vectors = embed(canonical_texts + qwen_texts, model, processor, args.device)
            canonical_vectors, qwen_vectors = vectors[:5], vectors[5:]
            sim = (canonical_vectors @ qwen_vectors.T).tolist()
            for canonical_idx, qwen_idx, value in align(sim):
                if canonical_idx is not None and qwen_idx is not None:
                    matched[canonical_idx] = (qwen_idx, float(value))
                    all_cosines.append(float(value))
                elif canonical_idx is None and qwen_idx is not None:
                    dropped.append(qwen_texts[qwen_idx])

        atoms = []
        phase_to_index = {phase: idx for idx, (phase, _) in enumerate(CANONICAL)}
        for atom in kin_entry["atoms"]:
            copied = dict(atom)
            idx = phase_to_index[copied["generic_phase"]]
            if idx in matched:
                qwen_idx, cosine = matched[idx]
                copied["content"] = qwen_texts[qwen_idx]
                copied["qwen_open_vocab_text"] = qwen_texts[qwen_idx]
                copied["semantic_cosine"] = cosine
            else:
                copied["content"] = CANONICAL[idx][1]
                copied["qwen_open_vocab_text"] = None
                copied["semantic_cosine"] = None
            copied["canonical_semantic_target"] = CANONICAL[idx][1]
            atoms.append(copied)
        result["episodes"][episode] = {"atoms": atoms, "qwen_available": episode in qwen["episodes"]}
        report["episodes"][episode] = {
            "qwen_available": episode in qwen["episodes"],
            "qwen_subtasks": qwen_texts,
            "matched_phases": sum(idx in matched for idx in range(5)),
            "dropped_qwen_subtasks": dropped,
            "mean_matched_cosine": (sum(value for _, value in matched.values()) / len(matched)) if matched else None,
        }
        print(f"episode {episode}: matched {len(matched)}/5 phases; Qwen emitted {len(qwen_texts)} subtasks")

    report["mean_matched_cosine"] = sum(all_cosines) / len(all_cosines) if all_cosines else None
    report["qwen_missing_episodes"] = [
        episode for episode in selected_episodes if episode not in qwen["episodes"]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    report_path = args.output.with_name(args.output.stem + "_semantic_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
