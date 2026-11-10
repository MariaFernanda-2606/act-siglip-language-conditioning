#!/usr/bin/env python3
"""Evaluate free-text Qwen subtasks against manual language using SigLIP cosine.

This is deliberately separate from temporal tIoU/Boundary MAE. It measures
whether Qwen's descriptions express the same actions as manual annotations,
even if their wording differs. A monotonic global alignment permits missing or
extra Qwen subtasks without changing their temporal order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, SiglipModel


GAP_QWEN = -0.10
GAP_MANUAL = -0.20


def embed(texts: list[str], model: SiglipModel, processor: AutoProcessor, device: str) -> torch.Tensor:
    inputs = processor(text=texts, padding="max_length", return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        vectors = model.get_text_features(**inputs)
    if hasattr(vectors, "pooler_output"):
        vectors = vectors.pooler_output
    return torch.nn.functional.normalize(vectors.float(), dim=1).cpu()


def align(similarity: list[list[float]]) -> list[tuple[int | None, int | None, float | None]]:
    """Global sequence alignment: (manual index, Qwen index, cosine)."""
    n, m = len(similarity), len(similarity[0]) if similarity else 0
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0], trace[i][0] = score[i - 1][0] + GAP_MANUAL, "manual_gap"
    for j in range(1, m + 1):
        score[0][j], trace[0][j] = score[0][j - 1] + GAP_QWEN, "qwen_gap"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i][j], trace[i][j] = max(
                (score[i - 1][j - 1] + similarity[i - 1][j - 1], "match"),
                (score[i - 1][j] + GAP_MANUAL, "manual_gap"),
                (score[i][j - 1] + GAP_QWEN, "qwen_gap"),
                key=lambda item: item[0],
            )
    output = []
    i, j = n, m
    while i or j:
        step = trace[i][j]
        if step == "match":
            output.append((i - 1, j - 1, float(similarity[i - 1][j - 1])))
            i, j = i - 1, j - 1
        elif step == "manual_gap":
            output.append((i - 1, None, None))
            i -= 1
        else:
            output.append((None, j - 1, None))
            j -= 1
    return list(reversed(output))


def manual_text(atom: dict) -> str:
    return str(atom.get("original_content") or atom["content"].replace("_", " "))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True, help="Output of export_lerobot_qwen_predictions.py")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--subset", choices=("development", "test", "all"), default="test")
    parser.add_argument("--model-id", default="google/siglip-base-patch16-224")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ground_truth = json.loads(args.ground_truth.read_text())
    qwen = json.loads(args.qwen.read_text())
    split = json.loads(args.split.read_text())
    if args.subset == "all":
        selected = set(ground_truth["episodes"])
    else:
        selected = {str(item) for item in split[args.subset]["episodes"]}

    print(f"Loading SigLIP text encoder: {args.model_id} on {args.device}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = SiglipModel.from_pretrained(args.model_id).to(args.device).eval()

    report = {"metric": "siglip_ordered_open_vocabulary_cosine_v1", "encoder": args.model_id,
              "subset": args.subset, "episodes": {}}
    cosine_values: list[float] = []
    total_manual, total_matched = 0, 0
    for episode in sorted(selected & set(qwen["episodes"]), key=int):
        manual_atoms = ground_truth["episodes"][episode]["atoms"]
        manual_atoms = sorted(manual_atoms, key=lambda atom: atom["start_frame"])
        manual_texts = [manual_text(atom) for atom in manual_atoms]
        qwen_texts = [str(atom.get("content", "")).strip() for atom in qwen["episodes"][episode]["atoms"]]
        total_manual += len(manual_texts)
        vectors = embed(manual_texts + qwen_texts, model, processor, args.device)
        similarity = (vectors[:len(manual_texts)] @ vectors[len(manual_texts):].T).tolist()
        rows = []
        for manual_idx, qwen_idx, cosine in align(similarity):
            row = {"manual_index": manual_idx, "qwen_index": qwen_idx, "cosine": cosine}
            if manual_idx is not None:
                row["manual_text"] = manual_texts[manual_idx]
            if qwen_idx is not None:
                row["qwen_text"] = qwen_texts[qwen_idx]
            rows.append(row)
            if cosine is not None:
                cosine_values.append(cosine)
                total_matched += 1
        report["episodes"][episode] = {"manual_subtasks": manual_texts, "qwen_subtasks": qwen_texts,
                                        "alignment": rows}

    requested = selected & set(ground_truth["episodes"])
    report["summary"] = {
        "episodes_requested": len(requested),
        "episodes_evaluated": len(report["episodes"]),
        "episodes_missing_qwen": sorted(requested - set(qwen["episodes"]), key=int),
        "manual_subtasks": total_manual,
        "matched_subtasks": total_matched,
        "semantic_coverage": total_matched / total_manual if total_manual else None,
        "mean_matched_cosine": sum(cosine_values) / len(cosine_values) if cosine_values else None,
        "mean_cosine_with_unmatched_as_zero": sum(cosine_values) / total_manual if total_manual else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
