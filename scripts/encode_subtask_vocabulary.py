#!/usr/bin/env python3
"""Encode every unique instruction of a language manifest with the SigLIP text encoder.

The table is keyed by text (not by subtask_id), so the same sentence coming from
Qwen and from the fallback is encoded once.  Episode-level task texts are added
as well, so the "global task only" baseline uses the same table.

Output (.npz):
    texts       [N]      unicode strings, row order of the table
    kinds       [N]      "subtask" | "task" | "subtask+task"
    embeddings  [N, D]   float32, L2-normalised SigLIP text features
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, SiglipModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="google/siglip-base-patch16-224")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --overwrite if intentional")

    kinds: dict[str, set[str]] = {}
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        kinds.setdefault(record["task_text"], set()).add("task")
        for segment in record["segments"]:
            kinds.setdefault(segment["subtask_text"], set()).add("subtask")
    texts = sorted(kinds)

    print(f"Encoding {len(texts)} unique texts with {args.model_id} on {args.device}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = SiglipModel.from_pretrained(args.model_id).to(args.device).eval()
    inputs = processor(text=texts, padding="max_length", return_tensors="pt").to(args.device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    if not isinstance(features, torch.Tensor):  # newer transformers return a model output
        features = features.pooler_output
    embeddings = torch.nn.functional.normalize(features.float(), dim=1).cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        texts=np.array(texts),
        kinds=np.array(["+".join(sorted(kinds[t])) for t in texts]),
        embeddings=embeddings.astype(np.float32),
        model_id=np.array(args.model_id),
        source_manifest=np.array(args.manifest.name),
    )
    print(f"Wrote table {embeddings.shape} to {args.output}")


if __name__ == "__main__":
    main()
