#!/usr/bin/env python3
"""Merge non-overlapping raw-Qwen prediction exports without changing texts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {"method": "qwen_raw_lerobot_merged", "episodes": {}}
    for path in args.inputs:
        document = json.loads(path.read_text())
        for episode, entry in document["episodes"].items():
            if episode in result["episodes"]:
                raise ValueError(f"Duplicate episode {episode} in {path}; refusing to choose one result")
            result["episodes"][episode] = entry
    result["episodes"] = dict(sorted(result["episodes"].items(), key=lambda item: int(item[0])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Merged {len(result['episodes'])} episodes into {args.output}")


if __name__ == "__main__":
    main()
