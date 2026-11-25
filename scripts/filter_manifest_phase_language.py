#!/usr/bin/env python3
"""Filter an open-vocabulary language manifest with phase and colour rules.

SigLIP text cosine does not separate opposite actions reliably (e.g. "close the
drawer" scores 0.82-0.84 against the canonical "open" phase), so a cosine
threshold cannot remove misaligned Qwen phrases.  This script keeps a Qwen
phrase only when

1. every colour it mentions matches the episode colour, and
2. its action verb is compatible with the segment's kinematic phase.

Rejected phrases are replaced by the colour-aware canonical fallback.  The
original Qwen phrase and the rejection reason are kept for provenance.  Segment
boundaries are never changed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

COLORS = ("red", "green", "yellow")
COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b")

# Patterns are matched against the lower-cased phrase with colours replaced by <c>.
PHASE_PATTERNS = {
    "pick": r"(pick up|grasp|lift|approach|extend (the )?(robot )?arm towards) the <c> cube",
    "move": r"move (the <c> cube )?towards? the <c> drawer",
    "open": r"open the (<c> )?drawer",
    "place": r"(place|insert) the <c> cube (in|inside|into) the (<c> )?drawer",
    "close": r"close the (<c> )?drawer",
}
PHASE_RE = {phase: re.compile(rf"^{pattern}$") for phase, pattern in PHASE_PATTERNS.items()}


def identifier(prefix: str, text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return f"{prefix}:{normalized}"


def rejection_reason(text: str, phase: str, colour: str) -> str | None:
    lowered = " ".join(text.lower().split())
    mentioned = set(COLOR_RE.findall(lowered))
    if mentioned and mentioned != {colour}:
        return f"wrong_colour:{'+'.join(sorted(mentioned))}"
    template = COLOR_RE.sub("<c>", lowered)
    if not PHASE_RE[phase].match(template):
        return "phase_mismatch"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="Input manifest JSONL (v1)")
    parser.add_argument("--output", type=Path, required=True, help="Filtered manifest JSONL")
    parser.add_argument("--vocabulary-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.resolve() == args.manifest.resolve():
        raise ValueError("Output must differ from the input manifest")
    for path in (args.output, args.vocabulary_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite if intentional")

    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    vocabulary: dict[str, str] = {}
    sources: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    rejected_by_phase: Counter[str] = Counter()

    for record in records:
        colour = record["cube_color"]
        for segment in record["segments"]:
            phase = segment["generic_phase"]
            segment["rejected_qwen_text"] = None
            segment["rejection_reason"] = None
            if segment["language_source"] == "qwen_open_vocab":
                reason = rejection_reason(segment["subtask_text"], phase, colour)
                if reason is not None:
                    segment["rejected_qwen_text"] = segment["subtask_text"]
                    segment["rejection_reason"] = reason
                    segment["subtask_text"] = segment["canonical_fallback_text"]
                    segment["subtask_id"] = identifier(f"fallback_{phase}_{colour}", segment["subtask_text"])
                    segment["language_source"] = "kinematic_fallback"
                    reasons[reason.split(":")[0]] += 1
                    rejected_by_phase[phase] += 1
            vocabulary[segment["subtask_id"]] = segment["subtask_text"]
            sources[segment["language_source"]] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    args.vocabulary_output.write_text(json.dumps({
        "schema_version": "act_open_vocabulary_language_manifest_v2",
        "source_manifest": args.manifest.name,
        "filter": "colour consistency + phase-compatible action verb",
        "phase_patterns": PHASE_PATTERNS,
        "frame_interval": "[start_frame, end_frame)",
        "episodes": len(records),
        "segment_language_source_counts": dict(sources),
        "rejection_reason_counts": dict(reasons),
        "rejections_by_phase": dict(rejected_by_phase),
        "subtask_vocabulary": dict(sorted(vocabulary.items())),
    }, indent=2) + "\n")

    print(f"Wrote {len(records)} episodes to {args.output}")
    print(f"Language sources: {dict(sources)}")
    print(f"Rejections: {dict(reasons)}  by phase: {dict(rejected_by_phase)}")
    print(f"Unique subtask texts: {len(vocabulary)}")


if __name__ == "__main__":
    main()
