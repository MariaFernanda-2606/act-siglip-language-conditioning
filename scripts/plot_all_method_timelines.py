#!/usr/bin/env python3
"""Compare temporal subtask annotations from all tested methods.

The figure intentionally uses a common phase colour palette across methods:
differences in a row therefore correspond to temporal segmentation, while the
text inside the open-vocabulary rows exposes the language that was retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PHASES = ("pick", "move", "open", "place", "close")
PHASE_LABELS = {
    "pick": "pick",
    "move": "move",
    "open": "open",
    "place": "place",
    "close": "close",
}
COLORS = {
    "pick": "#D84A4A",
    "move": "#377FC7",
    "open": "#4C9B5F",
    "place": "#D99A28",
    "close": "#8460AD",
    "unknown": "#8A949E",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def atoms(document: dict, episode: str) -> list[dict]:
    record = document.get("episodes", {}).get(str(episode), {})
    result = [
        atom
        for atom in record.get("atoms", [])
        if atom.get("style", "subtask") == "subtask"
    ]
    return sorted(result, key=lambda atom: atom["start_timestamp"])


def phase(atom: dict) -> str:
    value = atom.get("generic_phase")
    if value:
        return value.lower()
    text = atom.get("content", "unknown").lower()
    return next((item for item in PHASES if item in text), "unknown")


def shorten(text: str, limit: int = 31) -> str:
    text = text.replace("_", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def draw_row(ax: plt.Axes, row: int, label: str, group: list[dict], duration: float, show_text: bool) -> None:
    for atom in group:
        start = float(atom["start_timestamp"])
        end = float(atom["end_timestamp"])
        width = max(end - start, 0.01)
        current_phase = phase(atom)
        ax.broken_barh([(start, width)], (row - 0.34, 0.68), facecolors=COLORS.get(current_phase, COLORS["unknown"]), edgecolors="white", linewidth=1.0)
        label_text = shorten(atom.get("content", current_phase)) if show_text else PHASE_LABELS.get(current_phase, current_phase)
        # Do not force unreadable text into short intervals.
        if width / duration > 0.095:
            ax.text(start + width / 2, row, label_text, ha="center", va="center", color="white", fontsize=7.3, fontweight="normal", clip_on=True)


def plot(documents: list[tuple[str, dict, bool]], episodes: list[str], output: Path) -> None:
    rows_per_episode = len(documents)
    figure_height = max(7.4, 1.57 * len(episodes) * rows_per_episode + 1.4)
    fig, ax = plt.subplots(figsize=(16, figure_height))
    y_ticks, y_labels = [], []
    separator_positions = []
    all_rows = 0

    for episode_index, episode in enumerate(episodes):
        groups = [(name, atoms(document, episode), show_text) for name, document, show_text in documents]
        if not all(group for _, group, _ in groups):
            missing = [name for name, group, _ in groups if not group]
            raise ValueError(f"Episode {episode} is absent or empty in: {', '.join(missing)}")
        duration = max(atom["end_timestamp"] for _, group, _ in groups for atom in group)
        base = episode_index * (rows_per_episode + 1)
        for offset, (name, group, show_text) in enumerate(groups):
            row = base + offset
            draw_row(ax, row, name, group, duration, show_text)
            y_ticks.append(row)
            y_labels.append(f"Episódio {episode} — {name}")
            all_rows = max(all_rows, row)
        separator_positions.append(base + rows_per_episode - 0.02)

    ax.set_yticks(y_ticks, y_labels)
    ax.invert_yaxis()
    ax.set_ylim(all_rows + 0.8, -1.05)
    ax.set_xlabel("Tempo no episódio (s)", fontweight="bold")
    ax.set_title("Comparação qualitativa da segmentação temporal e da linguagem", loc="left", fontsize=16, fontweight="bold", pad=18)
    ax.grid(axis="x", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=9)
    for position in separator_positions[:-1]:
        ax.axhline(position + 0.55, color="#B8C0C7", linewidth=0.8, alpha=0.85)
    legend = [Patch(facecolor=COLORS[name], label=PHASE_LABELS[name]) for name in PHASES]
    ax.legend(handles=legend, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    fig.text(
        0.5,
        0.012,
        "Cores representam a fase. Nas linhas de vocabulário aberto, o texto exibido é a descrição produzida ou preservada pelo Qwen; "
        "as linhas híbridas usam as fronteiras cinemáticas.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--qwen-closed", type=Path, required=True)
    parser.add_argument("--qwen-open", type=Path, required=True)
    parser.add_argument("--kinematic", type=Path, required=True)
    parser.add_argument("--hybrid-open", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=str, default=["5", "55", "107"])
    parser.add_argument("--output", type=Path, required=True, help="Path without extension, or ending in .png/.pdf")
    args = parser.parse_args()
    output = args.output.with_suffix("") if args.output.suffix.lower() in {".png", ".pdf"} else args.output
    documents = [
        ("Manual", load(args.ground_truth), False),
        ("Qwen fechado", load(args.qwen_closed), False),
        ("Qwen aberto", load(args.qwen_open), True),
        ("Cinemático", load(args.kinematic), False),
        ("Híbrido aberto", load(args.hybrid_open), True),
    ]
    plot(documents, args.episodes, output)
    print(f"Wrote {output.with_suffix('.png')} and {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
