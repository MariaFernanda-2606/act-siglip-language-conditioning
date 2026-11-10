#!/usr/bin/env python3
"""Figuras em Matplotlib para os métodos mantidos na comparação final."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "Qwen puro fechado": "#E58A25",
    "Qwen puro aberto": "#8A5BB0",
    "Cinemático": "#31917F",
    "Híbrido aberto": "#254B76",
    "coverage": "#E58A25",
    "cosine": "#31917F",
    "conservative": "#3E7BC0",
    "fallback": "#98A2AD",
}
PHASES = ("pick", "move", "open", "place", "close")
PHASE_LABELS = ("pick", "move", "open", "place", "close")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(fig: plt.Figure, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def label_bars(ax: plt.Axes, bars, fmt: str) -> None:
    for bar in bars:
        value = bar.get_width()
        ax.text(value + ax.get_xlim()[1] * 0.015, bar.get_y() + bar.get_height() / 2, format(value, fmt), va="center", fontsize=10, fontweight="bold")


def temporal_figure(records: list[dict], output: Path) -> None:
    names = [record["name"] for record in records]
    colors = [COLORS[name] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, key, title, fmt in (
        (axes[0], "tiou", "tIoU temporal médio  ↑", ".3f"),
        (axes[1], "mae", "MAE de fronteira (s)  ↓", ".2f"),
    ):
        values = [record[key] for record in records]
        bars = ax.barh(names, values, color=colors, height=0.62)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 1 if key == "tiou" else max(values) * 1.23)
        label_bars(ax, bars, fmt)
    fig.suptitle("Comparação temporal no conjunto de teste (30 episódios)", fontsize=15, fontweight="bold", y=1.03)
    fig.text(0.5, -0.02, "O híbrido aberto usa as fronteiras cinemáticas; por isso, suas métricas temporais são iguais às do método cinemático.", ha="center", fontsize=10)
    save(fig, output)


def semantic_figure(closed: dict, semantic: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    f1 = closed["colour_aware_action_F1"]
    bar = axes[0].bar(["Vocabulário fechado\nF1 sensível à cor"], [f1], color=COLORS["Qwen puro fechado"], width=0.55)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Qwen com vocabulário fechado", loc="left", fontweight="bold")
    axes[0].set_ylabel("Pontuação ↑")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].set_axisbelow(True)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].bar_label(bar, labels=[f"{f1:.3f}"], padding=5, fontweight="bold")
    labels = ["Cobertura", "Cosseno\nalinhado", "Cosseno\nconservador"]
    values = [semantic["semantic_coverage"], semantic["mean_matched_cosine"], semantic["mean_cosine_with_unmatched_as_zero"]]
    bars = axes[1].bar(labels, values, color=[COLORS["coverage"], COLORS["cosine"], COLORS["conservative"]], width=0.62)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title("Qwen com vocabulário aberto", loc="left", fontweight="bold")
    axes[1].set_ylabel("Pontuação ↑")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].set_axisbelow(True)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=5, fontweight="bold")
    fig.suptitle("Saídas semânticas usam protocolos de avaliação diferentes", fontsize=15, fontweight="bold", y=1.03)
    fig.text(0.5, -0.02, "O cosseno em vocabulário aberto usa SigLIP; o cosseno conservador atribui zero às fases manuais omitidas pelo Qwen.", ha="center", fontsize=10)
    save(fig, output)


def phase_and_provenance(semantic: dict, manifest: Path, output: Path) -> None:
    phase_stats = defaultdict(lambda: [0, 0, []])
    for episode in semantic["episodes"].values():
        for row in episode["alignment"]:
            if row["manual_index"] is None:
                continue
            phase = row["manual_text"].split()[0].lower()
            phase_stats[phase][1] += 1
            if row["cosine"] is not None:
                phase_stats[phase][0] += 1
                phase_stats[phase][2].append(row["cosine"])
    coverage = [phase_stats[phase][0] / phase_stats[phase][1] for phase in PHASES]
    cosines = [np.mean(phase_stats[phase][2]) if phase_stats[phase][2] else 0.0 for phase in PHASES]
    sources = Counter()
    with manifest.open() as handle:
        for line in handle:
            sources.update(segment["language_source"] for segment in json.loads(line)["segments"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    x = np.arange(len(PHASES))
    width = 0.36
    b1 = axes[0].bar(x - width / 2, coverage, width, label="Cobertura", color=COLORS["coverage"])
    b2 = axes[0].bar(x + width / 2, cosines, width, label="Cosseno alinhado", color=COLORS["cosine"])
    axes[0].set_xticks(x, PHASE_LABELS)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Pontuação ↑")
    axes[0].set_title("Qwen aberto por fase manual", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].set_axisbelow(True)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].bar_label(b1, labels=[f"{v:.2f}" for v in coverage], padding=3, fontsize=8)
    axes[0].bar_label(b2, labels=[f"{v:.2f}" for v in cosines], padding=3, fontsize=8)
    qwen, fallback = sources["qwen_open_vocab"], sources["kinematic_fallback"]
    axes[1].pie([qwen, fallback], labels=[f"Texto livre do Qwen\n{qwen} ({qwen/(qwen+fallback):.1%})", f"Fallback cinemático\n{fallback} ({fallback/(qwen+fallback):.1%})"], colors=[COLORS["cosine"], COLORS["fallback"]], startangle=90, wedgeprops={"width": 0.52, "edgecolor": "white"}, textprops={"fontsize": 10})
    axes[1].set_title("Proveniência da linguagem no manifesto final", loc="left", fontweight="bold")
    fig.suptitle("A cobertura semântica evidencia o papel do fallback cinemático", fontsize=15, fontweight="bold", y=1.03)
    save(fig, output)


def component_matrix(output: Path) -> None:
    rows = [
        ["Qwen puro fechado", "rótulos fixos do Qwen", "Qwen", "—", "—"],
        ["Qwen puro aberto", "frases livres do Qwen", "Qwen", "—", "—"],
        ["Cinemático", "templates canônicos", "cinemática", "—", "sim"],
        ["Híbrido aberto", "frases livres do Qwen", "cinemática", "sim", "sim"],
    ]
    columns = ["Método", "Linguagem", "Fronteiras", "SigLIP", "Fallback"]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="left", colLoc="left", bbox=[0, 0.03, 1, 0.87])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.65)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#254B76")
            cell.set_text_props(color="white", weight="bold")
        elif row == len(rows):
            cell.set_facecolor("#DDEFEA")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#F4F7F9" if row % 2 else "#E8EEF2")
    ax.set_title("Componentes fornecidos por cada método considerado", loc="left", fontsize=15, fontweight="bold", pad=10)
    fig.text(0.5, -0.01, "O híbrido aberto preserva linguagem natural do Qwen e substitui seus intervalos temporais pouco confiáveis por fronteiras cinemáticas.", ha="center", fontsize=10)
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-closed", type=Path, required=True)
    parser.add_argument("--qwen-open", type=Path, required=True)
    parser.add_argument("--kinematic", type=Path, required=True)
    parser.add_argument("--hybrid-open", type=Path, required=True)
    parser.add_argument("--open-semantic", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    closed_qwen, open_qwen, kinematic, open_hybrid = [load(path) for path in (args.qwen_closed, args.qwen_open, args.kinematic, args.hybrid_open)]
    semantic = load(args.open_semantic)["summary"] | {"episodes": load(args.open_semantic)["episodes"]}
    records = [
        {"name": "Qwen puro fechado", "tiou": closed_qwen["mean_tIoU"], "mae": closed_qwen["boundary_MAE_seconds"]},
        {"name": "Qwen puro aberto", "tiou": open_qwen["mean_tIoU"], "mae": open_qwen["boundary_MAE_seconds"]},
        {"name": "Cinemático", "tiou": kinematic["mean_tIoU"], "mae": kinematic["boundary_MAE_seconds"]},
        {"name": "Híbrido aberto", "tiou": open_hybrid["mean_tIoU"], "mae": open_hybrid["boundary_MAE_seconds"]},
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporal_figure(records, args.output_dir / "01_temporal_comparison")
    semantic_figure(closed_qwen, semantic, args.output_dir / "02_semantic_comparison")
    phase_and_provenance(semantic, args.manifest, args.output_dir / "03_phase_coverage_and_provenance")
    component_matrix(args.output_dir / "04_method_components")
    print(f"Wrote PNG and PDF figures to {args.output_dir}")


if __name__ == "__main__":
    main()
