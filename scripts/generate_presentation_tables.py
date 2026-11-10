#!/usr/bin/env python3
"""Create slide-ready Matplotlib tables for the subtask annotation stage."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt


NAVY = "#254B76"
BLUE = "#3E7BC0"
TEAL = "#31917F"
ORANGE = "#E58A25"
PURPLE = "#8A5BB0"
LIGHT = "#F3F6F8"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save_table(title: str, columns: list[str], rows: list[list[str]], output: Path, widths: list[float], note: str = "") -> None:
    wrap_widths = [max(18, int(width * 112)) for width in widths]
    wrapped_columns = [textwrap.fill(value, width) for value, width in zip(columns, wrap_widths)]
    # Matplotlib renders $...$ with MathText (LaTeX-like syntax). Do not wrap
    # equations because inserting a newline inside the expression breaks it.
    wrapped_rows = [[value if "$" in value else textwrap.fill(value, width) for value, width in zip(row, wrap_widths)] for row in rows]
    row_lines = [max(cell.count("\n") + 1 for cell in row) for row in wrapped_rows]
    fig, ax = plt.subplots(figsize=(15, max(4.2, 1.8 + sum(row_lines) * 0.52)))
    ax.axis("off")
    table = ax.table(cellText=wrapped_rows, colLabels=wrapped_columns, cellLoc="left", colLoc="left", colWidths=widths, bbox=[0.01, 0.10 if note else 0.03, 0.98, 0.80 if note else 0.86])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        elif row % 2:
            cell.set_facecolor(LIGHT)
        else:
            cell.set_facecolor("white")
        if row > 0:
            cell.set_height(0.075 * row_lines[row - 1])
            if col == 1 and "$" in cell.get_text().get_text():
                cell.get_text().set_fontsize(10)
    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", color=NAVY, pad=18)
    if note:
        fig.text(0.02, 0.025, note, fontsize=10, color="#52606D")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-closed", type=Path, required=True)
    parser.add_argument("--qwen-open", type=Path, required=True)
    parser.add_argument("--kinematic", type=Path, required=True)
    parser.add_argument("--hybrid-open", type=Path, required=True)
    parser.add_argument("--open-semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    qc, qo, kin, ho = [load(path) for path in (args.qwen_closed, args.qwen_open, args.kinematic, args.hybrid_open)]
    semantic = load(args.open_semantic)["summary"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_table(
        "Métodos avaliados para a rotulação de subtarefas",
        ["Método", "Fonte da linguagem", "Limites temporais", "Papel do Qwen / SigLIP"],
        [
            ["Qwen puro (fechado)", "Rótulos fixos específicos de cor", "Timestamps do Qwen", "Qwen prevê rótulos de ação e tempos."],
            ["Qwen puro (aberto)", "Frases imperativas livres", "Timestamps do Qwen", "Qwen descreve eventos visíveis em vocabulário aberto."],
            ["Cinemático", "Templates canônicos", "Cinética dos estados do robô", "Não utiliza Qwen ou SigLIP."],
            ["Híbrido (aberto)", "Frase livre do Qwen ou fallback colorido", "Cinética dos estados do robô", "SigLIP alinha texto livre do Qwen à semântica da fase."],
        ],
        args.output_dir / "01_methods", [0.20, 0.24, 0.22, 0.32],
    )

    save_table(
        "Resultados no conjunto de teste: 30 episódios anotados manualmente",
        ["Método", "tIoU médio ↑", "MAE de fronteira ↓", "Resultado semântico"],
        [
            ["Qwen puro (fechado)", f"{qc['mean_tIoU']:.3f}", f"{qc['boundary_MAE_seconds']:.2f} s", f"F1 sensível à cor = {qc['colour_aware_action_F1']:.3f}"],
            ["Qwen puro (aberto)", f"{qo['mean_tIoU']:.3f}", f"{qo['boundary_MAE_seconds']:.2f} s", f"Cosseno = {semantic['mean_matched_cosine']:.3f}; cobertura = {semantic['semantic_coverage']:.3f}"],
            ["Cinemático", f"{kin['mean_tIoU']:.3f}", f"{kin['boundary_MAE_seconds']:.2f} s", "Não gera linguagem livre"],
            ["Híbrido (aberto)", f"{ho['mean_tIoU']:.3f}", f"{ho['boundary_MAE_seconds']:.2f} s", f"Cosseno = {semantic['mean_matched_cosine']:.3f}; conservador = {semantic['mean_cosine_with_unmatched_as_zero']:.3f}"],
        ],
        args.output_dir / "02_test_results", [0.24, 0.15, 0.18, 0.41],
        "tIoU e MAE de fronteira avaliam os intervalos temporais. F1 de rótulo fixo e cosseno de vocabulário aberto usam protocolos semânticos diferentes e não devem ser comparados diretamente.",
    )

    save_table(
        "Métricas utilizadas na avaliação da rotulação",
        ["Métrica", "Fórmula", "O que mede", "Direção desejada"],
        [
            ["tIoU médio", r"$\frac{1}{N}\sum_{i=1}^{N}\frac{|I_i^{pred}\cap I_i^{GT}|}{|I_i^{pred}\cup I_i^{GT}|}$", "Sobreposição temporal entre intervalos previstos e manuais", "Maior; 1 é perfeito"],
            ["MAE de fronteira", r"$\frac{1}{B}\sum_{i=1}^{B}|\hat{b}_i-b_i|$", "Erro absoluto dos tempos de transição, em segundos", "Menor; 0 s é perfeito"],
            ["F1 de ação", r"$F_1=2\,\frac{P\,R}{P+R}$", "Concordância exata de rótulos de ação no vocabulário fechado", "Maior; 1 é perfeito"],
            ["Cosseno (SigLIP)", r"$\frac{e_{Qwen}\cdot e_{GT}}{\|e_{Qwen}\|\,\|e_{GT}\|}$", "Similaridade semântica entre frase livre do Qwen e frase manual", "Maior; 1 é muito similar"],
            ["Cobertura semântica", r"$\frac{N_{correspondidas}}{N_{manuais}}$", "Fração de fases manuais para as quais Qwen emitiu uma frase alinhada", "Maior; 1 é completo"],
            ["Cosseno conservador", r"$\frac{1}{N}\sum_{i=1}^{N}\tilde{c}_i,\;\tilde{c}_i\in\{c_i,0\}$", "Qualidade semântica penalizada por fases omitidas pelo Qwen", "Maior; 1 é completo e similar"],
        ],
        args.output_dir / "03_metrics", [0.15, 0.32, 0.34, 0.19],
    )

    save_table(
        "Planejado versus realizado na etapa de rotulação",
        ["Atividade planejada", "Status", "Evidência / saída"],
        [
            ["Converter o dataset para formato compatível com LeRobot", "Concluído", "Dataset convertido de v2 para v3."],
            ["Criar ground truth manual", "Concluído", "45 episódios: 15 de desenvolvimento e 30 de teste cego."],
            ["Avaliar anotações geradas por Qwen", "Concluído", "Qwen fechado e aberto avaliados por métricas temporais e semânticas."],
            ["Desenvolver segmentação híbrida", "Concluído", "Fronteiras cinemáticas combinadas à semântica do Qwen."],
            ["Anotar conjunto completo de treinamento", "Concluído", "261 episódios utilizáveis; 1.305 segmentos em cinco fases."],
            ["Preparar dados linguísticos para o ACT", "Concluído", "Manifesto JSONL aberto com 62 textos únicos."],
            ["Integrar embeddings SigLIP ao treinamento do ACT", "Próxima etapa", "Banco de embeddings, alteração do dataloader, treinamento e avaliação do ACT."],
        ],
        args.output_dir / "04_planned_vs_completed", [0.40, 0.17, 0.41],
    )
    print(f"Wrote PNG and PDF tables to {args.output_dir}")


if __name__ == "__main__":
    main()
