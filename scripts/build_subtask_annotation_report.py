#!/usr/bin/env python3
"""Build a detailed Portuguese PDF report for the subtask-annotation stage."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "resumo_rotulacao_subtarefas.pdf"


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def table(data: list[list[str]], widths: list[float]) -> Table:
    value = Table(data, colWidths=[width * cm for width in widths], repeatRows=1, hAlign="LEFT")
    value.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#254B76")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("LEADING", (0, 0), (-1, -1), 10.3),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D8E0E8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F3F6F8"), colors.white]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return value


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8E0E8"))
    canvas.line(1.8 * cm, 1.45 * cm, A4[0] - 1.8 * cm, 1.45 * cm)
    canvas.setFillColor(colors.HexColor("#52606D"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(1.8 * cm, 0.95 * cm, "ACT + SigLIP - Relatorio da etapa de rotulacao de subtarefas")
    canvas.drawRightString(A4[0] - 1.8 * cm, 0.95 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def add_figure(story, path: str, caption: str, styles) -> None:
    figure = ROOT / path
    if figure.exists():
        image = Image(str(figure), width=16.2 * cm, height=7.0 * cm, kind="proportional")
        story.extend([Spacer(1, 0.15 * cm), image, Spacer(1, 0.08 * cm), p(caption, styles["caption"]), Spacer(1, 0.28 * cm)])


def main() -> None:
    qwen_closed = read("outputs/qwen/qwen7b_test_metrics.json")
    qwen_open = read("outputs/qwen/open_vocab/manual45_test_raw_temporal_metrics.json")
    kinematic = read("outputs/kinematic/test/kinematic_test_metrics.json")
    hybrid_closed = read("outputs/hybrid/all/hybrid_qwen7b_all_metrics.json")
    hybrid_open = read("outputs/hybrid/open_vocab/manual45_test_temporal_metrics.json")
    semantic = read("outputs/qwen/open_vocab/manual45_test_semantic_metrics.json")["summary"]
    manifest_path = ROOT / "datasets_annotation/language_manifests/open_vocab/all261_open_vocab_hybrid_v1.jsonl"
    manifest = [json.loads(line) for line in manifest_path.open()]
    source_counts = {"qwen_open_vocab": 0, "kinematic_fallback": 0}
    for record in manifest:
        for segment in record["segments"]:
            source_counts[segment["language_source"]] += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=1.8 * cm, leftMargin=1.8 * cm, topMargin=1.7 * cm, bottomMargin=2.0 * cm)
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=25, leading=30, textColor=colors.HexColor("#254B76"), alignment=TA_LEFT, spaceAfter=12),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontSize=13, leading=18, textColor=colors.HexColor("#52606D"), spaceAfter=24),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=colors.HexColor("#254B76"), spaceBefore=12, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#254B76"), spaceBefore=8, spaceAfter=5),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=10.2, leading=14.3, spaceAfter=8),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontSize=10.2, leading=14.0, leftIndent=13, firstLineIndent=-9, spaceAfter=4),
        "caption": ParagraphStyle("caption", parent=base["Normal"], fontSize=8.5, leading=10.5, textColor=colors.HexColor("#52606D"), alignment=TA_CENTER),
        "formula": ParagraphStyle("formula", parent=base["BodyText"], fontName="Courier", fontSize=9.2, leading=13, leftIndent=12, backColor=colors.HexColor("#F3F6F8"), borderPadding=6, spaceAfter=8),
    }
    story = []

    story.extend([
        Spacer(1, 2.4 * cm),
        p("Rotulacao hibrida de subtarefas para ACT condicionado por linguagem", styles["title"]),
        p("Relatorio detalhado da etapa de anotacao, avaliacao e preparacao do manifesto linguistico", styles["subtitle"]),
        Spacer(1, 0.6 * cm),
        Table([[p("<b>Escopo</b><br/>Segmentacao temporal e linguagem de subtarefas para demonstracoes de manipulacao de cubos e gavetas.", styles["body"]), p("<b>Resultado central</b><br/>Qwen fornece semantica linguistica; a cinemática fornece fronteiras temporais confiaveis.", styles["body"])]], colWidths=[8.0 * cm, 8.0 * cm], style=[("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF4F7")), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8E0E8")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 12)]),
        Spacer(1, 1.3 * cm),
        p("Resumo executivo", styles["h1"]),
        p("O projeto pretende adaptar o ACT para receber linguagem durante o treinamento e a execucao. Para isso, as demonstracoes precisam ser associadas a instrucoes que indiquem a subtarefa ativa. Nesta etapa, foram comparadas abordagens baseadas apenas em Qwen, apenas em cinemática e em combinacoes hibridas. A solucao adotada separa a responsabilidade semantica da responsabilidade temporal.", styles["body"]),
        p("O Qwen em vocabulário aberto gerou frases semanticamente adequadas quando emitiu uma subtarefa, mas apresentou baixa cobertura e fronteiras temporais imprecisas. O segmentador cinemático apresentou desempenho temporal superior. A anotacao final combina ambos e gera um manifesto linguistico para 261 episodios utilizaveis.", styles["body"]),
        PageBreak(),
    ])

    story.extend([
        p("1. Contexto e objetivo", styles["h1"]),
        p("A tarefa de cada demonstracao consiste em pegar um cubo colorido e coloca-lo na gaveta da cor correspondente. Ha tres configuracoes de cor: vermelho, verde e amarelo. Cada demonstracao pode ser descrita pela sequencia ordenada de cinco fases: pegar o cubo, mover em direcao a gaveta, abrir a gaveta, colocar o cubo e fechar a gaveta.", styles["body"]),
        p("O objetivo da etapa de rotulacao foi produzir, para cada demonstracao, intervalos temporais de subtarefas acompanhados por linguagem natural. Esses rotulos serao posteriormente convertidos em embeddings textuais SigLIP e usados como condicao para os chunks de acao do ACT.", styles["body"]),
        p("2. Dados e protocolo de avaliacao", styles["h1"]),
        p("O dataset original foi convertido de LeRobot v2 para v3. Foram identificados tres episodios com apenas um frame, que nao podem ser segmentados cinematicamente porque a estimacao de velocidade exige pelo menos dois frames. O conjunto final contem 261 episodios utilizaveis.", styles["body"]),
        table([
            ["Componente", "Quantidade / configuracao"],
            ["Dataset utilizavel", "261 episodios"],
            ["Ground truth manual", "45 episodios"],
            ["Desenvolvimento", "15 episodios, usados para ajustar parametros cinemáticos"],
            ["Teste cego", "30 episodios, usado para os resultados finais"],
            ["Ontologia manual", "pick, move, open, place, close, com cor"],
        ], [5.1, 11.1]),
        Spacer(1, 0.25 * cm),
        p("A separacao entre desenvolvimento e teste evita avaliar os parametros no mesmo subconjunto utilizado para escolha de suavizacao e duracao minima do segmentador cinemático.", styles["body"]),
        p("3. Metodos explorados", styles["h1"]),
        table([
            ["Metodo", "Linguagem", "Fronteiras", "Uso de Qwen / SigLIP"],
            ["Qwen puro fechado", "Rotulos fixos de cubo e gaveta", "Timestamps estimados por Qwen", "Qwen tenta resolver semantica e tempo"],
            ["Qwen puro aberto", "Frases imperativas livres", "Timestamps estimados por Qwen", "Prompt padrao do LeRobot; vocabulário aberto"],
            ["Cinemático", "Templates canônicos", "Estados e velocidades do robô", "Nao usa Qwen"],
            ["Hibrido fechado", "Templates + cor inferida", "Cinética", "Qwen fornece principalmente a cor"],
            ["Hibrido aberto", "Frase Qwen ou fallback", "Cinética", "SigLIP alinha frases livres; fallback completa fases ausentes"],
        ], [3.1, 4.2, 3.6, 5.3]),
        PageBreak(),
    ])

    story.extend([
        p("4. Por que Qwen puro foi insuficiente", styles["h1"]),
        p("Nas configuracoes puras, Qwen precisava inferir tanto a semantica quanto os instantes de inicio e fim de cada evento a partir de contato visual amostrado. Ele conseguiu reconhecer cores e varias acoes, mas frequentemente fundiu fases intermediarias, especialmente mover e abrir, e errou os instantes de transicao.", styles["body"]),
        p("O prompt aberto atual e o mesmo prompt padrao do LeRobot usado nos testes abertos anteriores. A diferenca da abordagem atual nao e um novo prompt que melhora o tempo: e o uso da saida. As frases abertas sao preservadas como informacao semantica, enquanto os timestamps de Qwen sao descartados e substituidos por fronteiras cinemáticas.", styles["body"]),
        add_figure(story, "outputs/figures/method_comparison_matplotlib/01_temporal_comparison.png", "Figura 1 - Comparacao temporal no teste cego. Os hibridos compartilham as fronteiras do metodo cinemático.", styles),
        p("5. Metricas", styles["h1"]),
        p("As metricas temporais comparam intervalos previstos e intervalos manuais. Para uma subtarefa i, tIoU e a razao entre a intersecao e a uniao dos intervalos:", styles["body"]),
        p("mean tIoU = (1/N) sum_i |I_pred,i intersect I_GT,i| / |I_pred,i union I_GT,i|", styles["formula"]),
        p("A Boundary MAE mede a distancia absoluta, em segundos, entre cada fronteira prevista e a fronteira manual:", styles["body"]),
        p("Boundary MAE = (1/B) sum_i |b_pred,i - b_GT,i|", styles["formula"]),
        p("Para vocabulário fechado, o F1 de acao e definido por F1 = 2PR/(P+R), com P como precisao e R como revocacao. Para vocabulário aberto, frases nao devem ser avaliadas por igualdade literal. Assim, foi usado o cosseno entre embeddings SigLIP:", styles["body"]),
        p("cos(e_Qwen, e_GT) = (e_Qwen dot e_GT) / (||e_Qwen|| ||e_GT||)", styles["formula"]),
        p("A cobertura semantica e a proporcao de fases manuais para as quais Qwen emitiu uma frase alinhada. O cosseno conservador atribui zero para fases omitidas por Qwen, combinando qualidade textual e cobertura.", styles["body"]),
        PageBreak(),
    ])

    temporal_rows = [
        ["Qwen puro fechado", f"{qwen_closed['mean_tIoU']:.3f}", f"{qwen_closed['boundary_MAE_seconds']:.2f} s", f"F1 sensivel a cor = {qwen_closed['colour_aware_action_F1']:.3f}"],
        ["Qwen puro aberto", f"{qwen_open['mean_tIoU']:.3f}", f"{qwen_open['boundary_MAE_seconds']:.2f} s", f"Cosseno = {semantic['mean_matched_cosine']:.3f}; cobertura = {semantic['semantic_coverage']:.3f}"],
        ["Cinemático", f"{kinematic['mean_tIoU']:.3f}", f"{kinematic['boundary_MAE_seconds']:.2f} s", "Nao gera linguagem livre"],
        ["Hibrido fechado", f"{hybrid_closed['mean_tIoU']:.3f}", f"{hybrid_closed['boundary_MAE_seconds']:.2f} s", f"F1 sensivel a cor = {hybrid_closed['colour_aware_action_F1']:.3f}"],
        ["Hibrido aberto", f"{hybrid_open['mean_tIoU']:.3f}", f"{hybrid_open['boundary_MAE_seconds']:.2f} s", f"Cosseno = {semantic['mean_matched_cosine']:.3f}; conservador = {semantic['mean_cosine_with_unmatched_as_zero']:.3f}"],
    ]
    story.extend([
        p("6. Resultados", styles["h1"]),
        table([["Metodo", "tIoU medio", "MAE de fronteira", "Resultado semantico"]] + temporal_rows, [3.2, 2.2, 3.0, 7.8]),
        Spacer(1, 0.25 * cm),
        p("O resultado temporal mais importante e a diferenca entre Qwen puro e cinemática. O metodo cinemático reduz a MAE de aproximadamente 7-9 segundos para 1,77 segundos e aumenta o tIoU para 0,638. Como os dois hibridos usam as mesmas fronteiras, eles mantem esse desempenho temporal.", styles["body"]),
        p("No teste aberto, Qwen apresentou cosseno medio de 0,959 nas subtarefas emitidas. Isso indica que, quando produz uma frase, ela e semanticamente semelhante ao texto manual. Entretanto, a cobertura foi 0,500: apenas 75 das 150 fases manuais tiveram correspondencia. O cosseno conservador de 0,480 penaliza explicitamente as fases omitidas.", styles["body"]),
        add_figure(story, "outputs/figures/method_comparison_matplotlib/02_semantic_comparison.png", "Figura 2 - Protocolos semanticos: F1 para rotulos fechados e cosseno/cobertura para linguagem aberta.", styles),
        add_figure(story, "outputs/figures/method_comparison_matplotlib/03_phase_coverage_and_provenance.png", "Figura 3 - Cobertura por fase e proveniencia da linguagem no manifesto final.", styles),
        p("7. Interpretacao", styles["h1"]),
        p("A solucao hibrida aberta e a mais adequada para a proxima etapa do projeto. Ela nao afirma que Qwen rotulou autonomamente todas as fases. Pelo contrario, a proveniencia de cada segmento e mantida: Qwen fornece texto quando ha frase alinhada; a cinemática fornece sempre a estrutura temporal; e o fallback produz linguagem colorida quando Qwen omite uma fase.", styles["body"]),
        PageBreak(),
    ])

    story.extend([
        p("8. Manifesto linguistico para o ACT", styles["h1"]),
        p("O manifesto linguistico e a interface entre a rotulacao hibrida e o treinamento do ACT. Ele e um arquivo JSONL com uma linha por episodio. Cada registro contem a meta global, cor do cubo e da gaveta, e cinco segmentos com inicio/fim em frames e segundos, texto ativo, fase generica, fonte da linguagem e similaridade semantica quando disponivel.", styles["body"]),
        p("Durante o treinamento, o dataloader deve localizar o segmento que contem o frame inicial de cada chunk de acao. O texto da subtarefa ativa e entao convertido em embedding SigLIP e usado para condicionar o ACT. Chunks que atravessarem fronteiras devem ser excluidos na primeira implementacao, para que um chunk corresponda a apenas uma instrucao.", styles["body"]),
        table([
            ["Propriedade do manifesto", "Valor"],
            ["Episodios", str(len(manifest))],
            ["Segmentos totais", str(source_counts["qwen_open_vocab"] + source_counts["kinematic_fallback"])],
            ["Texto livre de Qwen", f"{source_counts['qwen_open_vocab']} ({source_counts['qwen_open_vocab'] / 1305:.1%})"],
            ["Fallback cinemático", f"{source_counts['kinematic_fallback']} ({source_counts['kinematic_fallback'] / 1305:.1%})"],
            ["Textos de subtarefa unicos", "62"],
        ], [6.2, 10.0]),
        Spacer(1, 0.2 * cm),
        p("Exemplo conceitual: observacao + estado do robo + embedding da subtarefa ativa -> chunk de acoes. A instrucao global pode ser mantida como contexto adicional, enquanto a instrucao de subtarefa fornece condicionamento local e temporalmente alinhado.", styles["formula"]),
        add_figure(story, "outputs/figures/method_comparison_matplotlib/04_method_components.png", "Figura 4 - Componentes efetivamente fornecidos por cada metodo avaliado.", styles),
        p("9. Planejado versus realizado", styles["h1"]),
        table([
            ["Atividade", "Status", "Resultado"],
            ["Conversao do dataset para LeRobot v3", "Concluido", "Dataset v3 pronto para anotacao."],
            ["Ground truth manual", "Concluido", "45 episodios: 15 desenvolvimento e 30 teste."],
            ["Avaliacao de Qwen aberto e fechado", "Concluido", "Metricas temporais e semanticas calculadas."],
            ["Segmentacao hibrida", "Concluido", "Fronteiras cinemáticas e linguagem Qwen/fallback."],
            ["Anotacao completa", "Concluido", "261 episodios e 1.305 segmentos."],
            ["Manifesto linguistico", "Concluido", "JSONL pronto para o dataloader do ACT."],
            ["Embeddings SigLIP e treinamento ACT", "Proxima etapa", "Extrair embeddings, adaptar dataloader e treinar."],
        ], [6.0, 2.7, 7.5]),
        p("10. Limitacoes e proximos passos", styles["h1"]),
        p("A avaliacao atual mostra que Qwen nao e um segmentador temporal confiavel para esta tarefa. Alem disso, a cobertura semantica do Qwen aberto e limitada: mover e abrir sao frequentemente omitidos. O fallback resolve a completude do conjunto, mas deve ser reportado como fonte distinta de linguagem. O trabalho ainda nao avaliou o impacto das instrucoes abertas no desempenho do ACT, que e a proxima validacao experimental.", styles["body"]),
        p("Os proximos passos sao: (1) extrair embeddings SigLIP para os 62 textos unicos; (2) adaptar o dataloader para recuperar a subtarefa ativa por chunk; (3) treinar um ACT sem linguagem, com linguagem canônica e com o manifesto hibrido aberto; e (4) avaliar generalizacao para combinacoes de cor nao demonstradas e comandos composicionais.", styles["body"]),
    ])
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
