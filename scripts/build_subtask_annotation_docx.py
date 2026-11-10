#!/usr/bin/env python3
"""Create the detailed Portuguese Word report for the annotation stage."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "documents" / "relatorio_rotulacao_hibrida_subtarefas.docx"
BLUE = "254B76"
LIGHT_BLUE = "EAF1F5"
GREY = "D9E1E8"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), fill)
    tc_pr.append(element)


def borders(cell) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    border = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "6")
        node.set(qn("w:color"), GREY)
        border.append(node)
    tc_pr.append(border)


def set_cell_text(cell, text: str, bold: bool = False, white: bool = False) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.size = Pt(9.3)
    run.font.name = "Aptos"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    if white:
        run.font.color.rgb = RGBColor(255, 255, 255)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    borders(cell)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.width = Inches(widths[index])
        shade(cell, BLUE)
        set_cell_text(cell, header, bold=True, white=True)
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        for index, text in enumerate(row):
            cells[index].width = Inches(widths[index])
            if row_index % 2 == 0:
                shade(cells[index], LIGHT_BLUE)
            set_cell_text(cells[index], text)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def equation(doc: Document, latex: str, name: str) -> None:
    """Render a display equation to a high-resolution image for stable DOCX layout."""
    formulas = ROOT / "qa" / "docx_rotulacao" / "formulas"
    formulas.mkdir(parents=True, exist_ok=True)
    image = formulas / f"{name}.png"
    fig = plt.figure(figsize=(6.3, 0.52), dpi=300)
    fig.patch.set_alpha(0)
    fig.text(0.5, 0.5, f"${latex}$", ha="center", va="center", fontsize=15, color="black")
    plt.axis("off")
    fig.savefig(image, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.add_run().add_picture(str(image), width=Inches(4.5))


def add_body(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph(style="Body Text")
    paragraph.add_run(text)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item)


def add_figure(doc: Document, relative: str, caption: str) -> None:
    path = ROOT / relative
    if not path.exists():
        return
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Inches(6.35))
    caption_p = doc.add_paragraph(style="Caption")
    caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_p.add_run(caption)


def heading(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def main() -> None:
    qwen_closed = load("outputs/qwen/qwen7b_test_metrics.json")
    qwen_open = load("outputs/qwen/open_vocab/manual45_test_raw_temporal_metrics.json")
    kinematic = load("outputs/kinematic/test/kinematic_test_metrics.json")
    hybrid_closed = load("outputs/hybrid/all/hybrid_qwen7b_all_metrics.json")
    hybrid_open = load("outputs/hybrid/open_vocab/manual45_test_temporal_metrics.json")
    semantic = load("outputs/qwen/open_vocab/manual45_test_semantic_metrics.json")["summary"]
    manifest_path = ROOT / "datasets_annotation/language_manifests/open_vocab/all261_open_vocab_hybrid_v1.jsonl"
    manifest = [json.loads(line) for line in manifest_path.open()]
    qwen_segments = sum(s["language_source"] == "qwen_open_vocab" for e in manifest for s in e["segments"])
    fallback_segments = sum(s["language_source"] == "kinematic_fallback" for e in manifest for s in e["segments"])

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)

    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    styles["Normal"]._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    styles["Normal"].font.size = Pt(10.8)
    body = styles["Body Text"]
    body.font.name = "Aptos"
    body._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    body._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    body.font.size = Pt(10.8)
    body.paragraph_format.line_spacing = 1.18
    body.paragraph_format.space_after = Pt(7)
    for name, size in (("Title", 24), ("Heading 1", 16), ("Heading 2", 12)):
        style = styles[name]
        style.font.name = "Aptos Display"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Aptos Display")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos Display")
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
    styles["Heading 1"].paragraph_format.space_before = Pt(14)
    styles["Heading 1"].paragraph_format.space_after = Pt(7)
    styles["Heading 2"].paragraph_format.space_before = Pt(10)
    styles["Heading 2"].paragraph_format.space_after = Pt(5)
    styles["Caption"].font.name = "Aptos"
    styles["Caption"].font.size = Pt(8.5)

    title = doc.add_paragraph(style="Title")
    title.add_run("Rotulação híbrida de subtarefas para ACT condicionado por linguagem")
    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(18)
    run = subtitle.add_run("Relatório técnico da etapa de anotação automática, avaliação e geração do manifesto linguístico")
    run.italic = True
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(70, 80, 90)

    heading(doc, "Resumo executivo")
    add_body(doc, "Este documento descreve a etapa de rotulação automática de subtarefas para demonstrações robóticas de pegar um cubo colorido e colocá-lo na gaveta correspondente. O resultado central é um método híbrido em vocabulário aberto: a cinemática define fronteiras temporais estáveis e completas; o Qwen fornece descrições naturais quando disponíveis; e um fallback baseado na fase e na tarefa completa as descrições ausentes.")
    add_body(doc, "A abordagem foi escolhida após comparar Qwen puro em vocabulário aberto, Qwen puro com vocabulário fechado, segmentação cinemática e variantes híbridas. O manifesto linguístico resultante contém 261 episódios utilizáveis e 1.305 segmentos, prontos para serem associados a embeddings textuais SigLIP no treinamento do ACT.")

    heading(doc, "1 Contexto e objetivo")
    add_body(doc, "A tarefa do dataset consiste em pegar um cubo vermelho, amarelo ou verde e colocá-lo na gaveta da mesma cor. Cada episódio contém imagens, estados do robô e ações demonstradas. A instrução global da tarefa é útil, mas não especifica qual ação local está ativa em cada momento da trajetória.")
    add_body(doc, "Para condicionar o ACT por linguagem de forma temporalmente informativa, cada demonstração deve ser dividida em subtarefas. A ontologia usada na avaliação manual possui cinco fases ordenadas: pegar o cubo, mover até a gaveta, abrir a gaveta, colocar o cubo e fechar a gaveta.")
    equation(doc, r"\mathrm{pick}\ \rightarrow\ \mathrm{move}\ \rightarrow\ \mathrm{open}\ \rightarrow\ \mathrm{place}\ \rightarrow\ \mathrm{close}", "phase_sequence")
    add_body(doc, "O objetivo da anotação é associar cada intervalo temporal a uma descrição textual. Isso permite que uma janela de observações e ações do robô seja treinada com a instrução local correspondente, e não apenas com a instrução global do episódio.")

    heading(doc, "2 Dados e protocolo de avaliação")
    add_body(doc, "O dataset foi convertido da estrutura LeRobot v2 para v3, necessária para o ambiente LeRobot Annotate. Três episódios possuíam somente um frame e não puderam ser segmentados cinematicamente, pois o cálculo de velocidade requer pelo menos dois instantes. O conjunto final contém 261 episódios utilizáveis.")
    add_table(doc, ["Componente", "Configuração"], [
        ["Dataset utilizável", "261 episódios"],
        ["Referência manual", "45 episódios com cinco subtarefas canônicas e cor"],
        ["Desenvolvimento", "15 episódios usados para ajustar decisões do método"],
        ["Teste", "30 episódios reservados para avaliação final"],
        ["Câmera principal", "observation.images.cam_low"],
    ], [2.35, 4.65])
    add_body(doc, "A fase de desenvolvimento funciona como validação. Nela podem ser ajustados parâmetros como suavização, duração mínima dos segmentos, estratégia de alinhamento e escolhas de fallback. Os 30 episódios de teste não devem orientar esses ajustes; eles permanecem reservados para produzir uma estimativa mais honesta do desempenho final em episódios não usados na calibração.")

    heading(doc, "3 Modelo visual linguístico e configuração")
    add_body(doc, "O servidor local foi verificado antes dos experimentos. O modelo efetivamente disponibilizado foi Qwen Qwen2.5 VL 7B Instruct, acessado por endpoint compatível com OpenAI em http://127.0.0.1:8000/v1. Trata-se de um modelo visual-linguístico que recebe imagens e instruções textuais e produz texto como saída.")
    add_body(doc, "O LeRobot Annotate envia uma seleção de frames do episódio ao Qwen. Nos testes abertos, foram usados aproximadamente um frame por segundo, até 40 frames por prompt e até oito subtarefas. A câmera inferior foi escolhida por oferecer uma visão mais útil da garra, do cubo e da interação com a gaveta.")

    heading(doc, "4 Métodos explorados")
    add_table(doc, ["Método", "Linguagem", "Fronteiras temporais", "Papel do Qwen"], [
        ["Qwen puro fechado", "Rótulos fixos com ação e cor", "Estimadas pelo Qwen", "Prediz semântica e tempo"],
        ["Qwen puro aberto", "Frases livres", "Estimadas pelo Qwen", "Descreve o vídeo em linguagem natural"],
        ["Cinemático", "Templates canônicos", "Estados e velocidades do robô", "Não utilizado"],
        ["Híbrido aberto", "Frase do Qwen ou fallback", "Segmentador cinemático", "Semântica livre, não tempo"],
    ], [1.55, 1.75, 1.85, 1.85])

    heading(doc, "5 Qwen puro em vocabulário aberto")
    add_body(doc, "A primeira abordagem usou o prompt padrão de planejamento de subtarefas do LeRobot Annotate, sem impor uma lista fixa de ações. O objetivo era verificar se o Qwen conseguiria gerar descrições úteis para o condicionamento linguístico. Ele produziu frases como pick up the red cube, place the red cube in the red drawer e close the red drawer.")
    add_body(doc, "A principal qualidade observada foi semântica. Para as subtarefas que puderam ser alinhadas com a referência manual, a similaridade média de cosseno entre embeddings SigLIP foi 0,959. Isso indica que, quando o Qwen emite uma descrição, ela costuma ter significado próximo da anotação humana, mesmo sem usar as mesmas palavras.")
    add_body(doc, "Entretanto, o método aberto apresentou cobertura semântica de aproximadamente 0,500. Ele frequentemente omitiu as fases intermediárias mover e abrir, resumindo o episódio em pegar, colocar e fechar. Além disso, suas estimativas de tempo foram ruins: tIoU médio de 0,330 e Boundary MAE de 9,08 s.")

    heading(doc, "6 Qwen puro em vocabulário fechado")
    add_body(doc, "Também foi testado um prompt específico para a tarefa, que restringia a saída a cinco fases e incorporava a cor do cubo e da gaveta. Esse formato melhora a consistência nominal dos rótulos, por exemplo pick_red_cube e place_red_cube_in_red_drawer.")
    add_body(doc, "O F1 de ações sensível à cor foi 0,961, mostrando que o modelo normalmente escolheu os rótulos corretos quando a resposta foi restrita. Porém, a restrição de vocabulário não resolveu o problema temporal: o tIoU médio foi 0,220 e o Boundary MAE foi 7,25 s. Portanto, a dificuldade principal do Qwen não era somente escolher o nome da ação, mas localizar com precisão as transições no vídeo.")

    heading(doc, "7 Segmentação cinemática")
    add_body(doc, "A segmentação cinemática usa sinais temporais diretamente ligados à execução do robô, como estados articulares, velocidades, evolução do efetuador final e abertura ou fechamento da garra. Esses sinais são mais densos e mais próximos dos eventos físicos da demonstração do que uma descrição produzida a partir de frames amostrados.")
    add_body(doc, "Mudanças de comportamento cinemático ajudam a localizar as fases: aproximação e fechamento da garra em pick, deslocamento após agarrar o cubo em move, interação com a gaveta em open, reposicionamento e soltura em place e nova interação com a gaveta em close. O método também impõe a ordem das cinco fases, evitando sequências fisicamente incoerentes.")
    add_body(doc, "No conjunto de teste, a segmentação cinemática alcançou tIoU médio de 0,638 e Boundary MAE de 1,77 s. Ela é claramente mais adequada para responder quando cada subtarefa acontece. Porém, a cinemática não produz linguagem aberta nem entende visualmente cores e objetos; sua saída isolada ficaria limitada a templates estruturais.")
    add_figure(doc, "outputs/figures/method_comparison_matplotlib/01_temporal_comparison.png", "Figura 1. Comparação temporal no conjunto de teste. As variantes híbridas usam as mesmas fronteiras do método cinemático.")

    heading(doc, "8 Métricas de avaliação")
    heading(doc, "8.1 Métricas temporais", level=2)
    add_body(doc, "A tIoU mede quanto o intervalo temporal previsto se sobrepõe ao intervalo manual da mesma fase. Quanto mais próximo de um, melhor a coincidência temporal.")
    equation(doc, r"\mathrm{tIoU}=\frac{|I_{\mathrm{previsto}}\cap I_{\mathrm{manual}}|}{|I_{\mathrm{previsto}}\cup I_{\mathrm{manual}}|}", "tiou")
    add_body(doc, "O Boundary MAE mede, em segundos, o erro médio dos instantes de transição entre fases. Quanto mais próximo de zero, melhor.")
    equation(doc, r"\mathrm{Boundary\ MAE}=\frac{1}{B}\sum_i\left|b_{\mathrm{previsto},i}-b_{\mathrm{manual},i}\right|", "boundary_mae")
    heading(doc, "8.2 Métricas de linguagem", level=2)
    add_body(doc, "No vocabulário fechado, o F1 combina precisão e revocação para verificar a classificação das ações. Essa métrica é apropriada quando os rótulos pertencem a uma lista fixa. Já no vocabulário aberto, igualdade literal de texto não é adequada: pick up the red cube e grab the red block podem expressar a mesma intenção.")
    equation(doc, r"F1=2\frac{\mathrm{Precisao}\cdot\mathrm{Revocacao}}{\mathrm{Precisao}+\mathrm{Revocacao}}", "f1")
    add_body(doc, "Por isso, as frases abertas são comparadas por similaridade de cosseno entre embeddings textuais SigLIP. A cobertura semântica mede a fração de fases manuais para as quais o Qwen produziu uma frase alinhada. O cosseno conservador atribui zero às fases omitidas, combinando qualidade das frases e cobertura.")
    equation(doc, r"\cos(e_{\mathrm{Qwen}},e_{\mathrm{manual}})=\frac{e_{\mathrm{Qwen}}\cdot e_{\mathrm{manual}}}{\Vert e_{\mathrm{Qwen}}\Vert\Vert e_{\mathrm{manual}}\Vert}", "cosine")

    heading(doc, "9 Por que o método híbrido é necessário")
    add_body(doc, "Os resultados mostram que Qwen e cinemática têm capacidades complementares. O Qwen é útil para produzir linguagem natural que menciona objetos, cores e relações. A cinemática é útil para encontrar limites temporais estáveis e garantir que todas as fases estejam presentes. Nenhuma das duas fontes, isoladamente, satisfaz simultaneamente os requisitos de cobertura, precisão temporal e riqueza linguística.")
    add_body(doc, "No híbrido em vocabulário aberto, o Qwen não é usado como fonte final de timestamps. Seus intervalos temporais são descartados porque os resultados mostraram erros elevados. A cinemática define os cinco segmentos finais. O Qwen é usado como fonte de texto: suas frases são convertidas em embeddings SigLIP e alinhadas, respeitando a ordem, às fases canônicas.")
    add_body(doc, "Quando uma frase do Qwen é semanticamente compatível com uma fase, ela é preservada. Por exemplo, pick up the green cube pode ser mantida para pick. Quando o Qwen omite uma fase, o método usa um fallback construído a partir da fase cinemática e dos metadados da tarefa, como move the green cube to the green drawer ou open the green drawer.")
    equation(doc, r"\mathrm{Anotacao\ hibrida}=\mathrm{fronteira\ cinematica}+\mathrm{texto\ Qwen}+\mathrm{fallback}", "hybrid")
    add_body(doc, "A importância do híbrido é que ele não mascara as limitações do Qwen. O manifesto registra a origem de cada texto. Assim, é possível saber quais segmentos receberam linguagem realmente emitida pelo modelo e quais foram completados por fallback. O procedimento aumenta a cobertura para todos os segmentos sem atribuir falsamente ao Qwen descrições que ele não produziu.")
    add_figure(doc, "outputs/figures/method_comparison_matplotlib/05_qualitative_all_methods_timelines.png", "Figura 2. Comparação qualitativa entre referência manual, Qwen fechado e aberto, cinemática e híbridos. As cores representam as cinco fases; as linhas abertas mostram a linguagem preservada.")

    heading(doc, "10 Resultados consolidados")
    add_table(doc, ["Método", "tIoU médio", "Boundary MAE", "Resultado de linguagem"], [
        ["Qwen fechado", f"{qwen_closed['mean_tIoU']:.3f}", f"{qwen_closed['boundary_MAE_seconds']:.2f} s", f"F1 com cor: {qwen_closed['colour_aware_action_F1']:.3f}"],
        ["Qwen aberto", f"{qwen_open['mean_tIoU']:.3f}", f"{qwen_open['boundary_MAE_seconds']:.2f} s", f"Cosseno: {semantic['mean_matched_cosine']:.3f}; cobertura: {semantic['semantic_coverage']:.3f}"],
        ["Cinemático", f"{kinematic['mean_tIoU']:.3f}", f"{kinematic['boundary_MAE_seconds']:.2f} s", "Templates canônicos"],
        ["Híbrido aberto", f"{hybrid_open['mean_tIoU']:.3f}", f"{hybrid_open['boundary_MAE_seconds']:.2f} s", f"Cosseno conservador: {semantic['mean_cosine_with_unmatched_as_zero']:.3f}"],
        ["Híbrido fechado", f"{hybrid_closed['mean_tIoU']:.3f}", f"{hybrid_closed['boundary_MAE_seconds']:.2f} s", f"F1 com cor: {hybrid_closed['colour_aware_action_F1']:.3f}"],
    ], [1.4, 1.25, 1.35, 3.0])
    add_body(doc, "As métricas do híbrido aberto são temporalmente iguais às da cinemática porque ambos usam exatamente as mesmas fronteiras. Isso não é uma vantagem artificial atribuída ao Qwen; é uma consequência explícita da divisão de responsabilidades do método. A contribuição do Qwen no híbrido deve ser analisada por cobertura, similaridade semântica e, na etapa seguinte, pelo efeito no treinamento e na generalização do ACT.")

    heading(doc, "11 Manifesto linguístico")
    add_body(doc, "O manifesto linguístico é o artefato que conecta a rotulação automática ao treinamento. Ele é armazenado em JSONL, com um registro por episódio. Cada registro contém a instrução global, as cores, cinco segmentos temporais, a fase canônica, o texto ativo, a fonte do texto e, quando aplicável, a similaridade semântica do alinhamento.")
    add_table(doc, ["Propriedade", "Valor"], [
        ["Episódios", str(len(manifest))],
        ["Segmentos totais", str(qwen_segments + fallback_segments)],
        ["Segmentos com texto Qwen", f"{qwen_segments} ({qwen_segments / (qwen_segments + fallback_segments):.1%})"],
        ["Segmentos preenchidos por fallback", f"{fallback_segments} ({fallback_segments / (qwen_segments + fallback_segments):.1%})"],
        ["Textos de subtarefa únicos", "62"],
    ], [3.2, 3.8])
    add_body(doc, "A proveniência permite comparações futuras entre treinamento com linguagem aberta do Qwen, linguagem canônica de fallback e a combinação das duas. Também permite auditar automaticamente quais fases dependem mais de fallback e como a qualidade de linguagem se relaciona com o desempenho da política.")

    heading(doc, "12 Como o manifesto entra no ACT")
    add_body(doc, "No treinamento, o dataloader recebe uma janela temporal de imagens, estados e ações. A partir do episódio e do frame inicial da janela, ele consulta o manifesto para recuperar a subtarefa ativa. O texto dessa subtarefa é passado pelo encoder textual do SigLIP, produzindo um embedding que será usado como condicionamento adicional do ACT.")
    equation(doc, r"(o_t,s_t,e_{\mathrm{SigLIP}}(\mathrm{subtarefa}_t))\ \longrightarrow\ \hat{a}_{t:t+H}", "act_conditioning")
    add_body(doc, "Na implementação inicial, é recomendado descartar janelas que atravessem uma fronteira de subtarefa. Assim, cada chunk de ações estará associado a uma única instrução local. A instrução global pode continuar como contexto adicional, enquanto a subtarefa oferece um sinal linguístico alinhado ao estágio atual da execução.")

    heading(doc, "13 Próximos passos")
    add_bullets(doc, [
        "Extrair e armazenar embeddings SigLIP para os textos únicos do manifesto.",
        "Adaptar o dataloader do ACT para recuperar a subtarefa ativa por frame ou chunk.",
        "Treinar baselines sem linguagem, com linguagem canônica e com o manifesto híbrido aberto.",
        "Avaliar se o condicionamento textual melhora desempenho e generalização composicional para novas combinações de cor e comandos.",
        "Reportar separadamente desempenho temporal, cobertura semântica, proveniência da linguagem e desempenho final do ACT.",
    ])

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("ACT com SigLIP  |  Relatório da etapa de rotulação de subtarefas")
    footer.runs[0].font.size = Pt(8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
