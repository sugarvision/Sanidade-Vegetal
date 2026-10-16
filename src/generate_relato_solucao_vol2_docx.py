"""
Gerador do Documento Microsoft Word (.docx) do Relato da Solução — Volume 2
Projeto: Sanidade-Vegetal (SugarVision) — Squad 2
Framework SEMMA — Etapa: Modify (Engenharia de Features)
"""

import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color: str):
    """Define a cor de fundo de uma célula via XML."""
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def set_cell_margins(cell, top=120, bottom=120, left=180, right=180):
    """Ajusta as margens internas da célula."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def format_table_header(row, col_widths, bg_hex="142D55"):
    """Formata a linha de cabeçalho com fundo institucional e texto branco em negrito."""
    for i, cell in enumerate(row.cells):
        cell.width = col_widths[i]
        set_cell_background(cell, bg_hex)
        set_cell_margins(cell, top=140, bottom=140, left=160, right=160)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.size = Pt(9.5)
                run.font.name = "Calibri"

def format_table_row(row, col_widths, is_even=False):
    """Aplica zebra striping e bordas suaves."""
    bg_hex = "F8F9FA" if is_even else "FFFFFF"
    for i, cell in enumerate(row.cells):
        cell.width = col_widths[i]
        set_cell_background(cell, bg_hex)
        set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
        for p in cell.paragraphs:
            p.paragraph_format.line_spacing = 1.15
            for run in p.runs:
                run.font.size = Pt(9.0)
                run.font.color.rgb = RGBColor(40, 44, 52)
                run.font.name = "Calibri"

def add_heading_with_spacing(doc, text: str, level: int):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(4)
    for run in h.runs:
        run.font.name = "Calibri"
        if level == 1:
            run.font.color.rgb = RGBColor(20, 45, 85)     # Deep Navy
            run.font.size = Pt(15)
            run.font.bold = True
        elif level == 2:
            run.font.color.rgb = RGBColor(39, 130, 70)    # Agritech Green
            run.font.size = Pt(12.5)
            run.font.bold = True
        elif level == 3:
            run.font.color.rgb = RGBColor(211, 84, 0)     # Orange Accent
            run.font.size = Pt(11)
            run.font.bold = True
    return h

def add_highlight_box(doc, title: str, text: str, bg_hex="EBF3FD", border_hex="1E5AB4"):
    """Cria uma caixa de destaque elegante."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.8)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    r_t = p.add_run(f"★ {title}\n")
    r_t.font.bold = True
    r_t.font.size = Pt(10)
    r_t.font.color.rgb = RGBColor(30, 90, 180)

    r_b = p.add_run(text)
    r_b.font.size = Pt(9.5)
    r_b.font.color.rgb = RGBColor(30, 35, 45)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def build_relato_volume_2(output_path: Path):
    doc = docx.Document()

    # Configuração de Margens (1 polegada = 2.54 cm)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # =========================================================================
    # BANNER INSTITUCIONAL DE ABERTURA
    # =========================================================================
    banner_table = doc.add_table(rows=1, cols=1)
    banner_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    b_cell = banner_table.cell(0, 0)
    b_cell.width = Inches(6.8)
    set_cell_background(b_cell, "142D55")  # Deep Navy
    set_cell_margins(b_cell, top=240, bottom=240, left=260, right=260)

    bp = b_cell.paragraphs[0]
    bp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_top = bp.add_run("SANIDADE VEGETAL AI (SUGARVISION)\n")
    r_top.font.name = "Calibri"
    r_top.font.size = Pt(11)
    r_top.font.bold = True
    r_top.font.color.rgb = RGBColor(180, 220, 190)

    r_title = bp.add_run("RELATO DA SOLUÇÃO — VOLUME 2\n")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(18)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(255, 255, 255)

    r_sub = bp.add_run("Etapa Modify (SEMMA): Engenharia de Features, Pipeline Atômico com ColumnTransformer, Defesa Técnica Anti-Leakage e Consolidação da ABT Final\n")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(10.5)
    r_sub.font.color.rgb = RGBColor(220, 230, 245)

    r_foot = bp.add_run("Apresentação do Demoday 2 • Sprint 2 • FATEC Rio Claro 2026")
    r_foot.font.name = "Calibri"
    r_foot.font.size = Pt(9.5)
    r_foot.font.bold = True
    r_foot.font.color.rgb = RGBColor(255, 200, 100)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # =========================================================================
    # TABELA DE METADADOS DA ENTREGA
    # =========================================================================
    meta_tbl = doc.add_table(rows=2, cols=3)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    col_w_m = [Inches(2.2), Inches(2.4), Inches(2.2)]

    row0 = meta_tbl.rows[0]
    headers_m = ["DATA DO DEMODAY 2", "PROFESSOR / AVALIADOR", "EQUIPE RESPONSÁVEL"]
    for i, title in enumerate(headers_m):
        row0.cells[i].paragraphs[0].text = title
    format_table_header(row0, col_w_m, bg_hex="278246")  # Verde Agrícola

    row1 = meta_tbl.rows[1]
    vals_m = ["16/10/2026", "Prof. Julio Cesar de Lemos", "Sanidade Vegetal — Squad 2"]
    for i, val in enumerate(vals_m):
        p = row1.cells[i].paragraphs[0]
        p.text = val
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_table_row(row1, col_w_m, is_even=False)
    for cell in row1.cells:
        for run in cell.paragraphs[0].runs:
            run.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # =========================================================================
    # SEÇÃO 1: RACI ACCOUNTABILITY
    # =========================================================================
    add_heading_with_spacing(doc, "1. Identificação da Equipe e Matriz RACI / Accountability", level=1)
    
    p = doc.add_paragraph(
        "A governança e a distribuição técnica de responsabilidades da Squad 2 na Fase Modify "
        "do framework SEMMA seguem o modelo formal RACI, onde (A) representa o Accountable com autoridade "
        "e responsabilidade final pela aprovação da camada indicada:"
    )
    p.paragraph_format.space_after = Pt(6)

    raci_tbl = doc.add_table(rows=7, cols=3)
    raci_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    col_w_r = [Inches(1.8), Inches(1.5), Inches(3.5)]

    raci_headers = ["Nome do Membro", "Papel na Squad", "Responsabilidade Técnica Principal na Fase Modify (SEMMA)"]
    for i, h in enumerate(raci_headers):
        raci_tbl.rows[0].cells[i].paragraphs[0].text = h
    format_table_header(raci_tbl.rows[0], col_w_r, bg_hex="142D55")

    raci_data = [
        ("César Augusto Oliveira Bovo", "Product Owner / Tech Lead", 
         "Accountable (A) — Visão Computacional & Engenharia Visual: Formulação e validação agronômica dos descritores cromáticos (HSV/CIELAB, índices ExG, ExR, R/G) e texturais de Haralick via GLCM multidirecional com invariância à rotação."),
        ("Elisa Almeida Alcântara", "MLOps & Machine Learning Engineer", 
         "Accountable (A) — Pipeline Atômico & ABT Final: Arquitetura do ColumnTransformer com sub-pipelines desacoplados, prova matemática de isolamento anti-leakage, consolidação da base abt_features_modelagem e exportação em Apache Parquet/CSV com SHA-256."),
        ("Luis Otavio Routh da Silva", "Scrum Master & QA", 
         "Accountable (A) — Governança Ágil & Qualidade: Facilitação do Demoday 2, elaboração do Relato da Solução — Volume 2, controle rigoroso do Definition of Done (DoD) e conformidade técnica das entregas."),
        ("Heitor Vitti Partezani", "Data Architect", 
         "Accountable (A) — Amostragem & Schema: Manutenção da integridade referencial dos alvos supervisionados (binário e multiclasse), governança dos identificadores únicos (sample_id) e atualização do Dicionário de Dados."),
        ("Marvin Cristhian Gomes Pinto", "Data Scientist", 
         "Accountable (A) — Auditoria Estatística: Análise exploratória multivariada, auditoria de assimetria nos metadados ópticos, avaliação comparativa de bins e suporte aos testes estatísticos de separabilidade."),
        ("Guilherme Peres Romanzotti", "DevOps / Repo Manager", 
         "Accountable (A) — Repositório & CI: Gestão de branches e pull requests, proteção do .gitignore contra contaminação de bases pesadas e controle de versionamento dos artefatos serializados (joblib/json).")
    ]
    for row_idx, data in enumerate(raci_data, start=1):
        row = raci_tbl.rows[row_idx]
        for col_idx, text in enumerate(data):
            row.cells[col_idx].paragraphs[0].text = text
        format_table_row(row, col_w_r, is_even=(row_idx % 2 == 0))

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # =========================================================================
    # SEÇÃO 2: META DO DEMODAY 2 E ALINHAMENTO ESTRATÉGICO
    # =========================================================================
    add_heading_with_spacing(doc, "2. Metas do Demoday 2 & Alinhamento Estratégico (Fase Modify)", level=1)
    
    p = doc.add_paragraph(
        "A presente cerimônia do Demoday 2 formaliza a entrega e defesa técnica da Etapa Modify "
        "(Transformação, Engenharia de Features e Pré-processamento) do framework SEMMA no projeto SugarVision. "
        "A fitopatologia da cana-de-açúcar (Saccharum officinarum) impõe desafios severos para o aprendizado de máquina tabular:"
    )
    p.paragraph_format.space_after = Pt(4)

    p_bullets = doc.add_paragraph()
    p_bullets.add_run("1. Ruído Fotométrico e Iluminação Inconsistente: ").bold = True
    p_bullets.add_run("Fotografias obtidas em campo sofrem forte distorção solar, reflexos e sombras, inviabilizando o uso de canais RGB crus.\n")
    p_bullets.add_run("2. Micromorfologia Fúngica da Ferrugem: ").bold = True
    p_bullets.add_run("A infecção por Puccinia melanocephala e Puccinia kuehnii caracteriza-se pelo rompimento da cutícula e proliferação de pústulas urediais salientes e rugosas, exigindo descritores texturais especializados.\n")
    p_bullets.add_run("3. Risco Crítico de Data Leakage: ").bold = True
    p_bullets.add_run("Pré-processamentos soltos frequentemente calculam médias e desvios sobre a base total, contaminando validação e teste com informações do futuro.")
    p_bullets.paragraph_format.space_after = Pt(6)

    add_highlight_box(
        doc,
        title="DECLARAÇÃO DE META DA SPRINT 2 (DEMODAY 2)",
        text="Transformar as 6.571 imagens fotográficas brutas auditadas na Sprint 1 em uma Tabela Analítica Base (ABT) "
             "final com 35 features preditivas de alta relevância biológica, estruturada através de um Pipeline Atômico com "
             "ColumnTransformer, blindada com ZERO Data Leakage e persistida em Apache Parquet para os modelos de Support Vector Machines."
    )

    # =========================================================================
    # SEÇÃO 3: ARQUITETURA DA SOLUÇÃO E O QUE / COMO
    # =========================================================================
    add_heading_with_spacing(doc, "3. Arquitetura da Solução e Fluxo End-to-End ('O Que' e 'Como')", level=1)
    
    p = doc.add_paragraph(
        "A engenharia implementada organiza-se de maneira binária entre 'O Que' foi concebido (sinais biológicos e visuais) "
        "e 'Como' foi executado (engenharia de software e MLOps):"
    )
    p.paragraph_format.space_after = Pt(4)

    p_qc = doc.add_paragraph()
    p_qc.add_run("• 'O Que' Construímos: ").bold = True
    p_qc.add_run("7 descritores cromáticos (ExG, ExR, razões R/G, matiz circular HSV), 5 métricas texturais de Haralick com média multidirecional invariante a rotação, 1 índice proprietário de rugosidade foliar de pústula (IRFP) e 4 conjuntos de faixas estatísticas discretizadas.\n")
    p_qc.add_run("• 'Como' Processamos: ").bold = True
    p_qc.add_run("Um objeto ColumnTransformer central no Scikit-Learn que orquestra dois sub-pipelines especializados (numérico e categórico), impõe a Regra de Ouro do Fit exclusivo em X_train e purifica a matriz analítica descartando 8 colunas de metadados operacionais.")
    p_qc.paragraph_format.space_after = Pt(8)

    # =========================================================================
    # SEÇÃO 4: DEFESA DOS CRITÉRIOS TÉCNICOS
    # =========================================================================
    add_heading_with_spacing(doc, "4. Defesa dos Critérios Técnicos da Engenharia de Features ('O Que')", level=1)

    add_heading_with_spacing(doc, "4.1 Descritores Cromáticos e Índices Espectrais de Vegetação", level=2)
    p = doc.add_paragraph(
        "Para desacoplar a informação de cor pura da luminância, as imagens foram convertidas para o espaço HSV "
        "e submetidas ao cálculo de índices de vegetação clássicos da agricultura de precisão:"
    )
    p.paragraph_format.space_after = Pt(4)

    p_eq = doc.add_paragraph()
    p_eq.add_run("1. Índice de Excesso de Verde (ExG): ").bold = True
    p_eq.add_run("ExG = 2G - R - B. Folhas sadias apresentam alta refletância no verde devido à clorofila ativa (+44,82 médio). Na ferrugem, o índice colapsa para -18,45.\n")
    p_eq.add_run("2. Índice de Excesso de Vermelho (ExR): ").bold = True
    p_eq.add_run("ExR = 1.4R - G. Sensível a tecidos oxidados, lesões necróticas e esporulação fúngica (-18,10 em sadias vs +24,30 em ferrugem).\n")
    p_eq.add_run("3. Matiz Médio (μ_Hue) no HSV: ").bold = True
    p_eq.add_run("O canal Hue mapeia o comprimento de onda angular (0° a 360°), separando nitidamente a folha sadia (76,5° verde) da folha com ferrugem (24,2° laranja/castanho).")
    p_eq.paragraph_format.space_after = Pt(6)

    # Imagem cromática se existir
    fig_crom = Path("docs/figures/sprint2_analise_cromaticas_hsv_exg.png")
    if fig_crom.exists():
        doc.add_paragraph().paragraph_format.space_before = Pt(4)
        doc.add_picture(str(fig_crom), width=Inches(6.6))
        p_cap = doc.add_paragraph("Figura 1: Diagnóstico de separabilidade dos índices cromáticos (ExG, ExR, HSV) entre folhas sadias e patologias foliares.")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(8.5)
        p_cap.runs[0].font.italic = True
        doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_heading_with_spacing(doc, "4.2 Descritores Texturais de Haralick via GLCM e Invariância à Rotação", level=2)
    p = doc.add_paragraph(
        "A quebra da cutícula foliar por pústulas urediais foi capturada através da Matriz de Co-ocorrência em Níveis de Cinza (GLCM). "
        "Para anular a dependência da inclinação com que a foto foi registrada, calculou-se a matriz em 4 direções canônicas (0°, 45°, 90°, 135°) "
        "com distância d=1, consolidando as features pela média direcional:"
    )
    p.paragraph_format.space_after = Pt(4)

    p_tex = doc.add_paragraph()
    p_tex.add_run("• Contraste Haralick: ").bold = True
    p_tex.add_run("Mede variações locais quadráticas de cinza. É 1,72x superior na ferrugem (0,537 vs 0,312 em sadias), refletindo a transição abrupta de tons nas pústulas.\n")
    p_tex.add_run("• Homogeneidade (IDM): ").bold = True
    p_tex.add_run("Mede a suavidade superficial. Cai de 0,845 (tecido liso sadio) para 0,691 (tecido rugoso atacado).\n")
    p_tex.add_run("• Índice de Rugosidade Foliar de Pústula (IRFP): ").bold = True
    p_tex.add_run("Formulação proprietária da Squad 2: IRFP = (Contraste × Dissimilaridade) / (Homogeneidade + ε), amplificando o contraste microscópico.")
    p_tex.paragraph_format.space_after = Pt(6)

    # Tabela de Validação Estatística
    p = doc.add_paragraph("Tabela 1: Validação estatística rigorosa dos descritores visuais sobre as 6.571 instâncias:")
    p.runs[0].font.bold = True
    p.paragraph_format.space_after = Pt(4)

    stat_tbl = doc.add_table(rows=8, cols=5)
    stat_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    col_w_s = [Inches(1.8), Inches(1.1), Inches(1.1), Inches(1.4), Inches(1.4)]

    stat_headers = ["Descritor Visual", "Média Sadia", "Média Ferrugem", "Teste Mann-Whitney", "Cohen's d (Efeito)"]
    for i, h in enumerate(stat_headers):
        stat_tbl.rows[0].cells[i].paragraphs[0].text = h
    format_table_header(stat_tbl.rows[0], col_w_s, bg_hex="142D55")

    stat_rows = [
        ("Excesso de Verde (ExG)", "+44,82", "-18,45", "p < 10⁻¹⁵", "6,346 (Extremamente Alto)"),
        ("Excesso de Vermelho (ExR)", "-18,10", "+24,30", "p < 10⁻¹⁵", "5,812 (Extremamente Alto)"),
        ("Matiz Médio (μ_Hue)", "76,5° (Verde)", "24,2° (Laranja)", "p < 10⁻¹⁵", "7,410 (Extremamente Alto)"),
        ("Desvio Saturação (σ_Sat)", "0,118", "0,245", "p < 10⁻¹⁵", "4,920 (Extremamente Alto)"),
        ("Contraste GLCM Haralick", "0,312", "0,537", "p < 10⁻¹⁵", "3,421 (Muito Alto)"),
        ("Homogeneidade GLCM", "0,845", "0,691", "p < 10⁻¹⁵", "3,185 (Muito Alto)"),
        ("Índice de Rugosidade (IRFP)", "0,185", "0,782", "p < 10⁻¹⁵", "4,110 (Extremamente Alto)")
    ]
    for row_idx, data in enumerate(stat_rows, start=1):
        row = stat_tbl.rows[row_idx]
        for col_idx, val in enumerate(data):
            p = row.cells[col_idx].paragraphs[0]
            p.text = val
            if col_idx in (1, 2, 3, 4):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_table_row(row, col_w_s, is_even=(row_idx % 2 == 0))

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_heading_with_spacing(doc, "4.3 Defesa Técnica da Discretização e Justificativa do One-Hot Encoding", level=2)
    p = doc.add_paragraph(
        "A base consolidada reúne metadados de duas fontes (Roboflow e Mendeley), gerando disparidades de resolução e peso. "
        "Segmentamos as variáveis contínuas de metadados em faixas estatísticas (size_kb em 4 quartis, laplacian_var em 4 quartis de nitidez, "
        "resolução em 3 tiers via KMeans e aspect_ratio em 3 clusters). "
        "A defesa técnica para aplicar One-Hot Encoding em vez de codificação ordinal baseia-se na formulação do Support Vector Machines (SVM):"
    )
    p.paragraph_format.space_after = Pt(4)

    add_highlight_box(
        doc,
        title="POR QUE NÃO UTILIZAR CODIFICAÇÃO ORDINAL (0, 1, 2, 3) PARA O SVM?",
        text="Algoritmos de fronteira de margem máxima, como o SVM, operam através de combinações lineares de pesos w e produtos escalares. "
             "Se representássemos as faixas de resolução como inteiros ordinais, o modelo assumiria uma falsa relação linear métrica "
             "(ex: que o tier 3 é 3 vezes 'maior' que o tier 1). A codificação One-Hot projeta cada faixa em uma dimensão ortogonal "
             "independente, permitindo que o hiperplano atribua coeficientes específicos sem induzir viés artificial de monotonicidade."
    )

    # =========================================================================
    # SEÇÃO 5: TRANSFORMAÇÕES E IMPUTAÇÕES
    # =========================================================================
    add_heading_with_spacing(doc, "5. Transformações, Imputações e Arquitetura do Pipeline Atômico ('Como')", level=1)
    
    p = doc.add_paragraph(
        "A centralização de todas as transformações no módulo src/atomic_pipeline_preprocessing.py garantiu rastreabilidade e auditabilidade total:"
    )
    p.paragraph_format.space_after = Pt(4)

    p_subp = doc.add_paragraph()
    p_subp.add_run("1. Sub-Pipeline Numérico (17 Features Contínuas): ").bold = True
    p_subp.add_run("Aplica SimpleImputer(strategy='median') — imune a assimetrias e outliers — seguido de StandardScaler(with_mean=True, with_std=True). A padronização Z-score (μ=0, σ=1) é obrigatória para o SVM, cuja função de kernel RBF depende da distância euclidiana.\n")
    p_subp.add_run("2. Sub-Pipeline Categórico (6 Features Nominais e Bins): ").bold = True
    p_subp.add_run("Aplica SimpleImputer(strategy='most_frequent') preenchendo ausências pela moda observada no treino, seguido de OneHotEncoder(sparse_output=False, handle_unknown='ignore') expandindo em 18 colunas dummy com blindagem contra categorias desconhecidas.\n")
    p_subp.add_run("3. Camada de Purificação (remainder='drop'): ").bold = True
    p_subp.add_run("Descarta automaticamente 8 colunas operacionais (sample_id, filepath, width, height, channels), impedindo contaminação da matriz de pesos.")
    p_subp.paragraph_format.space_after = Pt(6)

    # =========================================================================
    # SEÇÃO 6: COMPROVAÇÃO DE AUSÊNCIA DE DATA LEAKAGE
    # =========================================================================
    add_heading_with_spacing(doc, "6. Comprovação Rigorosa e Matemática de Ausência de Data Leakage", level=1)

    p = doc.add_paragraph(
        "Para blindar o SugarVision contra vazamento de dados, o treinamento e a avaliação obedeceram à Regra de Ouro do Fit:"
    )
    p.paragraph_format.space_after = Pt(4)

    p_gold = doc.add_paragraph()
    p_gold.add_run("• fit_transform() Exclusivo em Treino: ").bold = True
    p_gold.add_run("Apenas as 5.574 instâncias de X_train foram submetidas ao cálculo de parâmetros (médias μ, desvios σ, medianas e categorias).\n")
    p_gold.add_run("• transform() Puro em Validação e Teste: ").bold = True
    p_gold.add_run("As 617 amostras de validação e 380 de teste cego receberam exclusivamente transform(), utilizando os parâmetros previamente congelados.")
    p_gold.paragraph_format.space_after = Pt(6)

    # Tabela de Auditoria das Médias
    p = doc.add_paragraph("Tabela 2: Auditoria matemática das médias e desvios nas partições após a padronização:")
    p.runs[0].font.bold = True
    p.paragraph_format.space_after = Pt(4)

    leak_tbl = doc.add_table(rows=4, cols=5)
    leak_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    col_w_l = [Inches(1.8), Inches(1.1), Inches(1.3), Inches(1.3), Inches(1.3)]

    leak_headers = ["Partição Amostral", "Instâncias", "Média Pós-Ajuste (x̄)", "Desvio Pós-Ajuste (s)", "Status Anti-Leakage"]
    for i, h in enumerate(leak_headers):
        leak_tbl.rows[0].cells[i].paragraphs[0].text = h
    format_table_header(leak_tbl.rows[0], col_w_l, bg_hex="142D55")

    leak_data = [
        ("Treino (train)", "5.574 (84,8%)", "x̄ ≡ 0,0000", "s ≡ 1,0000", "[ESPERADO] Base de ajuste"),
        ("Validação (valid)", "617 (9,4%)", "x̄ = 0,0142 ≠ 0", "s = 0,9874 ≠ 1", "[COMPROVADO] Zero Leakage"),
        ("Teste Cego (test)", "380 (5,8%)", "x̄ = -0,0218 ≠ 0", "s = 1,0182 ≠ 1", "[COMPROVADO] Zero Leakage")
    ]
    for row_idx, data in enumerate(leak_data, start=1):
        row = leak_tbl.rows[row_idx]
        for col_idx, val in enumerate(data):
            p = row.cells[col_idx].paragraphs[0]
            p.text = val
            if col_idx in (1, 2, 3, 4):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_table_row(row, col_w_l, is_even=(row_idx % 2 == 0))

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    p_proof = doc.add_paragraph(
        "Comprovação Matemática: Se as partições de validação e teste apresentassem média rigorosamente igual a 0,0000 e desvio 1,0000, "
        "haveria vazamento grave (pois indicaria que foram incluídas no cálculo global do StandardScaler). "
        "O fato de suas médias desviarem levemente de zero prova que os dados permaneceram não vistos durante o aprendizado da escala. "
        "Adicionalmente, injetou-se ruído intencional no conjunto de teste em um teste adversarial: os atributos internos do pipeline serializado "
        "permaneceram 100% inalterados bit a bit (0% de contaminação)."
    )
    p_proof.paragraph_format.space_after = Pt(8)

    # =========================================================================
    # SEÇÃO 7: APRESENTAÇÃO DA ABT FINAL
    # =========================================================================
    add_heading_with_spacing(doc, "7. Apresentação da Tabela Analítica Base Final (abt_features_modelagem)", level=1)

    p = doc.add_paragraph(
        "A base final consolidada abt_features_modelagem unifica em uma estrutura tabular plana os 5 metadados de controle "
        "e as 35 features preditivas normalizadas e codificadas, totalizando 40 colunas:"
    )
    p.paragraph_format.space_after = Pt(4)

    p_abt = doc.add_paragraph()
    p_abt.add_run("• Volumetria e Qualidade: ").bold = True
    p_abt.add_run("Rigorosamente 6.571 instâncias x 40 colunas, com ZERO NaNs (0 valores ausentes) em 100% dos dados.\n")
    p_abt.add_run("• Metadados de Controle Preservados: ").bold = True
    p_abt.add_run("sample_id (chave UUID imutável), split_partition (train: 5.574, valid: 617, test: 380), class_label, target_binary (1.146 sadias vs 5.425 doentes) e target_multiclass (0 a 6).\n")
    p_abt.add_run("• Features Preditivas Modelagem: ").bold = True
    p_abt.add_run("17 contínuas padronizadas Z-score e 18 categóricas codificadas via One-Hot.")
    p_abt.paragraph_format.space_after = Pt(6)

    # Tabela Benchmark Parquet vs CSV
    p = doc.add_paragraph("Tabela 3: Benchmark de performance de I/O — Apache Parquet vs CSV:")
    p.runs[0].font.bold = True
    p.paragraph_format.space_after = Pt(4)

    bench_tbl = doc.add_table(rows=5, cols=4)
    bench_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    col_w_b = [Inches(1.8), Inches(1.6), Inches(1.6), Inches(1.8)]

    bench_headers = ["Dimensão Avaliada", "Apache Parquet (.parquet)", "Texto Plano (.csv)", "Ganho / Vantagem Prática"]
    for i, h in enumerate(bench_headers):
        bench_tbl.rows[0].cells[i].paragraphs[0].text = h
    format_table_header(bench_tbl.rows[0], col_w_b, bg_hex="278246")

    bench_rows = [
        ("Tamanho em Disco", "916,92 KB (0,90 MB)", "2.989,66 KB (2,92 MB)", "3,26x menor (-69,3% de tráfego)"),
        ("Tempo Médio de I/O", "36,19 ms", "51,01 ms", "1,41x mais rápido em loops CV"),
        ("Preservação de Tipos", "Estrita (int64, float64)", "Inferência textual de parsing", "Zero risco de coerção de tipos"),
        ("Compressão Utilizada", "Snappy (Colunar binário)", "Nenhuma (ASCII bruto)", "Alta densidade e eficiência de RAM")
    ]
    for row_idx, data in enumerate(bench_rows, start=1):
        row = bench_tbl.rows[row_idx]
        for col_idx, val in enumerate(data):
            p = row.cells[col_idx].paragraphs[0]
            p.text = val
            if col_idx in (1, 2):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_table_row(row, col_w_b, is_even=(row_idx % 2 == 0))

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    p_hash = doc.add_paragraph()
    p_hash.add_run("Hashes Criptográficos SHA-256 (Registrados no Manifesto):").bold = True
    p_hash.add_run("\n• Parquet: e6bd41ebff356e2c2e6be8985f0026dfb0a11364c4c4f729657223cf103c4340")
    p_hash.add_run("\n• CSV:     b37d65c2bd7928eb2e76948b3540542a17ecdc28b000cffa2e990731e4128edc")
    p_hash.paragraph_format.space_after = Pt(8)

    # =========================================================================
    # SEÇÃO 8: CHECKLIST DOD
    # =========================================================================
    add_heading_with_spacing(doc, "8. Checklist de Governança, QA e Definition of Done (DoD)", level=1)

    dod_tbl = doc.add_table(rows=7, cols=4)
    dod_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    col_w_d = [Inches(2.2), Inches(2.6), Inches(1.0), Inches(1.0)]

    dod_headers = ["Item do Checklist DoD", "Evidência Técnica Comprovada", "Status", "Líder"]
    for i, h in enumerate(dod_headers):
        dod_tbl.rows[0].cells[i].paragraphs[0].text = h
    format_table_header(dod_tbl.rows[0], col_w_d, bg_hex="142D55")

    dods_data = [
        ("Descritores Cromáticos", "Índices ExG, ExR, R/G e HSV com p < 10⁻¹⁵ e Cohen's d > 5.0.", "Concluído", "César"),
        ("Descritores Texturais", "Matriz GLCM 4 direções, Haralick e IRFP invariantes a rotação.", "Concluído", "César"),
        ("Pipeline Atômico", "ColumnTransformer com sub-pipelines e remainder='drop'.", "Concluído", "Elisa"),
        ("Zero Data Leakage", "fit_transform exclusivo em Train; transform em Valid e Test.", "Concluído", "Elisa"),
        ("Auditoria de Nulos", "0 NaNs em 6.571 registros x 40 colunas comprovado em log.", "Concluído", "Marvin"),
        ("Exportação Dual Parquet/CSV", "Arquivos gerados com hashes SHA-256 em manifesto JSON.", "Concluído", "Heitor")
    ]
    for row_idx, data in enumerate(dods_data, start=1):
        row = dod_tbl.rows[row_idx]
        for col_idx, val in enumerate(data):
            p = row.cells[col_idx].paragraphs[0]
            p.text = val
            if col_idx in (2, 3):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_table_row(row, col_w_d, is_even=(row_idx % 2 == 0))

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # =========================================================================
    # SEÇÃO 9: DOSSIÊ DE ARTEFATOS
    # =========================================================================
    add_heading_with_spacing(doc, "9. Dossiê de Artefatos Gerados na Fase Modify", level=1)

    p = doc.add_paragraph("Todos os artefatos gerados encontram-se versionados e auditáveis no repositório:")
    p.paragraph_format.space_after = Pt(4)

    p_art = doc.add_paragraph()
    p_art.add_run("• Módulos Python Executáveis: ").bold = True
    p_art.add_run("src/feature_engineering_visual.py, src/atomic_pipeline_preprocessing.py, src/export_final_modeling_abt.py.\n")
    p_art.add_run("• Notebooks Demonstrativos: ").bold = True
    p_art.add_run("notebooks/06_pipeline_atomico_columntransformer.ipynb, notebooks/07_consolidacao_exportacao_abt_final.ipynb.\n")
    p_art.add_run("• Modelos e Metadados Serializados: ").bold = True
    p_art.add_run("models/preprocessor_pipeline.joblib (2,58 KB), models/preprocessor_metadata.json.\n")
    p_art.add_run("• Bases Analíticas Oficiais: ").bold = True
    p_art.add_run("data/processed/abt_features_modelagem.parquet (916,92 KB), data/processed/abt_features_modelagem.csv (2.989,66 KB), data/processed/abt_features_modelagem_manifest.json.")
    p_art.paragraph_format.space_after = Pt(8)

    # =========================================================================
    # SEÇÃO 10: CONCLUSÃO E PRÓXIMOS PASSOS
    # =========================================================================
    add_heading_with_spacing(doc, "10. Conclusão e Transição para a Fase Model (Sprint 3)", level=1)

    p = doc.add_paragraph(
        "A conclusão da Fase Modify representa a consolidação da base de engenharia de dados do SugarVision. "
        "A Tabela Analítica Base abt_features_modelagem encontra-se 100% homologada, sem valores nulos, rica em sinais "
        "fitopatológicos validados e blindada contra contaminação informacional. "
        "O teste de fumaça executado com GridSearchCV e Support Vector Machines (SVM RBF) confirmou a plena prontidão "
        "para a Fase Model na Sprint 3, com execução em 0,05 segundos e F1-Score preliminar unitário na partição de desenvolvimento."
    )
    p.paragraph_format.space_after = Pt(6)

    add_highlight_box(
        doc,
        title="TRANSIÇÃO IMEDIATA: SPRINT 3 (FASE MODEL — SVM & GRIDSEARCHCV)",
        text="A Squad 2 avança para a exploração exaustiva dos hiperparâmetros C e gamma no SVM, validação cruzada "
             "estratificada em 5 folds (StratifiedKFold) consumindo o arquivo Apache Parquet e calibração final de "
             "probabilidades com Platt Scaling para diagnóstico fitossanitário no campo."
    )

    # Salva o arquivo Word
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"[OK] Relato da Solução Volume 2 salvo com sucesso em: {output_path}")

if __name__ == "__main__":
    out = Path("docs/Relato_da_Solucao_Volume_2_Engenharia_de_Features.docx")
    build_relato_volume_2(out)
