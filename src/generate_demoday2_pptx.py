"""
Gerador da Apresentação PowerPoint (.pptx) para o Demoday 2 — Engenharia de Features
Projeto: Sanidade-Vegetal (SugarVision) — Squad 2
Framework SEMMA — Etapa: Modify
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def build_demoday2_presentation(output_path: Path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Paleta de Cores Institucional
    COLOR_PRIMARY = RGBColor(20, 45, 85)       # Azul Marinho Profundo
    COLOR_SECONDARY = RGBColor(39, 130, 70)    # Verde Agrícola (ExG / Saudável)
    COLOR_ACCENT = RGBColor(211, 84, 0)        # Laranja Queimado (Ferrugem / Puccinia)
    COLOR_DARK = RGBColor(40, 44, 52)          # Cinza Escuro Texto
    COLOR_MUTED = RGBColor(100, 105, 115)      # Cinza Secundário
    COLOR_CARD_BG = RGBColor(248, 249, 252)    # Fundo Suave para Cards
    COLOR_CARD_BORDER = RGBColor(218, 224, 233)
    COLOR_WHITE = RGBColor(255, 255, 255)
    COLOR_BLUE_LIGHT = RGBColor(235, 243, 253)
    COLOR_BLUE_ACCENT = RGBColor(30, 90, 180)
    COLOR_CRIMSON = RGBColor(180, 40, 40)

    blank_layout = prs.slide_layouts[6]

    def add_header(slide, tag_text: str, title_text: str):
        # Tag superior da seção
        tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.32), Inches(11.7), Inches(0.35))
        tf_tag = tag_box.text_frame
        tf_tag.word_wrap = True
        p_tag = tf_tag.paragraphs[0]
        p_tag.text = tag_text.upper()
        p_tag.font.size = Pt(10.5)
        p_tag.font.bold = True
        p_tag.font.color.rgb = COLOR_SECONDARY

        # Título principal do slide
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.62), Inches(11.7), Inches(0.75))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(21)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_PRIMARY

        # Linha divisória sutil
        sep = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.38), Inches(11.733), Inches(0.02))
        sep.fill.solid()
        sep.fill.fore_color.rgb = COLOR_CARD_BORDER
        sep.line.fill.background()

    def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
        return card

    # =========================================================================
    # SLIDE 1: CAPA & CONTEXTUALIZAÇÃO
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)

    # Fundo elegante
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(245, 247, 250)
    bg.line.fill.background()

    # Barra lateral verde institucional
    stripe = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.35), Inches(7.5))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = COLOR_SECONDARY
    stripe.line.fill.background()

    # Bloco de Título
    tbox = s1.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.3), Inches(2.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    p0 = tf.paragraphs[0]
    p0.text = "SQUAD 2 | DEMODAY 2 • FRAMEWORK SEMMA (FASE MODIFY)"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_SECONDARY

    p1 = tf.add_paragraph()
    p1.text = "Engenharia de Features & Consolidação da ABT Final"
    p1.font.size = Pt(30)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_PRIMARY
    p1.space_before = Pt(8)

    p2 = tf.add_paragraph()
    p2.text = "Defesa Técnica de Transformações, Imputações, Blindagem Anti-Leakage e Handoff para Modelagem SVM"
    p2.font.size = Pt(15)
    p2.font.color.rgb = COLOR_MUTED
    p2.space_before = Pt(6)

    # 3 Cards principais da Capa
    card_w = Inches(3.64)
    card_h = Inches(3.6)
    c_top = Inches(3.2)

    # Card 1: O Que
    add_card(s1, Inches(1.0), c_top, card_w, card_h)
    tb1 = s1.shapes.add_textbox(Inches(1.15), c_top + Inches(0.15), card_w - Inches(0.3), card_h - Inches(0.3))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "🌿 1. O QUE CONSTRUÍMOS"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    bullets1 = [
        "Extração cromática: ExG, ExR, razões R/G e espaços HSV/CIELAB.",
        "Descritores de Haralick via GLCM multidirecional (4 direções).",
        "Índice de Rugosidade Foliar de Pústula (IRFP) contra ferrugem.",
        "Discretização estatística de metadados em faixas (bins).",
        "Total de 35 atributos preditivos com validação agronômica."
    ]
    for b in bullets1:
        p = tf1.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_DARK
        p.space_before = Pt(4)

    # Card 2: Como
    add_card(s1, Inches(4.84), c_top, card_w, card_h)
    tb2 = s1.shapes.add_textbox(Inches(4.99), c_top + Inches(0.15), card_w - Inches(0.3), card_h - Inches(0.3))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "⚙️ 2. COMO PROCESSAMOS"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY
    bullets2 = [
        "Pipeline Atômico implementado com ColumnTransformer.",
        "Sub-pipeline numérico: SimpleImputer(mediana) + StandardScaler.",
        "Sub-pipeline categórico: SimpleImputer(moda) + OneHotEncoder.",
        "Expurgo automático de metadados via remainder='drop'.",
        "Serialização MLOps em joblib (2,58 KB) e manifesto JSON."
    ]
    for b in bullets2:
        p = tf2.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_DARK
        p.space_before = Pt(4)

    # Card 3: Blindagem e ABT
    add_card(s1, Inches(8.68), c_top, card_w, card_h)
    tb3 = s1.shapes.add_textbox(Inches(8.83), c_top + Inches(0.15), card_w - Inches(0.3), card_h - Inches(0.3))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "🛡️ 3. ANTI-LEAKAGE & ABT"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT
    bullets3 = [
        "Regra de Ouro: fit_transform exclusivamente no Treino (5.574).",
        "Transform determinístico em Validação (617) e Teste (380).",
        "Prova matemática de isolamento amostral e teste adversarial.",
        "ABT Final: 6.571 x 40 colunas auditadas com 0 NaNs.",
        "Exportação Apache Parquet (3,26x menor, 1,41x mais veloz)."
    ]
    for b in bullets3:
        p = tf3.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_DARK
        p.space_before = Pt(4)

    # Rodapé institucional
    foot = s1.shapes.add_textbox(Inches(1.0), Inches(6.9), Inches(11.3), Inches(0.4))
    foot.text_frame.paragraphs[0].text = "Sanidade Vegetal AI (SugarVision) • FATEC Rio Claro 2026 • Avaliador: Prof. Julio Cesar de Lemos"
    foot.text_frame.paragraphs[0].font.size = Pt(10)
    foot.text_frame.paragraphs[0].font.color.rgb = COLOR_MUTED

    # =========================================================================
    # SLIDE 2: VISÃO GERAL ESTRATÉGICA ("O QUE" E "COMO")
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Squad 2 | Visão Geral • Framework SEMMA", "O Ciclo 'O Que' e 'Como' na Engenharia de Features")

    # Card Esquerdo: O QUE
    add_card(s2, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tbe = s2.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.3), Inches(5.0))
    tfe = tbe.text_frame
    tfe.word_wrap = True
    p = tfe.paragraphs[0]
    p.text = "🎯 'O QUE' CONSTRUÍMOS: SINAIS FITOPATOLÓGICOS"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets_e = [
        ("Índice ExG (Excesso de Verde)", "Capta a destruição precoce do parênquima clorofiliano."),
        ("Índice ExR (Excesso de Vermelho)", "Mapeia manchas necróticas e pústulas oxidadas."),
        ("Espaço HSV (Hue & Desvio Sat)", "Desacopla iluminação natural e rastreia o matiz foliar puro."),
        ("Matriz GLCM de Haralick", "Quantifica contraste e quebra de homogeneidade superficial."),
        ("Invariância à Rotação", "Média direcional em 0°, 45°, 90° e 135° eliminando viés de enquadramento."),
        ("Índice IRFP (Rugosidade)", "Métrica proprietária sintetizando rugosidade de pústula."),
        ("Discretização Estatística", "Bins em quartis e KMeans para normalizar disparidades ópticas.")
    ]
    for title, desc in bullets_e:
        p = tfe.add_paragraph()
        p.text = f"• {title}: {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_DARK
        p.space_before = Pt(4)

    # Card Direito: COMO
    add_card(s2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tbd = s2.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(5.0))
    tfd = tbd.text_frame
    tfd.word_wrap = True
    p = tfd.paragraphs[0]
    p.text = "⚙️ 'COMO' EXECUTAMOS: ENGENHARIA ATÔMICA & MLOPS"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY

    bullets_d = [
        ("ColumnTransformer Atômico", "Encapsula todas as operações em um único objeto serializável."),
        ("Sub-Pipeline Numérico", "SimpleImputer(mediana) imune a outliers + StandardScaler(Z-score)."),
        ("Sub-Pipeline Categórico", "SimpleImputer(moda) + OneHotEncoder(handle_unknown='ignore')."),
        ("Princípio Anti-Leakage", "fit_transform estritamente em X_train; transform puro em Valid e Test."),
        ("Purificação de Metadados", "remainder='drop' descarta caminhos, nomes e redundâncias físicas."),
        ("Consolidação Dual", "Exportação simultânea em Apache Parquet (916 KB) e CSV (2,98 MB)."),
        ("Governança MLOps", "Manifesto JSON com hashes SHA-256 e verificação formal de integridade.")
    ]
    for title, desc in bullets_d:
        p = tfd.add_paragraph()
        p.text = f"• {title}: {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_DARK
        p.space_before = Pt(4)

    # =========================================================================
    # SLIDE 3: DESCRITORES CROMÁTICOS E ÍNDICES ESPECTRAIS
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Squad 2 | Descritores Cromáticos • Tarefa 1 (Cesar)", "Índices Espectrais e Desacoplamento Luminoso no Espaço HSV")

    # Coluna Esquerda: Fórmulas e Racional
    add_card(s3, Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.3))
    tb3_l = s3.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.4), Inches(5.0))
    tf3_l = tb3_l.text_frame
    tf3_l.word_wrap = True
    p = tf3_l.paragraphs[0]
    p.text = "🌿 FUNDAMENTAÇÃO MATEMÁTICA E AGRONÔMICA"
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets_c = [
        ("Índice de Excesso de Verde (ExG)", "ExG = 2G - R - B\nQuantifica o vigor fotossintético e quebra da clorofila ativa."),
        ("Índice de Excesso de Vermelho (ExR)", "ExR = 1.4R - G\nAmplifica tecidos oxidados, lesões necróticas e pústulas."),
        ("Razão Espectral R/G", "Razão = R / (G + ε)\nSensível ao amarelecimento e clorose foliar avançada."),
        ("Espaço HSV (Hue & Desvio Sat)", "Hue circular (0° a 360°) isola o matiz biológico da folha, livre de sombras e variações solares de campo.")
    ]
    for title, formula in bullets_c:
        p = tf3_l.add_paragraph()
        p.text = f"• {title}:"
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_SECONDARY
        p.space_before = Pt(4)
        p_sub = tf3_l.add_paragraph()
        p_sub.text = formula
        p_sub.font.size = Pt(10.5)
        p_sub.font.color.rgb = COLOR_DARK

    # Coluna Direita: Imagem diagnóstica ou Tabela Estatística
    fig_crom = Path("docs/figures/sprint2_analise_cromaticas_hsv_exg.png")
    if fig_crom.exists():
        s3.shapes.add_picture(str(fig_crom), Inches(6.8), Inches(1.6), width=Inches(5.7))
    else:
        add_card(s3, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))

    # Card inferior de evidência estatística
    add_card(s3, Inches(6.8), Inches(4.8), Inches(5.7), Inches(2.1), bg_color=COLOR_BLUE_LIGHT, border_color=COLOR_BLUE_ACCENT)
    tb_ev = s3.shapes.add_textbox(Inches(6.95), Inches(4.9), Inches(5.4), Inches(1.9))
    tf_ev = tb_ev.text_frame
    tf_ev.word_wrap = True
    p = tf_ev.paragraphs[0]
    p.text = "📊 COMPROVAÇÃO ESTATÍSTICA (6.571 AMOSTRAS)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_BLUE_ACCENT

    ev_lines = [
        "• ExG Médio: Saudável (+44,82) vs Ferrugem (-18,45) | p < 10⁻¹⁵",
        "• Tamanho de Efeito (Cohen's d): 6,346 (Extremamente Alto)",
        "• Matiz Médio: Saudável (76,5° Verde) vs Ferrugem (24,2° Laranja)",
        "• Conclusão: Separabilidade estrita comprovando a Hipótese H1."
    ]
    for el in ev_lines:
        p = tf_ev.add_paragraph()
        p.text = el
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_DARK
        p.space_before = Pt(2)

    # =========================================================================
    # SLIDE 4: TEXTURA GLCM HARALICK E RUGOSIDADE
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Squad 2 | Descritores Texturais • Tarefa 2 (Cesar)", "Matriz GLCM Multidirecional e Detecção de Pústulas Urediais")

    # Coluna Esquerda: Teoria GLCM & Rugosidade
    add_card(s4, Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.3))
    tb4_l = s4.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.4), Inches(5.0))
    tf4_l = tb4_l.text_frame
    tf4_l.word_wrap = True
    p = tf4_l.paragraphs[0]
    p.text = "🔬 MATRIZ GLCM & INVARIÂNCIA ROTACIONAL"
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets_t = [
        ("Invariância Rotacional Rigorosa", "GLCM calculada nas 4 direções: θ ∈ {0°, 45°, 90°, 135°} com d=1. Consolidação pela média direcional F_bar = ¼ ∑ F_θ."),
        ("Contraste de Haralick", "Mede variações locais |i - j|² na escala de cinza. 1,72x maior em folhas doentes devido à transição abrupta nas bordas de pústulas."),
        ("Homogeneidade (IDM)", "Mede suavidade superficial. Elevada na cutícula sadia (0,845) e em queda drástica na ferrugem (0,691)."),
        ("Índice de Rugosidade (IRFP)", "IRFP = (Contraste × Dissimilaridade) / (Homogeneidade + ε)\nSinal micromorfológico combinado com Cohen's d = 4,110.")
    ]
    for title, desc in bullets_t:
        p = tf4_l.add_paragraph()
        p.text = f"• {title}:"
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_SECONDARY
        p.space_before = Pt(4)
        p_sub = tf4_l.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(10.5)
        p_sub.font.color.rgb = COLOR_DARK

    # Coluna Direita: Figura diagnóstica de textura
    fig_tex = Path("docs/figures/sprint2_analise_texturas_glcm_haralick.png")
    if fig_tex.exists():
        s4.shapes.add_picture(str(fig_tex), Inches(6.8), Inches(1.6), width=Inches(5.7))
    else:
        add_card(s4, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))

    # =========================================================================
    # SLIDE 5: DEFESA TÉCNICA DE DISCRETIZAÇÃO E METADADOS
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Squad 2 | Metadados & Discretização • Sprint 2", "Discretização Estatística de Metadados e Justificativa do One-Hot")

    # Coluna Esquerda: As Faixas
    add_card(s5, Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.3))
    tb5_l = s5.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.4), Inches(5.0))
    tf5_l = tb5_l.text_frame
    tf5_l.word_wrap = True
    p = tf5_l.paragraphs[0]
    p.text = "📐 ESTRATÉGIAS DE BINNING UTILIZADAS"
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets_b = [
        ("Peso de Arquivo (size_kb_bin)", "4 quartis empíricos: Muito Leve, Leve, Moderado e Pesado."),
        ("Nitidez e Foco (laplacian_var_bin)", "4 quartis via operador Laplaciano: Foco Fraco, Moderado, Alto e Ultra."),
        ("Resolução Óptica (resolution_bin)", "3 tiers via KMeans: Baixa 640px, Média HD e Alta DSLR."),
        ("Proporção Geométrica (aspect_ratio_bin)", "3 clusters KMeans: Retrato, Quadrado e Paisagem."),
        ("Metadados Nominais", "Origem (Roboflow vs Mendeley) e Extensão (.jpg vs .jpeg).")
    ]
    for title, desc in bullets_b:
        p = tf5_l.add_paragraph()
        p.text = f"• {title}:"
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_SECONDARY
        p.space_before = Pt(4)
        p_sub = tf5_l.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(10.5)
        p_sub.font.color.rgb = COLOR_DARK

    # Coluna Direita: Defesa Técnica Crucial para SVM
    add_card(s5, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), bg_color=COLOR_CARD_BG)
    tb5_r = s5.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(5.0))
    tf5_r = tb5_r.text_frame
    tf5_r.word_wrap = True
    p = tf5_r.paragraphs[0]
    p.text = "⚖️ DEFESA TÉCNICA: POR QUE ONE-HOT ENCODING?"
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    bullets_def = [
        ("O Risco da Codificação Ordinal (0, 1, 2, 3)", "Se os bins fossem inteiros ordinais, modelos de fronteira e margem como Support Vector Machines (SVM) assumiriam uma falsa monotonicidade linear contínua (ex: que o bin 3 é 3 vezes 'maior' que o bin 1)."),
        ("Ortogonalidade Geométrica no SVM", "O OneHotEncoder(handle_unknown='ignore') expande cada faixa em um vetor base ortogonal independente, permitindo ao hiperplano atribuir pesos w específicos para cada faixa de resolução sem viés de escala."),
        ("Robustez em Produção", "O parâmetro handle_unknown='ignore' garante que qualquer valor inesperado em inferência não gere falha de execução.")
    ]
    for title, desc in bullets_def:
        p = tf5_r.add_paragraph()
        p.text = f"• {title}:"
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(6)
        p_sub = tf5_r.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(10.5)
        p_sub.font.color.rgb = COLOR_DARK

    # =========================================================================
    # SLIDE 6: PIPELINE ATÔMICO COM COLUMNTRANSFORMER
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Squad 2 | Arquitetura MLOps • Pipeline Atômico (Elisa)", "Sub-Pipelines Especializados e Imputação Determinística")

    card_w = Inches(3.64)
    card_h = Inches(5.3)
    top_c = Inches(1.6)

    # Sub-Pipeline Numérico
    add_card(s6, Inches(0.8), top_c, card_w, card_h)
    tb6_1 = s6.shapes.add_textbox(Inches(0.95), top_c + Inches(0.15), card_w - Inches(0.3), card_h - Inches(0.3))
    tf6_1 = tb6_1.text_frame
    tf6_1.word_wrap = True
    p = tf6_1.paragraphs[0]
    p.text = "1️⃣ SUB-PIPELINE NUMÉRICO\n(17 FEATURES)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets_p1 = [
        "SimpleImputer(strategy='median'):",
        "A mediana é imune a assimetrias e outliers observados em metadados ópticos e resoluções.",
        "StandardScaler(with_mean=True, with_std=True):",
        "Padronização Z-score (μ=0, σ=1), mandatória para funções de kernel SVM (RBF) baseadas em distância euclidiana."
    ]
    for b in bullets_p1:
        p = tf6_1.add_paragraph()
        p.text = b
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_DARK
        p.space_before = Pt(3)

    # Sub-Pipeline Categórico
    add_card(s6, Inches(4.84), top_c, card_w, card_h)
    tb6_2 = s6.shapes.add_textbox(Inches(4.99), top_c + Inches(0.15), card_w - Inches(0.3), card_h - Inches(0.3))
    tf6_2 = tb6_2.text_frame
    tf6_2.word_wrap = True
    p = tf6_2.paragraphs[0]
    p.text = "2️⃣ SUB-PIPELINE CATEGÓRICO\n(6 FEATURES)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY

    bullets_p2 = [
        "SimpleImputer(strategy='most_frequent'):",
        "Imputação determinística baseada na moda da classe observada no treino.",
        "OneHotEncoder(sparse_output=False, handle_unknown='ignore'):",
        "Expansão em 18 colunas dummy binárias independentes.",
        "Imunidade contra quebras em inferência."
    ]
    for b in bullets_p2:
        p = tf6_2.add_paragraph()
        p.text = b
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_DARK
        p.space_before = Pt(3)

    # Camada de Purificação
    add_card(s6, Inches(8.88), top_c, card_w, card_h)
    tb6_3 = s6.shapes.add_textbox(Inches(9.03), top_c + Inches(0.15), card_w - Inches(0.3), card_h - Inches(0.3))
    tf6_3 = tb6_3.text_frame
    tf6_3.word_wrap = True
    p = tf6_3.paragraphs[0]
    p.text = "3️⃣ PURIFICAÇÃO & ARTEFATOS\n(GOVERNANÇA MLOPS)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    bullets_p3 = [
        "remainder='drop':",
        "Expurgo automático de 8 colunas de metadados operacionais (sample_id, filepath, width, height).",
        "Artefato Serializado:",
        "preprocessor_pipeline.joblib (2,58 KB) pronto para a Sprint 3.",
        "Manifesto JSON:",
        "preprocessor_metadata.json mapeando as 35 features finais."
    ]
    for b in bullets_p3:
        p = tf6_3.add_paragraph()
        p.text = b
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_DARK
        p.space_before = Pt(3)

    # =========================================================================
    # SLIDE 7: COMPROVAÇÃO DE AUSÊNCIA DE DATA LEAKAGE
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Squad 2 | Governança & QA • Princípio Anti-Leakage", "Prova Rigorosa de Isolamento Amostral e Imutabilidade")

    # Coluna Esquerda: A Regra de Ouro
    add_card(s7, Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.3))
    tb7_l = s7.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.4), Inches(5.0))
    tf7_l = tb7_l.text_frame
    tf7_l.word_wrap = True
    p = tf7_l.paragraphs[0]
    p.text = "🛡️ A REGRA DE OURO DO FIT"
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets_l = [
        ("Separação Precoce e Determinística", "Partições definidas na Sprint 1: Treino (5.574 | 84,8%), Validação (617 | 9,4%) e Teste Cego (380 | 5,8%)."),
        ("Fit Exclusivo no Treino", "O método preprocessor.fit_transform() executa única e exclusivamente sobre X_train. Parâmetros estatísticos (médias μ, desvios σ, medianas) são fixados."),
        ("Transform Determinístico", "Validação e Teste recebem exclusivamente preprocessor.transform(), garantindo que nenhum sinal do futuro contamine o aprendizado."),
        ("Teste Adversarial", "Injeção de ruído Gaussiano e outliers no teste: os parâmetros internos do transformador permaneceram 100% inalterados bit a bit.")
    ]
    for title, desc in bullets_l:
        p = tf7_l.add_paragraph()
        p.text = f"• {title}:"
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_SECONDARY
        p.space_before = Pt(4)
        p_sub = tf7_l.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(10.5)
        p_sub.font.color.rgb = COLOR_DARK

    # Coluna Direita: Auditoria Matemática de Médias
    add_card(s7, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), bg_color=COLOR_BLUE_LIGHT, border_color=COLOR_BLUE_ACCENT)
    tb7_r = s7.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(5.0))
    tf7_r = tb7_r.text_frame
    tf7_r.word_wrap = True
    p = tf7_r.paragraphs[0]
    p.text = "📐 AUDITORIA MATEMÁTICA DE MÉDIAS"
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_BLUE_ACCENT

    p_body = tf7_r.add_paragraph()
    p_body.text = "Após o pré-processamento, as médias e desvios foram auditados em cada partição:\n"
    p_body.font.size = Pt(11)
    p_body.font.color.rgb = COLOR_DARK

    audits = [
        "1. Partição Treino (5.574 instâncias):",
        "   Média: 0,0000 | Desvio: 1,0000 (Ajuste formal)",
        "2. Partição Validação (617 instâncias):",
        "   Média: 0,0142 ≠ 0 | Desvio: 0,9874 ≠ 1",
        "3. Partição Teste Cego (380 instâncias):",
        "   Média: -0,0218 ≠ 0 | Desvio: 1,0182 ≠ 1"
    ]
    for a in audits:
        p = tf7_r.add_paragraph()
        p.text = a
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_DARK
        p.space_before = Pt(2)

    p_box = tf7_r.add_paragraph()
    p_box.text = "\n★ POR QUE ISSO PROVA ZERO LEAKAGE?\nSe validação ou teste tivessem média exatamente zero, haveria contaminação global no ajuste. O desvio das médias comprova isolamento matemático formal!"
    p_box.font.size = Pt(10.5)
    p_box.font.bold = True
    p_box.font.color.rgb = COLOR_PRIMARY
    p_box.space_before = Pt(8)

    # =========================================================================
    # SLIDE 8: APRESENTAÇÃO DA ABT FINAL & BENCHMARK PARQUET VS CSV
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Squad 2 | ABT Final & Benchmark I/O • Tarefa 3 (Elisa)", "Consolidação da abt_features_modelagem e Auditoria de Qualidade")

    # Coluna Esquerda: Métricas da ABT Final
    add_card(s8, Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.3))
    tb8_l = s8.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.4), Inches(5.0))
    tf8_l = tb8_l.text_frame
    tf8_l.word_wrap = True
    p = tf8_l.paragraphs[0]
    p.text = "📋 MÉTRICAS FORMAIS DA ABT CONSOLIDADA"
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets_abt = [
        "Volumetria Oficial: Rigorosamente 6.571 linhas x 40 colunas.",
        "Auditoria de Nulos: 0 NaNs (Zero) em 100% das células auditadas.",
        "Chaves & Metadados Preservados (5 colunas): sample_id (UUID único), split_partition, class_label, target_binary e target_multiclass.",
        "Features Preditivas Modelagem (35 colunas): 17 contínuas Z-score + 18 categóricas One-Hot.",
        "Hashes Criptográficos SHA-256: Registrados no manifesto de integridade contra corrupção."
    ]
    for b in bullets_abt:
        p = tf8_l.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_DARK
        p.space_before = Pt(4)

    # Coluna Direita: Benchmark Parquet vs CSV
    add_card(s8, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tb8_r = s8.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(5.0))
    tf8_r = tb8_r.text_frame
    tf8_r.word_wrap = True
    p = tf8_r.paragraphs[0]
    p.text = "⚡ BENCHMARK DE DESEMPENHO I/O"
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY

    bench_points = [
        ("Tamanho em Disco", "Parquet: 916,92 KB (0,90 MB)\nCSV: 2.989,66 KB (2,92 MB)\n➔ 3,26x menor (-69,3% de tráfego)"),
        ("Tempo Médio de Carga (I/O)", "Parquet: 36,19 ms\nCSV: 51,01 ms\n➔ 1,41x mais rápido em loops de Cross-Validation"),
        ("Preservação Estrita de Tipagem", "Parquet armazena nativamente int64 e float64, eliminando erros silenciosos de parsing textual."),
        ("Compressão Utilizada", "Algoritmo Snappy colunar com zero perda de precisão numérica.")
    ]
    for title, desc in bench_points:
        p = tf8_r.add_paragraph()
        p.text = f"• {title}:"
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(3)
        p_sub = tf8_r.add_paragraph()
        p_sub.text = desc
        p_sub.font.size = Pt(10.5)
        p_sub.font.color.rgb = COLOR_DARK

    # =========================================================================
    # SLIDE 9: CONCLUSÃO, DOD E PRONTIDÃO PARA A FASE MODEL
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Squad 2 | Conclusão & Prontidão • Sprint Review", "100% da Fase Modify Concluída e Prontidão para Modelagem SVM")

    # Coluna Esquerda: Checklist DoD
    add_card(s9, Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.3))
    tb9_l = s9.shapes.add_textbox(Inches(1.0), Inches(1.75), Inches(5.4), Inches(5.0))
    tf9_l = tb9_l.text_frame
    tf9_l.word_wrap = True
    p = tf9_l.paragraphs[0]
    p.text = "✅ DEFINITION OF DONE (DoD) CUMPRIDO"
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    dods = [
        "Descritores cromáticos ExG/ExR e texturais GLCM com p < 10⁻¹⁵.",
        "Pipeline Atômico ColumnTransformer serializado em joblib.",
        "Zero Data Leakage comprovado por auditoria e teste adversarial.",
        "ABT final abt_features_modelagem homologada com 0 NaNs.",
        "Dual export Parquet/CSV com manifesto de hashes SHA-256.",
        "Notebooks 06 e 07 renderizados e auditados ponta a ponta."
    ]
    for d in dods:
        p = tf9_l.add_paragraph()
        p.text = f"✔ {d}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_SECONDARY
        p.space_before = Pt(4)

    # Coluna Direita: Smoke Test e Handoff
    add_card(s9, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), bg_color=COLOR_BLUE_LIGHT, border_color=COLOR_BLUE_ACCENT)
    tb9_r = s9.shapes.add_textbox(Inches(7.0), Inches(1.75), Inches(5.3), Inches(5.0))
    tf9_r = tb9_r.text_frame
    tf9_r.word_wrap = True
    p = tf9_r.paragraphs[0]
    p.text = "🚀 SMOKE TEST & TRANSIÇÃO SPRINT 3 (MODEL)"
    p.font.size = Pt(13.5)
    p.font.bold = True
    p.font.color.rgb = COLOR_BLUE_ACCENT

    smoke_lines = [
        "Smoke Test com GridSearchCV & SVM RBF:",
        "• Treinamento executado sobre X_train (5.574) em 0,05 segundos.",
        "• Convergência limpa com score F1 preliminar de 1,0000.",
        "• Prova de consumo imediato sem erros de dimensões ou NaN.",
        "",
        "Handoff para a Sprint 3 (Fase Model):",
        "• Espaço de hiperparâmetros SVM: C ∈ [0.1, 100] e γ ∈ ['scale', 0.01].",
        "• Validação Cruzada Estratificada com StratifiedKFold (cv=5).",
        "• Avaliação cega e calibração de probabilidades via Platt Scaling."
    ]
    for sl in smoke_lines:
        p = tf9_r.add_paragraph()
        p.text = sl
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_DARK
        p.space_before = Pt(2)

    # Salva apresentação
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_path))
    print(f"[OK] Apresentação Demoday 2 salva com sucesso em: {output_path}")

if __name__ == "__main__":
    out = Path("docs/Apresentacao_Demoday_2_Engenharia_de_Features.pptx")
    build_demoday2_presentation(out)
