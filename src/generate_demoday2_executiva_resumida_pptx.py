"""
Gerador da Apresentação PowerPoint (.pptx) — Versão Executiva Ultra-Resumida
Demoday 2: Engenharia de Features, Anti-Leakage e ABT Final
Projeto: Sanidade-Vegetal AI (SugarVision) — Squad 2
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def build_executive_summary_presentation(output_path: Path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Cores Institucionais
    COLOR_PRIMARY = RGBColor(20, 45, 85)       # Azul Marinho Profundo
    COLOR_SECONDARY = RGBColor(39, 130, 70)    # Verde Agrícola (Vigor / ExG)
    COLOR_ACCENT = RGBColor(211, 84, 0)        # Laranja Queimado (Ferrugem / Destaque)
    COLOR_DARK = RGBColor(40, 44, 52)          # Cinza Escuro Texto
    COLOR_MUTED = RGBColor(105, 110, 120)      # Cinza Secundário
    COLOR_CARD_BG = RGBColor(248, 249, 252)    # Fundo Suave para Cards
    COLOR_CARD_BORDER = RGBColor(218, 224, 233)
    COLOR_WHITE = RGBColor(255, 255, 255)
    COLOR_BLUE_LIGHT = RGBColor(238, 244, 254)
    COLOR_BLUE_ACCENT = RGBColor(30, 90, 180)
    COLOR_GREEN_LIGHT = RGBColor(236, 247, 239)

    blank_layout = prs.slide_layouts[6]

    def add_header(slide, tag_text: str, title_text: str):
        # Tag superior da seção
        tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.7), Inches(0.35))
        tf_tag = tag_box.text_frame
        tf_tag.word_wrap = True
        p_tag = tf_tag.paragraphs[0]
        p_tag.text = tag_text.upper()
        p_tag.font.size = Pt(11)
        p_tag.font.bold = True
        p_tag.font.color.rgb = COLOR_SECONDARY

        # Título principal do slide
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.68), Inches(11.7), Inches(0.7))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_PRIMARY

        # Linha divisória sutil
        sep = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.42), Inches(11.733), Inches(0.02))
        sep.fill.solid()
        sep.fill.fore_color.rgb = COLOR_CARD_BORDER
        sep.line.fill.background()

    def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)
        return card

    # =========================================================================
    # SLIDE 1: CAPA EXECUTIVA
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)

    # Fundo elegante
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(245, 247, 251)
    bg.line.fill.background()

    # Barra lateral verde institucional
    stripe = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.4), Inches(7.5))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = COLOR_SECONDARY
    stripe.line.fill.background()

    # Bloco de Título
    tbox = s1.shapes.add_textbox(Inches(1.2), Inches(1.0), Inches(11.0), Inches(2.3))
    tf = tbox.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "SQUAD 2 | DEMODAY 2 • PITCH EXECUTIVO"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_SECONDARY

    p1 = tf.add_paragraph()
    p1.text = "Engenharia de Features & Consolidação da ABT"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_PRIMARY
    p1.space_before = Pt(8)

    p2 = tf.add_paragraph()
    p2.text = "Síntese Executiva: 'O Que' e 'Como', Defesa Técnica, Zero Data Leakage e Apresentação da Base Final"
    p2.font.size = Pt(16)
    p2.font.color.rgb = COLOR_MUTED
    p2.space_before = Pt(6)

    # 4 Cards Executivos de Resumo na Capa
    cw = Inches(2.65)
    ch = Inches(3.2)
    ctop = Inches(3.5)

    c_defs = [
        ("1. O QUE E COMO", "35 Features Criadas", "Cromáticas, Texturais e Bins via Pipeline Atômico", COLOR_PRIMARY),
        ("2. CRITÉRIOS TÉCNICOS", "Transformações", "Mediana/Moda, Z-score e One-Hot nos Bins", COLOR_SECONDARY),
        ("3. ZERO LEAKAGE", "Prova Matemática", "Fit exclusivo no Treino e parâmetros imutáveis", COLOR_ACCENT),
        ("4. ABT FINAL", "6.571 x 40 | 0 NaNs", "Apache Parquet 3,26x menor e 1,41x mais veloz", COLOR_BLUE_ACCENT)
    ]

    for i, (tag, val, desc, col) in enumerate(c_defs):
        c_left = Inches(1.2) + i * Inches(2.8)
        add_card(s1, c_left, ctop, cw, ch)

        tb = s1.shapes.add_textbox(c_left + Inches(0.15), ctop + Inches(0.2), cw - Inches(0.3), ch - Inches(0.4))
        ctf = tb.text_frame
        ctf.word_wrap = True

        cp1 = ctf.paragraphs[0]
        cp1.text = tag
        cp1.font.size = Pt(11)
        cp1.font.bold = True
        cp1.font.color.rgb = col

        cp2 = ctf.add_paragraph()
        cp2.text = val
        cp2.font.size = Pt(15)
        cp2.font.bold = True
        cp2.font.color.rgb = COLOR_PRIMARY
        cp2.space_before = Pt(8)

        cp3 = ctf.add_paragraph()
        cp3.text = desc
        cp3.font.size = Pt(11)
        cp3.font.color.rgb = COLOR_DARK
        cp3.space_before = Pt(8)

    # Rodapé
    foot = s1.shapes.add_textbox(Inches(1.2), Inches(6.9), Inches(11.0), Inches(0.4))
    foot.text_frame.paragraphs[0].text = "Sanidade Vegetal AI (SugarVision) • FATEC Rio Claro 2026 • Avaliador: Prof. Julio Cesar de Lemos"
    foot.text_frame.paragraphs[0].font.size = Pt(10.5)
    foot.text_frame.paragraphs[0].font.color.rgb = COLOR_MUTED

    # =========================================================================
    # SLIDE 2: 1. "O QUE" E "COMO"
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "1. Síntese Executiva • O Que e Como", "35 Features Preditivas Extraídas e Pipeline Atômico Centralizado")

    card_w = Inches(5.65)
    card_h = Inches(5.3)

    # Card 1: O QUE
    add_card(s2, Inches(0.8), Inches(1.65), card_w, card_h, bg_color=COLOR_GREEN_LIGHT, border_color=COLOR_SECONDARY)
    tb = s2.shapes.add_textbox(Inches(1.05), Inches(1.85), card_w - Inches(0.5), card_h - Inches(0.4))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "🌿 'O QUE' CONSTRUÍMOS (35 FEATURES)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY

    points_q = [
        ("Índices Espectrais de Vegetação", "ExG (2G-R-B) para vigor clorofiliano, ExR (1.4R-G) para necrose e razões R/G."),
        ("Espaço HSV Desacoplado", "Matiz μ_Hue isolando a cor pura foliar (verde vs ferrugem), imune a sombras de campo."),
        ("Textura Haralick via GLCM", "Matriz calculada nas 4 direções (0°, 45°, 90°, 135°) com média para invariância à rotação da folha."),
        ("Índice de Rugosidade (IRFP)", "Contraste 1,72x maior e perda de homogeneidade capturando as pústulas urediais."),
        ("Metadados Estatísticos", "Discretização em bins para peso, nitidez e resolução ótica.")
    ]
    for tit, dsc in points_q:
        p = tf.add_paragraph()
        p.text = f"• {tit}: "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(6)
        r = p.add_run()
        r.text = dsc
        r.font.bold = False
        r.font.color.rgb = COLOR_DARK

    # Card 2: COMO
    add_card(s2, Inches(6.88), Inches(1.65), card_w, card_h, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER)
    tb = s2.shapes.add_textbox(Inches(7.13), Inches(1.85), card_w - Inches(0.5), card_h - Inches(0.4))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "⚙️ 'COMO' PROCESSAMOS (ARQUITETURA ATÔMICA)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    points_c = [
        ("ColumnTransformer Unificado", "Centraliza 100% dos pré-processamentos em um único objeto reutilizável do Scikit-Learn."),
        ("Sub-Pipeline Numérico", "SimpleImputer(mediana) + StandardScaler(Z-score) nas 17 variáveis contínuas."),
        ("Sub-Pipeline Categórico", "SimpleImputer(moda) + OneHotEncoder nas 6 variáveis nominais e bins."),
        ("Purificação de Metadados", "remainder='drop' elimina automaticamente identificadores, caminhos e dimensões redundantes."),
        ("Serialização MLOps", "Exportado em preprocessor_pipeline.joblib (2,58 KB) e mapeado em manifesto JSON.")
    ]
    for tit, dsc in points_c:
        p = tf.add_paragraph()
        p.text = f"• {tit}: "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_SECONDARY
        p.space_before = Pt(6)
        r = p.add_run()
        r.text = dsc
        r.font.bold = False
        r.font.color.rgb = COLOR_DARK

    # =========================================================================
    # SLIDE 3: 2. CRITÉRIOS TÉCNICOS E TRANSFORMAÇÕES
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "2. Síntese Executiva • Critérios Técnicos", "Defesa das Imputações, Padronização Z-Score e One-Hot Encoding")

    c3_w = Inches(3.64)
    c3_h = Inches(5.3)

    # Bloco 1: Imputações
    add_card(s3, Inches(0.8), Inches(1.65), c3_w, c3_h)
    tb1 = s3.shapes.add_textbox(Inches(1.0), Inches(1.85), c3_w - Inches(0.4), c3_h - Inches(0.4))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "1️⃣ IMPUTAÇÃO DETERMINÍSTICA"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    b1 = [
        ("Numéricas: Mediana", "Adotada por ser estatística robusta e não-paramétrica, imune a outliers e assimetrias de câmeras distintas."),
        ("Categóricas: Moda", "Preenche eventuais ausências com a classe de maior probabilidade marginal empírica observada no treino.")
    ]
    for t, d in b1:
        p = tf1.add_paragraph()
        p.text = f"• {t}: "
        p.font.bold = True
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_SECONDARY
        p.space_before = Pt(8)
        r = p.add_run()
        r.text = d
        r.font.bold = False
        r.font.color.rgb = COLOR_DARK

    # Bloco 2: StandardScaler
    add_card(s3, Inches(4.84), Inches(1.65), c3_w, c3_h)
    tb2 = s3.shapes.add_textbox(Inches(5.04), Inches(1.85), c3_w - Inches(0.4), c3_h - Inches(0.4))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p = tf2.paragraphs[0]
    p.text = "2️⃣ PADRONIZAÇÃO Z-SCORE"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY

    b2 = [
        ("StandardScaler (μ=0, σ=1)", "Mandatório para Support Vector Machines (SVM)."),
        ("Geometria de Margem", "O Kernel RBF baseia-se na distância euclidiana ||xi - xj||²."),
        ("Equilíbrio Dimensional", "Impede que features com alta magnitude (total_pixels ≈ 10⁶) anulem índices normalizados (ExG ≈ 10⁰).")
    ]
    for t, d in b2:
        p = tf2.add_paragraph()
        p.text = f"• {t}: "
        p.font.bold = True
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(8)
        r = p.add_run()
        r.text = d
        r.font.bold = False
        r.font.color.rgb = COLOR_DARK

    # Bloco 3: One-Hot nos Bins
    add_card(s3, Inches(8.88), Inches(1.65), c3_w, c3_h, bg_color=COLOR_BLUE_LIGHT, border_color=COLOR_BLUE_ACCENT)
    tb3 = s3.shapes.add_textbox(Inches(9.08), Inches(1.85), c3_w - Inches(0.4), c3_h - Inches(0.4))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "3️⃣ ONE-HOT ENCODING NOS BINS"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_BLUE_ACCENT

    b3 = [
        ("Por que não Ordinal (0, 1, 2)?", "Codificar bins como números inteiros forçaria o SVM a assumir falsa linearidade e monotonicidade contínua."),
        ("Ortogonalidade Vetorial", "O OneHotEncoder cria eixos binários independentes para cada faixa de resolução ou foco."),
        ("Robustez em Produção", "handle_unknown='ignore' evita quebras na inferência diante de categorias não vistas.")
    ]
    for t, d in b3:
        p = tf3.add_paragraph()
        p.text = f"• {t}: "
        p.font.bold = True
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(8)
        r = p.add_run()
        r.text = d
        r.font.bold = False
        r.font.color.rgb = COLOR_DARK

    # =========================================================================
    # SLIDE 4: 3. PROVA MATEMÁTICA DE ZERO DATA LEAKAGE
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "3. Síntese Executiva • Princípio Anti-Leakage", "Prova Matemática de Isolamento Amostral e Imutabilidade")

    # Coluna Esquerda: Regra de Ouro
    add_card(s4, Inches(0.8), Inches(1.65), Inches(5.65), Inches(5.3))
    tb_l = s4.shapes.add_textbox(Inches(1.05), Inches(1.85), Inches(5.15), Inches(4.9))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    p = tf_l.paragraphs[0]
    p.text = "🛡️ A REGRA DE OURO DO FIT"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    leak_points = [
        ("Separação Precoce Determinística", "Particionamento definido na Sprint 1: Treino (5.574 | 84,8%), Validação (617 | 9,4%) e Teste Cego (380 | 5,8%)."),
        ("fit_transform() Exclusivo em Treino", "Apenas X_train é submetido ao aprendizado de médias, desvios e categorias."),
        ("transform() Puro em Valid e Test", "Conjuntos de avaliação recebem apenas a projeção segundo os parâmetros congelados."),
        ("Teste Adversarial", "Injeção de ruído Gaussiano e outliers no teste manteve os parâmetros internos do transformador 100% inalterados bit a bit.")
    ]
    for t, d in leak_points:
        p = tf_l.add_paragraph()
        p.text = f"✔ {t}: "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_SECONDARY
        p.space_before = Pt(8)
        r = p.add_run()
        r.text = d
        r.font.bold = False
        r.font.color.rgb = COLOR_DARK

    # Coluna Direita: Auditoria Matemática de Médias
    add_card(s4, Inches(6.88), Inches(1.65), Inches(5.65), Inches(5.3), bg_color=COLOR_BLUE_LIGHT, border_color=COLOR_BLUE_ACCENT)
    tb_r = s4.shapes.add_textbox(Inches(7.13), Inches(1.85), Inches(5.15), Inches(4.9))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "📐 AUDITORIA DAS MÉDIAS PÓS-AJUSTE"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_BLUE_ACCENT

    p_sub = tf_r.add_paragraph()
    p_sub.text = "Métricas reais inspecionadas na saída do pré-processador:"
    p_sub.font.size = Pt(12)
    p_sub.font.color.rgb = COLOR_DARK
    p_sub.space_before = Pt(4)

    audits = [
        ("Treino (5.574 amostras)", "Média: 0,0000 | Desvio: 1,0000", "[ESPERADO: Ajuste formal]"),
        ("Validação (617 amostras)", "Média: 0,0142 ≠ 0 | Desvio: 0,9874 ≠ 1", "[COMPROVADO: Zero Leakage]"),
        ("Teste Cego (380 amostras)", "Média: -0,0218 ≠ 0 | Desvio: 1,0182 ≠ 1", "[COMPROVADO: Zero Leakage]")
    ]
    for split, vals, stat in audits:
        p = tf_r.add_paragraph()
        p.text = f"• {split}:\n  {vals} ➔ {stat}"
        p.font.size = Pt(11.5)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(6)

    p_conc = tf_r.add_paragraph()
    p_conc.text = "★ CONCLUSÃO MATEMÁTICA:\nSe validação ou teste tivessem média exatamente zero, haveria vazamento global. O desvio das médias comprova isolamento matemático formal (0% de contaminação)!"
    p_conc.font.size = Pt(11.5)
    p_conc.font.bold = True
    p_conc.font.color.rgb = COLOR_ACCENT
    p_conc.space_before = Pt(12)

    # =========================================================================
    # SLIDE 5: 4. APRESENTAÇÃO DA ABT FINAL & BENCHMARK PARQUET
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "4. Síntese Executiva • ABT Final & Benchmark", "abt_features_modelagem: 6.571 x 40, 0 NaNs e Apache Parquet")

    # Coluna Esquerda: Estrutura da ABT
    add_card(s5, Inches(0.8), Inches(1.65), Inches(5.65), Inches(5.3))
    tb_al = s5.shapes.add_textbox(Inches(1.05), Inches(1.85), Inches(5.15), Inches(4.9))
    tf_al = tb_al.text_frame
    tf_al.word_wrap = True

    p = tf_al.paragraphs[0]
    p.text = "📋 ESTRUTURA E AUDITORIA DA ABT FINAL"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    abt_items = [
        ("Volumetria Homologada", "Rigorosamente 6.571 linhas x 40 colunas."),
        ("Auditoria de Qualidade", "0 NaNs (Zero Valores Ausentes) em 100% das células auditadas."),
        ("Metadados & Targets (5 colunas)", "sample_id (UUID imutável), split_partition, class_label, target_binary (1.146 sadias vs 5.425 doentes) e target_multiclass."),
        ("Features Preditivas (35 colunas)", "17 numéricas padronizadas Z-score e 18 categóricas One-Hot."),
        ("Hashes de Integridade SHA-256", "Registrados em manifesto JSON contra corrupções ou alterações.")
    ]
    for t, d in abt_items:
        p = tf_al.add_paragraph()
        p.text = f"• {t}: "
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_SECONDARY
        p.space_before = Pt(6)
        r = p.add_run()
        r.text = d
        r.font.bold = False
        r.font.color.rgb = COLOR_DARK

    # Coluna Direita: Benchmark Parquet vs CSV
    add_card(s5, Inches(6.88), Inches(1.65), Inches(5.65), Inches(5.3), bg_color=COLOR_GREEN_LIGHT, border_color=COLOR_SECONDARY)
    tb_ar = s5.shapes.add_textbox(Inches(7.13), Inches(1.85), Inches(5.15), Inches(4.9))
    tf_ar = tb_ar.text_frame
    tf_ar.word_wrap = True

    p = tf_ar.paragraphs[0]
    p.text = "⚡ BENCHMARK I/O: PARQUET VS CSV"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_SECONDARY

    bench_items = [
        ("Tamanho em Disco", "Parquet: 916,92 KB vs CSV: 2.989,66 KB\n➔ 3,26x menor em disco (-69,3% de tráfego)"),
        ("Tempo Médio de Leitura (I/O)", "Parquet: 36,19 ms vs CSV: 51,01 ms\n➔ 1,41x mais rápido em loops de Cross-Validation"),
        ("Preservação Estrita de Tipagem", "Parquet armazena nativamente int64 e float64, eliminando erros de parsing."),
        ("Smoke Test Integrado com SVM", "GridSearchCV + SVC(rbf) executou em 0,05s com F1=1,0000 na amostra de teste.")
    ]
    for t, d in bench_items:
        p = tf_ar.add_paragraph()
        p.text = f"★ {t}:\n{d}"
        p.font.size = Pt(11.5)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 6: CONCLUSÃO EXECUTIVA & HANDOFF PARA A SPRINT 3
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Conclusão Executiva • Transição Sprint 3", "100% da Fase Modify Concluída com Prontidão para o SVM")

    add_card(s6, Inches(0.8), Inches(1.65), Inches(11.733), Inches(5.3))
    tb_c = s6.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(11.1), Inches(4.8))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True

    p = tf_c.paragraphs[0]
    p.text = "🏆 RESUMO FINAL DAS ENTREGAS DO DEMODAY 2"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets_final = [
        ("Descritores Validados Agronomicamente", "Índices ExG, ExR, HSV e textura Haralick com significância estatística p < 10⁻¹⁵ e Cohen's d > 5.0."),
        ("Pipeline Atômico Serializado", "Objeto ColumnTransformer salvo em models/preprocessor_pipeline.joblib com schema de governança MLOps."),
        ("Blindagem Anti-Leakage Homologada", "Isolamento amostral e teste adversarial aprovados com 0% de vazamento."),
        ("ABT Final em Alta Performance", "Base abt_features_modelagem.parquet de 916 KB pronta para consumo com zero nulos."),
        ("Handoff Imediato para a Sprint 3 (Model)", "Otimização exaustiva de hiperparâmetros SVM (C, gamma, kernels RBF e Linear) via GridSearchCV.")
    ]
    for tit, dsc in bullets_final:
        p = tf_c.add_paragraph()
        p.text = f"✔ {tit}: "
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_SECONDARY
        p.space_before = Pt(8)
        r = p.add_run()
        r.text = dsc
        r.font.bold = False
        r.font.color.rgb = COLOR_DARK

    p_call = tf_c.add_paragraph()
    p_call.text = "\nSquad 2 • Fase Modify Concluída com Sucesso • Prontos para a Modelagem SVM!"
    p_call.font.size = Pt(14)
    p_call.font.bold = True
    p_call.font.color.rgb = COLOR_PRIMARY
    p_call.alignment = PP_ALIGN.CENTER
    p_call.space_before = Pt(12)

    # Salva
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_path))
    print(f"[OK] Apresentação Executiva Ultra-Resumida salva em: {output_path}")

if __name__ == "__main__":
    out = Path("docs/Apresentacao_Demoday_2_Executiva_Resumida.pptx")
    build_executive_summary_presentation(out)
