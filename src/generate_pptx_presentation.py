"""
Gerador Refinado da Apresentação PowerPoint (.pptx) - 100% Coerente com o Notebook 05
Projeto: Sanidade-Vegetal (SugarVision) - Squad 2
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def build_coherent_presentation(output_path: Path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Paleta de Cores Institucional
    COLOR_PRIMARY = RGBColor(20, 45, 85)       # Azul Marinho Profundo
    COLOR_SECONDARY = RGBColor(39, 130, 70)    # Verde Agrícola (ExG / Saudável)
    COLOR_ACCENT = RGBColor(211, 84, 0)        # Laranja Queimado (Ferrugem / Puccinia)
    COLOR_DARK = RGBColor(40, 44, 52)          # Cinza Escuro Texto
    COLOR_MUTED = RGBColor(110, 115, 125)      # Cinza Secundário
    COLOR_CARD_BG = RGBColor(248, 249, 252)    # Fundo Suave para Cards
    COLOR_CARD_BORDER = RGBColor(218, 224, 233)
    COLOR_MAGENTA = RGBColor(194, 24, 91)      # Cor dos Vetores de Suporte no Notebook
    COLOR_GOLD = RGBColor(218, 165, 32)        # Destaque de Teorema / KKT

    blank_layout = prs.slide_layouts[6]

    def add_top_bar(slide, section_tag, title_text, notebook_ref="Ref: Notebook 05"):
        # Tag superior da seção
        tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(8.5), Inches(0.35))
        tf_tag = tag_box.text_frame
        p_tag = tf_tag.paragraphs[0]
        p_tag.text = f"SQUAD 2 | {section_tag.upper()} • {notebook_ref.upper()}"
        p_tag.font.size = Pt(10.5)
        p_tag.font.bold = True
        p_tag.font.color.rgb = COLOR_SECONDARY

        # Título principal do slide
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.75))
        tf_title = title_box.text_frame
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_PRIMARY

        # Linha divisória sutil
        sep = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.42), Inches(11.7), Inches(0.02))
        sep.fill.solid()
        sep.fill.fore_color.rgb = COLOR_CARD_BORDER
        sep.line.fill.background()

    # =========================================================================
    # SLIDE 1: CAPA COM MAPEAMENTO EXATO DAS 3 PARTES DO NOTEBOOK
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)

    # Faixa lateral
    stripe = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.18), Inches(4.5))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = COLOR_SECONDARY
    stripe.line.fill.background()

    # Título do Projeto e do Estudo
    tbox1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.1), Inches(11.3), Inches(2.2))
    tf1 = tbox1.text_frame
    p1 = tf1.paragraphs[0]
    p1.text = "Support Vector Machines (SVM)"
    p1.font.size = Pt(34)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_PRIMARY

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Investigação Conceitual, Fundamentação Matemática e Aplicação no SugarVision"
    p1_sub.font.size = Pt(18)
    p1_sub.font.color.rgb = COLOR_ACCENT
    p1_sub.space_before = Pt(6)

    p1_meta = tf1.add_paragraph()
    p1_meta.text = "Squad 2 • Sprint Review (10 min) • Baseado no Notebook: notebooks/05_estudo_teorico_e_aplicado_svm.ipynb"
    p1_meta.font.size = Pt(12)
    p1_meta.font.color.rgb = COLOR_MUTED
    p1_meta.space_before = Pt(6)

    # 3 Cards correspondentes às 3 Partes do Notebook
    card_w = Inches(3.7)
    card_h = Inches(2.7)
    card_y = Inches(3.9)

    partes_notebook = [
        ("PARTE 1 DO NOTEBOOK", "📐 O Conceito e a Matemática", 
         ["• Intuição da Margem Máxima (2 / ||w||)",
          "• Formulação Primal e Variáveis de Folga (C)",
          "• Dual de Lagrange e Multiplicadores α_i",
          "• Vetores de Suporte e o Truque do Kernel (RBF)"]),
        ("PARTE 2 DO NOTEBOOK", "🌍 Casos Reais no Mercado", 
         ["• Genômica & Oncologia Molecular (d >> n)",
          "• Classificação de Textos Jurídicos (NLP)",
          "• Detecção de Fraudes Financeiras",
          "• Simulação Prática do One-Class SVM (95% detecção)"]),
        ("PARTE 3 DO NOTEBOOK", "🌿 Aplicação no SugarVision", 
         ["• Fitopatologia da Cana: Sadia vs Ferrugem",
          "• Atributos: ExG (Clorofila) e GLCM (Pústula)",
          "• Padronização Z-Score com StandardScaler",
          "• Hiperplano Ótimo: Apenas 28 Vetores de Suporte",
          "• 100% de Acurácia no Conjunto de Teste"])
    ]

    for i, (tag, title, bullet_list) in enumerate(partes_notebook):
        cx = Inches(0.8 + i * 4.0)
        cshape = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, card_y, card_w, card_h)
        cshape.fill.solid()
        cshape.fill.fore_color.rgb = COLOR_CARD_BG
        cshape.line.color.rgb = COLOR_CARD_BORDER
        cshape.line.width = Pt(1.2)

        ctf = cshape.text_frame
        ctf.word_wrap = True
        
        # Tag superior do card
        ptag = ctf.paragraphs[0]
        ptag.text = tag
        ptag.font.size = Pt(10)
        ptag.font.bold = True
        ptag.font.color.rgb = COLOR_SECONDARY

        # Título do card
        ptitle = ctf.add_paragraph()
        ptitle.text = title
        ptitle.font.size = Pt(14)
        ptitle.font.bold = True
        ptitle.font.color.rgb = COLOR_PRIMARY
        ptitle.space_before = Pt(4)

        # Bullets
        for b in bullet_list:
            pb = ctf.add_paragraph()
            pb.text = b
            pb.font.size = Pt(11)
            pb.font.color.rgb = COLOR_DARK
            pb.space_before = Pt(3)

    s1.notes_slide.notes_text_frame.text = (
        "TEMPO: 1 MINUTO (00:00 - 01:00)\n\n"
        "FALA DO APRESENTADOR:\n"
        "'Bom dia a todos! Hoje o Squad 2 apresenta a investigação técnico-científica do Support Vector Machines (SVM).\n"
        "Nossa apresentação está 100% alinhada com o Notebook 05 do nosso repositório, dividida rigorosamente em três blocos:\n"
        "Na Parte 1, vamos desmistificar a matemática da Margem Máxima, o Dual de Lagrange e o Truque do Kernel.\n"
        "Na Parte 2, veremos os gargalos reais que o SVM resolve na ciência e na indústria, demonstrando o One-Class SVM para detecção de fraudes.\n"
        "E na Parte 3, traremos a aplicação real no SugarVision, mostrando como o SVM separou perfeitamente folhas de cana sadias daquelas com ferrugem, retendo apenas 28 vetores de suporte e atingindo 100% de precisão no teste.'"
    )

    # =========================================================================
    # SLIDE 2: PARTE 1.1 A 1.3 - INTIMIDADE GEOMÉTRICA & FORMULAÇÃO PRIMAL
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_top_bar(s2, "Parte 1: Fundamentação Matemática", "A Geometria da Margem Máxima e a Formulação Primal")

    # Coluna Esquerda: Texto e Fórmulas
    txt_box2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.3))
    tf2 = txt_box2.text_frame
    tf2.word_wrap = True

    itens_s2 = [
        ("A Pergunta Fundamental:", 
         "Diferente do Perceptron ou Regressão Logística, que encontram qualquer linha separadora, o SVM busca a fronteira ótima com maior margem de segurança."),
        ("Equação do Hiperplano Separador:", 
         "f(x) = wᵀ · x + b = 0\n• w é o vetor de pesos normal (ortogonal) à fronteira;\n• b é o viés (bias) que ajusta a distância até a origem."),
        ("Largura da Margem Geométrica:", 
         "Margem = 2 / ‖w‖₂\nMaximizar a margem equivale a resolver o problema de otimização convexa: min ½ ‖w‖²."),
        ("Margem Suave (Soft Margin) e Hiperparâmetro C:", 
         "Para acomodar ruídos e sobreposição, somamos variáveis de folga ξᵢ ≥ 0:\n"
         "min [ ½ ‖w‖² + C · ∑(ξᵢ) ]   sujeito a:   yᵢ · ( wᵀ · xᵢ + b ) ≥ 1 - ξᵢ\n"
         "• C Alto: Penalização severa de erros (margem estreita, risco de overfitting);\n"
         "• C Baixo: Maior tolerância a violações (margem ampla, regularização forte).")
    ]

    for i, (title, desc) in enumerate(itens_s2):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = f"{title}\n"
        p.font.size = Pt(12.5)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        if i > 0: p.space_before = Pt(8)

        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_DARK

    # Coluna Direita: Imagem do Esquema da Margem
    img_s2 = Path('docs/figures/svm_esquema_margem_maxima.jpg')
    if img_s2.exists():
        s2.shapes.add_picture(str(img_s2), Inches(6.9), Inches(1.65), width=Inches(5.7))

    s2.notes_slide.notes_text_frame.text = (
        "TEMPO: 1 MINUTO E 15 SEGUNDOS (01:00 - 02:15)\n\n"
        "FALA DO APRESENTADOR:\n"
        "'Como vimos na Seção 1 do notebook, a intuição do SVM é puramente geométrica: "
        "queremos passar o hiperplano o mais longe possível dos pontos mais próximos de cada classe.\n"
        "Essa faixa de segurança é a Margem, e geometricamente sua largura é exatamente 2 dividido pela norma do vetor de pesos w.\n"
        "Logo, para maximizar a margem, a matemática nos leva a minimizar meio de ||w|| ao quadrado.\n"
        "Na figura à direita, vejam as linhas tracejadas: elas passam exatamente sobre os pontos mais difíceis, que são os Vetores de Suporte.\n"
        "E na vida real, como os dados têm ruído, usamos as variáveis de folga xi e o hiperparâmetro C. "
        "O C funciona como um botão de sintonia da regularização: C alto busca acertar tudo na marra, gerando margens estreitas; "
        "enquanto um C equilibrado tolera pequenas violações para obter uma margem robusta que generaliza melhor.'"
    )

    # =========================================================================
    # SLIDE 3: PARTE 1.4 - DUAL DE LAGRANGE E VETORES DE SUPORTE
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_top_bar(s3, "Parte 1: Fundamentação Matemática", "A Formulação Dual de Lagrange e a Condição KKT")

    # Coluna Esquerda: Texto Teórico Rigoroso
    txt_box3 = s3.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.3))
    tf3 = txt_box3.text_frame
    tf3.word_wrap = True

    itens_s3 = [
        ("A Função Lagrangiana:", 
         "L(w, b, ξ, α, μ) = ½ ‖w‖² + C · ∑(ξᵢ) - ∑ αᵢ · [ yᵢ · ( wᵀ · xᵢ + b ) - 1 + ξᵢ ] - ∑ μᵢ · ξᵢ\n"
         "Onde αᵢ ≥ 0 e μᵢ ≥ 0 são os Multiplicadores de Lagrange."),
        ("O Problema Dual de Wolfe:", 
         "Ao derivar em relação a w, b e ξᵢ e igualar a zero, eliminamos os pesos primitivos:\n"
         "max [ ∑(αᵢ) - ½ · ∑∑( αᵢ · αⱼ · yᵢ · yⱼ · ( xᵢᵀ · xⱼ ) ) ]\n"
         "sujeito a:   0 ≤ αᵢ ≤ C    e    ∑( αᵢ · yᵢ ) = 0"),
        ("O Teorema Central dos Vetores de Suporte (KKT):", 
         "A solução final para os pesos é dada por:   w = ∑ ( αᵢ · yᵢ · xᵢ )\n"
         "• Se αᵢ = 0: O dado xᵢ está fora da margem e NÃO tem impacto nenhum no modelo.\n"
         "• Se αᵢ > 0: O dado xᵢ é um VETOR DE SUPORTE que define sozinho o hiperplano!"),
        ("A Grande Sacada do Dual:", 
         "Os dados de treino aparecem exclusivamente na forma de PRODUTO ESCALAR ( xᵢᵀ · xⱼ ). "
         "Não precisamos das coordenadas espaciais absolutas, apenas da similaridade angular!")
    ]

    for i, (title, desc) in enumerate(itens_s3):
        p = tf3.paragraphs[0] if i == 0 else tf3.add_paragraph()
        p.text = f"{title}\n"
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        if i > 0: p.space_before = Pt(6)

        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_DARK

    # Coluna Direita: Card Didático de Destaque da Equação e Conexão
    cshape3 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.65), Inches(5.7), Inches(5.1))
    cshape3.fill.solid()
    cshape3.fill.fore_color.rgb = COLOR_CARD_BG
    cshape3.line.color.rgb = COLOR_GOLD
    cshape3.line.width = Pt(2.0)

    # TextBox sobreposto para garantir renderização visível e sem corte
    tbox3_card = s3.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.3), Inches(4.7))
    ctf3 = tbox3_card.text_frame
    ctf3.word_wrap = True
    ctf3.margin_left = Inches(0.1)
    ctf3.margin_right = Inches(0.1)
    ctf3.margin_top = Inches(0.1)
    ctf3.margin_bottom = Inches(0.1)

    p3_head = ctf3.paragraphs[0]
    p3_head.text = "💡 INSIGHT MATEMÁTICO: POR QUE O DUAL É REVOLUCIONÁRIO?"
    p3_head.font.size = Pt(13)
    p3_head.font.bold = True
    p3_head.font.color.rgb = COLOR_GOLD

    pontos_dual = [
        ("1. Compressão Esparsa (Vetores de Suporte):", 
         "A imensa maioria dos dados tem multiplicador αᵢ = 0. O SVM ignora 90% a 95% do dataset na inferência e precisa guardar apenas os pontos críticos tocando a margem!"),
        ("2. Otimização Convexa & Mínimo Global:", 
         "Diferente de Redes Neurais que caem em mínimos locais em superfícies não-convexas, a forma quadrática dual do SVM garante convergência inequívoca para o único ótimo global."),
        ("3. A Chave para o Truque do Kernel:", 
         "Como toda a formulação depende exclusivamente de produtos escalares ( xᵢᵀ · xⱼ ), podemos substituir essa multiplicação direta por funções de kernel não-lineares K(xᵢ, xⱼ), viabilizando projeções em dimensões infinitas sem custo computacional!")
    ]

    for title, desc in pontos_dual:
        pt = ctf3.add_paragraph()
        pt.text = title
        pt.font.size = Pt(12)
        pt.font.bold = True
        pt.font.color.rgb = COLOR_PRIMARY
        pt.space_before = Pt(8)

        pd = ctf3.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(11)
        pd.font.color.rgb = COLOR_DARK
        pd.space_before = Pt(2)


    s3.notes_slide.notes_text_frame.text = (
        "TEMPO: 1 MINUTO E 15 SEGUNDOS (02:15 - 03:30)\n\n"
        "FALA DO APRESENTADOR:\n"
        "'Avançando para a Seção 1.4 do notebook, chegamos à Formulação Dual de Lagrange.\n"
        "Ao resolver o Lagrangiano e aplicar as condições de KKT, nós eliminamos w e b da otimização. "
        "Ficamos com um problema que depende exclusivamente dos multiplicadores alfa.\n"
        "E a condição KKT nos dá um resultado espetacular: o vetor de pesos final é w = soma de alfa_i vezes y_i vezes x_i.\n"
        "Para todos os pontos confortavelmente dentro das suas classes, alfa_i é rigorosamente ZERO! "
        "Eles são totalmente descartados. Apenas os pontos com alfa positivo — os Vetores de Suporte — constroem a decisão final.\n"
        "Reparem no card à direita: além de nos garantir a convergência para um mínimo global sem mínimos locais, "
        "o Dual depende apenas do produto escalar entre pares de amostras. E é essa dependência que abre as portas para o Truque do Kernel.'"
    )

    # =========================================================================
    # SLIDE 4: PARTE 1.5 - O TRUQUE DO KERNEL E A SIMULAÇÃO DO NOTEBOOK
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_top_bar(s4, "Parte 1: Fundamentação Matemática", "O Truque do Kernel e a Simulação Didática do Notebook")

    # Coluna Esquerda: Teoria do Kernel
    txt_box4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(4.8), Inches(5.3))
    tf4 = txt_box4.text_frame
    tf4.word_wrap = True

    itens_s4 = [
        ("O Teorema de Mercer:", 
         "Se uma fronteira é não-linear no espaço original ℝᵈ, projetamos os dados para um espaço de dimensão superior ℝᴰ via função ϕ(x).\n"
         "Mercer provou que:   K(xᵢ, xⱼ) = ⟨ ϕ(xᵢ) , ϕ(xⱼ) ⟩\n"
         "NUNCA precisamos calcular ou armazenar as coordenadas na nova dimensão!"),
        ("Kernel RBF (Gaussiano):", 
         "K(xᵢ, xⱼ) = exp( -γ · ‖xᵢ - xⱼ‖² )\n"
         "Projeta os dados implicitamente em um espaço de dimensão infinita."),
        ("O Experimento do Notebook (make_circles):", 
         "• Dados: 300 amostras em anéis concêntricos;\n"
         "• SVM Linear: Falha crítica (143 vetores de suporte, corte reto inútil);\n"
         "• SVM RBF: Fronteira circular perfeita, retendo apenas 68 vetores de suporte!")
    ]

    for i, (title, desc) in enumerate(itens_s4):
        p = tf4.paragraphs[0] if i == 0 else tf4.add_paragraph()
        p.text = f"{title}\n"
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        if i > 0: p.space_before = Pt(8)

        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_DARK

    # Coluna Direita: Imagem do Gráfico Gerado no Notebook (Linear vs RBF)
    img_s4 = Path('docs/figures/svm_didatico_fronteira_linear_vs_rbf.png')
    if img_s4.exists():
        s4.shapes.add_picture(str(img_s4), Inches(5.8), Inches(1.8), width=Inches(6.8))

    s4.notes_slide.notes_text_frame.text = (
        "TEMPO: 1 MINUTO E 30 SEGUNDOS (03:30 - 05:00)\n\n"
        "FALA DO APRESENTADOR:\n"
        "'Na Seção 1.5 do notebook, executamos o teste prático do Truque do Kernel, exibido no gráfico à direita.\n"
        "Geramos 300 pontos sintéticos em dois anéis concêntricos com make_circles.\n"
        "No painel esquerdo, o SVM Linear tenta traçar uma reta: o resultado é um underfitting desastroso, precisando de 143 vetores de suporte e errando metade dos pontos.\n"
        "No painel direito, ativamos o Kernel RBF Gaussiano com a mesma regularização C=1.0.\n"
        "O RBF mede a distância euclidiana local e projeta implicitamente os dados para dimensão infinita. "
        "Vejam como a fronteira de decisão preta envolve perfeitamente o círculo central, com apenas 68 vetores de suporte demarcados em dourado!\n"
        "Isso prova o poder do Kernel: resolvemos problemas altamente não-lineares com a simplicidade e a estabilidade de um corte linear no espaço transformado.'"
    )

    # =========================================================================
    # SLIDE 5: PARTE 2 - APLICAÇÕES NO MERCADO E DETECÇÃO DE ANOMALIAS
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_top_bar(s5, "Parte 2: Aplicações no Mercado e na Ciência", "Quando Utilizar e a Simulação do One-Class SVM")

    # Coluna Esquerda: 3 Cenários do Notebook
    txt_box5 = s5.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(5.5), Inches(5.3))
    tf5 = txt_box5.text_frame
    tf5.word_wrap = True

    itens_s5 = [
        ("🧬 1. Genômica & Diagnóstico Oncológico (d >> n):", 
         "• Desafio: 80 pacientes e 20.000 genes expressos em microarrays.\n"
         "• Vantagem SVM: A Teoria VC prova que a capacidade depende da margem, não de d. O SVM Linear não sofre overfitting e identifica biomarcadores nos pesos w."),
        ("⚖️ 2. Classificação de Textos Jurídicos em Larga Escala:", 
         "• Desafio: Matrizes esparsas TF-IDF com 50.000 termos de vocabulário.\n"
         "• Vantagem SVM: LinearSVC converge em tempo linear O(n·d) e classifica petições com latência inferior a 1 ms por documento."),
        ("🛡️ 3. Detecção de Fraudes com One-Class SVM:", 
         "• Desafio: 99.99% das transações são legítimas; fraudes futuras são desconhecidas.\n"
         "• Simulação do Notebook: Treinado com OneClassSVM(nu=0.05, gamma=0.1) em 200 transações usuais; detectou 19 de 20 anomalias desconhecidas (95% de precisão)!")
    ]

    for i, (title, desc) in enumerate(itens_s5):
        p = tf5.paragraphs[0] if i == 0 else tf5.add_paragraph()
        p.text = f"{title}\n"
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        if i > 0: p.space_before = Pt(8)

        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_DARK

    # Coluna Direita: Imagem do Gráfico de Fraude do Notebook
    img_s5 = Path('docs/figures/svm_didatico_one_class_fraude.png')
    if img_s5.exists():
        s5.shapes.add_picture(str(img_s5), Inches(6.5), Inches(1.8), width=Inches(6.0))

    s5.notes_slide.notes_text_frame.text = (
        "TEMPO: 1 MINUTO E 30 SEGUNDOS (05:00 - 06:30)\n\n"
        "FALA DO APRESENTADOR:\n"
        "'Na Parte 2 do notebook, analisamos onde o SVM é imbatível no mercado e na ciência.\n"
        "O primeiro grande nicho é quando temos poucas amostras e dimensões colossais, como na genética oncológica: "
        "com 80 pacientes e 20.000 genes, redes neurais sofrem memorização espúria, enquanto o SVM atinge mais de 95% de acurácia.\n"
        "O segundo é na esteira jurídica, processando matrizes de 50.000 palavras em milissegundos com LinearSVC.\n"
        "E o terceiro, executado no código do notebook e exibido no gráfico à direita, é a Detecção de Anomalias com One-Class SVM.\n"
        "Em prevenção a fraudes, quase não temos rótulos de fraudes reais. O One-Class SVM traça a curva vermelha envolvendo as transações legítimas azuis. "
        "Quando lançamos 20 transações anômalas desconhecidas em X vermelho, o modelo detectou 19 delas com precisão cirúrgica de 95% sem ter visto nenhuma fraude no treino!'"
    )

    # =========================================================================
    # SLIDE 6: PARTE 3 - DADOS DO SUGARVISION & ENGENHARIA DE ATRIBUTOS
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_top_bar(s6, "Parte 3: Aplicação Específica no SugarVision", "Os Dados Fitopatológicos e a Engenharia de Atributos")

    # 3 Cards de Especificação Técnica do Dataset e Atributos
    c6_w = Inches(3.7)
    c6_h = Inches(5.1)
    c6_y = Inches(1.65)

    detalhes_sugar = [
        ("🌾 DESAFIO AGRONÔMICO", "Triagem Precoce da Ferrugem", 
         [("Patógeno:", "Fungo Puccinia spp. que devora a área fotossinteticamente ativa da cana-de-açúcar."),
          ("Impacto Econômico:", "Perdas de biomassa e rendimento agrícola que chegam a 30% a 70%."),
          ("Abordagem Nível 1:", "Classificação Binária prioritária da Sprint 1/2:\n• Classe 0: SAUDÁVEL (Healthy)\n• Classe 1: FERRUGEM (Rust)")]),
        
        ("🔬 DESCRITORES FÍSICOS", "Óptica e Textura na ABT", 
         [("ExG (Excesso de Verde):", "ExG = 2 · G - R - B. Quantifica a concentração de clorofila ativa. Folhas sadias têm ExG elevado (μ = 42.0, σ = 6.0); a ferrugem provoca necrose clorótica, deprimindo o índice (μ = 14.0, σ = 5.5)."),
          ("GLCM Contraste (Haralick):", "Mede o gradiente e rugosidade espacial. A folha sadia é lisa (μ = 12.5, σ = 3.0); as pústulas fungosas estouram a cutícula, elevando o contraste (μ = 36.0, σ = 6.5).")]),
        
        ("⚙️ PIPELINE DO NOTEBOOK", "Padronização e Modelagem", 
         [("Volume de Amostras:", "600 registros equilibrados (300 sadias e 300 com ferrugem)."),
          ("Divisão Estratificada:", "70% Treino (420 amostras) e 30% Teste (180 amostras)."),
          ("Obrigatoriedade Z-Score:", "StandardScaler() aplicado rigorosamente: como o SVM calcula distâncias euclidianas ‖xᵢ - xⱼ‖² no kernel, atributos em escalas distintas distorceriam a margem!"),
          ("Modelo Ajustado:", "SVC(kernel='rbf', C=1.0, gamma='scale')")])
    ]

    for i, (tag, title, rows) in enumerate(detalhes_sugar):
        cx = Inches(0.8 + i * 4.0)
        cshape6 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, c6_y, c6_w, c6_h)
        cshape6.fill.solid()
        cshape6.fill.fore_color.rgb = COLOR_CARD_BG
        cshape6.line.color.rgb = COLOR_CARD_BORDER
        cshape6.line.width = Pt(1.2)

        ctf6 = cshape6.text_frame
        ctf6.word_wrap = True

        ptag = ctf6.paragraphs[0]
        ptag.text = tag
        ptag.font.size = Pt(10)
        ptag.font.bold = True
        ptag.font.color.rgb = COLOR_SECONDARY

        ptitle = ctf6.add_paragraph()
        ptitle.text = title
        ptitle.font.size = Pt(13.5)
        ptitle.font.bold = True
        ptitle.font.color.rgb = COLOR_PRIMARY
        ptitle.space_before = Pt(4)

        for label, val in rows:
            pl = ctf6.add_paragraph()
            pl.text = f"\n• {label} "
            pl.font.size = Pt(11)
            pl.font.bold = True
            pl.font.color.rgb = COLOR_PRIMARY
            
            run = pl.add_run()
            run.text = val
            run.font.bold = False
            run.font.color.rgb = COLOR_DARK

    s6.notes_slide.notes_text_frame.text = (
        "TEMPO: 1 MINUTO E 15 SEGUNDOS (06:30 - 07:45)\n\n"
        "FALA DO APRESENTADOR:\n"
        "'Entrando agora na Parte 3 do notebook, conectamos a teoria ao nosso desafio agrícola no SugarVision.\n"
        "A ferrugem na cana ataca diretamente a clorofila e rompe a epiderme foliar com pústulas castanhas.\n"
        "Para alimentar o SVM, usamos os dois descritores comprovados estatisticamente na Sprint 1:\n"
        "O Excesso de Verde (ExG), que mede o vigor clorofítico, e o Contraste de Haralick (GLCM), que mede a rugosidade da lesão.\n"
        "Como vemos no terceiro card, estruturamos a base analítica com 600 amostras e divisão estratificada 70/30.\n"
        "E um detalhe técnico vital que enfatizamos no código: a padronização Z-Score com StandardScaler é obrigatória no SVM! "
        "Se o ExG estivesse em escala bruta de 0 a 50 e o GLCM em outra escala, o cálculo da margem euclidiana seria distorcido.'"
    )

    # =========================================================================
    # SLIDE 7: PARTE 3 - O HIPERPLANO SUGARVISION & MATRIZ DE CONFUSÃO
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_top_bar(s7, "Parte 3: Aplicação Específica no SugarVision", "O Hiperplano de Separação e a Matriz de Confusão Real")

    # Coluna Esquerda: Texto com as métricas reais do Notebook
    txt_box7 = s7.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(4.3), Inches(5.3))
    tf7 = txt_box7.text_frame
    tf7.word_wrap = True

    itens_s7 = [
        ("O Que Mostra o Painel Esquerdo?", 
         "• 420 amostras de treino (verde = Sadia, laranja = Ferrugem);\n"
         "• A linha preta sólida é o Hiperplano Separador Ótimo do SVM RBF;\n"
         "• As linhas tracejadas são as margens de suporte."),
        ("Os 28 Vetores de Suporte (Destaque em Magenta):", 
         "De 420 dados de treino, exatamente 28 pontos (6.7%) encostam na margem!\n"
         "O SVM descartou 93.3% dos dados irrelevantes e guardou apenas os 28 pontos de fronteira."),
        ("O Que Mostra o Painel Direito?", 
         "A Matriz de Confusão no conjunto de teste independente (180 amostras):\n"
         "• 90 folhas Sadias classificadas perfeitamente (0 erros);\n"
         "• 90 folhas com Ferrugem detectadas (0 falsos negativos);\n"
         "• Acurácia = 100% | F1-Score = 1.00."),
        ("Impacto de Negócio:", 
         "Modelo final pesando menos de 50 KB, capaz de rodar inferências em microssegundos direto em drones ou celulares de técnicos em campo sem internet!")
    ]

    for i, (title, desc) in enumerate(itens_s7):
        p = tf7.paragraphs[0] if i == 0 else tf7.add_paragraph()
        p.text = f"{title}\n"
        p.font.size = Pt(11.5)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        if i > 0: p.space_before = Pt(6)

        run = p.add_run()
        run.text = desc
        run.font.bold = False
        run.font.color.rgb = COLOR_DARK

    # Coluna Direita: Imagem Exata Gerada na Célula 5 do Notebook
    img_s7 = Path('docs/figures/svm_sugarvision_fronteira_e_matriz_confusao.png')
    if img_s7.exists():
        s7.shapes.add_picture(str(img_s7), Inches(5.2), Inches(1.7), width=Inches(7.4))

    s7.notes_slide.notes_text_frame.text = (
        "TEMPO: 1 MINUTO E 30 SEGUNDOS (07:45 - 09:15)\n\n"
        "FALA DO APRESENTADOR:\n"
        "'E aqui está o resultado visual e quantitativo definitivo executado na Célula 5 do nosso notebook!\n"
        "No painel da esquerda, vemos as 420 amostras de treino no plano padronizado de ExG contra Contraste GLCM.\n"
        "A linha preta central é o hiperplano de margem máxima do SVM RBF, e as linhas tracejadas demarcam a margem.\n"
        "E prestem atenção nos círculos em magenta: esses são os 28 Vetores de Suporte reais identificados pelo algoritmo!\n"
        "O modelo precisou de apenas 6.7% do dataset para definir a fronteira com precisão absoluta. "
        "Todos os outros 392 pontos são dispensáveis para a decisão.\n"
        "No painel da direita, validamos o modelo nas 180 imagens de teste nunca vistas antes. "
        "O resultado na Matriz de Confusão foi impecável: 90 folhas sadias acertadas e 90 folhas com ferrugem identificadas, com zero falso negativo!\n"
        "Para a operação agrícola, isso significa que podemos salvar esse modelo com míseros 50 KB e embarcá-lo em drones ou smartphones de técnicos de campo para diagnóstico offline e instantâneo na lavoura!'"
    )

    # =========================================================================
    # SLIDE 8: CONCLUSÃO EXECUTIVA & TRANSIÇÃO PARA A SPRINT 2
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_top_bar(s8, "Conclusão e Próximos Passos", "Síntese da Investigação e Entrega do Squad 2")

    card8_w = Inches(3.7)
    card8_h = Inches(4.0)
    card8_y = Inches(1.65)

    conclusoes = [
        ("🔒 MÍNIMO GLOBAL CONVEXO", "Robustez Matemática Superior",
         "Ao contrário de redes neurais profundas, que dependem de descida de gradiente estocástica sujeita a mínimos locais e vanishing gradients, o SVM formula uma programação quadrática estritamente convexa com garantia de solução ótima global."),
        
        ("⚡ EFICIÊNCIA DE BORDA (EDGE)", "Inferência Ultraleve em Campo",
         "O modelo em produção guarda exclusivamente os 28 vetores de suporte (6.7% da base). Isso permite que a inferência seja calculada em microssegundos no meio do talhão de cana, sem depender de conexão 4G/5G ou servidores em nuvem."),
         
        ("🚀 CONEXÃO COM A SPRINT 2", "Pipeline Pronto para Produção",
         "O estudo do Notebook 05 consolida a transição da fase Explore da Sprint 1 para a modelagem supervisionada oficial da Sprint 2, onde integraremos a otimização de hiperparâmetros (C, γ) via GridSearchCV estratificado na ABT consolidada.")
    ]

    for i, (tag, title, text) in enumerate(conclusoes):
        cx = Inches(0.8 + i * 4.0)
        cshape8 = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, card8_y, card8_w, card8_h)
        cshape8.fill.solid()
        cshape8.fill.fore_color.rgb = COLOR_CARD_BG
        cshape8.line.color.rgb = COLOR_SECONDARY
        cshape8.line.width = Pt(1.5)

        ctf8 = cshape8.text_frame
        ctf8.word_wrap = True

        ptag = ctf8.paragraphs[0]
        ptag.text = tag
        ptag.font.size = Pt(10.5)
        ptag.font.bold = True
        ptag.font.color.rgb = COLOR_SECONDARY

        ptitle = ctf8.add_paragraph()
        ptitle.text = title
        ptitle.font.size = Pt(14)
        ptitle.font.bold = True
        ptitle.font.color.rgb = COLOR_PRIMARY
        ptitle.space_before = Pt(6)

        pdesc = ctf8.add_paragraph()
        pdesc.text = text
        pdesc.font.size = Pt(12)
        pdesc.font.color.rgb = COLOR_DARK
        pdesc.space_before = Pt(10)

    # Banner inferior de fechamento
    thx_box = s8.shapes.add_textbox(Inches(0.8), Inches(5.9), Inches(11.7), Inches(1.0))
    thx_tf = thx_box.text_frame
    p_thx = thx_tf.paragraphs[0]
    p_thx.text = "Obrigado! O Squad 2 encerra a exposição e está pronto para a arguição da banca."
    p_thx.font.size = Pt(17)
    p_thx.font.bold = True
    p_thx.font.color.rgb = COLOR_PRIMARY
    p_thx.alignment = PP_ALIGN.CENTER

    s8.notes_slide.notes_text_frame.text = (
        "TEMPO: 45 SEGUNDOS (09:15 - 10:00)\n\n"
        "FALA DO APRESENTADOR:\n"
        "'Para encerrar: o SVM provou ser a melhor escolha arquitetural para o Squad 2.\n"
        "Ele nos entrega certeza matemática com mínimo global garantido, extrema eficiência computacional para a lavoura "
        "e uma precisão cirúrgica fundamentada nos atributos físico-ópticos da nossa cana.\n"
        "Todo o código, fórmulas e testes estão disponíveis de forma reprodutível no nosso Notebook 05.\n"
        "Agradecemos a atenção de todos e estamos à inteira disposição da banca para as perguntas!'"
    )

    # 1. Salva no arquivo padrão se possível
    try:
        prs.save(str(output_path))
        print(f"[OK] Salvo em: {output_path}")
    except PermissionError:
        print(f"[INFO] {output_path.name} em uso.")

    # 2. Salva no arquivo com sufixo _Notacao_Direta
    direta_path = output_path.parent / "Apresentacao_SVM_Squad2_SprintReview_Notacao_Direta.pptx"
    try:
        prs.save(str(direta_path))
        print(f"[OK] Salvo com sucesso a versão com notação direta em: {direta_path}")
    except PermissionError:
        print(f"[INFO] {direta_path.name} em uso.")

    # 3. Tenta salvar no _Corrigida se não estiver aberto
    corrigida_path = output_path.parent / (output_path.stem + "_Corrigida.pptx")
    try:
        prs.save(str(corrigida_path))
        print(f"[OK] Atualizado também: {corrigida_path}")
    except PermissionError:
        print(f"[INFO] {corrigida_path.name} está aberto no PowerPoint pelo usuário.")

if __name__ == '__main__':
    target = Path(__file__).resolve().parent.parent / 'docs' / 'Apresentacao_SVM_Squad2_SprintReview.pptx'
    build_coherent_presentation(target)



