"""
Gerador de Documento Editável Word (.docx)
Título: Diagnóstico Inteligente na Cana-de-Açúcar (Transição da Fase Modify para Model)
Projeto: Sanidade-Vegetal (SugarVision) — Framework SEMMA
"""

import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

# Cores institucionais
COLOR_NAVY = RGBColor(20, 45, 85)       # #142D55
COLOR_GREEN = RGBColor(39, 130, 70)     # #278246
COLOR_DARK_GREEN = RGBColor(20, 83, 45) # #14532D
COLOR_ORANGE = RGBColor(211, 84, 0)     # #D35400
COLOR_TEXT = RGBColor(40, 44, 52)       # #282C34
COLOR_MUTED = RGBColor(100, 116, 139)   # #64748B
COLOR_CODE = RGBColor(30, 41, 59)       # #1E293B

def set_cell_background(cell, hex_color: str):
    """Aplica cor de fundo a uma célula."""
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def set_cell_margins(cell, top=140, bottom=140, left=180, right=180):
    """Configura margens internas de uma célula em dxa."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_borders(cell, top="none", bottom="none", left="none", right="none", 
                     color="CCCCCC", sz="4"):
    """Configura bordas específicas de uma célula."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    borders_dict = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
    for border_name, border_style in borders_dict.items():
        if border_style != "none":
            b_el = OxmlElement(f'w:{border_name}')
            b_el.set(qn('w:val'), border_style)
            b_el.set(qn('w:sz'), sz)
            b_el.set(qn('w:space'), '0')
            b_el.set(qn('w:color'), color)
            tcBorders.append(b_el)
        else:
            b_el = OxmlElement(f'w:{border_name}')
            b_el.set(qn('w:val'), 'none')
            tcBorders.append(b_el)
    tcPr.append(tcBorders)

def add_callout_box(doc, title: str, text: str, bg_hex="F0FDF4", border_color="22C55E"):
    """Adiciona caixa de destaque com barra lateral colorida."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.8)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)
    set_cell_borders(cell, left="single", color=border_color, sz="24") # barra lateral de 3pt

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15

    r_title = p.add_run(f"{title}\n")
    r_title.font.name = "Calibri"
    r_title.font.bold = True
    r_title.font.size = Pt(10.5)
    r_title.font.color.rgb = COLOR_DARK_GREEN

    r_text = p.add_run(text)
    r_text.font.name = "Calibri"
    r_text.font.size = Pt(10)
    r_text.font.color.rgb = COLOR_TEXT

    # Espaçamento após tabela
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(4)

def add_code_block(doc, filename_tag: str, code_text: str, caption: str = ""):
    """Adiciona um bloco de código formatado e sombreado."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.8)
    set_cell_background(cell, "F8FAFC")
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    set_cell_borders(cell, top="single", bottom="single", left="single", right="single", color="CBD5E1", sz="6")

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.05

    # Cabeçalho do arquivo
    r_tag = p.add_run(f"📁 {filename_tag}\n")
    r_tag.font.name = "Consolas"
    r_tag.font.bold = True
    r_tag.font.size = Pt(8.5)
    r_tag.font.color.rgb = RGBColor(14, 116, 144) # teal

    # Código em Consolas
    r_code = p.add_run(code_text.strip())
    r_code.font.name = "Consolas"
    r_code.font.size = Pt(8.5)
    r_code.font.color.rgb = COLOR_CODE

    if caption:
        p_cap = cell.add_paragraph()
        p_cap.paragraph_format.space_before = Pt(4)
        p_cap.paragraph_format.space_after = Pt(0)
        r_cap = p_cap.add_run(f"💡 {caption}")
        r_cap.font.name = "Calibri"
        r_cap.font.italic = True
        r_cap.font.size = Pt(8.5)
        r_cap.font.color.rgb = COLOR_MUTED

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(6)

def build_document(output_path: Path):
    doc = docx.Document()

    # Configuração de Margens (A4 / 2.5 cm em todas as bordas)
    for section in doc.sections:
        section.top_margin = Inches(0.9)
        section.bottom_margin = Inches(0.9)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)

        # Cabeçalho do documento
        header = section.header
        p_head = header.paragraphs[0]
        p_head.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r_h1 = p_head.add_run("🌿 SugarVision AI • Transição Modify → Model (Framework SEMMA)")
        r_h1.font.name = "Calibri"
        r_h1.font.size = Pt(8.5)
        r_h1.font.color.rgb = COLOR_MUTED

        # Rodapé do documento
        footer = section.footer
        p_foot = footer.paragraphs[0]
        p_foot.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r_f1 = p_foot.add_run("Copyright © 2026 SugarVision — Heitor Vitti, César Bovo, Guilherme Romanzotti, Marvin Cristhian, Elisa Alcantara e Luis Otávio.")
        r_f1.font.name = "Calibri"
        r_f1.font.size = Pt(8)
        r_f1.font.color.rgb = COLOR_MUTED

    # =========================================================================
    # BANNER DE TÍTULO EXECUTIVO
    # =========================================================================
    tbl_title = doc.add_table(rows=1, cols=1)
    tbl_title.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_t = tbl_title.cell(0, 0)
    c_t.width = Inches(6.8)
    set_cell_background(c_t, "F0FDF4")
    set_cell_margins(c_t, top=180, bottom=180, left=220, right=220)
    set_cell_borders(c_t, top="single", bottom="single", left="single", right="single", color="BBF7D0", sz="12")

    p_t = c_t.paragraphs[0]
    p_t.paragraph_format.space_before = Pt(2)
    p_t.paragraph_format.space_after = Pt(2)
    p_t.paragraph_format.line_spacing = 1.15

    r_pre = p_t.add_run("RELATO TÉCNICO DE TRANSIÇÃO METODOLÓGICA\n")
    r_pre.font.name = "Calibri"
    r_pre.font.bold = True
    r_pre.font.size = Pt(9.5)
    r_pre.font.color.rgb = RGBColor(22, 101, 52)

    r_main = p_t.add_run("Diagnóstico Inteligente na Cana-de-Açúcar\n")
    r_main.font.name = "Calibri"
    r_main.font.bold = True
    r_main.font.size = Pt(17)
    r_main.font.color.rgb = COLOR_NAVY

    r_sub = p_t.add_run("(Transição da Fase Modify para Model — Framework SEMMA)")
    r_sub.font.name = "Calibri"
    r_sub.font.italic = True
    r_sub.font.size = Pt(11.5)
    r_sub.font.color.rgb = COLOR_GREEN

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(4)
    p_spacer.paragraph_format.space_after = Pt(4)

    # =========================================================================
    # OBJETIVO
    # =========================================================================
    add_callout_box(
        doc,
        title="🎯 Objetivo do Relato:",
        text="Explicar como o projeto saiu do preparo dos dados foliares e entrou na fase de construção e validação da inteligência artificial para o campo.",
        bg_hex="F0FDF4",
        border_color="16A34A"
    )

    # =========================================================================
    # PASSO 1
    # =========================================================================
    p1_h = doc.add_heading("Passo 1: O Contexto e a Virada de Chave (Modify → Model)", level=1)
    p1_h.paragraph_format.space_before = Pt(10)
    p1_h.paragraph_format.space_after = Pt(4)
    for r in p1_h.runs:
        r.font.name = "Calibri"
        r.font.bold = True
        r.font.color.rgb = COLOR_NAVY

    p1_txt = doc.add_paragraph()
    p1_txt.paragraph_format.space_after = Pt(4)
    p1_txt.paragraph_format.line_spacing = 1.15
    p1_txt.add_run("• Ideia central: ").bold = True
    p1_txt.add_run("Entender o ciclo de evolução do projeto — de organizar os dados para ensinar a máquina a pensar.\n")
    p1_txt.add_run("• O Ciclo do Projeto (Metodologia SEMMA): ").bold = True
    p1_txt.add_run("O projeto segue uma trilha de engenharia de dados com cinco etapas: Amostragem, Exploração, Modificação (Modify), Modelar (Model) e Avaliar.\n")
    p1_txt.add_run("• O que foi concluído na fase Modify: ").bold = True
    p1_txt.add_run("Na etapa anterior (Modify), o time organizou, limpou e padronizou uma base com 6.571 fotos de folhas de cana. Dessas fotos, foram extraídas 35 medidas visuais (cor, brilho, rugosidade e textura) sem distorções estatísticas.\n")
    p1_txt.add_run("• A Transição para a fase Model: ").bold = True
    p1_txt.add_run("Este relato marca exatamente a virada de chave para a fase de Modelagem (Model). O foco deixou de ser arrumar a tabela e passou a ser: qual inteligência artificial aprende melhor a diagnosticar as 7 condições da lavoura (folha sadia e 6 doenças)?.")

    # Código do Passo 1
    code_p1 = """# src/atomic_pipeline_preprocessing.py — Taxonomia de Atributos da ABT Consolidada (6.571 instâncias)
NUMERICAL_FEATURES = [
    'mean_hue', 'std_saturation', 'exg_index', 'exr_index', 'rg_ratio',
    'indice_clorose_necrose', 'hue_dispersion', 'glcm_contrast',
    'glcm_homogeneity', 'glcm_dissimilarity', 'glcm_energy',
    'indice_rugosidade_pustula', 'laplacian_var', 'size_kb',
    'total_pixels', 'resolution_mp', 'aspect_ratio'
]

DISCRETIZED_FEATURES = [
    'size_kb_bin', 'laplacian_var_bin', 'resolution_bin', 'aspect_ratio_bin'
]

CATEGORICAL_NOMINAL_FEATURES = ['dataset_source', 'extension']
ALL_CATEGORICAL_FEATURES = CATEGORICAL_NOMINAL_FEATURES + DISCRETIZED_FEATURES

# src/vision_inference_engine.py — Extração determinística de índices bio-espectrais
# ExG: sensível à clorofila e tecido sadio | ExR: marcador fúngico de necrose e pústulas
exg_pixel = 2.0 * g - r - b
exr_pixel = 1.4 * r - g
exg_index = float(np.mean(exg_pixel))
exr_index = float(np.mean(exr_pixel))
indice_clorose_necrose = float(((np.mean(r) + 1.0) / (np.mean(g) + 1.0)) / (std_saturation + 1e-4))"""
    add_code_block(doc, "src/atomic_pipeline_preprocessing.py & src/vision_inference_engine.py", code_p1,
                   "Estruturação estrita das 35 features preparadas na fase Modify e prontas para alimentação na fase Model.")

    # =========================================================================
    # PASSO 2
    # =========================================================================
    p2_h = doc.add_heading("Passo 2: Estabelecendo a Régua de Comparação (Os Baselines)", level=1)
    p2_h.paragraph_format.space_before = Pt(10)
    p2_h.paragraph_format.space_after = Pt(4)
    for r in p2_h.runs:
        r.font.name = "Calibri"
        r.font.bold = True
        r.font.color.rgb = COLOR_NAVY

    p2_txt = doc.add_paragraph()
    p2_txt.paragraph_format.space_after = Pt(4)
    p2_txt.paragraph_format.line_spacing = 1.15
    p2_txt.add_run("• Ideia central: ").bold = True
    p2_txt.add_run("Antes de usar algoritmos avançados, precisamos provar se métodos simples já resolveriam o problema.\n")
    p2_txt.add_run("• O \"Chute Proporcional\" (Dummy): ").bold = True
    p2_txt.add_run("Se alguém apenas chutasse os diagnósticos respeitando a proporção em que cada doença aparece no campo, acertaria apenas 16,40% das vezes. Esse é o piso mínimo de partida.\n")
    p2_txt.add_run("• As Regras Simples (Árvores e Fórmulas Clássicas):\n").bold = True
    p2_txt.add_run("  - Criamos uma árvore de decisão com perguntas diretas (ex.: \"A textura da folha está muito áspera? Há tons avermelhados de necrose?\").\n")
    p2_txt.add_run("  - Esses métodos básicos alcançaram entre 60% e 68% de acurácia.\n")
    p2_txt.add_run("• O Risco do Desbalanceamento: ").bold = True
    p2_txt.add_run("Por padrão, essas fórmulas simples tendiam a ignorar doenças raras, focando apenas no que era comum. Ao ativar o balanceamento de classes (class_weight='balanced'), forçamos o sistema a enxergar ameaças raras, resgatando diagnósticos da patologia Grassy Shoot de 0% para 35% de sensibilidade.")

    # Código do Passo 2
    code_p2 = """# src/baseline_models_benchmark.py — Concepção dos Modelos Baselines e Ponderação de Classes
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 1. Classificador Ingênuo Proporcional (Piso estocástico de 16,40%)
def build_dummy_baseline(strategy: str = 'stratified', random_state: int = 42) -> DummyClassifier:
    return DummyClassifier(strategy=strategy, random_state=random_state)

# 2. Regressão Logística Multiclasse com Penalidade L2 (Ridge)
def build_logistic_regression_baseline(C: float = 1.0, class_weight: str = 'balanced', random_state: int = 42):
    return LogisticRegression(
        penalty='l2', C=C, class_weight=class_weight,
        max_iter=1000, random_state=random_state, solver='lbfgs'
    )

# 3. Árvore de Decisão Rasa (Profundidade Máxima = 4)
def build_decision_tree_baseline(max_depth: int = 4, class_weight: str = 'balanced', random_state: int = 42):
    return DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight=class_weight,   # Evita o colapso e omissão de patologias raras
        criterion='gini',
        random_state=random_state
    )"""
    add_code_block(doc, "src/baseline_models_benchmark.py", code_p2,
                   "Construção da régua de comparação mínima e ativação de pesos balanceados contra desbalanceamento amostral.")

    # =========================================================================
    # PASSO 3
    # =========================================================================
    p3_h = doc.add_heading("Passo 3: O Modelo Campeão (SVM) e a Validação Agronômica", level=1)
    p3_h.paragraph_format.space_before = Pt(10)
    p3_h.paragraph_format.space_after = Pt(4)
    for r in p3_h.runs:
        r.font.name = "Calibri"
        r.font.bold = True
        r.font.color.rgb = COLOR_NAVY

    p3_txt = doc.add_paragraph()
    p3_txt.paragraph_format.space_after = Pt(4)
    p3_txt.paragraph_format.line_spacing = 1.15
    p3_txt.add_run("• Ideia central: ").bold = True
    p3_txt.add_run("Traçar uma cerca matemática segura e confirmar se a inteligência faz sentido na lavoura.\n")
    p3_txt.add_run("• A Fronteira do SVM (Support Vector Machine): ").bold = True
    p3_txt.add_run("Adotamos o SVM, um algoritmo que traça a fronteira de separação mais ampla e segura possível entre folhas doentes e sadias.\n")
    p3_txt.add_run("• Superioridade Estatística Comprovada: ").bold = True
    p3_txt.add_run("O SVM campeão alcançou 69,18% de acurácia. Por meio de testes estatísticos formais provamos que esse ganho não foi sorte, mas uma evolução real frente a todos os métodos simples.\n")
    p3_txt.add_run("• Conferência com a Biologia da Planta:\n").bold = True
    p3_txt.add_run("  - Quando a IA encontrava tons verdes intensos e textura uniforme, ela classificava a planta como sadia.\n")
    p3_txt.add_run("  - Quando encontrava manchas alaranjadas e quebra na película da folha, diagnosticava ferrugem.\n")
    p3_txt.add_run("  - Isso comprovou que o modelo aprendeu sintomas agronômicos reais, e não ruídos da imagem.")

    # Código do Passo 3
    code_p3 = """# src/atomic_svm_pipeline.py — Parametrização do Estimador SVM RBF Campeão
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

svm_champion = SVC(
    kernel='rbf',
    C=10.0,                      # Margem suave balanceando penalidade e generalização
    gamma=0.01,                  # Alcance ótimo de influência dos vetores de suporte
    probability=True,            # Habilita calibração de probabilidades a posteriori (Platt Scaling)
    class_weight='balanced',     # Ponderação agronômica para classes minoritárias
    random_state=42
)

# src/statistical_comparison_framework.py — Testes Estatísticos Formais Pareados (5 Folds)
from scipy.stats import ttest_rel, wilcoxon

t_stat, p_val_ttest = ttest_rel(svm_fold_scores, baseline_fold_scores)
w_stat, p_val_wilcoxon = wilcoxon(svm_fold_scores, baseline_fold_scores)
# Comprovação de superioridade com significância estatística rigorosa (p < 0.05)
is_statistically_superior = (p_val_ttest < 0.05) and (np.mean(svm_fold_scores) > np.mean(baseline_fold_scores))"""
    add_code_block(doc, "src/atomic_svm_pipeline.py & src/statistical_comparison_framework.py", code_p3,
                   "Parametrização ótima do SVM campeão e comprovação de superioridade com teste de hipótese estatística.")

    # =========================================================================
    # PASSO 4
    # =========================================================================
    p4_h = doc.add_heading("Passo 4: O \"Truque dos Kernels\" e o Teste de Sol e Sombra", level=1)
    p4_h.paragraph_format.space_before = Pt(10)
    p4_h.paragraph_format.space_after = Pt(4)
    for r in p4_h.runs:
        r.font.name = "Calibri"
        r.font.bold = True
        r.font.color.rgb = COLOR_NAVY

    p4_txt = doc.add_paragraph()
    p4_txt.paragraph_format.space_after = Pt(4)
    p4_txt.paragraph_format.line_spacing = 1.15
    p4_txt.add_run("• Ideia central: ").bold = True
    p4_txt.add_run("Projetar dados complexos e testar a resistência da IA contra o clima da fazenda.\n")
    p4_txt.add_run("• Separando o que Parece Misturado (Kernel Trick): ").bold = True
    p4_txt.add_run("Quando as lesões foliares se sobrepõem visualmente, o algoritmo utiliza funções matemáticas chamadas Kernels para encontrar ângulos onde a separação se torna clara.\n")
    p4_txt.add_run("• O Teste de Estresse no Campo: ").bold = True
    p4_txt.add_run("Na lavoura real há poeira, sol forte e nuvens gerando sombras. Simulamos essas perturbações no computador.\n")
    p4_txt.add_run("• Descoberta Prática:\n").bold = True
    p4_txt.add_run("  - O modelo matematicamente mais sofisticado (Kernel RBF) teve uma queda de 26,5 pontos percentuais sob sombra severa, pois depende excessivamente da iluminação local.\n")
    p4_txt.add_run("  - Versões lineares e polinomiais mostraram-se muito mais estáveis sob iluminação adversa.\n")
    p4_txt.add_run("  - Essa análise gera recomendações diretas para a captura de fotos no campo.")

    # Código do Passo 4
    code_p4 = """# src/svm_kernel_trick_study.py — Simulação de Ruído Fotométrico e Sombra Severa
CHROMA_FEATURES = ['mean_hue', 'std_saturation', 'exg_index', 'exr_index', 'rg_ratio', 'indice_clorose_necrose']

# 1. Perturbação de Poeira e Sol Forte (Ruído Gaussiano nos canais cromáticos)
def add_photometric_noise(X, train_std, level, rng):
    X_pert = X.copy()
    for col in CHROMA_FEATURES:
        X_pert[col] += rng.normal(0.0, level * train_std[col], size=len(X_pert))
    return X_pert

# 2. Perturbação de Sombra Severa (Fator multiplicativo f de atenuação exponencial)
SHADOW_POWERS = {
    'exg_index': 1, 'exr_index': 1,
    'laplacian_var': 2, 'glcm_contrast': 2,
    'glcm_dissimilarity': 1, 'indice_rugosidade_pustula': 3
}

def apply_shadow(X, factor: float):
    X_shadow = X.copy()
    for col, power in SHADOW_POWERS.items():
        X_shadow[col] = X_shadow[col] * (factor ** power)
    return X_shadow

# Resultado do estudo: sob sombra f=0.4, o Kernel RBF sofreu queda de 26,5 p.p.
# indicando a necessidade de protocolos rígidos de iluminação difusa para a captura."""
    add_code_block(doc, "src/svm_kernel_trick_study.py", code_p4,
                   "Funções de estresse computacional simulando condições adversas reais de campo (sombras e sol direto).")

    # =========================================================================
    # PASSO 5
    # =========================================================================
    p5_h = doc.add_heading("Passo 5: Blindagem, Segurança e Governança de Software", level=1)
    p5_h.paragraph_format.space_before = Pt(10)
    p5_h.paragraph_format.space_after = Pt(4)
    for r in p5_h.runs:
        r.font.name = "Calibri"
        r.font.bold = True
        r.font.color.rgb = COLOR_NAVY

    p5_txt = doc.add_paragraph()
    p5_txt.paragraph_format.space_after = Pt(4)
    p5_txt.paragraph_format.line_spacing = 1.15
    p5_txt.add_run("• Ideia central: ").bold = True
    p5_txt.add_run("Garantir que o modelo não cometa \"fraude\" e seja rastreável de ponta a ponta.\n")
    p5_txt.add_run("• Blindagem Contra \"Cola\" (Anti-Leakage): ").bold = True
    p5_txt.add_run("Construímos um duto de dados unificado (Pipeline) para garantir que a IA nunca tenha acesso prévio às imagens do teste cego, eliminando qualquer risco de aprendizado artificial viciado.\n")
    p5_txt.add_run("• Assinatura Digital (Hashes SHA-256): ").bold = True
    p5_txt.add_run("Todos os modelos e planilhas gerados receberam códigos criptográficos de integridade. Isso assegura que o arquivo levado a campo é idêntico ao auditado no laboratório.\n")
    p5_txt.add_run("• Certeza Calibrada: ").bold = True
    p5_txt.add_run("O sistema não apenas aponta um diagnóstico, mas calcula a probabilidade real de acerto (ex.: “88% de certeza para ferrugem”), fornecendo segurança para a tomada de decisão do agrônomo.")

    # Código do Passo 5
    code_p5 = """# 1. Blindagem Anti-Leakage: Encapsulamento Atômico do Pré-processador e Classificador
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), NUMERICAL_FEATURES),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ALL_CATEGORICAL_FEATURES)
    ],
    remainder='drop'
)
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('svm', svm_champion)])

# 2. Assinatura Digital Criptográfica (SHA-256) dos Artefatos de Produção
def calculate_file_hash(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

# 3. Certeza Calibrada via Platt Scaling e Margem Geométrica (src/vision_inference_engine.py)
probas = full_pipeline.predict_proba(df_feat)[0]
confidence_pct = float(round(np.max(probas) * 100.0, 2))  # Ex: 88,4% de probabilidade
dfunc = full_pipeline.decision_function(df_feat)[0]
delta_margin = float(np.sort(dfunc)[-1] - np.sort(dfunc)[-2])  # Margem hiperdimensional Δ"""
    add_code_block(doc, "src/atomic_svm_pipeline.py & src/export_final_modeling_abt.py", code_p5,
                   "Pipeline atômico anti-vazamento, assinatura criptográfica SHA-256 e calibração de incerteza operacional.")

    # =========================================================================
    # PASSO 6
    # =========================================================================
    p6_h = doc.add_heading("Passo 6: Conclusão da Fase Model e os Próximos Passos (Fase Assess)", level=1)
    p6_h.paragraph_format.space_before = Pt(10)
    p6_h.paragraph_format.space_after = Pt(4)
    for r in p6_h.runs:
        r.font.name = "Calibri"
        r.font.bold = True
        r.font.color.rgb = COLOR_NAVY

    p6_txt = doc.add_paragraph()
    p6_txt.paragraph_format.space_after = Pt(4)
    p6_txt.paragraph_format.line_spacing = 1.15
    p6_txt.add_run("• Ideia central: ").bold = True
    p6_txt.add_run("O dever de casa da modelagem foi concluído com sucesso; agora o foco é o valor econômico e o uso no campo.\n")
    p6_txt.add_run("• Critérios Atendidos na Modelagem:\n").bold = True
    p6_txt.add_run("  - Engenharia sólida: Códigos reprodutíveis, blindados e auditados.\n")
    p6_txt.add_run("  - Eficácia matemática: SVM validado estatisticamente como o melhor modelo.\n")
    p6_txt.add_run("  - Coerência prática: Decisões alinhadas à fitopatologia da cana.\n")
    p6_txt.add_run("• O Próximo Passo — Fase Assess (Avaliação de Negócio e Campo):\n").bold = True
    p6_txt.add_run("  - Matriz de Custo Financeiro: Calcular o impacto econômico de não identificar a tempo doenças de quarentena, nas quais um erro custa muito mais caro que um alarme falso.\n")
    p6_txt.add_run("  - Microsserviço de Diagnóstico: Transformar esse modelo campeão em um serviço conectado a aplicativo móvel para leitura fotográfica em tempo real nas fazendas.")

    # Código do Passo 6
    code_p6 = """# 1. Matriz de Custo Financeiro Assimétrico (src/svm_agronomic_error_audit.py)
@dataclass(frozen=True)
class SugarcaneEconomicParameters:
    expected_yield_tons_per_ha: float = 85.0
    price_per_ton_brl: float = 145.0             # Receita bruta: R$ 12.325,00/ha
    rust_yield_loss_fraction: float = 0.35       # 35% de quebra por ferrugem omitida
    rust_epidemic_spread_multiplier: float = 1.80# Disseminação eólica para talhões vizinhos
    fungicide_cost_per_ha: float = 485.00        # Custo do insumo + aplicação

    @property
    def cost_fn_rust(self) -> float:
        # Falso Negativo de Ferrugem: R$ 10.017,25/ha (perda de safra + reforma precoce de soqueira)
        return (self.expected_yield_tons_per_ha * self.price_per_ton_brl * self.rust_yield_loss_fraction * self.rust_epidemic_spread_multiplier) + 2252.50

    @property
    def cost_fp_fungicide(self) -> float:
        # Falso Positivo: R$ 485,00/ha (aplicação desnecessária de fungicida químico)
        return self.fungicide_cost_per_ha

    # Razão de Assimetria = 20,65x: Omitir um foco de ferrugem é 20 vezes mais custoso que um alarme falso!

# 2. Microsserviço de Diagnóstico em Tempo Real (src/app_demoday5_sugarvision.py)
import streamlit as st
from src.vision_inference_engine import VisionInferenceEngine

engine = VisionInferenceEngine("models/svm_pipeline_optimal.joblib")
uploaded_leaf = st.file_uploader("Capture a foto da folha no campo:", type=["jpg", "png"])
if uploaded_leaf:
    result = engine.diagnose_image(uploaded_leaf.read())
    st.write(f"Diagnóstico: {result['consolidated_class']}")
    st.write(f"Certeza Calibrada: {result['confidence_pct']:.1f}%")
    st.write(f"Status do SLA: {result['sla_status']}")"""
    add_code_block(doc, "src/svm_agronomic_error_audit.py & src/app_demoday5_sugarvision.py", code_p6,
                   "Matriz de impacto financeiro por hectare e implementação do microsserviço de diagnóstico em tempo real.")

    # =========================================================================
    # CARD DE ENCERRAMENTO E SÍNTESE
    # =========================================================================
    tbl_end = doc.add_table(rows=1, cols=1)
    tbl_end.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_e = tbl_end.cell(0, 0)
    c_e.width = Inches(6.8)
    set_cell_background(c_e, "142D55")
    set_cell_margins(c_e, top=160, bottom=160, left=200, right=200)

    p_e = c_e.paragraphs[0]
    p_e.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_e.paragraph_format.line_spacing = 1.15

    r_e1 = p_e.add_run("💡 SÍNTESE DA TRANSIÇÃO MODIFY → MODEL:\n")
    r_e1.font.name = "Calibri"
    r_e1.font.bold = True
    r_e1.font.size = Pt(11)
    r_e1.font.color.rgb = RGBColor(255, 200, 100) # gold

    r_e2 = p_e.add_run(
        "\"Com o encerramento da fase Modify, a equipe transformou 6.571 imagens fotográficas em uma matriz analítica blindada com 35 descritores. "
        "Na fase Model, provou-se a superioridade do SVM RBF ótimo (69,18% de acurácia com 100% de sensibilidade para Ferrugem), estabelecendo as bases "
        "para a fase Assess com matriz de custo financeiro e microsserviço de campo para o Demoday.\""
    )
    r_e2.font.name = "Calibri"
    r_e2.font.italic = True
    r_e2.font.size = Pt(10)
    r_e2.font.color.rgb = RGBColor(255, 255, 255)

    # Salva o documento
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"[OK] Documento Word salvo com sucesso em: {output_path}")

if __name__ == "__main__":
    out = Path("docs/Diagnostico_Inteligente_CanadeAcucar_Transicao_Modify_Model.docx")
    build_document(out)

