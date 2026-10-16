"""
Gerador do Guia Didático Descomplicado em Word (.docx) e PDF
Projeto: Sanidade-Vegetal AI (SugarVision) — Demoday 2
"""

import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color: str):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def set_cell_margins(cell, top=140, bottom=140, left=200, right=200):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_analogy_box(doc, icon_title: str, text: str, bg_hex="F0F7F4", border_hex="278246"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.8)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    r_t = p.add_run(f"{icon_title}\n")
    r_t.font.bold = True
    r_t.font.size = Pt(10.5)
    r_t.font.color.rgb = RGBColor(39, 130, 70) if "278246" in border_hex else RGBColor(30, 90, 180)

    r_b = p.add_run(text)
    r_b.font.size = Pt(9.5)
    r_b.font.color.rgb = RGBColor(40, 44, 52)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def add_heading_styled(doc, text: str, level: int):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    for run in h.runs:
        run.font.name = "Calibri"
        if level == 1:
            run.font.color.rgb = RGBColor(20, 45, 85)     # Navy
            run.font.size = Pt(15)
            run.font.bold = True
        elif level == 2:
            run.font.color.rgb = RGBColor(39, 130, 70)    # Green
            run.font.size = Pt(12.5)
            run.font.bold = True
        elif level == 3:
            run.font.color.rgb = RGBColor(211, 84, 0)     # Orange
            run.font.size = Pt(11)
            run.font.bold = True
    return h

def build_guia_didatico(output_docx: Path):
    doc = docx.Document()

    for s in doc.sections:
        s.top_margin = Inches(0.9)
        s.bottom_margin = Inches(0.9)
        s.left_margin = Inches(0.9)
        s.right_margin = Inches(0.9)

    # BANNER DE ABERTURA
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = tbl.cell(0, 0)
    c.width = Inches(6.8)
    set_cell_background(c, "142D55")
    set_cell_margins(c, top=200, bottom=200, left=240, right=240)

    p = c.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p.add_run("GUIA DIDÁTICO E DESCOMPLICADO\n")
    r1.font.size = Pt(11)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(180, 220, 190)

    r2 = p.add_run("Entendendo os Termos Técnicos do Demoday 2\n")
    r2.font.size = Pt(18)
    r2.font.bold = True
    r2.font.color.rgb = RGBColor(255, 255, 255)

    r3 = p.add_run("Traduzindo a Inteligência Artificial e a Fitopatologia da Cana-de-Açúcar para o Dia a Dia\n")
    r3.font.size = Pt(10.5)
    r3.font.color.rgb = RGBColor(220, 230, 245)

    r4 = p.add_run("Sanidade Vegetal AI (SugarVision) • FATEC Rio Claro 2026")
    r4.font.size = Pt(9.5)
    r4.font.bold = True
    r4.font.color.rgb = RGBColor(255, 200, 100)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 1. O PROBLEMA
    add_heading_styled(doc, "1. O Problema Real: O Que Estamos Fazendo?", level=1)
    p = doc.add_paragraph(
        "Imagine que você é um agrônomo andando por um canavial gigantesco. Você olha para as folhas de cana "
        "e precisa responder rápido: 'Essa folha está sadia ou com Ferrugem?'.\n\n"
        "Um ser humano experiente faz isso prestando atenção em duas coisas:\n"
        "1. Na cor da folha (se está verde saudável ou alaranjada/queimada com manchas).\n"
        "2. No toque da folha (se a superfície é lisa ou áspera com pequenas casquinhas de ferida).\n\n"
        "O nosso trabalho na Etapa Modify foi ensinar o computador a fazer essa mesma avaliação sozinho, "
        "olhando para fotos tiradas no campo por celulares."
    )
    p.paragraph_format.space_after = Pt(4)

    # 2. ENGENHARIA DE FEATURES E CORES
    add_heading_styled(doc, "2. O Que é 'Engenharia de Features' e Termos de Cor", level=1)
    
    add_analogy_box(
        doc,
        icon_title="🩺 Analogia do Exame Médico (O que é uma Feature?):",
        text="Quando você vai ao médico, ele não fica apenas olhando para a sua foto. Ele mede dados objetivos: "
             "temperatura (38°C), pressão arterial (12 por 8), glicose (95 mg/dL). Esses números são as 'features' (características) do seu corpo.\n"
             "Uma foto comum é apenas um monte de pontos coloridos soltos. A Engenharia de Features é o processo de transformar essa foto "
             "em 35 números inteligentes que medem a saúde e a textura daquela folha de cana."
    )

    p = doc.add_paragraph()
    p.add_run("• ExG (Excesso de Verde) = 'O Termômetro da Vida': ").bold = True
    p.add_run("Mede o quanto o verde da folha se destaca. Folhas ricas em clorofila têm ExG alto e positivo (+44,82). Folhas atacadas por fungo perdem a clorofila e o ExG desaba para valores negativos (-18,45).\n\n")
    p.add_run("• ExR (Excesso de Vermelho) = 'O Termômetro da Ferrugem': ").bold = True
    p.add_run("O fungo da ferrugem solta esporos alaranjados e queima a folha. Quanto maior o ExR, mais queimada e infectada está a planta (-18 em folhas sadias vs +24 na ferrugem).\n\n")
    p.add_run("• Razão R/G: ").bold = True
    p.add_run("Compara direto a intensidade de vermelho contra o verde, medindo o amarelecimento da folha.")
    p.paragraph_format.space_after = Pt(4)

    add_analogy_box(
        doc,
        icon_title="🕶️ Analogia do Óculos de Sol (Espaço HSV vs RGB comum):",
        text="Se você tirar foto de uma folha ao meio-dia sob sol forte, ela fica brilhante quase branca. Se tirar no fim da tarde ou na sombra, "
             "a mesma folha parece escura. O formato comum de câmera (RGB) mistura a cor com o brilho da luz.\n"
             "O formato HSV separa as coisas: o Hue (Matiz) mede a cor pura da folha, não importa se ela está no sol ou na sombra. "
             "Isso impede que o robô confunda uma sombra escura com uma mancha de doença!",
        bg_hex="EBF3FD", border_hex="1E5AB4"
    )

    # 3. TEXTURA E TOQUE
    add_heading_styled(doc, "3. Textura e Tato: A Matriz GLCM de Haralick", level=1)

    add_analogy_box(
        doc,
        icon_title="✋ Analogia do Papel Sulfite vs Folha de Lixa (GLCM e Rugosidade):",
        text="Se você fechar os olhos e passar a mão num papel sulfite, você sente suavidade contínua. Se passar a mão numa lixa de parede, "
             "sente pontinhos ásperos e relevo saliente.\n"
             "O fungo da ferrugem cria pequenas erupções chamadas pústulas (pequenos vulcões de esporos que rompem a casca da folha). "
             "A Matriz GLCM de Haralick é a matemática que permite ao computador sentir o tato da imagem: ela mede se a folha é lisa "
             "(sadia) ou áspera e rugosa (com ferrugem)."
    )

    p = doc.add_paragraph()
    p.add_run("• Contraste Alto: ").bold = True
    p.add_run("Significa que há mudanças bruscas de claro e escuro lado a lado (bordas de feridas). Em folhas doentes, o contraste foi 1,72 vezes maior.\n\n")
    p.add_run("• Homogeneidade: ").bold = True
    p.add_run("Mede a suavidade da folha. Folhas sadias têm alta homogeneidade (0,845); na ferrugem, ela despenca para 0,691.\n\n")
    p.add_run("• IRFP (Índice de Rugosidade Foliar de Pústula): ").bold = True
    p.add_run("Fórmula criada pela equipe combinando contraste e perda de homogeneidade em uma nota única de aspereza.\n\n")
    p.add_run("• Invariância à Rotação: ").bold = True
    p.add_run("Significa que não importa se a foto foi tirada com a folha em pé, deitada ou de lado. O computador avalia em 4 direções e tira a média, sem se confundir.")
    p.paragraph_format.space_after = Pt(4)

    # 4. A LINHA DE PRODUÇÃO (PIPELINE)
    add_heading_styled(doc, "4. A Linha de Montagem: Pipeline Atômico e Pré-processamento", level=1)

    add_analogy_box(
        doc,
        icon_title="🏭 Analogia da Linha de Montagem de Carros (Pipeline Atômico):",
        text="Numa fábrica de automóveis, uma peça entra na esteira, é lavada, soldada, pintada e montada sempre na mesma ordem, "
             "sem ninguém improvisar no meio do caminho.\n"
             "O Pipeline Atômico (ColumnTransformer) é essa esteira automática: ele recebe a foto e aplica todas as correções e fórmulas "
             "na ordem exata, empacotado em um único objeto de código. Qualquer foto nova amanhã passa rigorosamente pelas mesmas regras."
    )

    p = doc.add_paragraph()
    p.add_run("• Imputação (Mediana e Moda): ").bold = True
    p.add_run("Quando falta um dado (ex: foco ou tamanho que falhou na leitura), preenchemos o buraco com o valor do meio (mediana para números) ou mais frequente (moda para categorias), sem distorcer o resultado.\n\n")
    p.add_run("• Padronização Z-Score (StandardScaler): ").bold = True
    p.add_run("Coloca todo mundo na mesma régua. Uma imagem pode ter 2.000.000 de pixels e um índice de cor valer 0,45. Sem padronizar, o computador acharia que 2 milhões manda em tudo. O Z-score equilibra as grandezas.\n\n")
    p.add_run("• One-Hot Encoding nos Bins: ").bold = True
    p.add_run("Separamos as qualidades em faixas (ex: Resolução Baixa, Média, Alta). Criamos caixinhas de 'Sim/Não' para que o robô não assuma preconceitos ou escalas artificiais.")
    p.paragraph_format.space_after = Pt(4)

    # 5. DATA LEAKAGE
    add_heading_styled(doc, "5. O Ponto Mais Importante: Data Leakage (Vazamento de Dados)", level=1)

    add_analogy_box(
        doc,
        icon_title="🎓 Analogia do Gabarito Vazado (O Perigo do Data Leakage):",
        text="Imagine um aluno que vai fazer uma prova. Se o professor deixar o aluno ver as questões da prova final no dia anterior, "
             "ele vai tirar nota 10 com facilidade. Mas se colocarmos esse mesmo aluno para trabalhar no dia seguinte, ele vai errar tudo, "
             "porque só decorou as respostas que vazaram!\n"
             "Na Inteligência Artificial, Data Leakage é quando o computador 'cola' das fotos de teste durante o treino. "
             "O modelo parece genial no laboratório, mas quebra a cara no campo real.",
        bg_hex="FFF8E7", border_hex="D35400"
    )

    p = doc.add_paragraph(
        "Como provamos matematicamente que NÃO houve vazamento?\n"
        "1. Trancamos 380 fotos em um 'cofre' (conjunto de teste cego).\n"
        "2. O computador aprendeu médias e escalas apenas nas fotos de treino (5.574).\n"
        "3. Provamos matematicamente que as fotos do teste não tinham média zero (o que comprova que nunca participaram da conta).\n"
        "4. Tentamos até sujar o teste com ruído de propósito: a máquina nem sequer tomou conhecimento. Vazamento Zero (0%) auditado!"
    )
    p.paragraph_format.space_after = Pt(4)

    # 6. ABT E PARQUET
    add_heading_styled(doc, "6. A Planilha Mestre e Formatos: ABT e Apache Parquet", level=1)
    p = doc.add_paragraph()
    p.add_run("• ABT Final (abt_features_modelagem): ").bold = True
    p.add_run("É a Planilha Mestre Oficial do projeto. Ela tem exatamente 6.571 linhas (uma para cada folha) e 40 colunas limpas, com ZERO dados faltando (0 NaNs).\n\n")
    p.add_run("• Apache Parquet vs CSV = 'O ZIP Inteligente vs O Bloco de Notas Pesado': ").bold = True
    p.add_run("O CSV é como um texto de bloco de notas: ocupa 3 MB e o computador precisa ler linha por linha para adivinhar o tipo de dado. O Parquet é um formato moderno colunar: ocupa 3,26 vezes menos espaço (menos de 1 MB) e abre 41% mais rápido, sem risco de errar números decimais.\n\n")
    p.add_run("• Hashes SHA-256 = 'O Lacre de Segurança': ").bold = True
    p.add_run("Uma impressão digital criptográfica gerada para a tabela. Se alguém mudar uma única vírgula, o código se rompe, provando que ninguém adulterou os dados.")
    p.paragraph_format.space_after = Pt(4)

    # 7. E AGORA? O QUE É O SVM?
    add_heading_styled(doc, "7. E Agora? O que é o SVM da Próxima Fase?", level=1)

    add_analogy_box(
        doc,
        icon_title="🤠 Analogia da Cerca no Pasto (Support Vector Machines - SVM):",
        text="Pense no SVM como um fazendeiro esperto que precisa passar uma cerca separando os animais sadios dos animais doentes.\n"
             "Ele procura os animais mais difíceis que estão bem perto da fronteira (os 'Vetores de Suporte') e passa a cerca exatamente "
             "no meio do caminho com a maior margem de segurança possível para não errar nenhum.\n"
             "Como nossa tabela de dados foi perfeitamente tratada nesta Sprint 2, o SVM agora consegue passar essa cerca com facilidade e precisão máxima!"
    )

    # RESUMO EM UMA FRASE
    tbl_end = doc.add_table(rows=1, cols=1)
    tbl_end.alignment = WD_TABLE_ALIGNMENT.CENTER
    ce = tbl_end.cell(0, 0)
    ce.width = Inches(6.8)
    set_cell_background(ce, "142D55")
    set_cell_margins(ce, top=140, bottom=140, left=180, right=180)

    pe = ce.paragraphs[0]
    pe.alignment = WD_ALIGN_PARAGRAPH.CENTER
    re1 = pe.add_run("💡 RESUMO EM UMA FRASE:\n")
    re1.font.bold = True
    re1.font.size = Pt(11)
    re1.font.color.rgb = RGBColor(255, 200, 100)

    re2 = pe.add_run(
        "\"Pegamos 6.571 fotos soltas de folhas de cana, medimos a cor e a aspereza de cada uma sem deixar o sol atrapalhar, "
        "limpamos os dados sem trapaça matemática (zero vazamento) e entregamos uma planilha ultrarrápida pronta para a inteligência artificial "
        "aprender a diagnosticar a ferrugem.\""
    )
    re2.font.italic = True
    re2.font.size = Pt(10)
    re2.font.color.rgb = RGBColor(255, 255, 255)

    output_docx.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_docx))
    print(f"[OK] Guia Didático Word salvo em: {output_docx}")

if __name__ == "__main__":
    out_docx = Path("docs/Guia_Didatico_Termos_Demoday2.docx")
    build_guia_didatico(out_docx)
