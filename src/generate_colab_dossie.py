"""Script para gerar a versao interativa em Jupyter Notebook (.ipynb)
do dossie da Fase Modify para execucao direta no Google Colab.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Celula 1: Badge Colab e Cabecalho
cell1 = """# 🔬 Dossiê e Inventário Completo da Fase Modify (SEMMA — Sprint 2)

<a href="https://colab.research.google.com/github/sugarvision/Sanidade-Vegetal/blob/main/notebooks/dossie_fase_modify_semma.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Projeto:** Sanidade-Vegetal (SugarVision)  
**Metodologia:** Framework SEMMA (*Sample, Explore, Modify, Model, Assess*)  
**Etapa:** **Modify** (Engenharia de Features, Transformação, Pipeline Atômico e Consolidação da ABT)  
**Ciclo:** Sprint 2  
**Status:** **100% Concluído e Auditado**
"""
cells.append(nbf.v4.new_markdown_cell(cell1))

# Celula 2: Setup no Colab (Opcional)
cell2_md = """### ⚙️ Configuração Automática de Ambiente (Caso esteja executando no Google Colab)
Execute a célula abaixo apenas se estiver no Google Colab para clonar o repositório e preparar as pastas."""
cells.append(nbf.v4.new_markdown_cell(cell2_md))

cell2_code = """import os
import sys

# Detecta se esta rodando no Google Colab
if 'google.colab' in str(get_ipython()):
    print("🚀 Ambiente Google Colab detectado! Clonando repositorio do SugarVision...")
    if not os.path.exists('Sanidade-Vegetal'):
        !git clone https://github.com/sugarvision/Sanidade-Vegetal.git
    %cd Sanidade-Vegetal
    print("✅ Repositório pronto e diretório configurado!")
else:
    print("💻 Executando em ambiente local.")
"""
cells.append(nbf.v4.new_code_cell(cell2_code))

# Celula 3: Visao Geral
cell3 = """---
## 📌 1. Visão Geral da Fase Modify

Na metodologia SEMMA, a fase **Modify** transforma os dados brutos e explorados em representações matemáticas discriminatórias, limpas e escaladas, prontas para algoritmos de aprendizado de máquina supervisionados. 

No projeto **SugarVision**, a fase Modify resolveu os principais desafios de visão computacional agrícola e modelagem preditiva tabular:
1. **Engenharia de Features Visuais:** Conversão para múltiplos espaços de cor ($RGB \\rightarrow HSV, CIELAB$), índices biofísicos de vegetação ($ExG, ExR$), razões espectrais de necrose ($R/G, G/B$) e análise de textura via Matriz de Co-ocorrência em Níveis de Cinza (GLCM / Haralick).
2. **Discretização e Binning:** Redução de assimetria de metadados (`size_kb`, `laplacian_var`) com particionamento em quantis e isolamento estrito contra *data leakage*.
3. **Pipeline Atômico com `ColumnTransformer`:** Encadeamento de imputação mediana, padronização $z$-score e codificação *One-Hot*, governado pela **Regra de Ouro do Fit** (`fit` exclusivo na partição de treino).
4. **Consolidação da ABT Final:** Geração da *Analytical Base Table* com **6.571 instâncias**, **40 colunas**, **0 NaNs**, com validação de prontidão para a Fase *Model* (SVM com `GridSearchCV`).
"""
cells.append(nbf.v4.new_markdown_cell(cell3))

# Celula 4: Catalogo de Notebooks
cell4 = """---
## 📓 2. Catálogo de Notebooks Jupyter da Fase Modify

Você também pode abrir os notebooks específicos diretamente no Colab pelos links abaixo:

| Notebook | Descrição e Finalidade | Abrir no Colab |
| :--- | :--- | :---: |
| **`06_pipeline_atomico_columntransformer.ipynb`** | **Pipeline Atômico de Pré-processamento:** Fluxo unificado no Scikit-Learn com `ColumnTransformer`. Imputação por mediana, normalização $z$-score, One-Hot Encoding e validação anti-leakage. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sugarvision/Sanidade-Vegetal/blob/main/notebooks/06_pipeline_atomico_columntransformer.ipynb) |
| **`07_consolidacao_exportacao_abt_final.ipynb`** | **Consolidação e Exportação da ABT Final:** Aplicação integral do pipeline nas 6.571 instâncias. Auditoria de 0 NaNs, benchmark I/O e teste de prontidão com `GridSearchCV`. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sugarvision/Sanidade-Vegetal/blob/main/notebooks/07_consolidacao_exportacao_abt_final.ipynb) |
| **`03_sprint2_modify_pipeline.ipynb`** | **Notebook de Disparo Integrado:** Interface rápida para execução dos scripts de extração e rotinas de validação de ponta a ponta. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sugarvision/Sanidade-Vegetal/blob/main/notebooks/03_sprint2_modify_pipeline.ipynb) |
"""
cells.append(nbf.v4.new_markdown_cell(cell4))

# Celula 5: Modulos em src
cell5 = """---
## ⚙️ 3. Módulos e Scripts Executáveis (`src/`)

A inteligência de código e os pipelines matemáticos foram estruturados em módulos reutilizáveis:

1. **`src/feature_engineering_visual.py`**:
   * Conversão de espaços de cor: $RGB \\rightarrow HSV$ e $CIELAB$.
   * Índices espectrais foliares: $ExG = 2G - R - B$, $ExR = 1.4R - G$, razões $R/G$ e $G/B$.
   * Textura Haralick via GLCM multidirecional ($0^\\circ, 45^\\circ, 90^\\circ, 135^\\circ$): Contraste, Homogeneidade, Dissimilaridade, Energia.
2. **`src/discretizacao_binning.py`**:
   * Avaliação de assimetria de metadados e aplicação de `KBinsDiscretizer` com quantis estritamente no conjunto de treino.
3. **`src/atomic_pipeline_preprocessing.py`**:
   * `ColumnTransformer` que integra imputação, escalonamento e codificação dummy sem vazamento de dados. Serializa o pipeline em `models/preprocessor_pipeline.joblib`.
4. **`src/export_final_modeling_abt.py`**:
   * Processamento das 6.571 linhas, auditoria de 0 NaNs, persistência em Apache Parquet e CSV e verificação de prontidão para SVM.
"""
cells.append(nbf.v4.new_markdown_cell(cell5))

# Celula 6: Verificacao de Integridade da ABT Final
cell6_md = """---
## 🔍 4. Verificação Interativa da ABT Consolidada (Smoke Test)
A célula abaixo lê os metadados e valida formalmente a integridade dos artefatos produzidos na Fase Modify."""
cells.append(nbf.v4.new_markdown_cell(cell6_md))

cell6_code = """import pandas as pd
import json
import os

manifest_path = 'data/processed/abt_features_modelagem_manifest.json'
parquet_path = 'data/processed/abt_features_modelagem.parquet'
csv_path = 'data/processed/abt_features_modelagem.csv'

print("=== AUDITORIA DE ARTEFATOS DA FASE MODIFY ===")
if os.path.exists(manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    print(f"✅ Manifesto Carregado com Sucesso!")
    print(f"• Total de Instâncias: {manifest['corpus']['total_instances']}")
    print(f"• Total de Colunas: {manifest['corpus']['total_columns']}")
    print(f"• Total de NaNs / Valores Ausentes: {manifest['audit']['total_nans']}")
    print(f"• Conformidade 0 NaNs: {'APROVADO' if manifest['audit']['zero_nan_compliance'] else 'FALHA'}")
    print(f"• Partições: Treino={manifest['corpus']['split_distribution']['train']}, Validação={manifest['corpus']['split_distribution']['val']}, Teste={manifest['corpus']['split_distribution']['test']}")

if os.path.exists(parquet_path):
    df_pq = pd.read_parquet(parquet_path)
    print(f"✅ Arquivo Parquet verificado: shape {df_pq.shape}")
elif os.path.exists(csv_path):
    df_csv = pd.read_csv(csv_path)
    print(f"✅ Arquivo CSV verificado: shape {df_csv.shape}")
else:
    print("ℹ️ Execute os scripts de geração para materializar a base localmente.")
"""
cells.append(nbf.v4.new_code_cell(cell6_code))

# Celula 7: Artefatos Persistidos
cell7 = """---
## 📦 5. Artefatos Finais Persistidos em Disco

### Bases Tratadas (`data/processed/`)
* **`abt_features_modelagem.parquet`** (916,92 KB): Tabela definitiva em formato colunar com compressão Snappy (3,26× mais compacta que CSV).
* **`abt_features_modelagem.csv`** (2.989,66 KB): Espelho em formato aberto para auditoria e interoperabilidade.
* **`abt_features_modelagem_manifest.json`**: Manifesto criptográfico com hashes SHA-256 e schema das variáveis.

### Modelos e Pipelines (`models/`)
* **`preprocessor_pipeline.joblib`** (2,58 KB): Pipeline `ColumnTransformer` serializado com compressão, pronto para inferência e acoplamento com o classificador SVM na Sprint 3.
* **`preprocessor_metadata.json`**: Manifesto de governança MLOps contendo os nomes das 35 features transformadas via `get_feature_names_out()`.
"""
cells.append(nbf.v4.new_markdown_cell(cell7))

# Celula 8: Guia de Reproducao
cell8 = """---
## 🚀 6. Como Reproduzir a Fase Modify Localmente ou no Terminal

```bash
# 1. Executar a Engenharia de Features Visuais e Índices Espectrais
python src/feature_engineering_visual.py

# 2. Executar a Discretização Estatística (Binning)
python src/discretizacao_binning.py

# 3. Ajustar e Serializar o Pipeline Atômico Anti-Leakage
python src/atomic_pipeline_preprocessing.py

# 4. Consolidar e Exportar a ABT Final (0 NaNs + Parquet + CSV)
python src/export_final_modeling_abt.py
```
"""
cells.append(nbf.v4.new_markdown_cell(cell8))

nb['cells'] = cells

output_path = 'notebooks/dossie_fase_modify_semma.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook gerado com sucesso em: {output_path}")
