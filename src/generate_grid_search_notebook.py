"""
Script utilitário para gerar e executar o notebook 09_grid_search_exaustivo_svm.ipynb
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 3 — Framework SEMMA (Fase: Model)
Responsável: Elisa (Engenharia de Machine Learning & MLOps)
"""

import os
import sys
import json
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient


def create_and_run_gridsearch_notebook():
    base_dir = Path(__file__).resolve().parent.parent
    nb_path = base_dir / "notebooks" / "09_grid_search_exaustivo_svm.ipynb"
    
    nb = nbf.v4.new_notebook()
    nb['metadata'] = {
        'kernelspec': {
            'display_name': 'Python 3 (SugarVision)',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.13'
        }
    }
    
    cells = []
    
    # -------------------------------------------------------------------------
    # Célula 1: Título e Cabeçalho
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""# 🔎 Otimização de Hiperparâmetros via Grid Search Exaustivo (GridSearchCV)
**Projeto:** Sanidade-Vegetal (SugarVision) — Detecção Inteligente de Fitopatologias em Cana-de-Açúcar  
**Sprint:** 3 — Framework SEMMA (Fase: Model)  
**Responsável:** Elisa (`EA`) — Engenharia de Machine Learning & MLOps  
**Dataset Base:** `data/processed/abt_sanidade_vegetal.csv` (6.571 instâncias)  

---

## 📌 Objetivo da Entrega
Configurar, executar e auditar a **busca exaustiva em grade (`GridSearchCV`)** acoplada ao **Pipeline Atômico Scikit-Learn** homologado na Tarefa 1. 

A exploração mapeia o espaço de busca dos hiperparâmetros para três famílias de funções de kernel (**Linear, RBF e Polinomial**) através de **validação cruzada estratificada em 5 folds** (`StratifiedKFold`), orientada pelo **$F_1$-Score Macro** (`scoring='f1_macro'`), extraindo formalmente a combinação ótima de hiperparâmetros (`best_params_`), o melhor score médio (`best_score_`) e o estimador ajustado para produção (`best_estimator_`).

### ✅ Checklist Oficial da Tarefa:
1. **Definir os grids de hiperparâmetros para os kernels Linear, RBF e Polinomial.**
2. **Configurar a validação cruzada estratificada com `cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.**
3. **Definir a métrica de otimização principal (`scoring='f1_macro'`).**
4. **Extrair os melhores parâmetros (`best_params_`), o melhor score (`best_score_`) e o estimador final ajustado.**"""))

    # -------------------------------------------------------------------------
    # Célula 2: Imports e Configuração de Ambiente
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# 1. Configuração do ambiente e importação das dependências
import os
import sys
import json
import time
from pathlib import Path

# Adicionar diretório src ao path
sys.path.insert(0, str(Path.cwd().parent / "src" if Path.cwd().name == "notebooks" else Path.cwd() / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn import set_config
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss
)
import joblib

from atomic_svm_pipeline import (
    build_atomic_svm_pipeline,
    TARGET_COLUMNS,
    SPLIT_COLUMN,
    NUMERICAL_FEATURES
)
from grid_search_svm_optimization import (
    get_svm_param_grid,
    CLASS_NAMES_MAP,
    TARGET_NAMES
)

set_config(display="diagram")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

print(f"Scikit-Learn versão: {sklearn.__version__}")
print(f"Pandas versão      : {pd.__version__}")
print(f"NumPy versão       : {np.__version__}")"""))

    # -------------------------------------------------------------------------
    # Célula 3: Seção 1 - Ingestão da Base e Particionamento
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 1. Carregamento dos Dados e Preparação das Partições

Utilizamos a partição determinística `split_partition == 'train'` (5.574 instâncias) para alimentar a validação cruzada do `GridSearchCV`. As partições de `valid` (617) e `test` (380) permanecem estritamente blindadas para aferição da capacidade de generalização externa."""))

    # -------------------------------------------------------------------------
    # Célula 4: Código da Seção 1
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""data_path = Path.cwd().parent / "data" / "processed" / "abt_sanidade_vegetal.csv" if Path.cwd().name == "notebooks" else Path.cwd() / "data" / "processed" / "abt_sanidade_vegetal.csv"
df = pd.read_csv(data_path)

df_train = df[df[SPLIT_COLUMN] == 'train'].copy()
df_valid = df[df[SPLIT_COLUMN] == 'valid'].copy()
df_test = df[df[SPLIT_COLUMN] == 'test'].copy()

X_train = df_train.drop(columns=TARGET_COLUMNS)
y_train = df_train['target_multiclass']

X_valid = df_valid.drop(columns=TARGET_COLUMNS)
y_valid = df_valid['target_multiclass']

X_test = df_test.drop(columns=TARGET_COLUMNS)
y_test = df_test['target_multiclass']

print(f"ABT Carregada:")
print(f"- Treino (Base do GridSearch) : {len(X_train)} instâncias")
print(f"- Validação Independente      : {len(X_valid)} instâncias")
print(f"- Teste Cego Final            : {len(X_test)} instâncias")
print(f"Distribuição de Classes no Treino:")
print(pd.Series(y_train.map(CLASS_NAMES_MAP)).value_counts().to_string())"""))

    # -------------------------------------------------------------------------
    # Célula 5: Seção 2 - Estruturação do Grid de Hiperparâmetros
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 2. Estruturação do Grid de Hiperparâmetros Desacoplado

Para evitar desperdício computacional (ex: testar `gamma` em kernel linear), estruturamos o grid como uma **lista de 3 dicionários mutuamente exclusivos**:

1. **Kernel Linear:** Varia a constante de regularização $C \in [0.1, 1.0, 10.0, 100.0]$.
2. **Kernel RBF:** Combina $C \in [0.1, 1.0, 10.0, 100.0]$ com $\gamma \in [\text{'scale'}, \text{'auto'}, 0.01, 0.1]$.
3. **Kernel Polinomial:** Explora $C \in [0.1, 1.0, 10.0]$, $\gamma \in [\text{'scale'}, \text{'auto'}]$ e grau $d \in [2, 3]$.

$$\text{Espaço Amostral} = 4 + (4 \times 4) + (3 \times 2 \times 2) = 4 + 16 + 12 = 32 \text{ candidatos}$$
$$\text{Total de Modelos Ajustados} = 32 \text{ candidatos} \times 5 \text{ folds} = 160 \text{ ajustes}$$"""))

    # -------------------------------------------------------------------------
    # Célula 6: Código da Seção 2
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""param_grid = get_svm_param_grid()
print("Grade Hiperparamétrica Estruturada:")
for idx, grid_dict in enumerate(param_grid, start=1):
    kernel_name = grid_dict['svm__kernel'][0]
    n_comb = np.prod([len(v) for k, v in grid_dict.items() if k != 'svm__kernel'])
    print(f"Sub-grade {idx} [{kernel_name.upper()}]: {n_comb} combinações -> {grid_dict}")"""))

    # -------------------------------------------------------------------------
    # Célula 7: Seção 3 - Validação Cruzada Estratificada e Métrica f1_macro
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 3. Validação Cruzada Estratificada & Métrica Macro $F_1$

### Por que `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`?
Garante que a proporção das 7 classes fitopatológicas seja mantida rigorosamente idêntica em cada um dos 5 subconjuntos de validação e treino.

### Por que `scoring='f1_macro'`?
Em cenários agrícolas reais com patologias de baixa ocorrência (ex: *Grassy Shoot* com apenas 150 amostras de treino versus *Red Rot* com 1.125), a métrica de Acurácia é enganosa. O **$F_1$-Macro** atribui peso idêntico a cada classe:
$$\text{F1-Macro} = \frac{1}{K} \sum_{k=1}^K F_1^{(k)}$$
penalizando qualquer modelo que ignore patologias raras."""))

    # -------------------------------------------------------------------------
    # Célula 8: Código da Seção 3 e Execução / Leitura dos Resultados
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""models_dir = Path.cwd().parent / "models" if Path.cwd().name == "notebooks" else Path.cwd() / "models"
ranking_path = models_dir / "svm_gridsearch_results.csv"
optimal_model_path = models_dir / "svm_pipeline_optimal.joblib"
metadata_path = models_dir / "svm_gridsearch_metadata.json"

# Carregar o modelo ótimo ajustado
best_estimator = joblib.load(optimal_model_path)

# Carregar ranking completo dos 32 candidatos
results_df = pd.read_csv(ranking_path)

# Carregar manifesto de metadados
with open(metadata_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)

print(f"Status da Busca Exaustiva : {meta['production_status']}")
print(f"Total de Ajustes Realizados: {meta['search_configuration']['total_fits']}")
print(f"Tempo Total de Busca       : {meta['search_configuration']['execution_time_seconds']}s")
print(f"Melhor Score CV (F1-Macro) : {meta['best_cv_score_f1_macro']:.4f}")
print(f"Hiperparâmetros Campeões   : {meta['best_hyperparameters']}")"""))

    # -------------------------------------------------------------------------
    # Célula 9: Seção 4 - Análise do Ranking e Melhores Modelos
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 4. Ranking Completo dos Modelos Candidatos

Apresentamos a tabela de desempenho das melhores configurações avaliadas durante a busca exaustiva."""))

    # -------------------------------------------------------------------------
    # Célula 10: Código da Seção 4
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""print("TOP 10 MELHORES MODELOS DO GRID SEARCH EXAUSTIVO:")
cols_show = ['rank_test_f1_macro', 'param_svm__kernel', 'param_svm__C', 'param_svm__gamma', 'param_svm__degree', 'mean_test_f1_macro', 'std_test_f1_macro', 'mean_test_accuracy']
display_df = results_df[[c for c in cols_show if c in results_df.columns]].head(10)
print(display_df.to_string(index=False))"""))

    # -------------------------------------------------------------------------
    # Célula 11: Seção 5 - Diagnósticos Visuais
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 5. Visualizações e Diagnósticos Técnicos

Exibição dos gráficos diagnósticos:
1. **Comparação de Eficácia por Kernel** (Linear vs RBF vs Polinomial).
2. **Superfície de Resposta do Kernel RBF** ($C$ vs $\gamma$).
3. **Matriz de Confusão Otimizada** no conjunto de validação.
4. **Ganho de Performance por Patologia** frente ao baseline."""))

    # -------------------------------------------------------------------------
    # Célula 12: Código da Seção 5
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# 1. Comparação entre Kernels
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
kernel_perf = results_df.groupby('param_svm__kernel')['mean_test_f1_macro'].agg(['max', 'mean']).reset_index()
sns.barplot(data=kernel_perf, x='param_svm__kernel', y='max', palette='Blues_d', ax=ax)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 4), textcoords='offset points')
ax.set_title("Desempenho Máximo por Família de Kernel (F1-Macro Médio CV)", fontsize=12, fontweight='bold')
ax.set_xlabel("Kernel do SVM")
ax.set_ylabel("F1-Score Macro Máximo")
ax.set_ylim(0.4, 0.75)
plt.tight_layout()
plt.show()"""))

    # -------------------------------------------------------------------------
    # Célula 13: Matriz de Confusão Otimizada
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# Predição no conjunto de validação com o modelo campeão
y_val_pred = best_estimator.predict(X_valid)

fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=120)
cm = confusion_matrix(y_valid, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TARGET_NAMES)
disp.plot(cmap='Blues', ax=ax, colorbar=True, values_format='d')
ax.set_title(f"Matriz de Confusão Otimizada — SVM {meta['best_hyperparameters'].get('svm__kernel', 'rbf').upper()}", fontsize=13, fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()"""))

    # -------------------------------------------------------------------------
    # Célula 14: Seção 6 - Avaliação nos Conjuntos Externos (Validação e Teste Cego)
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 6. Avaliação Externa de Generalização (Validação & Teste)

Apresentamos o relatório de métricas formal no conjunto de validação independente (617 instâncias) e no conjunto de teste cego final (380 instâncias)."""))

    # -------------------------------------------------------------------------
    # Célula 15: Código da Seção 6
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""print("RELATÓRIO DE CLASSIFICAÇÃO NO CONJUNTO DE VALIDAÇÃO (617 amostras):")
print(classification_report(y_valid, y_val_pred, target_names=TARGET_NAMES, digits=4))

# Teste Cego Final
y_test_pred = best_estimator.predict(X_test)
print("RELATÓRIO DE CLASSIFICAÇÃO NO TESTE CEGO FINAL (380 amostras):")
print(classification_report(y_test, y_test_pred, target_names=TARGET_NAMES, digits=4))"""))

    nb['cells'] = cells
    
    print(f"Gravando notebook 09 preliminar em: {nb_path}")
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Executando notebook 09 de ponta a ponta com NotebookClient...")
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()
    
    print(f"Salvando notebook executado com saídas completas em: {nb_path}")
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"[OK] Notebook 09 executado e renderizado com 100% de sucesso!")
    return nb_path


if __name__ == '__main__':
    create_and_run_gridsearch_notebook()
