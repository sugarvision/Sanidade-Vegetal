"""
Script utilitário para gerar e executar o notebook 08_pipeline_atomico_svm_anti_leakage.ipynb
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


def create_and_run_svm_notebook():
    base_dir = Path(__file__).resolve().parent.parent
    nb_path = base_dir / "notebooks" / "08_pipeline_atomico_svm_anti_leakage.ipynb"
    
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
    cells.append(nbf.v4.new_markdown_cell("""# 🌿 Integração Atômica do Estimador SVM ao Pipeline Anti-Leakage
**Projeto:** Sanidade-Vegetal (SugarVision) — Detecção Inteligente de Fitopatologias em Cana-de-Açúcar  
**Sprint:** 3 — Framework SEMMA (Fase: Model)  
**Responsável:** Elisa (`EA`) — Engenharia de Machine Learning & MLOps  
**Dataset Base:** `data/processed/abt_sanidade_vegetal.csv` (6.571 instâncias)  

---

## 📌 Objetivo da Entrega
Construir, validar e homologar a integração atômica do estimador **Support Vector Classifier (SVC)** ao objeto unificado de `Pipeline` do Scikit-Learn (acoplado ao `ColumnTransformer` da Sprint 2). 

Esta etapa estabelece o contrato formal de inferência (`.fit()`, `.predict()`, `.predict_proba()`), comprova a ausência absoluta de *Data Leakage* em validação cruzada estratificada em 5 folds e prepara a esteira para a execução subsequente do **Grid Search Exaustivo (`GridSearchCV`)**.

### ✅ Checklist Oficial da Tarefa:
1. **Construir o objeto Pipeline Scikit-Learn completo** (Pré-processador da Sprint 2 + Estimador SVC).
2. **Garantir que o escalonamento (`StandardScaler`) e transformações ocorram isoladamente dentro de cada fold.**
3. **Validar a compatibilidade dos métodos `.fit()`, `.predict()` e `.predict_proba()`.**
4. **Assegurar ausência total de *Data Leakage* na execução combinada.**"""))

    # -------------------------------------------------------------------------
    # Célula 2: Imports e Configuração de Ambiente
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# 1. Configuração do ambiente e importação das dependências
import os
import sys
import json
import time
import hashlib
from pathlib import Path

# Adicionar diretório src ao path
sys.path.insert(0, str(Path.cwd().parent / "src" if Path.cwd().name == "notebooks" else Path.cwd() / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss,
    ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
import joblib

# Ativar visualização de diagramas HTML para pipelines no Scikit-Learn
set_config(display="diagram")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

print(f"Scikit-Learn versão: {sklearn.__version__}")
print(f"Pandas versão      : {pd.__version__}")
print(f"NumPy versão       : {np.__version__}")
print(f"Joblib versão      : {joblib.__version__}")"""))

    # -------------------------------------------------------------------------
    # Célula 3: Seção 1 - Ingestão da ABT e Construtor do Pipeline
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 1. Construção do Pipeline Atômico Scikit-Learn

A arquitetura unifica o pré-processador da Sprint 2 (`ColumnTransformer`, com `SimpleImputer`, `StandardScaler` e `OneHotEncoder`) ao estimador `SVC(probability=True, class_weight='balanced')`.

Essa integração garante que qualquer chamada de transformação ou ajuste seja atômica: dados brutos entram na ponta inicial do pipeline e probabilidades calibradas saem na extremidade final."""))

    # -------------------------------------------------------------------------
    # Célula 4: Código da Seção 1
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""from atomic_pipeline_preprocessing import (
    build_atomic_column_transformer,
    NUMERICAL_FEATURES,
    ALL_CATEGORICAL_FEATURES,
    TARGET_COLUMNS,
    SPLIT_COLUMN
)
from atomic_svm_pipeline import build_atomic_svm_pipeline

# Carregar ABT oficial
data_path = Path.cwd().parent / "data" / "processed" / "abt_sanidade_vegetal.csv" if Path.cwd().name == "notebooks" else Path.cwd() / "data" / "processed" / "abt_sanidade_vegetal.csv"
df = pd.read_csv(data_path)
print(f"ABT Carregada: {df.shape[0]} instâncias x {df.shape[1]} colunas.")

# Instanciar Pipeline Atômico
full_pipeline = build_atomic_svm_pipeline(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42,
    class_weight='balanced'
)

# Renderizar diagrama estrutural do pipeline
full_pipeline"""))

    # -------------------------------------------------------------------------
    # Célula 5: Seção 2 - Validação dos Métodos de Inferência
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 2. Validação Formal dos Métodos `.fit()`, `.predict()` e `.predict_proba()`

O estimador `SVC` requer calibração explícita via `probability=True` (baseada no método de Platt Scaling e validação cruzada interna de 5 folds) para produzir probabilidades fidedignas.

Validamos aqui o cumprimento de todo o protocolo de inferência nos dois níveis definidos no escopo:
1. **Multiclasse (Nível 2):** Diagnóstico diferencial das 7 classes fitopatológicas.
2. **Binário (Nível 1):** Triagem rápida folha Sadia ($y=0$) vs Patológica ($y=1$)."""))

    # -------------------------------------------------------------------------
    # Célula 6: Código da Seção 2
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# Segregação de partições (Treino vs Validação)
df_train = df[df[SPLIT_COLUMN] == 'train'].copy()
df_valid = df[df[SPLIT_COLUMN] == 'valid'].copy()

X_train = df_train.drop(columns=TARGET_COLUMNS)
y_train_multi = df_train['target_multiclass']
y_train_bin = df_train['target_binary']

X_valid = df_valid.drop(columns=TARGET_COLUMNS)
y_valid_multi = df_valid['target_multiclass']
y_valid_bin = df_valid['target_binary']

# Mapeamento oficial de rótulos multiclasse
class_names_map = {
    0: 'GRASSY SHOOT',
    1: 'HEALTHY',
    2: 'LEAF SCALD',
    3: 'MOSAIC',
    4: 'RED ROT',
    5: 'RUST',
    6: 'YELLOW LEAF'
}
target_names = [class_names_map[i] for i in range(7)]

print(f"Treino   : {len(X_train)} amostras")
print(f"Validação: {len(X_valid)} amostras")

# 1. Ajuste do Pipeline Multiclasse (.fit)
t0 = time.perf_counter()
full_pipeline.fit(X_train, y_train_multi)
t_fit = time.perf_counter() - t0
print(f"[OK] .fit() concluído em {t_fit:.2f}s com {full_pipeline.named_steps['svm'].support_vectors_.shape[0]} vetores de suporte.")

# 2. Predição (.predict)
y_pred_multi = full_pipeline.predict(X_valid)
print(f"[OK] .predict() concluído. Formato: {y_pred_multi.shape}")

# 3. Predição Probabilística (.predict_proba)
y_proba_multi = full_pipeline.predict_proba(X_valid)
print(f"[OK] .predict_proba() concluído. Formato: {y_proba_multi.shape}")

# Verificação estocástica estrita
row_sums = y_proba_multi.sum(axis=1)
np.testing.assert_allclose(row_sums, np.ones(len(X_valid)), rtol=1e-5, atol=1e-5)
print(f"[OK] Conformidade Estocástica: 100% das linhas somam 1.0 (Platt Scaling calibrado).")"""))

    # -------------------------------------------------------------------------
    # Célula 7: Seção 3 - Avaliação Baseline e Matriz de Confusão
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 3. Avaliação de Performance Baseline & Calibração

Apresentamos o relatório de classificação completo, matriz de confusão e curvas de calibração de probabilidade sobre o conjunto de validação independente."""))

    # -------------------------------------------------------------------------
    # Célula 8: Código da Seção 3
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# Relatório de Classificação Multiclasse
print("RELATÓRIO DE CLASSIFICAÇÃO MULTICLASSE (BASELINE RBF):")
print(classification_report(y_valid_multi, y_pred_multi, target_names=target_names, digits=4))

# Visualização da Matriz de Confusão
fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
cm = confusion_matrix(y_valid_multi, y_pred_multi)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap='Blues', ax=ax, colorbar=True, values_format='d')
plt.title("Matriz de Confusão — SVM Baseline RBF (Multiclasse)", fontsize=13, fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Salvar figura para documentação
fig_dir = Path.cwd().parent / "docs" / "figures" if Path.cwd().name == "notebooks" else Path.cwd() / "docs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "sprint3_svm_matriz_confusao_baseline.png", dpi=150)
plt.show()"""))

    # -------------------------------------------------------------------------
    # Célula 9: Código de Calibração de Probabilidades
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# Curvas de Calibração (Reliability Diagram) para as principais patologias
fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

for class_idx in [1, 4, 5]: # Healthy, Red Rot, Rust
    prob_pos = y_proba_multi[:, class_idx]
    y_true_binary = (y_valid_multi == class_idx).astype(int)
    prob_true, prob_pred = calibration_curve(y_true_binary, prob_pos, n_bins=5)
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f"Classe: {class_names_map[class_idx]}")

ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfeitamente Calibrado')
ax.set_title("Diagrama de Confiabilidade de Probabilidades (Platt Scaling)", fontsize=12, fontweight='bold')
ax.set_xlabel("Probabilidade Média Predita")
ax.set_ylabel("Fração de Positivos Reais")
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(fig_dir / "sprint3_svm_curvas_calibracao.png", dpi=150)
plt.show()"""))

    # -------------------------------------------------------------------------
    # Célula 10: Seção 4 - Auditoria de Isolamento de Folds (Anti-Leakage)
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 4. Auditoria Matemática de Isolamento por Fold em StratifiedKFold

### Prova Formal de Ausência de Vazamento de Dados (Anti-Leakage):
Para assegurar ausência total de contaminação estatística:
1. As médias ($\mu^{(k)}$) e desvios padrão ($\sigma^{(k)}$) calculados pelo `StandardScaler` **devem ser derivados estritamente do subconjunto de treino de cada fold**.
2. Os parâmetros estatísticos aprendidos em cada fold devem ser distintos entre si ($\mu^{(i)} \neq \mu^{(j)}$ para $i \neq j$) e diferir da média global da população, comprovando que o conjunto de validação permaneceu estritamente isolado."""))

    # -------------------------------------------------------------------------
    # Célula 11: Código da Seção 4
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""from atomic_svm_pipeline import audit_stratified_kfold_isolation

# Executar bateria de 5 folds com auditoria matemática em cada partição
cv_audit_results = audit_stratified_kfold_isolation(
    df=df,
    target_col='target_multiclass',
    n_splits=5,
    random_state=42
)

# Visualização gráfica do isolamento estatístico entre folds
fold_df = pd.DataFrame(cv_audit_results['fold_results'])

fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=120)
sns.barplot(data=fold_df, x='fold', y='f1_macro', palette='Blues_d', ax=ax1)
ax1.axhline(cv_audit_results['mean_f1_macro'], color='red', linestyle='--', 
            label=f"Média CV: {cv_audit_results['mean_f1_macro']:.4f} (±{cv_audit_results['std_f1_macro']:.4f})")
ax1.set_title("Consistência de Performance (F1-Macro) nos 5 Folds Estratificados", fontsize=12, fontweight='bold')
ax1.set_xlabel("Fold de Validação Cruzada")
ax1.set_ylabel("F1-Score (Macro)")
ax1.set_ylim(0.5, 0.65)
ax1.legend(loc='lower right')
plt.tight_layout()
plt.savefig(fig_dir / "sprint3_svm_kfold_isolation.png", dpi=150)
plt.show()"""))

    # -------------------------------------------------------------------------
    # Célula 12: Seção 5 - Prova de Imutabilidade Pós-Fit (Injeção Adversarial)
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 5. Prova de Imutabilidade Pós-Fit (Blindagem Adversarial)

Demonstramos empiricamente que a passagem de valores aberrantes ou anômalos no momento da inferência não altera os coeficientes do hiperplano do SVM nem as estatísticas do pré-processador."""))

    # -------------------------------------------------------------------------
    # Célula 13: Código da Seção 5
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""from atomic_svm_pipeline import validate_adversarial_non_leakage

adversarial_audit = validate_adversarial_non_leakage(full_pipeline, X_valid)
print(f"Status da Blindagem Adversarial: {adversarial_audit['status']}")
print(f"Risco de Vazamento Residual     : {adversarial_audit['leakage_risk_percentage']}%")"""))

    # -------------------------------------------------------------------------
    # Célula 14: Seção 6 - Governança MLOps e Serialização
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## 6. Governança MLOps & Prontidão para o GridSearchCV

Validamos a serialização do artefato `models/svm_pipeline_baseline.joblib` e a integridade do manifesto de governança `models/svm_pipeline_metadata.json`."""))

    # -------------------------------------------------------------------------
    # Célula 15: Código da Seção 6
    # -------------------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""models_dir = Path.cwd().parent / "models" if Path.cwd().name == "notebooks" else Path.cwd() / "models"
meta_path = models_dir / "svm_pipeline_metadata.json"

with open(meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)

print("MANIFESTO DE GOVERNANÇA MLOPS (Resumo Executivo):")
print(f"- Nome do Pipeline       : {meta['pipeline_name']}")
print(f"- Hash SHA-256           : {meta['artifact']['sha256']}")
print(f"- Tamanho do Artefato    : {meta['artifact']['size_kb']} KB")
print(f"- F1-Macro Médio (5-Fold): {meta['anti_leakage_audit']['mean_cv_f1_macro']}")
print(f"- Status Anti-Leakage    : {meta['anti_leakage_audit']['status']}")
print(f"- Prontidão p/ GridSearch: {meta['readiness_for_gridsearch']['status']}")
print(f"- Grade Recomendada      : {meta['readiness_for_gridsearch']['recommended_param_grid']}")"""))

    nb['cells'] = cells
    
    print(f"Gravando notebook preliminar em: {nb_path}")
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Executando notebook de ponta a ponta com NotebookClient...")
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()
    
    print(f"Salvando notebook executado com saídas completas em: {nb_path}")
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print("[OK] Notebook 08 executado e gravado com 100% de sucesso!")
    return nb_path


if __name__ == '__main__':
    create_and_run_svm_notebook()
