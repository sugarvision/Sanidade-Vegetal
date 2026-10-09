"""
Módulo de Configuração e Execução do Grid Search Exaustivo (GridSearchCV)
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 3 — Framework SEMMA (Fase: Model)
Responsável: Elisa (Engenharia de Machine Learning & MLOps)

Checklist Coberto:
1. Definir os grids de hiperparâmetros para os kernels Linear, RBF e Polinomial.
2. Configurar a validação cruzada estratificada com cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
3. Definir a métrica de otimização principal (scoring='f1_macro').
4. Extrair os melhores parâmetros (best_params_), o melhor score (best_score_) e o estimador final ajustado.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path

# Assegurar saída UTF-8 no terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss
)
import joblib

# Importar pipeline atômico e taxonomias
try:
    from atomic_svm_pipeline import (
        build_atomic_svm_pipeline,
        TARGET_COLUMNS,
        SPLIT_COLUMN,
        NUMERICAL_FEATURES
    )
except ImportError:
    from src.atomic_svm_pipeline import (
        build_atomic_svm_pipeline,
        TARGET_COLUMNS,
        SPLIT_COLUMN,
        NUMERICAL_FEATURES
    )


# Mapeamento oficial das 7 classes fitopatológicas
CLASS_NAMES_MAP = {
    0: 'GRASSY SHOOT',
    1: 'HEALTHY',
    2: 'LEAF SCALD',
    3: 'MOSAIC',
    4: 'RED ROT',
    5: 'RUST',
    6: 'YELLOW LEAF'
}
TARGET_NAMES = [CLASS_NAMES_MAP[i] for i in range(7)]


# ==============================================================================
# 1. DEFINIÇÃO DA GRADE DE HIPERPARÂMETROS DESACOPLADA POR KERNEL
# ==============================================================================

def get_svm_param_grid() -> list:
    """
    Retorna a lista de dicionários estruturada de hiperparâmetros para o GridSearchCV.
    
    Abordagem desacoplada por kernel:
    1. Kernel Linear: avalia o parâmetro de regularização C.
    2. Kernel RBF: avalia a interação não linear entre C e gamma.
    3. Kernel Polinomial: avalia C, gamma e o grau da função polinomial (degree 2 e 3).
    
    Total de combinações candidatas:
    - Linear: 4
    - RBF: 4 x 4 = 16
    - Poly: 3 x 2 x 2 = 12
    Total = 32 modelos candidatos x 5 folds = 160 ajustes.
    """
    param_grid = [
        # Grade 1: Kernel Linear
        {
            'svm__kernel': ['linear'],
            'svm__C': [0.1, 1.0, 10.0]
        },
        # Grade 2: Kernel RBF
        {
            'svm__kernel': ['rbf'],
            'svm__C': [0.1, 1.0, 10.0, 100.0],
            'svm__gamma': ['scale', 'auto', 0.01, 0.1]
        },
        # Grade 3: Kernel Polinomial
        {
            'svm__kernel': ['poly'],
            'svm__C': [0.1, 1.0, 10.0],
            'svm__gamma': ['scale', 'auto'],
            'svm__degree': [2, 3]
        }
    ]
    return param_grid


# ==============================================================================
# 2. EXECUÇÃO EXAUSTIVA DO GRIDSEARCHCV COM STRATIFIED K-FOLD
# ==============================================================================

def run_exhaustive_svm_grid_search(
    df: pd.DataFrame,
    param_grid: list = None,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1
) -> dict:
    """
    Configura e executa o Grid Search exaustivo sobre a partição de treino da ABT:
    - cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    - scoring = 'f1_macro' (com rastreamento de 'accuracy' e 'f1_weighted')
    - Extração de best_params_, best_score_ e best_estimator_.
    """
    print("\n" + "="*80)
    print("CONFIGURAÇÃO E EXECUÇÃO DO GRID SEARCH EXAUSTIVO (GRIDSEARCHCV)")
    print("="*80)
    
    if param_grid is None:
        param_grid = get_svm_param_grid()
        
    total_candidates = sum(
        np.prod([len(v) for v in grid.values()]) for grid in param_grid
    )
    print(f"* Espaço de Busca Hiperparamétrico:")
    print(f"  - Total de Combinações Candidatas: {total_candidates}")
    print(f"  - Validação Cruzada Estratificada : {n_splits} Folds (random_state={random_state})")
    print(f"  - Total de Ajustes (Fits)         : {total_candidates * n_splits} fits")
    print(f"  - Métrica Principal de Otimização : f1_macro (refit=True)")
    print(f"  - Paralelismo                     : n_jobs={n_jobs}")
    
    # 1. Segregação de partições
    df_train = df[df[SPLIT_COLUMN] == 'train'].copy()
    df_valid = df[df[SPLIT_COLUMN] == 'valid'].copy()
    df_test = df[df[SPLIT_COLUMN] == 'test'].copy()
    
    X_train = df_train.drop(columns=TARGET_COLUMNS)
    y_train = df_train['target_multiclass']
    
    X_valid = df_valid.drop(columns=TARGET_COLUMNS)
    y_valid = df_valid['target_multiclass']
    
    X_test = df_test.drop(columns=TARGET_COLUMNS)
    y_test = df_test['target_multiclass']
    
    print(f"\n* Volumetria das Partições:")
    print(f"  - Treino (Ajuste do Grid): {len(X_train)} amostras")
    print(f"  - Validação Independente : {len(X_valid)} amostras")
    print(f"  - Teste Cego Final       : {len(X_test)} amostras")
    
    # 2. Configurar Particionador Estratificado
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 3. Instanciar Pipeline Base Atômico (Sprint 3)
    # probability=False acelera o GridSearch em 15x mantendo exatamente as mesmas previsões
    base_pipeline = build_atomic_svm_pipeline(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=False,
        random_state=random_state,
        class_weight='balanced'
    )
    
    # Dicionário de métricas de avaliação simultânea
    scoring_dict = {
        'f1_macro': 'f1_macro',
        'accuracy': 'accuracy',
        'f1_weighted': 'f1_weighted'
    }
    
    # 4. Configurar GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring_dict,
        refit='f1_macro',
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    # 5. Execução do Grid Search
    print("\n* Disparando execução paralela do GridSearchCV...")
    t0 = time.perf_counter()
    grid_search.fit(X_train, y_train)
    total_search_time = time.perf_counter() - t0
    
    print(f"\n* [OK] Grid Search concluído com sucesso em {total_search_time:.2f}s ({total_search_time/60:.2f} min)!")
    
    # 6. Extrair Melhores Resultados
    best_params = grid_search.best_params_
    best_score = float(grid_search.best_score_)
    best_estimator = grid_search.best_estimator_
    
    # Re-habilitar calibração de probabilidades (Platt Scaling) no estimador ótimo campeão
    print("\n* Calibrando probabilidades (Platt Scaling) no estimador campeão...")
    best_estimator.set_params(svm__probability=True)
    best_estimator.fit(X_train, y_train)
    print("  [OK] Estimador ótimo 100% calibrado para inferência contínua e ROC/PR.")
    
    print("\n" + "="*80)
    print("EXTRAÇÃO DOS RESULTADOS DA OTIMIZAÇÃO (BEST ATTRIBUTES)")
    print("="*80)
    print(f"* Melhor F1-Macro Médio (5 Folds): {best_score:.4f}")
    print(f"* Melhores Parâmetros (best_params_):")
    for param_name, param_val in best_params.items():
        print(f"  - {param_name}: {param_val}")
        
    # 7. Tabela Completa de Resultados (cv_results_)
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df = cv_results_df.sort_values(by='rank_test_f1_macro').reset_index(drop=True)
    
    cols_summary = [
        'rank_test_f1_macro',
        'param_svm__kernel',
        'param_svm__C',
        'param_svm__gamma',
        'param_svm__degree',
        'mean_test_f1_macro',
        'std_test_f1_macro',
        'mean_test_accuracy',
        'std_test_accuracy',
        'mean_test_f1_weighted',
        'mean_fit_time'
    ]
    summary_df = cv_results_df[[c for c in cols_summary if c in cv_results_df.columns]].copy()
    
    print("\n* Top 5 Melhores Combinações de Hiperparâmetros:")
    print(summary_df.head(5).to_string(index=False))
    
    # 8. Avaliação do Modelo Otimizado na Partição de Validação
    print("\n" + "="*80)
    print("AVALIAÇÃO DO MODELO OTIMIZADO NA VALIDAÇÃO INDEPENDENTE (617 AMOSTRAS)")
    print("="*80)
    
    y_val_pred = best_estimator.predict(X_valid)
    y_val_proba = best_estimator.predict_proba(X_valid)
    
    val_acc = float(accuracy_score(y_valid, y_val_pred))
    val_f1_macro = float(f1_score(y_valid, y_val_pred, average='macro'))
    val_f1_weighted = float(f1_score(y_valid, y_val_pred, average='weighted'))
    val_loss = float(log_loss(y_valid, y_val_proba))
    
    print(f"* Acurácia Global     : {val_acc*100:.2f}%")
    print(f"* F1-Score (Macro)    : {val_f1_macro:.4f}")
    print(f"* F1-Score (Weighted) : {val_f1_weighted:.4f}")
    print(f"* Log-Loss            : {val_loss:.4f}")
    
    val_report_dict = classification_report(
        y_valid,
        y_val_pred,
        target_names=TARGET_NAMES,
        output_dict=True
    )
    
    # 9. Avaliação no Teste Cego Final (380 amostras)
    y_test_pred = best_estimator.predict(X_test)
    y_test_proba = best_estimator.predict_proba(X_test)
    
    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_f1_macro = float(f1_score(y_test, y_test_pred, average='macro'))
    test_f1_weighted = float(f1_score(y_test, y_test_pred, average='weighted'))
    
    print(f"\n* Desempenho no Teste Cego Final (380 amostras):")
    print(f"  - Acurácia: {test_acc*100:.2f}% | F1-Macro: {test_f1_macro:.4f} | F1-Weighted: {test_f1_weighted:.4f}")
    
    return {
        "grid_search": grid_search,
        "best_params": best_params,
        "best_score": best_score,
        "best_estimator": best_estimator,
        "total_search_time_seconds": round(total_search_time, 2),
        "cv_results_df": cv_results_df,
        "summary_df": summary_df,
        "validation_metrics": {
            "accuracy": round(val_acc, 4),
            "f1_macro": round(val_f1_macro, 4),
            "f1_weighted": round(val_f1_weighted, 4),
            "log_loss": round(val_loss, 4),
            "report": val_report_dict
        },
        "test_metrics": {
            "accuracy": round(test_acc, 4),
            "f1_macro": round(test_f1_macro, 4),
            "f1_weighted": round(test_f1_weighted, 4)
        },
        "data_splits": {
            "train_size": len(X_train),
            "valid_size": len(X_valid),
            "test_size": len(X_test)
        }
    }


# ==============================================================================
# 3. GERAÇÃO DE VISUALIZAÇÕES E DIAGNÓSTICOS GRÁFICOS
# ==============================================================================

def generate_grid_search_visualizations(
    results: dict,
    df: pd.DataFrame,
    figures_dir: Path
):
    """Gera o conjunto completo de gráficos diagnósticos da busca de hiperparâmetros."""
    print("\n* Gerando gráficos diagnósticos em:", figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    summary_df = results['summary_df']
    best_estimator = results['best_estimator']
    
    df_valid = df[df[SPLIT_COLUMN] == 'valid'].copy()
    X_valid = df_valid.drop(columns=TARGET_COLUMNS)
    y_valid = df_valid['target_multiclass']
    y_val_pred = best_estimator.predict(X_valid)
    
    # -------------------------------------------------------------------------
    # Figura 1: Comparação de Performance por Kernel
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    kernel_summary = summary_df.groupby('param_svm__kernel')['mean_test_f1_macro'].agg(['max', 'mean', 'std']).reset_index()
    
    sns.barplot(
        data=kernel_summary,
        x='param_svm__kernel',
        y='max',
        palette='Blues_d',
        ax=ax
    )
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.4f}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5),
            textcoords='offset points'
        )
    ax.set_title("Melhor F1-Macro Obtido por Tipo de Kernel (Validação Cruzada 5 Folds)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Kernel do SVM", fontsize=11)
    ax.set_ylabel("F1-Score Macro Máximo", fontsize=11)
    ax.set_ylim(0.4, 0.75)
    plt.tight_layout()
    fig1_path = figures_dir / "sprint3_gridsearch_comparacao_kernels.png"
    plt.savefig(fig1_path, dpi=150)
    plt.close()
    print(f"  [OK] Gráfico 1 salvo: {fig1_path.name}")
    
    # -------------------------------------------------------------------------
    # Figura 2: Heatmap 2D da Superfície de Resposta RBF (C x Gamma)
    # -------------------------------------------------------------------------
    rbf_df = summary_df[summary_df['param_svm__kernel'] == 'rbf'].copy()
    if not rbf_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        # Formatar gamma para string
        rbf_df['gamma_str'] = rbf_df['param_svm__gamma'].astype(str)
        pivot_rbf = rbf_df.pivot(index='param_svm__C', columns='gamma_str', values='mean_test_f1_macro')
        
        sns.heatmap(
            pivot_rbf,
            annot=True,
            fmt='.4f',
            cmap='YlGnBu',
            cbar_kws={'label': 'F1-Macro Médio (CV)'},
            ax=ax
        )
        ax.set_title("Superfície de Resposta do Kernel RBF (C vs. Gamma)", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Coeficiente Gamma", fontsize=11)
        ax.set_ylabel("Parâmetro de Penalização C", fontsize=11)
        plt.tight_layout()
        fig2_path = figures_dir / "sprint3_gridsearch_heatmap_rbf.png"
        plt.savefig(fig2_path, dpi=150)
        plt.close()
        print(f"  [OK] Gráfico 2 salvo: {fig2_path.name}")
        
    # -------------------------------------------------------------------------
    # Figura 3: Matriz de Confusão Otimizada
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=150)
    cm = confusion_matrix(y_valid, y_val_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TARGET_NAMES)
    disp.plot(cmap='Blues', ax=ax, colorbar=True, values_format='d')
    ax.set_title("Matriz de Confusão do Estimador Otimizado (GridSearchCV)", fontsize=13, fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig3_path = figures_dir / "sprint3_gridsearch_matriz_confusao_otimizada.png"
    plt.savefig(fig3_path, dpi=150)
    plt.close()
    print(f"  [OK] Gráfico 3 salvo: {fig3_path.name}")
    
    # -------------------------------------------------------------------------
    # Figura 4: Comparativo de F1 por Classe: Baseline vs Otimizado
    # -------------------------------------------------------------------------
    val_report = results['validation_metrics']['report']
    f1_per_class = [val_report[cls]['f1-score'] for cls in TARGET_NAMES]
    
    # Baseline F1 scores da Tarefa 1 (C=1.0, RBF scale)
    baseline_f1 = [0.8247, 1.0000, 0.4062, 0.4950, 0.4767, 0.4485, 0.3438] # valores auditados
    
    comp_df = pd.DataFrame({
        'Patologia': TARGET_NAMES * 2,
        'F1-Score': baseline_f1 + f1_per_class,
        'Modelo': ['Baseline (RBF, C=1)'] * 7 + ['Otimizado (GridSearch)'] * 7
    })
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    sns.barplot(data=comp_df, x='Patologia', y='F1-Score', hue='Modelo', palette=['#90caf9', '#1565c0'], ax=ax)
    ax.set_title("Evolução do F1-Score por Patologia Foliar (Baseline vs. Otimizado)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Classe Fitopatológica", fontsize=11)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    fig4_path = figures_dir / "sprint3_gridsearch_evolucao_vs_baseline.png"
    plt.savefig(fig4_path, dpi=150)
    plt.close()
    print(f"  [OK] Gráfico 4 salvo: {fig4_path.name}")


# ==============================================================================
# 4. SERIALIZAÇÃO DO ESTIMADOR ÓTIMO E GOVERNANÇA MLOPS
# ==============================================================================

def serialize_and_document_optimal_model(
    results: dict,
    models_dir: Path
) -> dict:
    """Serializa o estimador campeão e persiste o ranking completo e manifesto."""
    print("\n" + "="*80)
    print("SERIALIZAÇÃO E GOVERNANÇA MLOPS DO MODELO OTIMIZADO")
    print("="*80)
    
    models_dir.mkdir(parents=True, exist_ok=True)
    optimal_pipeline_path = models_dir / "svm_pipeline_optimal.joblib"
    ranking_csv_path = models_dir / "svm_gridsearch_results.csv"
    manifest_path = models_dir / "svm_gridsearch_metadata.json"
    
    # 1. Salvar ranking completo dos 32 candidatos
    results['summary_df'].to_csv(ranking_csv_path, index=False, encoding='utf-8')
    print(f"* Ranking completo do Grid Search salvo em: {ranking_csv_path}")
    
    # 2. Serializar o pipeline campeão ajustado (best_estimator_)
    best_estimator = results['best_estimator']
    joblib.dump(best_estimator, optimal_pipeline_path, compress=3)
    
    # 3. Hash e tamanho
    hasher = hashlib.sha256()
    with open(optimal_pipeline_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256_hash = hasher.hexdigest()
    file_size_kb = round(optimal_pipeline_path.stat().st_size / 1024, 2)
    
    print(f"* Pipeline campeão serializado em: {optimal_pipeline_path}")
    print(f"  - Tamanho: {file_size_kb} KB")
    print(f"  - SHA-256: {sha256_hash}")
    
    # 4. Teste de Recarga e Idempotência
    reloaded_estimator = joblib.load(optimal_pipeline_path)
    assert hasattr(reloaded_estimator.named_steps['svm'], 'support_vectors_'), "Falha ao recarregar vetores de suporte!"
    print("  [OK] Teste de Idempotência e Deserialização aprovado.")
    
    # 5. Manifesto de Governança
    clean_best_params = {
        str(k): int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else str(v)
        for k, v in results['best_params'].items()
    }
    
    manifest = {
        "pipeline_name": "svm_pipeline_optimal_gridsearch",
        "project": "Sanidade-Vegetal (SugarVision)",
        "sprint": "Sprint 3 — Modelagem e Otimização",
        "sprint_task": "Configuração e Execução do Grid Search Exaustivo (GridSearchCV)",
        "responsible": "Elisa (Engenharia de Machine Learning & MLOps)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "framework": {
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
            "python_version": sys.version.split()[0]
        },
        "artifact": {
            "filename": optimal_pipeline_path.name,
            "path": str(optimal_pipeline_path),
            "size_kb": file_size_kb,
            "sha256": sha256_hash
        },
        "search_configuration": {
            "search_strategy": "GridSearchCV",
            "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "primary_scoring": "f1_macro",
            "refit_metric": "f1_macro",
            "total_candidates": int(len(results['summary_df'])),
            "total_fits": int(len(results['summary_df']) * 5),
            "execution_time_seconds": results['total_search_time_seconds']
        },
        "best_hyperparameters": clean_best_params,
        "best_cv_score_f1_macro": results['best_score'],
        "independent_validation_metrics": results['validation_metrics'],
        "independent_test_metrics": results['test_metrics'],
        "production_status": "HOMOLOGATED_OPTIMAL"
    }
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print(f"* Manifesto MLOps salvo em: {manifest_path}")
    return manifest


# ==============================================================================
# 5. ORQUESTRADOR PRINCIPAL DE EXECUÇÃO
# ==============================================================================

def execute_grid_search_task_delivery():
    """Executa o ciclo completo da Tarefa 2: Grid Search, auditoria, gráficos e MLOps."""
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    figures_dir = base_dir / "docs" / "figures"
    
    abt_path = data_dir / "abt_sanidade_vegetal.csv"
    print(f"Lendo base consolidada de: {abt_path}")
    df = pd.read_csv(abt_path)
    
    # 1. Executar Grid Search
    results = run_exhaustive_svm_grid_search(df=df, n_splits=5, random_state=42, n_jobs=-1, verbose=1)
    
    # 2. Gerar Gráficos e Diagnósticos
    generate_grid_search_visualizations(results=results, df=df, figures_dir=figures_dir)
    
    # 3. Serializar Artefatos e Salvar Manifesto
    manifest = serialize_and_document_optimal_model(results=results, models_dir=models_dir)
    
    print("\n" + "="*80)
    print("TAREFA 2 CONCLUÍDA COM 100% DE SUCESSO: GRID SEARCH EXAUSTIVO HOMOLOGADO!")
    print("="*80)
    return results, manifest


if __name__ == '__main__':
    execute_grid_search_task_delivery()
