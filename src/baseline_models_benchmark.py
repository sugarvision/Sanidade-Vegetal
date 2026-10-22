"""
Módulo de Concepção, Treinamento e Benchmark de Modelos Baseline
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 3 — Framework SEMMA (Fase: Model)
Responsável: Elisa (Engenharia de Machine Learning & MLOps)

Checklist Coberto:
1. Instanciar e treinar um classificador ingênuo (DummyClassifier(strategy='stratified')) para fixar o piso de desempenho.
2. Treinar um modelo linear simples (LogisticRegression) com penalidade Ridge sobre a ABT normalizada.
3. Treinar uma árvore de decisão rasa (DecisionTreeClassifier(max_depth=4)) como baseline de interpretabilidade.
4. Implementar a ponderação de classes (class_weight='balanced') para mitigar o impacto do desbalanceamento amostral.
5. Consolidar uma tabela comparativa com as métricas preliminares (Acurácia, Precisão, Recall e F1-Score) dos baselines.
6. Comprovar o ganho estatístico obtido pelo SVM (Baseline RBF e Optimal RBF).
"""

import os
import sys
import json
import time
import hashlib
import warnings
from pathlib import Path

# Suprimir avisos desnecessários de bibliotecas externas
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

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
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss
)
from scipy.stats import ttest_rel
import joblib

# Importar construtor atômico e metadados de colunas
try:
    from atomic_pipeline_preprocessing import (
        build_atomic_column_transformer,
        NUMERICAL_FEATURES,
        ALL_CATEGORICAL_FEATURES,
        METADATA_DROPPED_COLUMNS,
        TARGET_COLUMNS,
        SPLIT_COLUMN
    )
except ImportError:
    from src.atomic_pipeline_preprocessing import (
        build_atomic_column_transformer,
        NUMERICAL_FEATURES,
        ALL_CATEGORICAL_FEATURES,
        METADATA_DROPPED_COLUMNS,
        TARGET_COLUMNS,
        SPLIT_COLUMN
    )


# Mapeamento oficial de patologias
CLASS_ID_TO_NAME = {
    0: "GRASSY SHOOT",
    1: "HEALTHY",
    2: "LEAF SCALD",
    3: "MOSAIC",
    4: "RED ROT",
    5: "RUST",
    6: "YELLOW LEAF"
}
CLASS_NAMES_ORDERED = [CLASS_ID_TO_NAME[i] for i in sorted(CLASS_ID_TO_NAME.keys())]


# ==============================================================================
# 1. FÁBRICAS DE INSTANCIAÇÃO DOS MODELOS BASELINE E REFERÊNCIA
# ==============================================================================

def build_dummy_baseline(
    strategy: str = 'stratified',
    random_state: int = 42
) -> DummyClassifier:
    """
    Instancia o Classificador Ingênuo (DummyClassifier) para estabelecer o piso
    estatístico de desempenho mínimo do projeto (baseline randômico empírico).
    """
    return DummyClassifier(strategy=strategy, random_state=random_state)


def build_logistic_regression_baseline(
    penalty: str = 'l2',
    C: float = 1.0,
    class_weight: str = 'balanced',
    max_iter: int = 1000,
    random_state: int = 42
) -> LogisticRegression:
    """
    Instancia a Regressão Logística com penalidade Ridge (L2) sobre a ABT normalizada.
    Atua como baseline linear probabilístico e paramétrico.
    """
    return LogisticRegression(
        penalty=penalty,
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs'
    )


def build_decision_tree_baseline(
    max_depth: int = 4,
    class_weight: str = 'balanced',
    criterion: str = 'gini',
    random_state: int = 42
) -> DecisionTreeClassifier:
    """
    Instancia a Árvore de Decisão rasa (max_depth=4) como baseline não linear de interpretabilidade.
    Permite inspeção visual direta de regras de decisão agronômicas.
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight=class_weight,
        criterion=criterion,
        random_state=random_state
    )


def build_svm_reference(
    kernel: str = 'rbf',
    C: float = 10.0,
    gamma: float = 0.01,
    class_weight: str = 'balanced',
    probability: bool = True,
    random_state: int = 42
) -> SVC:
    """
    Instancia o estimador SVM ótimo de referência (Sprint 3 Grid Search) para
    comprovação estatística de superioridade frente aos baselines.
    """
    return SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        probability=probability,
        random_state=random_state
    )


def build_atomic_baseline_pipeline(estimator_type: str = 'logistic_regression', **kwargs) -> Pipeline:
    """
    Encapsula o estimador baseline escolhido juntamente com o ColumnTransformer atômico
    para execução de ponta a ponta sobre dados brutos com garantia anti-leakage.
    """
    preprocessor = build_atomic_column_transformer()
    if estimator_type == 'dummy':
        clf = build_dummy_baseline(**kwargs)
    elif estimator_type == 'logistic_regression':
        clf = build_logistic_regression_baseline(**kwargs)
    elif estimator_type == 'decision_tree':
        clf = build_decision_tree_baseline(**kwargs)
    elif estimator_type == 'svm':
        clf = build_svm_reference(**kwargs)
    else:
        raise ValueError(f"Estimador desconhecido: {estimator_type}")
        
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        (estimator_type, clf)
    ])


# ==============================================================================
# 2. MOTOR DE VALIDAÇÃO CRUZADA E BENCHMARK MULTI-MÉTRICA
# ==============================================================================

def run_stratified_kfold_benchmark(
    models_dict: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Executa validação cruzada 5-Fold StratifiedKFold rigorosa para todos os modelos,
    coletando métricas fold a fold e calculando médias, desvios e folds individuais
    para testes de hipótese estatística.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_summary_records = []
    fold_raw_metrics = {name: {'accuracy': [], 'f1_macro': [], 'precision_macro': [], 'recall_macro': []} for name in models_dict.keys()}
    
    print("\n" + "="*85)
    print(f"BENCHMARK DE VALIDAÇÃO CRUZADA STRATIFIED K-FOLD (k={n_splits}) NO CONJUNTO DE TREINO (N={len(X_train)})")
    print("="*85)
    
    for name, model in models_dict.items():
        t0 = time.perf_counter()
        fold_accs = []
        fold_f1_macros = []
        fold_f1_weighteds = []
        fold_prec_macros = []
        fold_rec_macros = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
            X_tr_f, y_tr_f = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va_f, y_va_f = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            # Clonar ou instanciar cópia limpa do modelo
            clf = sklearn.base.clone(model)
            clf.fit(X_tr_f, y_tr_f)
            y_pred_f = clf.predict(X_va_f)
            
            acc = accuracy_score(y_va_f, y_pred_f)
            f1_mac = f1_score(y_va_f, y_pred_f, average='macro', zero_division=0)
            f1_wei = f1_score(y_va_f, y_pred_f, average='weighted', zero_division=0)
            prec_mac = precision_score(y_va_f, y_pred_f, average='macro', zero_division=0)
            rec_mac = recall_score(y_va_f, y_pred_f, average='macro', zero_division=0)
            
            fold_accs.append(acc)
            fold_f1_macros.append(f1_mac)
            fold_f1_weighteds.append(f1_wei)
            fold_prec_macros.append(prec_mac)
            fold_rec_macros.append(rec_mac)
            
            fold_raw_metrics[name]['accuracy'].append(acc)
            fold_raw_metrics[name]['f1_macro'].append(f1_mac)
            fold_raw_metrics[name]['precision_macro'].append(prec_mac)
            fold_raw_metrics[name]['recall_macro'].append(rec_mac)
            
        elapsed = time.perf_counter() - t0
        
        record = {
            "Modelo": name,
            "CV_Acuracia_Media": np.mean(fold_accs),
            "CV_Acuracia_Std": np.std(fold_accs),
            "CV_Precisao_Macro_Media": np.mean(fold_prec_macros),
            "CV_Precisao_Macro_Std": np.std(fold_prec_macros),
            "CV_Recall_Macro_Media": np.mean(fold_rec_macros),
            "CV_Recall_Macro_Std": np.std(fold_rec_macros),
            "CV_F1_Macro_Media": np.mean(fold_f1_macros),
            "CV_F1_Macro_Std": np.std(fold_f1_macros),
            "CV_F1_Weighted_Media": np.mean(fold_f1_weighteds),
            "Tempo_CV_Segundos": round(elapsed, 2)
        }
        cv_summary_records.append(record)
        
        print(f"[{name:44s}] Acc: {record['CV_Acuracia_Media']*100:5.2f}% (±{record['CV_Acuracia_Std']*100:4.2f}%) | "
              f"Prec: {record['CV_Precisao_Macro_Media']:5.4f} | Rec: {record['CV_Recall_Macro_Media']:5.4f} | "
              f"F1-Macro: {record['CV_F1_Macro_Media']:5.4f} (±{record['CV_F1_Macro_Std']:5.4f}) | "
              f"Tempo: {record['Tempo_CV_Segundos']:5.2f}s")
              
    df_cv_results = pd.DataFrame(cv_summary_records)
    return df_cv_results, fold_raw_metrics


# ==============================================================================
# 3. AVALIAÇÃO EM PARTIÇÃO HOLD-OUT (VALIDAÇÃO E TESTE) E NÍVEL DE CLASSE
# ==============================================================================

def evaluate_holdout_partitions(
    fitted_models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Avalia os modelos ajustados nas partições de Validação e Teste independentes.
    Gera métricas agregadas e relatórios detalhados por classe fitopatológica.
    """
    records = []
    per_class_records = []
    confusion_matrices = {}
    
    print("\n" + "="*85)
    print("AVALIAÇÃO EM PARTIÇÃO HOLD-OUT DE VALIDAÇÃO (N=617) E TESTE (N=380)")
    print("="*85)
    
    for name, model in fitted_models.items():
        # Predições na partição de validação
        t0 = time.perf_counter()
        y_pred_val = model.predict(X_valid)
        t_val = time.perf_counter() - t0
        
        acc_v = accuracy_score(y_valid, y_pred_val)
        prec_mac_v = precision_score(y_valid, y_pred_val, average='macro', zero_division=0)
        rec_mac_v = recall_score(y_valid, y_pred_val, average='macro', zero_division=0)
        f1_mac_v = f1_score(y_valid, y_pred_val, average='macro', zero_division=0)
        f1_wei_v = f1_score(y_valid, y_pred_val, average='weighted', zero_division=0)
        
        # Predições na partição de teste
        y_pred_test = model.predict(X_test)
        acc_t = accuracy_score(y_test, y_pred_test)
        prec_mac_t = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
        rec_mac_t = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
        f1_mac_t = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
        f1_wei_t = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        
        # Matriz de confusão da validação
        cm = confusion_matrix(y_valid, y_pred_val, labels=sorted(CLASS_ID_TO_NAME.keys()))
        confusion_matrices[name] = cm
        
        records.append({
            "Modelo": name,
            "Valid_Acuracia": acc_v,
            "Valid_Precisao_Macro": prec_mac_v,
            "Valid_Recall_Macro": rec_mac_v,
            "Valid_F1_Macro": f1_mac_v,
            "Valid_F1_Weighted": f1_wei_v,
            "Valid_Inferencia_ms": round(t_val * 1000, 2),
            "Test_Acuracia": acc_t,
            "Test_Precisao_Macro": prec_mac_t,
            "Test_Recall_Macro": rec_mac_t,
            "Test_F1_Macro": f1_mac_t,
            "Test_F1_Weighted": f1_wei_t
        })
        
        # Métricas por classe na Validação
        rep = classification_report(y_valid, y_pred_val, target_names=CLASS_NAMES_ORDERED, output_dict=True, zero_division=0)
        for class_name in CLASS_NAMES_ORDERED:
            per_class_records.append({
                "Modelo": name,
                "Classe": class_name,
                "Precision": rep[class_name]["precision"],
                "Recall": rep[class_name]["recall"],
                "F1-Score": rep[class_name]["f1-score"],
                "Suporte": rep[class_name]["support"]
            })
            
        print(f"[{name:44s}] Valid: Acc={acc_v*100:5.2f}% | F1-Macro={f1_mac_v:5.4f} | Prec={prec_mac_v:5.4f} | Rec={rec_mac_v:5.4f} || "
              f"Test: Acc={acc_t*100:5.2f}% | F1-Macro={f1_mac_t:5.4f}")
              
    df_holdout = pd.DataFrame(records)
    df_per_class = pd.DataFrame(per_class_records)
    return df_holdout, df_per_class, confusion_matrices


# ==============================================================================
# 4. COMPROVAÇÃO ESTATÍSTICA DO GANHO DO SVM (TESTE DE HIPÓTESES)
# ==============================================================================

def compute_statistical_gain_and_hypothesis_test(
    fold_raw_metrics: dict,
    reference_model_name: str = "SVM RBF Ótimo (C=10, γ=0.01, balanced)"
) -> pd.DataFrame:
    """
    Aplica o teste t pareado de Student (paired t-test) sobre os 5 folds de CV
    entre o modelo SVM de referência e cada um dos baselines.
    Comprova formalmente o ganho estatístico (p-value < 0.05).
    """
    ref_f1 = fold_raw_metrics[reference_model_name]['f1_macro']
    ref_acc = fold_raw_metrics[reference_model_name]['accuracy']
    
    stat_records = []
    print("\n" + "="*85)
    print(f"TESTE DE HIPÓTESE PAREADO (t-test) — COMPROVAÇÃO DO GANHO ESTATÍSTICO DO SVM")
    print(f"Hipótese Nula (H0): Desempenho do SVM = Desempenho do Baseline")
    print(f"Hipótese Alternativa (H1): Desempenho do SVM > Desempenho do Baseline (p-value < 0.05)")
    print("="*85)
    
    for name, metrics in fold_raw_metrics.items():
        if name == reference_model_name:
            continue
            
        base_f1 = metrics['f1_macro']
        base_acc = metrics['accuracy']
        
        # Teste t pareado para F1-Macro
        t_stat_f1, p_val_f1 = ttest_rel(ref_f1, base_f1)
        # Teste t pareado para Acurácia
        t_stat_acc, p_val_acc = ttest_rel(ref_acc, base_acc)
        
        gain_f1_abs = np.mean(ref_f1) - np.mean(base_f1)
        gain_f1_rel = (gain_f1_abs / np.mean(base_f1)) * 100 if np.mean(base_f1) > 0 else 0
        gain_acc_abs = np.mean(ref_acc) - np.mean(base_acc)
        gain_acc_rel = (gain_acc_abs / np.mean(base_acc)) * 100 if np.mean(base_acc) > 0 else 0
        
        stat_signif_f1 = "Sim (p < 0.01)" if p_val_f1 < 0.01 else ("Sim (p < 0.05)" if p_val_f1 < 0.05 else "Não (p >= 0.05)")
        stat_signif_acc = "Sim (p < 0.01)" if p_val_acc < 0.01 else ("Sim (p < 0.05)" if p_val_acc < 0.05 else "Não (p >= 0.05)")
        
        stat_records.append({
            "Baseline": name,
            "Ganho_Abs_F1_Macro": round(gain_f1_abs, 4),
            "Ganho_Rel_F1_Macro_Pct": round(gain_f1_rel, 2),
            "t_stat_F1": round(t_stat_f1, 3),
            "p_value_F1": round(p_val_f1, 6),
            "Significancia_F1": stat_signif_f1,
            "Ganho_Abs_Acuracia": round(gain_acc_abs, 4),
            "Ganho_Rel_Acuracia_Pct": round(gain_acc_rel, 2),
            "t_stat_Acuracia": round(t_stat_acc, 3),
            "p_value_Acuracia": round(p_val_acc, 6),
            "Significancia_Acuracia": stat_signif_acc
        })
        
        print(f"* Comparação com [{name}]:")
        print(f"  - Ganho F1-Macro : +{gain_f1_abs:5.4f} (+{gain_f1_rel:5.2f}%) | t={t_stat_f1:6.3f} | p-value={p_val_f1:8.6f} -> {stat_signif_f1}")
        print(f"  - Ganho Acurácia : +{gain_acc_abs*100:5.2f}% (+{gain_acc_rel:5.2f}%) | t={t_stat_acc:6.3f} | p-value={p_val_acc:8.6f} -> {stat_signif_acc}")
        
    return pd.DataFrame(stat_records)


# ==============================================================================
# 5. ANÁLISE DE INTERPRETABILIDADE E PONDERAÇÃO DE CLASSES
# ==============================================================================

def extract_decision_tree_interpretability(
    dt_model: DecisionTreeClassifier,
    feature_names: list,
    output_dir: Path
) -> dict:
    """
    Extrai regras de decisão textuais e gera visualização gráfica da árvore de decisão
    para validação agronômica e auditabilidade.
    """
    rules_text = export_text(dt_model, feature_names=feature_names)
    importances = pd.Series(dt_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    # Salvar regras em arquivo texto
    rules_path = output_dir / "baseline_decision_tree_rules.txt"
    with open(rules_path, 'w', encoding='utf-8') as f:
        f.write("=== REGRAS DE DECISÃO DA ÁRVORE RASA (max_depth=4) ===\n\n")
        f.write(rules_text)
        
    # Plotar árvore gráfica em alta resolução
    fig, ax = plt.subplots(figsize=(24, 12), dpi=200)
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=CLASS_NAMES_ORDERED,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax
    )
    plt.title("Baseline de Interpretabilidade: Árvore de Decisão Rasa (max_depth=4)\nSugarVision — Classificação Fitopatológica em Cana-de-Açúcar", fontsize=14, weight='bold', pad=15)
    plt.tight_layout()
    tree_plot_path = output_dir / "baseline_decision_tree_structure.png"
    fig.savefig(tree_plot_path)
    plt.close(fig)
    
    print(f"\n[OK] Árvore de decisão exportada com sucesso:")
    print(f"  - Regras textuais salvas em: {rules_path}")
    print(f"  - Gráfico da árvore salvo em: {tree_plot_path}")
    print(f"  - Top 5 Atributos Mais Relevantes na Árvore:")
    for feat, imp in importances.head(5).items():
        print(f"    * {feat:25s}: {imp*100:5.2f}%")
        
    return {
        "rules_text": rules_text,
        "feature_importances": importances.to_dict(),
        "tree_plot_path": str(tree_plot_path)
    }


def plot_benchmark_visualizations(
    df_cv: pd.DataFrame,
    df_holdout: pd.DataFrame,
    df_per_class: pd.DataFrame,
    confusion_matrices: dict,
    figures_dir: Path
):
    """
    Gera conjunto completo de gráficos diagnósticos para o dossiê da Sprint 3.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Gráfico de Barras: Comparativo Geral das Métricas de Validação Cruzada
    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    df_plot = df_cv.copy()
    # Ordem dos modelos
    metrics = ["CV_Acuracia_Media", "CV_Precisao_Macro_Media", "CV_Recall_Macro_Media", "CV_F1_Macro_Media"]
    metric_labels = ["Acurácia", "Precisão (Macro)", "Recall (Macro)", "F1-Score (Macro)"]
    
    x = np.arange(len(df_plot))
    width = 0.18
    
    colors = ['#2b5c8f', '#3690c0', '#67a9cf', '#02818a']
    for i, (m, label) in enumerate(zip(metrics, metric_labels)):
        ax.bar(x + i*width, df_plot[m], width, label=label, color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.5)
        for j, val in enumerate(df_plot[m]):
            ax.text(x[j] + i*width, val + 0.015, f"{val:.2f}", ha='center', va='bottom', fontsize=8, rotation=90)
            
    ax.set_title("Benchmark Comparativo dos Modelos Baseline vs SVM Referência (5-Fold CV Treino)", fontsize=13, weight='bold', pad=15)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df_plot["Modelo"], rotation=18, ha='right', fontsize=9, weight='semibold')
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Pontuação (0.0 - 1.0)", fontsize=11)
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.axhline(0.1428, color='crimson', linestyle='--', linewidth=1.2, label='Chute Aleatório Puro (1/7 = 14.3%)')
    plt.tight_layout()
    chart1_path = figures_dir / "baseline_models_benchmark_comparison.png"
    fig.savefig(chart1_path)
    plt.close(fig)
    print(f"[OK] Gráfico comparativo geral salvo em: {chart1_path}")
    
    # 2. Impacto da Ponderação de Classes (class_weight='balanced')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=200, sharey=True)
    
    lr_models = ["Regressão Logística L2 (Sem Ponderação)", "Regressão Logística L2 (class_weight='balanced')"]
    dt_models = ["Árvore Decisão rasa (Sem Ponderação)", "Árvore Decisão rasa (class_weight='balanced')"]
    
    df_lr = df_per_class[df_per_class["Modelo"].isin(lr_models)]
    df_dt = df_per_class[df_per_class["Modelo"].isin(dt_models)]
    
    sns.barplot(data=df_lr, x="Classe", y="Recall", hue="Modelo", ax=axes[0], palette=["#bdbdbd", "#238b45"])
    axes[0].set_title("Regressão Logística: Efeito de class_weight='balanced' no Recall", fontsize=11, weight='bold')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha='right', fontsize=8)
    axes[0].set_ylabel("Recall por Patologia", fontsize=10)
    axes[0].legend(loc='upper right', fontsize=8)
    
    sns.barplot(data=df_dt, x="Classe", y="Recall", hue="Modelo", ax=axes[1], palette=["#bdbdbd", "#fe9929"])
    axes[1].set_title("Árvore de Decisão: Efeito de class_weight='balanced' no Recall", fontsize=11, weight='bold')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel("")
    axes[1].legend(loc='upper right', fontsize=8)
    
    plt.suptitle("Mitigação do Desbalanceamento Amostral: Resgate das Classes Raras (Grassy Shoot e Leaf Scald)", fontsize=13, weight='bold', y=1.02)
    plt.tight_layout()
    chart2_path = figures_dir / "baseline_class_weight_impact.png"
    fig.savefig(chart2_path)
    plt.close(fig)
    print(f"[OK] Gráfico de impacto da ponderação salvo em: {chart2_path}")
    
    # 3. Matrizes de Confusão Comparativas (4 Modelos Chave na Validação)
    selected_cms = [
        ("Dummy (Stratified)", confusion_matrices.get("Dummy Classifier (Stratified)")),
        ("Decision Tree (Balanced)", confusion_matrices.get("Árvore Decisão rasa (class_weight='balanced')")),
        ("Logistic Regression (Ridge, Balanced)", confusion_matrices.get("Regressão Logística L2 (class_weight='balanced')")),
        ("SVM RBF Ótimo (Referência)", confusion_matrices.get("SVM RBF Ótimo (C=10, γ=0.01, balanced)"))
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=200)
    axes = axes.flatten()
    
    short_labels = ["GS", "HEA", "LS", "MOS", "RR", "RUST", "YL"]
    for i, (title, cm) in enumerate(selected_cms):
        if cm is not None:
            # Normalizar por linha (Recall por classe)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=short_labels,
                yticklabels=short_labels,
                ax=axes[i],
                cbar=False
            )
            axes[i].set_title(f"Matriz Normalizada: {title}", fontsize=11, weight='bold')
            axes[i].set_ylabel("Classe Real", fontsize=9)
            axes[i].set_xlabel("Classe Predita", fontsize=9)
            
    plt.suptitle("Matrizes de Confusão Normalizadas no Conjunto de Validação (N=617)", fontsize=14, weight='bold', y=0.99)
    plt.tight_layout()
    chart3_path = figures_dir / "baseline_confusion_matrices.png"
    fig.savefig(chart3_path)
    plt.close(fig)
    print(f"[OK] Matrizes de confusão salvas em: {chart3_path}")


# ==============================================================================
# 6. ORQUESTRADOR CENTRAL E EXPORTAÇÃO DE GOVERNANÇA MLOPS
# ==============================================================================

def execute_baseline_models_benchmark():
    """
    Executa o ciclo completo de benchmark, auditoria e exportação de artefatos.
    """
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    figures_dir = base_dir / "docs" / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    abt_modeling_path = data_dir / "abt_features_modelagem.csv"
    print(f"Carregando ABT Normalizada de: {abt_modeling_path}")
    df = pd.read_csv(abt_modeling_path)
    
    # Identificar colunas preditivas (35 features)
    feature_cols = [c for c in df.columns if c not in ['sample_id', 'split_partition', 'class_label', 'target_binary', 'target_multiclass']]
    print(f"* Total de Features Preditivas Normalizadas: {len(feature_cols)}")
    
    # Divisão determinística por partições oficiais
    train_df = df[df[SPLIT_COLUMN] == 'train'].reset_index(drop=True)
    valid_df = df[df[SPLIT_COLUMN] == 'valid'].reset_index(drop=True)
    test_df = df[df[SPLIT_COLUMN] == 'test'].reset_index(drop=True)
    
    X_train, y_train = train_df[feature_cols], train_df['target_multiclass']
    X_valid, y_valid = valid_df[feature_cols], valid_df['target_multiclass']
    X_test, y_test = test_df[feature_cols], test_df['target_multiclass']
    
    print(f"* Volumetria do Particionamento:")
    print(f"  - Treino     : {len(X_train)} instâncias (84,8%)")
    print(f"  - Validação  : {len(X_valid)} instâncias (9,4%)")
    print(f"  - Teste      : {len(X_test)} instâncias (5,8%)")
    
    # Instanciação do portfólio completo de modelos para o benchmark
    models_to_benchmark = {
        "Dummy Classifier (Stratified)": build_dummy_baseline(strategy='stratified', random_state=42),
        "Regressão Logística L2 (Sem Ponderação)": build_logistic_regression_baseline(penalty='l2', C=1.0, class_weight=None, random_state=42),
        "Regressão Logística L2 (class_weight='balanced')": build_logistic_regression_baseline(penalty='l2', C=1.0, class_weight='balanced', random_state=42),
        "Árvore Decisão rasa (Sem Ponderação)": build_decision_tree_baseline(max_depth=4, class_weight=None, random_state=42),
        "Árvore Decisão rasa (class_weight='balanced')": build_decision_tree_baseline(max_depth=4, class_weight='balanced', random_state=42),
        "SVM RBF Baseline (balanced)": SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=42),
        "SVM RBF Ótimo (C=10, γ=0.01, balanced)": build_svm_reference(kernel='rbf', C=10.0, gamma=0.01, class_weight='balanced', random_state=42)
    }
    
    # 1. Benchmark com Validação Cruzada 5-Fold
    df_cv, fold_raw_metrics = run_stratified_kfold_benchmark(
        models_dict=models_to_benchmark,
        X_train=X_train,
        y_train=y_train,
        n_splits=5,
        random_state=42
    )
    
    # 2. Ajuste dos modelos em 100% da base de treino e avaliação em Validação e Teste
    print("\n* Ajustando modelos no conjunto completo de Treino...")
    fitted_models = {}
    for name, model in models_to_benchmark.items():
        m_fit = sklearn.base.clone(model)
        m_fit.fit(X_train, y_train)
        fitted_models[name] = m_fit
        
    df_holdout, df_per_class, cms = evaluate_holdout_partitions(
        fitted_models=fitted_models,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test
    )
    
    # Unificar tabela mestre de benchmark
    df_benchmark_full = pd.merge(df_cv, df_holdout, on="Modelo")
    benchmark_csv_path = models_dir / "baseline_benchmark_results.csv"
    df_benchmark_full.to_csv(benchmark_csv_path, index=False, encoding='utf-8')
    print(f"\n[OK] Tabela consolidada de benchmark salva em: {benchmark_csv_path}")
    
    # Salvar métricas por classe
    per_class_csv_path = models_dir / "baseline_per_class_metrics.csv"
    df_per_class.to_csv(per_class_csv_path, index=False, encoding='utf-8')
    print(f"[OK] Métricas detalhadas por classe salvas em: {per_class_csv_path}")
    
    # 3. Teste de Hipótese Estatística de Ganho do SVM
    df_hypothesis = compute_statistical_gain_and_hypothesis_test(
        fold_raw_metrics=fold_raw_metrics,
        reference_model_name="SVM RBF Ótimo (C=10, γ=0.01, balanced)"
    )
    hyp_csv_path = models_dir / "baseline_statistical_significance.csv"
    df_hypothesis.to_csv(hyp_csv_path, index=False, encoding='utf-8')
    print(f"[OK] Auditoria de significância estatística salva em: {hyp_csv_path}")
    
    # 4. Extração de Interpretabilidade da Árvore de Decisão
    tree_model = fitted_models["Árvore Decisão rasa (class_weight='balanced')"]
    dt_interpretability = extract_decision_tree_interpretability(
        dt_model=tree_model,
        feature_names=feature_cols,
        output_dir=figures_dir
    )
    
    # 5. Geração de Gráficos Diagnósticos
    plot_benchmark_visualizations(
        df_cv=df_cv,
        df_holdout=df_holdout,
        df_per_class=df_per_class,
        confusion_matrices=cms,
        figures_dir=figures_dir
    )
    
    # 6. Serialização dos Artefatos de Modelagem (.joblib)
    artifacts = {}
    models_to_persist = {
        "baseline_dummy_model.joblib": fitted_models["Dummy Classifier (Stratified)"],
        "baseline_logistic_regression.joblib": fitted_models["Regressão Logística L2 (class_weight='balanced')"],
        "baseline_decision_tree.joblib": fitted_models["Árvore Decisão rasa (class_weight='balanced')"]
    }
    
    print("\n* Serializando modelos baseline em formato joblib...")
    for filename, model_obj in models_to_persist.items():
        save_path = models_dir / filename
        joblib.dump(model_obj, save_path)
        with open(save_path, 'rb') as f:
            h = hashlib.sha256(f.read()).hexdigest()
        size_kb = round(os.path.getsize(save_path) / 1024, 2)
        artifacts[filename] = {
            "path": str(save_path),
            "size_kb": size_kb,
            "sha256": h
        }
        print(f"  - {filename} ({size_kb} KB) -> SHA-256: {h}")
        
    # 7. Manifesto de Governança MLOps
    manifest = {
        "task_name": "Concepção, Treinamento e Benchmark de Modelos Baseline",
        "project": "Sanidade-Vegetal (SugarVision)",
        "sprint": "Sprint 3 — Framework SEMMA (Fase: Model)",
        "responsible": "Elisa (Engenharia de Machine Learning & MLOps)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "framework_versions": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "python": sys.version.split()[0]
        },
        "datasets": {
            "source_abt": str(abt_modeling_path),
            "total_instances": len(df),
            "train_instances": len(X_train),
            "valid_instances": len(X_valid),
            "test_instances": len(X_test),
            "feature_count": len(feature_cols)
        },
        "artifacts_persisted": artifacts,
        "benchmark_summary_cv_train": df_cv.to_dict(orient="records"),
        "benchmark_summary_holdout": df_holdout.to_dict(orient="records"),
        "statistical_hypothesis_test": df_hypothesis.to_dict(orient="records"),
        "decision_tree_top_features": list(dt_interpretability["feature_importances"].items())[:5]
    }
    
    manifest_path = models_dir / "baseline_models_metadata.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Manifesto MLOps de Governança salvo em: {manifest_path}")
    
    print("\n" + "="*85)
    print("CONCLUSAO COM 100% DE SUCESSO: BENCHMARK DE MODELOS BASELINE HOMOLOGADO!")
    print("="*85)
    return manifest


if __name__ == '__main__':
    execute_baseline_models_benchmark()
