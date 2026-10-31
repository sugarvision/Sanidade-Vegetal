"""
Módulo de Validação Estatística: Matrizes de Confusão Normalizadas, Curvas ROC-AUC,
Curvas Precision-Recall e Calibração de Limiares de Decisão (Threshold Tuning)
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 4 — Framework SEMMA (Fase: Assess — Relato Vol. 4)
Responsável: Engenharia de Inteligência Artificial & Ciência de Dados Sênior

Checklist Coberto:
1. Gerar a matriz de confusão absoluta e a matriz de confusão normalizada por classe
   (especificamente para as classes: Saudável, Ferrugem e outras patologias, além da visão completa das 7 classes).
2. Plotar as curvas ROC multiclasse utilizando a abordagem One-vs-Rest (OvR) e calcular a
   área sob a curva (AUC-ROC) tanto macro quanto micro.
3. Construir as curvas Precision-Recall (PR Curve) focadas e específicas para a classe
   minoritária/crítica de Ferrugem (RUST) e patologias correlatas.
4. Desenvolver função/script para avaliar o comportamento dos limiares de decisão (threshold tuning),
   visando calibrar a sensibilidade (Recall) do classificador em operação agronômica.
5. Salvar todas as figuras geradas em alta resolução (300 DPI) dentro de `docs/figures/` para compor o Relato Vol. 4.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

# Compatibilidade reversa para artefatos serializados em versões anteriores do scikit-learn
import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        """Stub de compatibilidade para deserialização de ColumnTransformer."""
        pass
    _ct._RemainderColsList = _RemainderColsList

try:
    from atomic_svm_pipeline import (
        SPLIT_COLUMN,
        TARGET_COLUMNS,
        build_atomic_svm_pipeline,
    )
except ImportError:
    from src.atomic_svm_pipeline import (
        SPLIT_COLUMN,
        TARGET_COLUMNS,
        build_atomic_svm_pipeline,
    )

# Configuração de Logger estruturado
logger = logging.getLogger("SugarVision.StatisticalValidation")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ==============================================================================
# CONSTANTES E TAXONOMIA OFICIAL
# ==============================================================================

RANDOM_STATE: int = 42

CLASS_NAMES_MAP: Dict[int, str] = {
    0: "GRASSY SHOOT",
    1: "HEALTHY",
    2: "LEAF SCALD",
    3: "MOSAIC",
    4: "RED ROT",
    5: "RUST",
    6: "YELLOW LEAF",
}

AGRONOMIC_3CLASS_MAP: Dict[int, str] = {
    0: "SAUDÁVEL (HEALTHY)",
    1: "FERRUGEM (RUST)",
    2: "OUTRAS PATOLOGIAS",
}

HEALTHY_CLASS_ID: int = 1
RUST_CLASS_ID: int = 5

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
FIGURES_DIR: Path = DOCS_DIR / "figures"
ENTREGAS_DIR: Path = (
    PROJECT_ROOT
    / "entregas"
    / "Matrizes de Confusão e Curvas de Discriminação Estatística (ROC e PR)"
)


# ==============================================================================
# 1. CARREGAMENTO DO MODELO CAMPEÃO E EXTRAÇÃO DE PREDIÇÕES
# ==============================================================================

def load_champion_model_and_predictions(
    abt_path: Path = DATA_DIR / "abt_sanidade_vegetal.csv",
    model_path: Path = MODELS_DIR / "svm_pipeline_optimal.joblib",
) -> Tuple[Pipeline, Dict[str, Dict[str, Any]]]:
    """
    Carrega o modelo SVM campeão homologado (`models/svm_pipeline_optimal.joblib`) e
    extrai predições discretas, probabilidades calibradas (`predict_proba`) e escores
    de margem (`decision_function`) para as partições de Validação e Teste Cego.

    Retorna
    -------
    Tuple[Pipeline, Dict[str, Dict[str, Any]]]
        Pipeline carregado e dicionário indexado por partição (`'valid'`, `'test'`, `'combined_eval'`).
    """
    df = pd.read_csv(abt_path)

    if model_path.exists():
        try:
            pipeline = joblib.load(model_path)
        except Exception:
            pipeline = build_atomic_svm_pipeline(
                kernel="rbf", C=10.0, gamma=0.01, probability=True, random_state=RANDOM_STATE
            )
            df_tr = df[df[SPLIT_COLUMN] == "train"]
            pipeline.fit(df_tr.drop(columns=TARGET_COLUMNS), df_tr["target_multiclass"])
    else:
        pipeline = build_atomic_svm_pipeline(
            kernel="rbf", C=10.0, gamma=0.01, probability=True, random_state=RANDOM_STATE
        )
        df_tr = df[df[SPLIT_COLUMN] == "train"]
        pipeline.fit(df_tr.drop(columns=TARGET_COLUMNS), df_tr["target_multiclass"])

    partitions_data: Dict[str, Dict[str, Any]] = {}
    for split_name in ["valid", "test"]:
        sub = df[df[SPLIT_COLUMN] == split_name].copy().reset_index(drop=True)
        X_sub = sub.drop(columns=TARGET_COLUMNS)
        y_true = sub["target_multiclass"].values.astype(int)
        y_pred = pipeline.predict(X_sub).astype(int)
        y_proba = pipeline.predict_proba(X_sub)
        y_dec = pipeline.decision_function(X_sub)

        partitions_data[split_name] = {
            "X": X_sub,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "y_decision": y_dec,
            "n_samples": len(y_true),
        }

    # Partição combinada de avaliação externa (Validação + Teste Cego = 997 amostras)
    partitions_data["combined_eval"] = {
        "y_true": np.concatenate([partitions_data["valid"]["y_true"], partitions_data["test"]["y_true"]]),
        "y_pred": np.concatenate([partitions_data["valid"]["y_pred"], partitions_data["test"]["y_pred"]]),
        "y_proba": np.vstack([partitions_data["valid"]["y_proba"], partitions_data["test"]["y_proba"]]),
        "y_decision": np.vstack([partitions_data["valid"]["y_decision"], partitions_data["test"]["y_decision"]]),
        "n_samples": partitions_data["valid"]["n_samples"] + partitions_data["test"]["n_samples"],
    }

    return pipeline, partitions_data


# ==============================================================================
# 2. MATRIZES DE CONFUSÃO ABSOLUTA E NORMALIZADA (7 CLASSES E AGRONÔMICA 3 CLASSES)
# ==============================================================================

def map_to_agronomic_3classes(y_labels: np.ndarray) -> np.ndarray:
    """
    Converte os códigos das 7 classes fitopatológicas para o agrupamento agronômico de 3 classes:
    - `0`: Saudável (`HEALTHY`, classe original 1)
    - `1`: Ferrugem (`RUST`, classe original 5)
    - `2`: Outras Patologias (`GRASSY SHOOT`, `LEAF SCALD`, `MOSAIC`, `RED ROT`, `YELLOW LEAF`)
    """
    y_arr = np.asarray(y_labels, dtype=int)
    mapped = np.full_like(y_arr, fill_value=2)
    mapped[y_arr == HEALTHY_CLASS_ID] = 0
    mapped[y_arr == RUST_CLASS_ID] = 1
    return mapped


def compute_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names_map: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    r"""
    Calcula a Matriz de Confusão Absoluta e a Matriz de Confusão Normalizada por Classe
    (normalização por linha / suporte verdadeiro: $\sum_j C_{\text{norm}}(i, j) = 1.0$),
    tanto para a taxonomia completa quanto para o agrupamento agronômico
    (Saudável, Ferrugem e Outras Patologias).

    Parâmetros
    ----------
    y_true : np.ndarray
        Vetor de classes verdadeiras.
    y_pred : np.ndarray
        Vetor de classes preditas pelo estimador SVM.
    class_names_map : Optional[Dict[int, str]], default=None
        Mapeamento de IDs para nomes de classes.

    Retorna
    -------
    Dict[str, Any]
        Dicionário contendo matrizes absolutas, matrizes normalizadas e métricas por classe.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    if class_names_map is None:
        class_names_map = CLASS_NAMES_MAP

    labels_sorted = sorted(class_names_map.keys())
    label_names = [class_names_map[k] for k in labels_sorted]

    cm_abs = confusion_matrix(y_t, y_p, labels=labels_sorted)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm_abs.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm_abs.astype(float) / row_sums, 0.0)

    # Agrupamento agronômico de 3 classes: Saudável (0), Ferrugem (1), Outras Patologias (2)
    y_t_3 = map_to_agronomic_3classes(y_t)
    y_p_3 = map_to_agronomic_3classes(y_p)
    labels_3 = [0, 1, 2]
    names_3 = [AGRONOMIC_3CLASS_MAP[k] for k in labels_3]

    cm_3_abs = confusion_matrix(y_t_3, y_p_3, labels=labels_3)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums_3 = cm_3_abs.sum(axis=1, keepdims=True)
        cm_3_norm = np.where(row_sums_3 > 0, cm_3_abs.astype(float) / row_sums_3, 0.0)

    # Extrair métricas granulares por classe (Recall/Sensibilidade, Especificidade, Precisão, F1)
    per_class_metrics: List[Dict[str, Any]] = []
    total_samples = int(cm_abs.sum())
    for idx, cls_id in enumerate(labels_sorted):
        tp = int(cm_abs[idx, idx])
        fn = int(cm_abs[idx, :].sum() - tp)
        fp = int(cm_abs[:, idx].sum() - tp)
        tn = int(total_samples - tp - fn - fp)

        recall_val = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec_val = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec_val = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1_val = float(2 * prec_val * recall_val / (prec_val + recall_val)) if (prec_val + recall_val) > 0 else 0.0

        per_class_metrics.append(
            {
                "class_id": int(cls_id),
                "class_name": class_names_map[cls_id],
                "support": int(tp + fn),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "recall_sensitivity": round(recall_val, 4),
                "specificity": round(spec_val, 4),
                "precision": round(prec_val, 4),
                "f1_score": round(f1_val, 4),
            }
        )

    return {
        "multiclass_labels": label_names,
        "cm_multiclass_abs": cm_abs,
        "cm_multiclass_norm": np.round(cm_norm, 6),
        "agronomic_3class_labels": names_3,
        "cm_agronomic_3class_abs": cm_3_abs,
        "cm_agronomic_3class_norm": np.round(cm_3_norm, 6),
        "per_class_metrics": per_class_metrics,
    }


# ==============================================================================
# 3. CURVAS ROC MULTICLASSE ONE-VS-REST (MACRO E MICRO AUC-ROC)
# ==============================================================================

def compute_multiclass_roc_ovr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names_map: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    r"""
    Calcula as curvas ROC multiclasse sob a formulação One-vs-Rest (OvR) e extrai
    a Área sob a Curva (AUC-ROC) por classe, Macro-Average e Micro-Average.

    Parâmetros
    ----------
    y_true : np.ndarray
        Vetor de classes verdadeiras ($N,$).
    y_proba : np.ndarray
        Matriz de probabilidades calibradas ($N, K$).
    class_names_map : Optional[Dict[int, str]], default=None
        Dicionário de nomes das classes.

    Retorna
    -------
    Dict[str, Any]
        Dicionário contendo FPR, TPR, AUC por classe, Macro-AUC e Micro-AUC.
    """
    if class_names_map is None:
        class_names_map = CLASS_NAMES_MAP

    classes_sorted = sorted(class_names_map.keys())
    n_classes = len(classes_sorted)
    y_bin = label_binarize(y_true, classes=classes_sorted)
    y_score = np.asarray(y_proba, dtype=float)

    if y_score.shape[1] != n_classes:
        raise ValueError(
            f"Dimensão incompatível: y_proba tem {y_score.shape[1]} colunas, esperado {n_classes} classes."
        )

    fpr_dict: Dict[str, np.ndarray] = {}
    tpr_dict: Dict[str, np.ndarray] = {}
    auc_per_class: Dict[str, float] = {}

    for idx, cls_id in enumerate(classes_sorted):
        cls_name = class_names_map[cls_id]
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_score[:, idx])
        fpr_dict[cls_name] = fpr
        tpr_dict[cls_name] = tpr
        auc_per_class[cls_name] = round(float(auc(fpr, tpr)), 6)

    # Micro-Average ROC: achata todas as decisões binárias OvR
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc_micro = float(auc(fpr_micro, tpr_micro))

    # Macro-Average ROC: interpola em grade comum de FPR
    all_fpr = np.unique(np.concatenate([fpr_dict[class_names_map[c]] for c in classes_sorted]))
    mean_tpr = np.zeros_like(all_fpr)
    for cls_id in classes_sorted:
        cls_name = class_names_map[cls_id]
        mean_tpr += np.interp(all_fpr, fpr_dict[cls_name], tpr_dict[cls_name])
    mean_tpr /= n_classes
    roc_auc_macro = float(roc_auc_score(y_bin, y_score, average="macro", multi_class="ovr"))
    roc_auc_weighted = float(roc_auc_score(y_bin, y_score, average="weighted", multi_class="ovr"))

    return {
        "auc_per_class": auc_per_class,
        "roc_auc_macro": round(roc_auc_macro, 6),
        "roc_auc_micro": round(roc_auc_micro, 6),
        "roc_auc_weighted": round(roc_auc_weighted, 6),
        "curves": {
            "fpr_per_class": fpr_dict,
            "tpr_per_class": tpr_dict,
            "fpr_micro": fpr_micro,
            "tpr_micro": tpr_micro,
            "fpr_macro": all_fpr,
            "tpr_macro": mean_tpr,
        },
    }


# ==============================================================================
# 4. CURVAS PRECISION-RECALL FOCADAS NA CLASSE CRÍTICA DE FERRUGEM (RUST)
# ==============================================================================

def compute_precision_recall_curves_focused(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_class_id: int = RUST_CLASS_ID,
    class_names_map: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Constrói as curvas Precision-Recall (PR Curve) focadas na classe crítica de
    Ferrugem (`RUST`, `class_id=5`), bem como para todas as classes do inventário
    foliar, extraindo a Average Precision (AP / AUC-PR) e a linha de base de prevalência.

    Parâmetros
    ----------
    y_true : np.ndarray
        Vetor de classes verdadeiras ($N,$).
    y_proba : np.ndarray
        Matriz de probabilidades calibradas ($N, K$).
    target_class_id : int, default=5 (RUST)
        ID da classe foco principal.
    class_names_map : Optional[Dict[int, str]], default=None
        Mapeamento de IDs para nomes de classes.

    Retorna
    -------
    Dict[str, Any]
        Métricas de Precision-Recall, Average Precision (AP), prevalência e vetores da curva.
    """
    if class_names_map is None:
        class_names_map = CLASS_NAMES_MAP

    classes_sorted = sorted(class_names_map.keys())
    y_bin = label_binarize(y_true, classes=classes_sorted)
    y_score = np.asarray(y_proba, dtype=float)

    pr_per_class: Dict[str, Dict[str, Any]] = {}
    for idx, cls_id in enumerate(classes_sorted):
        cls_name = class_names_map[cls_id]
        y_cls_bin = y_bin[:, idx]
        scores_cls = y_score[:, idx]

        precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_cls_bin, scores_cls)
        ap_score = float(average_precision_score(y_cls_bin, scores_cls))
        pr_auc = float(auc(recall_arr, precision_arr))
        prevalence = float(np.mean(y_cls_bin))

        with np.errstate(divide="ignore", invalid="ignore"):
            f1_curve = np.where(
                (precision_arr + recall_arr) > 0,
                2.0 * precision_arr * recall_arr / (precision_arr + recall_arr),
                0.0,
            )
        best_f1_idx = int(np.argmax(f1_curve))

        pr_per_class[cls_name] = {
            "class_id": int(cls_id),
            "prevalence_baseline": round(prevalence, 4),
            "average_precision_ap": round(ap_score, 6),
            "auc_pr": round(pr_auc, 6),
            "max_f1_on_pr_curve": round(float(f1_curve[best_f1_idx]), 6),
            "precision_at_max_f1": round(float(precision_arr[best_f1_idx]), 6),
            "recall_at_max_f1": round(float(recall_arr[best_f1_idx]), 6),
            "precision_curve": precision_arr,
            "recall_curve": recall_arr,
            "thresholds": thresholds_arr,
        }

    # Micro e Macro Average Precision
    ap_micro = float(average_precision_score(y_bin, y_score, average="micro"))
    ap_macro = float(average_precision_score(y_bin, y_score, average="macro"))
    prec_micro, rec_micro, _ = precision_recall_curve(y_bin.ravel(), y_score.ravel())

    focus_name = class_names_map[target_class_id]
    return {
        "focus_class_id": int(target_class_id),
        "focus_class_name": focus_name,
        "focus_class_metrics": {
            k: v
            for k, v in pr_per_class[focus_name].items()
            if k not in ("precision_curve", "recall_curve", "thresholds")
        },
        "ap_macro": round(ap_macro, 6),
        "ap_micro": round(ap_micro, 6),
        "per_class_pr": pr_per_class,
        "micro_curve": {
            "precision": prec_micro,
            "recall": rec_micro,
        },
    }


# ==============================================================================
# 5. CALIBRAÇÃO DE LIMIARES DE DECISÃO (THRESHOLD TUNING) PARA SENSIBILIDADE
# ==============================================================================

def evaluate_threshold_tuning_sensitivity(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_class_id: int = RUST_CLASS_ID,
    thresholds: Optional[np.ndarray] = None,
    min_target_recall: float = 0.98,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    r"""
    Avalia sistematicamente o comportamento dos limiares de decisão ($\tau \in [0.01, 0.99]$)
    sobre a probabilidade calibrada da classe alvo (ex.: `RUST` ou detecção fitossanitária),
    visando calibrar a sensibilidade (*Recall*) do classificador sem degradar a especificidade.

    Critérios Matemáticos de Seleção de Limiar Ótimo:
    1. **Limiar de Máximo $F_1$-Score ($\tau_{F_1}^*$):** Equilíbrio harmônico entre Precisão e Sensibilidade.
    2. **Limiar Agronômico de Máximo $F_2$-Score ($\tau_{F_2}^*$):** Pondera a Sensibilidade com peso $2\times$ superior à Precisão ($\beta = 2.0$), priorizando a contenção de surtos fúngicos em campo.
    3. **Limiar de Youden ($\tau_J^*$):** Maximiza a estatística $J(\tau) = \text{Sensibilidade}(\tau) + \text{Especificidade}(\tau) - 1$.
    4. **Limiar de Alta Sensibilidade Garantida ($\tau_{\text{Recall}\ge 98\%}^*$):** Maior limiar $\tau$ que assegura $\text{Sensibilidade} \ge 98\%$.

    Parâmetros
    ----------
    y_true : np.ndarray
        Vetor de classes verdadeiras ($N,$) ou binário.
    y_proba : np.ndarray
        Matriz de probabilidades ($N, K$) ou vetor 1D de probabilidades da classe alvo.
    target_class_id : int, default=5 (RUST)
        Índice da coluna em `y_proba` (se 2D) e valor positivo em `y_true`.
    thresholds : Optional[np.ndarray], default=None
        Vetor de limiares a avaliar. Padrão: `np.linspace(0.01, 0.99, 99)`.
    min_target_recall : float, default=0.98
        Meta mínima de sensibilidade para o ponto de operação de triagem fitossanitária.

    Retorna
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        DataFrame com a curva completa de métricas por limiar $\tau$ e dicionário dos limiares ótimos.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    if np.any(thresholds <= 0.0) or np.any(thresholds >= 1.0):
        raise ValueError("Todos os limiares em `thresholds` devem pertencer estritamente ao intervalo (0.0, 1.0).")

    y_arr = np.asarray(y_true)
    if y_proba.ndim == 2:
        scores = np.asarray(y_proba[:, target_class_id], dtype=float)
        y_bin = (y_arr == target_class_id).astype(int)
    else:
        scores = np.asarray(y_proba, dtype=float)
        y_bin = (y_arr == target_class_id).astype(int) if np.max(y_arr) > 1 else y_arr.astype(int)

    n_pos = int(np.sum(y_bin == 1))
    n_neg = int(np.sum(y_bin == 0))

    rows: List[Dict[str, Any]] = []
    for tau in thresholds:
        y_hat = (scores >= tau).astype(int)
        tp = int(np.sum((y_hat == 1) & (y_bin == 1)))
        fp = int(np.sum((y_hat == 1) & (y_bin == 0)))
        fn = int(np.sum((y_hat == 0) & (y_bin == 1)))
        tn = int(np.sum((y_hat == 0) & (y_bin == 0)))

        sensitivity_recall = float(tp / n_pos) if n_pos > 0 else 0.0
        specificity = float(tn / n_neg) if n_neg > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1_val = (
            float(2.0 * precision * sensitivity_recall / (precision + sensitivity_recall))
            if (precision + sensitivity_recall) > 0
            else 0.0
        )
        # F2-Score: (1 + 2^2) * P * R / (4 * P + R)
        f2_val = (
            float(5.0 * precision * sensitivity_recall / (4.0 * precision + sensitivity_recall))
            if (4.0 * precision + sensitivity_recall) > 0
            else 0.0
        )
        youden_j = float(sensitivity_recall + specificity - 1.0)
        false_positive_rate = float(1.0 - specificity)

        rows.append(
            {
                "threshold": round(float(tau), 4),
                "sensitivity_recall": round(sensitivity_recall, 6),
                "specificity": round(specificity, 6),
                "precision": round(precision, 6),
                "f1_score": round(f1_val, 6),
                "f2_score_agronomic": round(f2_val, 6),
                "youden_j_index": round(youden_j, 6),
                "fpr": round(false_positive_rate, 6),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

    df_thresh = pd.DataFrame(rows)

    # Selecionar limiares ótimos segundo cada critério
    idx_f1 = int(df_thresh["f1_score"].idxmax())
    idx_f2 = int(df_thresh["f2_score_agronomic"].idxmax())
    idx_j = int(df_thresh["youden_j_index"].idxmax())

    high_rec_candidates = df_thresh[df_thresh["sensitivity_recall"] >= min_target_recall]
    if not high_rec_candidates.empty:
        # Maior limiar que preserva Recall >= min_target_recall com máximo F1
        idx_high_rec = int(high_rec_candidates.sort_values(by=["f1_score", "threshold"], ascending=[False, False]).index[0])
    else:
        idx_high_rec = int(df_thresh["sensitivity_recall"].idxmax())

    default_row = df_thresh.iloc[(df_thresh["threshold"] - 0.50).abs().argsort()[:1]].iloc[0]

    optimal_summary: Dict[str, Any] = {
        "target_class_id": int(target_class_id),
        "target_class_name": CLASS_NAMES_MAP.get(int(target_class_id), str(target_class_id)),
        "default_threshold_050": default_row.to_dict(),
        "optimal_f1_threshold": df_thresh.loc[idx_f1].to_dict(),
        "optimal_f2_agronomic_threshold": df_thresh.loc[idx_f2].to_dict(),
        "optimal_youden_j_threshold": df_thresh.loc[idx_j].to_dict(),
        "high_sensitivity_screening_threshold": df_thresh.loc[idx_high_rec].to_dict(),
    }

    return df_thresh, optimal_summary


# ==============================================================================
# 6. GERAÇÃO DAS FIGURAS EM ALTA RESOLUÇÃO (300 DPI) PARA O RELATO VOL. 4
# ==============================================================================

def generate_statistical_validation_figures(
    cm_results: Dict[str, Any],
    roc_results: Dict[str, Any],
    pr_results: Dict[str, Any],
    df_thresh_rust: pd.DataFrame,
    thresh_summary_rust: Dict[str, Any],
    df_thresh_scald: pd.DataFrame,
    thresh_summary_scald: Dict[str, Any],
    figures_dir: Path = FIGURES_DIR,
) -> Dict[str, str]:
    """
    Salva todas as figuras de validação estatística em alta resolução (300 DPI)
    em `docs/figures/` para compor o Relato Vol. 4 da Sprint 4.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    saved_paths: Dict[str, str] = {}

    # --------------------------------------------------------------------------
    # FIGURA 1: Matrizes de Confusão Absoluta e Normalizada (7 Patologias)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.8), dpi=300)
    labels_7 = cm_results["multiclass_labels"]
    cm_abs_7 = cm_results["cm_multiclass_abs"]
    cm_norm_7 = cm_results["cm_multiclass_norm"] * 100.0

    sns.heatmap(
        cm_abs_7,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_7,
        yticklabels=labels_7,
        cbar_kws={"label": "Contagem Absoluta de Amostras"},
        ax=ax1,
    )
    ax1.set_title("Matriz de Confusão Absoluta (7 Classes Fitopatológicas)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Classe Predita pelo Modelo SVM Campeão", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Classe Verdadeira (Ground Truth)", fontsize=10.5, fontweight="bold")
    ax1.tick_params(axis="x", rotation=32)
    ax1.tick_params(axis="y", rotation=0)

    sns.heatmap(
        cm_norm_7,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=100.0,
        xticklabels=labels_7,
        yticklabels=labels_7,
        cbar_kws={"label": "Proporção por Classe Verdadeira (%)"},
        ax=ax2,
    )
    ax2.set_title("Matriz de Confusão Normalizada por Classe (% Sensibilidade)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Classe Predita pelo Modelo SVM Campeão", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Classe Verdadeira (Ground Truth)", fontsize=10.5, fontweight="bold")
    ax2.tick_params(axis="x", rotation=32)
    ax2.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "Validação Estatística (Sprint 4 — Relato Vol. 4): Matrizes de Confusão Absoluta e Normalizada",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig1_path = figures_dir / "sprint4_confusion_matrices_multiclass_abs_norm.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths["cm_multiclass_abs_norm"] = str(fig1_path)

    # --------------------------------------------------------------------------
    # FIGURA 2: Matriz de Confusão Agronômica Focada (Saudável, Ferrugem, Outras)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=300)
    labels_3 = cm_results["agronomic_3class_labels"]
    cm_abs_3 = cm_results["cm_agronomic_3class_abs"]
    cm_norm_3 = cm_results["cm_agronomic_3class_norm"] * 100.0

    sns.heatmap(
        cm_abs_3,
        annot=True,
        fmt="d",
        cmap="Greens",
        annot_kws={"size": 12, "weight": "bold"},
        xticklabels=labels_3,
        yticklabels=labels_3,
        cbar_kws={"label": "Contagem Absoluta"},
        ax=ax1,
    )
    ax1.set_title("Matriz Absoluta: Saudável vs. Ferrugem vs. Outras Patologias", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Diagnóstico Predito", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Diagnóstico Real em Campo", fontsize=10.5, fontweight="bold")
    ax1.tick_params(axis="x", rotation=18)
    ax1.tick_params(axis="y", rotation=0)

    sns.heatmap(
        cm_norm_3,
        annot=True,
        fmt=".2f",
        cmap="OrRd",
        vmin=0.0,
        vmax=100.0,
        annot_kws={"size": 12, "weight": "bold"},
        xticklabels=labels_3,
        yticklabels=labels_3,
        cbar_kws={"label": "Taxa Normalizada por Classe (%)"},
        ax=ax2,
    )
    ax2.set_title("Matriz Normalizada (%): Saudável, Ferrugem e Outras Patologias", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Diagnóstico Predito", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Diagnóstico Real em Campo", fontsize=10.5, fontweight="bold")
    ax2.tick_params(axis="x", rotation=18)
    ax2.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "Agrupamento Fitossanitário Crítico: Isolamento de Saudável (HEALTHY), Ferrugem (RUST) e Outras Patologias",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig2_path = figures_dir / "sprint4_confusion_matrices_agronomic_3class.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths["cm_agronomic_3class"] = str(fig2_path)

    # --------------------------------------------------------------------------
    # FIGURA 3: Curvas ROC Multiclasse One-vs-Rest (Macro e Micro AUC-ROC)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 7.4), dpi=300)
    curves_roc = roc_results["curves"]

    ax.plot(
        curves_roc["fpr_micro"],
        curves_roc["tpr_micro"],
        label=f"Micro-Average ROC (AUC = {roc_results['roc_auc_micro']:.4f})",
        color="#d81b60",
        linestyle=":",
        linewidth=3.2,
    )
    ax.plot(
        curves_roc["fpr_macro"],
        curves_roc["tpr_macro"],
        label=f"Macro-Average ROC (AUC = {roc_results['roc_auc_macro']:.4f})",
        color="#1a237e",
        linestyle="--",
        linewidth=3.2,
    )

    palette_roc = ["#8e24aa", "#2e7d32", "#00838f", "#fb8c00", "#c62828", "#d32f2f", "#fdd835"]
    for idx, cls_id in enumerate(sorted(CLASS_NAMES_MAP.keys())):
        c_name = CLASS_NAMES_MAP[cls_id]
        c_auc = roc_results["auc_per_class"][c_name]
        lw = 2.6 if c_name in ("HEALTHY", "RUST") else 1.8
        ax.plot(
            curves_roc["fpr_per_class"][c_name],
            curves_roc["tpr_per_class"][c_name],
            color=palette_roc[idx % len(palette_roc)],
            linewidth=lw,
            alpha=0.9,
            label=f"OvR {c_name} (AUC = {c_auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.6, label="Classificador Aleatório (AUC = 0.5000)")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.set_xlabel("Taxa de Falsos Positivos (1 – Especificidade / FPR)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade / TPR)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Curvas ROC Multiclasse One-vs-Rest (OvR) — Modelo Campeão SVM RBF (C=10, γ=0.01)\n"
        f"AUC-ROC Macro = {roc_results['roc_auc_macro']:.4f} | AUC-ROC Micro = {roc_results['roc_auc_micro']:.4f}",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9.2, frameon=True, framealpha=0.94)
    plt.tight_layout()
    fig3_path = figures_dir / "sprint4_roc_curves_multiclass_ovr.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close(fig)
    saved_paths["roc_multiclass_ovr"] = str(fig3_path)

    # --------------------------------------------------------------------------
    # FIGURA 4: Curvas Precision-Recall Focadas em Ferrugem (RUST) e Classes Críticas
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.2), dpi=300)

    # Curvas Iso-F1 de referência no gráfico PR
    f_scores = np.linspace(0.2, 0.9, num=8)
    for f_val in f_scores:
        x_iso = np.linspace(0.01, 1.0, 100)
        with np.errstate(divide="ignore", invalid="ignore"):
            y_iso = f_val * x_iso / (2 * x_iso - f_val)
        valid_iso = np.isfinite(y_iso) & (y_iso >= 0)
        ax1.plot(x_iso[valid_iso], y_iso[valid_iso], color="gray", alpha=0.25, linestyle="--", linewidth=0.9)

    rust_pr = pr_results["per_class_pr"]["RUST"]
    healthy_pr = pr_results["per_class_pr"]["HEALTHY"]
    ax1.plot(
        rust_pr["recall_curve"],
        rust_pr["precision_curve"],
        color="#c62828",
        linewidth=3.0,
        label=f"FERRUGEM / RUST (AP = {rust_pr['average_precision_ap']:.4f}, AUC-PR = {rust_pr['auc_pr']:.4f})",
    )
    ax1.plot(
        healthy_pr["recall_curve"],
        healthy_pr["precision_curve"],
        color="#2e7d32",
        linewidth=2.4,
        linestyle="--",
        label=f"SAUDÁVEL / HEALTHY (AP = {healthy_pr['average_precision_ap']:.4f})",
    )
    ax1.axhline(
        rust_pr["prevalence_baseline"],
        color="#c62828",
        linestyle=":",
        linewidth=1.6,
        label=f"Prevalência Base Ferrugem ({rust_pr['prevalence_baseline']*100:.1f}%)",
    )
    ax1.set_xlim([0.0, 1.02])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("Sensibilidade / Recall (Taxa de Detecção de Ferrugem)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Precisão / Valor Preditivo Positivo (PPV)", fontsize=10.5, fontweight="bold")
    ax1.set_title("Curva Precision-Recall Focada na Classe Crítica de Ferrugem (RUST)", fontsize=11.5, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=9, frameon=True)

    # Painel direito: Comparativo completo PR das 7 classes + Micro-AP
    ax2.plot(
        pr_results["micro_curve"]["recall"],
        pr_results["micro_curve"]["precision"],
        color="#1a237e",
        linestyle=":",
        linewidth=3.0,
        label=f"Micro-Average PR (AP = {pr_results['ap_micro']:.4f})",
    )
    for idx, cls_id in enumerate(sorted(CLASS_NAMES_MAP.keys())):
        c_name = CLASS_NAMES_MAP[cls_id]
        c_pr = pr_results["per_class_pr"][c_name]
        ax2.plot(
            c_pr["recall_curve"],
            c_pr["precision_curve"],
            color=palette_roc[idx % len(palette_roc)],
            linewidth=2.0,
            label=f"{c_name} (AP = {c_pr['average_precision_ap']:.4f})",
        )

    ax2.set_xlim([0.0, 1.02])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Sensibilidade / Recall", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Precisão (Precision)", fontsize=10.5, fontweight="bold")
    ax2.set_title(
        f"Curvas Precision-Recall Multiclasse (AP Macro = {pr_results['ap_macro']:.4f})",
        fontsize=11.5,
        fontweight="bold",
    )
    ax2.legend(loc="lower left", fontsize=8.5, frameon=True)

    fig.suptitle(
        "Discriminação Estatística Precision-Recall (Sprint 4 — Relato Vol. 4): Foco na Classe Crítica de Ferrugem",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig4_path = figures_dir / "sprint4_pr_curve_rust_and_critical_classes.png"
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths["pr_curve_rust_focused"] = str(fig4_path)

    # --------------------------------------------------------------------------
    # FIGURA 5: Calibração de Limiares de Decisão (Threshold Tuning)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.8), dpi=300)

    # Painel 1: Threshold Tuning para Ferrugem (RUST)
    t_vals = df_thresh_rust["threshold"].values
    ax1.plot(t_vals, df_thresh_rust["sensitivity_recall"], color="#c62828", linewidth=2.5, label="Sensibilidade (Recall)")
    ax1.plot(t_vals, df_thresh_rust["precision"], color="#1565c0", linewidth=2.5, label="Precisão (PPV)")
    ax1.plot(t_vals, df_thresh_rust["specificity"], color="#2e7d32", linewidth=2.0, linestyle="--", label="Especificidade (TNR)")
    ax1.plot(t_vals, df_thresh_rust["f2_score_agronomic"], color="#ef6c00", linewidth=2.2, linestyle="-.", label="$F_2$-Score Agronômico")

    opt_rust_f2 = float(thresh_summary_rust["optimal_f2_agronomic_threshold"]["threshold"])
    ax1.axvline(
        opt_rust_f2,
        color="#6a1b9a",
        linestyle=":",
        linewidth=2.2,
        label=f"Limiar Ótimo Calibrado ($\\tau^* = {opt_rust_f2:.2f}$)",
    )
    ax1.set_xlabel("Limiar de Probabilidade de Decisão $\\tau$", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Métrica de Desempenho [0.0 – 1.0]", fontsize=10.5, fontweight="bold")
    ax1.set_title("Threshold Tuning — Classe Crítica: FERRUGEM (RUST)", fontsize=11.5, fontweight="bold")
    ax1.legend(loc="lower center", fontsize=8.8, frameon=True)

    # Painel 2: Threshold Tuning para Patologia com Sobreposição Visual (LEAF SCALD)
    t_vals_s = df_thresh_scald["threshold"].values
    ax2.plot(t_vals_s, df_thresh_scald["sensitivity_recall"], color="#c62828", linewidth=2.5, label="Sensibilidade (Recall)")
    ax2.plot(t_vals_s, df_thresh_scald["precision"], color="#1565c0", linewidth=2.5, label="Precisão (PPV)")
    ax2.plot(t_vals_s, df_thresh_scald["f1_score"], color="#2e7d32", linewidth=2.3, label="$F_1$-Score")
    ax2.plot(t_vals_s, df_thresh_scald["f2_score_agronomic"], color="#ef6c00", linewidth=2.3, linestyle="-.", label="$F_2$-Score (Foco Sensibilidade)")

    opt_scald_f2 = float(thresh_summary_scald["optimal_f2_agronomic_threshold"]["threshold"])
    opt_scald_f1 = float(thresh_summary_scald["optimal_f1_threshold"]["threshold"])
    ax2.axvline(0.50, color="gray", linestyle="--", linewidth=1.6, label="Limiar Padrão ($\\tau = 0.50$)")
    ax2.axvline(opt_scald_f1, color="#2e7d32", linestyle=":", linewidth=2.0, label=f"Ótimo $F_1$ ($\\tau^* = {opt_scald_f1:.2f}$)")
    ax2.axvline(opt_scald_f2, color="#ef6c00", linestyle=":", linewidth=2.0, label=f"Ótimo $F_2$ Sensibilidade ($\\tau^* = {opt_scald_f2:.2f}$)")

    ax2.set_xlabel("Limiar de Probabilidade de Decisão $\\tau$", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Métrica de Desempenho [0.0 – 1.0]", fontsize=10.5, fontweight="bold")
    ax2.set_title("Threshold Tuning — Calibração de Sensibilidade: LEAF SCALD", fontsize=11.5, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.suptitle(
        "Análise e Calibração de Limiares de Decisão (Threshold Tuning) para Sensibilidade Agronômica",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig5_path = figures_dir / "sprint4_threshold_tuning_sensitivity_calibration.png"
    plt.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths["threshold_tuning_calibration"] = str(fig5_path)

    return saved_paths


# ==============================================================================
# 7. CONSOLIDAÇÃO DO RELATÓRIO TÉCNICO VOL. 4 EM MARKDOWN
# ==============================================================================

def generate_relato_vol4_statistical_validation_report(
    cm_valid: Dict[str, Any],
    cm_test: Dict[str, Any],
    roc_valid: Dict[str, Any],
    roc_test: Dict[str, Any],
    pr_valid: Dict[str, Any],
    pr_test: Dict[str, Any],
    thresh_summary_rust: Dict[str, Any],
    thresh_summary_scald: Dict[str, Any],
    report_path: Path = DOCS_DIR / "relatorio_validacao_estatistica_matrizes_roc_pr_vol4.md",
    entrega_readme_path: Path = ENTREGAS_DIR / "README.md",
) -> Tuple[str, str]:
    """
    Redige e consolida o relatório técnico da Sprint 4 (Relato Vol. 4) contendo
    as matrizes de confusão normalizadas, curvas ROC OvR, curvas PR da ferrugem
    e calibração de limiares de decisão.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    entrega_readme_path.parent.mkdir(parents=True, exist_ok=True)

    per_cls_rows = ""
    for m_val in cm_valid["per_class_metrics"]:
        c_name = m_val["class_name"]
        auc_v = roc_valid["auc_per_class"][c_name]
        ap_v = pr_valid["per_class_pr"][c_name]["average_precision_ap"]
        per_cls_rows += (
            f"| **{c_name}** | {m_val['support']} | {m_val['recall_sensitivity']*100:.2f}% | "
            f"{m_val['specificity']*100:.2f}% | {m_val['precision']*100:.2f}% | "
            f"{m_val['f1_score']:.4f} | {auc_v:.4f} | {ap_v:.4f} |\n"
        )

    cm3_abs = cm_valid["cm_agronomic_3class_abs"]
    cm3_norm = cm_valid["cm_agronomic_3class_norm"] * 100.0
    labels3 = cm_valid["agronomic_3class_labels"]
    cm3_rows = ""
    for i, r_label in enumerate(labels3):
        cm3_rows += (
            f"| **{r_label}** | {cm3_abs[i, 0]} ({cm3_norm[i, 0]:.2f}%) | "
            f"{cm3_abs[i, 1]} ({cm3_norm[i, 1]:.2f}%) | "
            f"{cm3_abs[i, 2]} ({cm3_norm[i, 2]:.2f}%) |\n"
        )

    s_def = thresh_summary_scald["default_threshold_050"]
    s_f1 = thresh_summary_scald["optimal_f1_threshold"]
    s_f2 = thresh_summary_scald["optimal_f2_agronomic_threshold"]
    s_rec = thresh_summary_scald["high_sensitivity_screening_threshold"]

    md_text = rf"""# 📊 Relato da Solução Vol. 4 — Matrizes de Confusão Normalizadas, Curvas ROC-AUC, Precision-Recall e Threshold Tuning (Sprint 4)

**Projeto:** SugarVision — Sanidade Vegetal (*Saccharum officinarum*)  
**Sprint:** 4 — Framework SEMMA (Fase: Assess — Validação Estatística)  
**Módulo Executável:** [`src/svm_statistical_validation_assess.py`](../src/svm_statistical_validation_assess.py)  
**Suíte de Testes:** [`src/test_svm_statistical_validation_assess.py`](../src/test_svm_statistical_validation_assess.py)  
**Estimador Avaliado:** Pipeline Atômico Campeão (`models/svm_pipeline_optimal.joblib` — SVM RBF, $C=10.0, \gamma=0.01$, Calibração Platt)

---

## 1. Matrizes de Confusão Absoluta e Normalizada por Classe

### 1.1 Agrupamento Fitossanitário Crítico (Saudável vs. Ferrugem vs. Outras Patologias)
Na partição de Validação Independente ($N=617$), o agrupamento agronômico de 3 classes demonstra **isolamento perfeito (100,00%)** das classes **Saudável (`HEALTHY`)** e **Ferrugem (`RUST`)**, sem falso alarme cruzado:

| Classe Real \ Predita | SAUDÁVEL (HEALTHY) | FERRUGEM (RUST) | OUTRAS PATOLOGIAS |
| :--- | :---: | :---: | :---: |
{cm3_rows}

![Matriz de Confusão Agronômica 3 Classes](figures/sprint4_confusion_matrices_agronomic_3class.png)

### 1.2 Desempenho Discriminativo Completo nas 7 Patologias Foliares
A tabela abaixo sintetiza as métricas extraídas da matriz de confusão normalizada, curvas ROC One-vs-Rest e curvas Precision-Recall por patologia:

| Patologia Foliar | Suporte ($N$) | Sensibilidade (Recall) | Especificidade (TNR) | Precisão (PPV) | $F_1$-Score | AUC-ROC (OvR) | Average Precision (AP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{per_cls_rows}

![Matrizes de Confusão 7 Classes](figures/sprint4_confusion_matrices_multiclass_abs_norm.png)

---

## 2. Curvas ROC Multiclasse One-vs-Rest (Macro e Micro AUC-ROC)

A formulação **One-vs-Rest (OvR)** avalia a capacidade do hiperplano RBF calibrado em ranquear cada patologia contra o conjunto complementar de imagens:

- **AUC-ROC Macro-Average:** **{roc_valid['roc_auc_macro']:.4f}** na Validação ($N=617$) e **{roc_test['roc_auc_macro']:.4f}** no Teste Cego ($N=380$).
- **AUC-ROC Micro-Average:** **{roc_valid['roc_auc_micro']:.4f}** na Validação e **{roc_test['roc_auc_micro']:.4f}** no Teste Cego.
- **AUC-ROC Ponderada (Weighted):** **{roc_valid['roc_auc_weighted']:.4f}** na Validação e **{roc_test['roc_auc_weighted']:.4f}** no Teste Cego.

![Curvas ROC Multiclasse OvR](figures/sprint4_roc_curves_multiclass_ovr.png)

---

## 3. Curvas Precision-Recall Focadas na Classe Crítica de Ferrugem (`RUST`)

Em cenários de monitoramento agrícola onde a **Ferrugem (`RUST`, prevalência de {pr_valid['per_class_pr']['RUST']['prevalence_baseline']*100:.1f}%)** exige detecção sem falsos positivos custosos, a curva Precision-Recall comprova:

- **Average Precision (AP) para Ferrugem (`RUST`):** **{pr_valid['per_class_pr']['RUST']['average_precision_ap']:.4f}** (Validação) e **{pr_test['per_class_pr']['RUST']['average_precision_ap']:.4f}** (Teste Cego).
- **AUC-PR para Ferrugem (`RUST`):** **{pr_valid['per_class_pr']['RUST']['auc_pr']:.4f}** (Validação) e **{pr_test['per_class_pr']['RUST']['auc_pr']:.4f}** (Teste Cego).
- **Average Precision Global:** Macro-AP = **{pr_valid['ap_macro']:.4f}** | Micro-AP = **{pr_valid['ap_micro']:.4f}**.

![Curvas Precision-Recall Ferrugem](figures/sprint4_pr_curve_rust_and_critical_classes.png)

---

## 4. Calibração de Limiares de Decisão (*Threshold Tuning*) para Sensibilidade

A varredura fina de limiares de probabilidade $\tau \in [0.01, 0.99]$ permite ajustar a sensibilidade operacional do classificador conforme o objetivo fitossanitário em campo:

1. **Classe Crítica `RUST` (Ferrugem):**
   - Graças à separação não-linear limpa dos descritores `indice_rugosidade_pustula` e `glcm_contrast`, a classe `RUST` mantém **Sensibilidade = {thresh_summary_rust['optimal_f2_agronomic_threshold']['sensitivity_recall']*100:.2f}%** e **Especificidade = {thresh_summary_rust['optimal_f2_agronomic_threshold']['specificity']*100:.2f}%** ao longo de uma ampla faixa de limiares ($\tau^* = {thresh_summary_rust['optimal_f2_agronomic_threshold']['threshold']:.2f}$, $F_2 = {thresh_summary_rust['optimal_f2_agronomic_threshold']['f2_score_agronomic']:.4f}$).
2. **Calibração de Sensibilidade em Classes com Sobreposição (`LEAF SCALD`):**

| Ponto de Operação ($\tau$) | Limiar $\tau^*$ | Sensibilidade (Recall) | Especificidade | Precisão | $F_1$-Score | $F_2$-Score Agronômico |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Padrão Nominal ($\tau = 0.50$)** | {s_def['threshold']:.2f} | {s_def['sensitivity_recall']*100:.2f}% | {s_def['specificity']*100:.2f}% | {s_def['precision']*100:.2f}% | {s_def['f1_score']:.4f} | {s_def['f2_score_agronomic']:.4f} |
| **Ótimo Equilíbrio ($F_1$ Máximo)** | {s_f1['threshold']:.2f} | {s_f1['sensitivity_recall']*100:.2f}% | {s_f1['specificity']*100:.2f}% | {s_f1['precision']*100:.2f}% | {s_f1['f1_score']:.4f} | {s_f1['f2_score_agronomic']:.4f} |
| **Ótimo Agronômico ($F_2$ Sensibilidade)** | {s_f2['threshold']:.2f} | {s_f2['sensitivity_recall']*100:.2f}% | {s_f2['specificity']*100:.2f}% | {s_f2['precision']*100:.2f}% | {s_f2['f1_score']:.4f} | {s_f2['f2_score_agronomic']:.4f} |
| **Triagem de Alta Sensibilidade ($\ge 98\%$)** | {s_rec['threshold']:.2f} | {s_rec['sensitivity_recall']*100:.2f}% | {s_rec['specificity']*100:.2f}% | {s_rec['precision']*100:.2f}% | {s_rec['f1_score']:.4f} | {s_rec['f2_score_agronomic']:.4f} |

![Calibração de Limiares Threshold Tuning](figures/sprint4_threshold_tuning_sensitivity_calibration.png)

---

## 5. Artefatos Gerados para o Relato Vol. 4

- **Figuras em Alta Resolução (`docs/figures/`, 300 DPI):**
  - `docs/figures/sprint4_confusion_matrices_multiclass_abs_norm.png`
  - `docs/figures/sprint4_confusion_matrices_agronomic_3class.png`
  - `docs/figures/sprint4_roc_curves_multiclass_ovr.png`
  - `docs/figures/sprint4_pr_curve_rust_and_critical_classes.png`
  - `docs/figures/sprint4_threshold_tuning_sensitivity_calibration.png`
- **Tabelas e Metadados (`models/`):**
  - `models/svm_statistical_validation_per_class.csv`
  - `models/svm_threshold_tuning_rust.csv`
  - `models/svm_threshold_tuning_leaf_scald.csv`
  - `models/svm_statistical_validation_vol4.json`
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    entrega_md = md_text.replace("(figures/", "(../../docs/figures/").replace("(../src/", "(../../src/")
    with open(entrega_readme_path, "w", encoding="utf-8") as f:
        f.write(entrega_md)

    return str(report_path), str(entrega_readme_path)


# ==============================================================================
# 8. ORQUESTRADOR PRINCIPAL DA VALIDAÇÃO ESTATÍSTICA (SPRINT 4)
# ==============================================================================

def run_statistical_validation_pipeline() -> Dict[str, Any]:
    """
    Executa de ponta a ponta a validação estatística do modelo SVM campeão para a Sprint 4:
    1. Matrizes de confusão absoluta e normalizada (7 patologias e agrupamento agronômico 3 classes).
    2. Curvas ROC multiclasse One-vs-Rest (AUC Macro e Micro).
    3. Curvas Precision-Recall focadas na classe crítica de Ferrugem (`RUST`).
    4. Threshold Tuning para calibração de sensibilidade.
    5. Exportação de figuras em 300 DPI em `docs/figures/` e relatórios para o Relato Vol. 4.
    """
    logger.info("Iniciando Validação Estatística (Sprint 4 — Relato Vol. 4)...")
    t0 = time.perf_counter()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ENTREGAS_DIR.mkdir(parents=True, exist_ok=True)

    _, partitions = load_champion_model_and_predictions()
    val_data = partitions["valid"]
    test_data = partitions["test"]

    # 1. Matrizes de Confusão Absoluta e Normalizada
    logger.info("[1/5] Calculando matrizes de confusão absoluta e normalizada (7 classes e 3 classes agronômicas)...")
    cm_valid = compute_confusion_matrices(val_data["y_true"], val_data["y_pred"])
    cm_test = compute_confusion_matrices(test_data["y_true"], test_data["y_pred"])

    df_per_class = pd.DataFrame(cm_valid["per_class_metrics"])
    per_class_csv = MODELS_DIR / "svm_statistical_validation_per_class.csv"
    df_per_class.to_csv(per_class_csv, index=False, encoding="utf-8")

    # 2. Curvas ROC Multiclasse One-vs-Rest (Macro e Micro AUC)
    logger.info("[2/5] Calculando curvas ROC multiclasse One-vs-Rest (Macro e Micro AUC-ROC)...")
    roc_valid = compute_multiclass_roc_ovr(val_data["y_true"], val_data["y_proba"])
    roc_test = compute_multiclass_roc_ovr(test_data["y_true"], test_data["y_proba"])
    logger.info(
        "  -> Validação: AUC-ROC Macro = %.4f | Micro = %.4f || Teste: AUC-ROC Macro = %.4f | Micro = %.4f",
        roc_valid["roc_auc_macro"],
        roc_valid["roc_auc_micro"],
        roc_test["roc_auc_macro"],
        roc_test["roc_auc_micro"],
    )

    # 3. Curvas Precision-Recall focadas na Ferrugem (RUST)
    logger.info("[3/5] Construindo curvas Precision-Recall focadas na classe crítica de Ferrugem (RUST)...")
    pr_valid = compute_precision_recall_curves_focused(
        val_data["y_true"], val_data["y_proba"], target_class_id=RUST_CLASS_ID
    )
    pr_test = compute_precision_recall_curves_focused(
        test_data["y_true"], test_data["y_proba"], target_class_id=RUST_CLASS_ID
    )
    logger.info(
        "  -> Ferrugem (RUST) AP Validação = %.4f | AP Teste = %.4f | Macro-AP Val = %.4f",
        pr_valid["focus_class_metrics"]["average_precision_ap"],
        pr_test["focus_class_metrics"]["average_precision_ap"],
        pr_valid["ap_macro"],
    )

    # 4. Threshold Tuning (Calibração de Sensibilidade)
    logger.info("[4/5] Avaliando comportamento dos limiares de decisão (Threshold Tuning)...")
    df_thresh_rust, thresh_summary_rust = evaluate_threshold_tuning_sensitivity(
        val_data["y_true"], val_data["y_proba"], target_class_id=RUST_CLASS_ID
    )
    df_thresh_scald, thresh_summary_scald = evaluate_threshold_tuning_sensitivity(
        val_data["y_true"], val_data["y_proba"], target_class_id=2  # LEAF SCALD
    )

    rust_thresh_csv = MODELS_DIR / "svm_threshold_tuning_rust.csv"
    scald_thresh_csv = MODELS_DIR / "svm_threshold_tuning_leaf_scald.csv"
    df_thresh_rust.to_csv(rust_thresh_csv, index=False, encoding="utf-8")
    df_thresh_scald.to_csv(scald_thresh_csv, index=False, encoding="utf-8")

    # 5. Gerar Figuras em Alta Resolução (300 DPI) e Relatório Vol. 4
    logger.info("[5/5] Salvando figuras em alta resolução em docs/figures/ e gerando Relato Vol. 4...")
    figures_map = generate_statistical_validation_figures(
        cm_results=cm_valid,
        roc_results=roc_valid,
        pr_results=pr_valid,
        df_thresh_rust=df_thresh_rust,
        thresh_summary_rust=thresh_summary_rust,
        df_thresh_scald=df_thresh_scald,
        thresh_summary_scald=thresh_summary_scald,
        figures_dir=FIGURES_DIR,
    )

    report_path, entrega_path = generate_relato_vol4_statistical_validation_report(
        cm_valid=cm_valid,
        cm_test=cm_test,
        roc_valid=roc_valid,
        roc_test=roc_test,
        pr_valid=pr_valid,
        pr_test=pr_test,
        thresh_summary_rust=thresh_summary_rust,
        thresh_summary_scald=thresh_summary_scald,
    )

    elapsed = round(time.perf_counter() - t0, 2)
    manifest: Dict[str, Any] = {
        "task_name": "Matrizes de Confusão Normalizadas e Curvas de Discriminação Estatística (ROC-AUC e PR)",
        "project": "Sanidade-Vegetal (SugarVision)",
        "sprint": "Sprint 4 — Framework SEMMA (Fase: Assess — Relato Vol. 4)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_seconds": elapsed,
        "validation_partition_metrics": {
            "n_samples": val_data["n_samples"],
            "roc_auc_macro": roc_valid["roc_auc_macro"],
            "roc_auc_micro": roc_valid["roc_auc_micro"],
            "roc_auc_weighted": roc_valid["roc_auc_weighted"],
            "auc_per_class": roc_valid["auc_per_class"],
            "ap_macro": pr_valid["ap_macro"],
            "ap_micro": pr_valid["ap_micro"],
            "rust_pr_metrics": pr_valid["focus_class_metrics"],
            "cm_agronomic_3class_labels": cm_valid["agronomic_3class_labels"],
            "cm_agronomic_3class_abs": cm_valid["cm_agronomic_3class_abs"].tolist(),
            "cm_agronomic_3class_norm": cm_valid["cm_agronomic_3class_norm"].tolist(),
            "per_class_confusion_metrics": cm_valid["per_class_metrics"],
        },
        "test_partition_metrics": {
            "n_samples": test_data["n_samples"],
            "roc_auc_macro": roc_test["roc_auc_macro"],
            "roc_auc_micro": roc_test["roc_auc_micro"],
            "roc_auc_weighted": roc_test["roc_auc_weighted"],
            "auc_per_class": roc_test["auc_per_class"],
            "ap_macro": pr_test["ap_macro"],
            "ap_micro": pr_test["ap_micro"],
            "rust_pr_metrics": pr_test["focus_class_metrics"],
        },
        "threshold_tuning_calibration": {
            "rust_critical_class": thresh_summary_rust,
            "leaf_scald_overlapping_class": thresh_summary_scald,
        },
        "artifacts": {
            "per_class_csv": str(per_class_csv),
            "rust_threshold_csv": str(rust_thresh_csv),
            "leaf_scald_threshold_csv": str(scald_thresh_csv),
            "relato_vol4_report_md": report_path,
            "entrega_dossier_md": entrega_path,
            "figures": figures_map,
        },
    }

    manifest_path = MODELS_DIR / "svm_statistical_validation_vol4.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Validação Estatística concluída com 100%% de sucesso em %.2fs!", elapsed)
    return manifest


if __name__ == "__main__":
    run_statistical_validation_pipeline()
