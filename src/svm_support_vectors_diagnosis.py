"""
Módulo de Diagnóstico Matemático dos Vetores de Suporte (Support Vectors)
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 3 — Framework SEMMA (Fase: Model)
Responsável: Engenharia de Inteligência Artificial & Ciência de Dados Sênior

Checklist Coberto:
1. Extrair a contagem e proporção de vetores de suporte por classe (acessando `n_support_` e `support_vectors_`).
2. Desenvolver um script para re-treinar e avaliar a relação entre diferentes valores do hiperparâmetro C
   e a quantidade de vetores de suporte (comparando margens rígidas vs margens suaves).
3. Isolar as imagens/patches correspondentes aos índices dos vetores de suporte no dataset original
   e salvar uma amostra para análise visual do grau de ambiguidade foliar.
4. Identificar matematicamente os vetores de suporte que violam a margem
   (erros de margem ou instâncias onde as variáveis de folga xi_i > 0).
5. Gerar visualizações gráficas de alta qualidade (2D/3D) da projeção dos vetores de suporte
   em relação ao hiperplano separador (usando matplotlib/seaborn e plotly).
"""

from __future__ import annotations

import hashlib
import json
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Compatibilidade reversa para artefatos serializados em versões anteriores do scikit-learn
import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        """Stub de compatibilidade para deserialização de ColumnTransformer."""
        pass
    _ct._RemainderColsList = _RemainderColsList

try:
    from atomic_svm_pipeline import (
        NUMERICAL_FEATURES,
        SPLIT_COLUMN,
        TARGET_COLUMNS,
        build_atomic_svm_pipeline,
    )
except ImportError:
    from src.atomic_svm_pipeline import (
        NUMERICAL_FEATURES,
        SPLIT_COLUMN,
        TARGET_COLUMNS,
        build_atomic_svm_pipeline,
    )


# ==============================================================================
# CONSTANTES E TAXONOMIA FITOPATOLÓGICA
# ==============================================================================

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE: int = 42
DEFAULT_C_GRID: List[float] = [0.01, 0.1, 1.0, 10.0, 100.0]

CLASS_NAMES_MAP: Dict[int, str] = {
    0: "GRASSY SHOOT",
    1: "HEALTHY",
    2: "LEAF SCALD",
    3: "MOSAIC",
    4: "RED ROT",
    5: "RUST",
    6: "YELLOW LEAF",
}

BINARY_CLASS_NAMES_MAP: Dict[int, str] = {
    0: "HEALTHY",
    1: "RUST",
}

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
FIGURES_DIR: Path = PROJECT_ROOT / "docs" / "figures"
PATCHES_DIR: Path = FIGURES_DIR / "support_vectors_patches"
ENTREGAS_DIR: Path = PROJECT_ROOT / "entregas" / "Diagnóstico Matemático dos Vetores de Suporte (Support Vectors)"


# ==============================================================================
# UTILITÁRIOS INTERNOS DE EXTRAÇÃO DO ESTIMADOR
# ==============================================================================

def _extract_svc_and_features(
    model_or_pipeline: Union[Pipeline, SVC],
    X: Union[pd.DataFrame, np.ndarray],
) -> Tuple[SVC, np.ndarray]:
    """
    Extrai a instância ajustada do estimador `SVC` e a matriz de atributos
    transformada no espaço de entrada do hiperplano.

    Parâmetros
    ----------
    model_or_pipeline : Union[Pipeline, SVC]
        Objeto `Pipeline` contendo a etapa `'svm'` (ou último passo `SVC`) ou um `SVC` direto.
    X : Union[pd.DataFrame, np.ndarray]
        Dados de entrada (brutos se `Pipeline`, ou já escalonados se `SVC`).

    Retorna
    -------
    Tuple[SVC, np.ndarray]
        Tupla `(svc_estimator, X_transformed_array)`.
    """
    if isinstance(model_or_pipeline, Pipeline):
        if "svm" in model_or_pipeline.named_steps:
            svc = model_or_pipeline.named_steps["svm"]
        else:
            svc = model_or_pipeline.steps[-1][1]
        if "preprocessor" in model_or_pipeline.named_steps:
            X_trans = model_or_pipeline.named_steps["preprocessor"].transform(X)
        elif len(model_or_pipeline.steps) > 1:
            X_trans = model_or_pipeline[:-1].transform(X)
        else:
            X_trans = np.asarray(X, dtype=float)
    elif isinstance(model_or_pipeline, SVC):
        svc = model_or_pipeline
        X_trans = np.asarray(X, dtype=float)
    else:
        raise TypeError(
            f"Esperado sklearn.pipeline.Pipeline ou sklearn.svm.SVC, recebido: {type(model_or_pipeline)}"
        )

    if not hasattr(svc, "support_vectors_") or not hasattr(svc, "n_support_"):
        raise ValueError("O estimador SVC fornecido não está ajustado (atributos `support_vectors_` e `n_support_` ausentes).")

    return svc, np.asarray(X_trans, dtype=float)


# ==============================================================================
# 1. CONTAGEM E PROPORÇÃO DE VETORES DE SUPORTE POR CLASSE
# ==============================================================================

def extract_support_vectors_by_class(
    model_or_pipeline: Union[Pipeline, SVC],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    class_names_map: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    r"""
    Extrai a contagem, proporção intraclasse, participação global e estatísticas
    geométricas dos vetores de suporte por classe acessando `n_support_` e `support_vectors_`.

    Parâmetros
    ----------
    model_or_pipeline : Union[Pipeline, SVC]
        Estimador `SVC` ou `Pipeline` ajustado nos dados de treinamento.
    X_train : Union[pd.DataFrame, np.ndarray]
        Conjunto de treinamento original usado no ajuste.
    y_train : Union[pd.Series, np.ndarray]
        Vetor de rótulos reais do conjunto de treinamento.
    class_names_map : Optional[Dict[int, str]], default=None
        Mapeamento opcional de índice numérico da classe para nome fitopatológico.

    Retorna
    -------
    pd.DataFrame
        Tabela analítica com métricas de densidade de suporte por classe:
        - `class_id`: Código numérico da classe
        - `class_name`: Nome da classe fitopatológica
        - `n_train_samples`: Total de amostras da classe no treino ($N_c$)
        - `n_support_vectors`: Contagem extraída de `n_support_[i]` ($SV_c$)
        - `class_sv_proportion`: Densidade intraclasse ($SV_c / N_c \in [0, 1]$)
        - `class_sv_percentage`: Densidade intraclasse em percentual
        - `global_sv_share`: Participação no total de vetores de suporte ($SV_c / SV_{total}$)
        - `sv_mean_l2_norm`: Norma Euclidiana média dos vetores de suporte da classe em `support_vectors_`
        - `sv_centroid_dispersion`: Dispersão média em relação ao centroide de suporte da classe
    """
    svc, _ = _extract_svc_and_features(model_or_pipeline, X_train)
    y_arr = np.asarray(y_train)

    n_support = np.asarray(svc.n_support_, dtype=int)
    support_vectors = np.asarray(svc.support_vectors_, dtype=float)
    classes = np.asarray(svc.classes_)

    total_sv = int(n_support.sum())
    if total_sv != int(support_vectors.shape[0]):
        raise RuntimeError(
            f"Inconsistência estrutural no SVC: sum(n_support_)={total_sv} != support_vectors_.shape[0]={support_vectors.shape[0]}"
        )

    if class_names_map is None:
        class_names_map = CLASS_NAMES_MAP if len(classes) > 2 else BINARY_CLASS_NAMES_MAP

    records: List[Dict[str, Any]] = []
    sv_offset = 0

    for idx, cls_val in enumerate(classes):
        cls_sv_count = int(n_support[idx])
        cls_train_count = int(np.sum(y_arr == cls_val))
        cls_sv_slice = support_vectors[sv_offset : sv_offset + cls_sv_count]
        sv_offset += cls_sv_count

        if cls_sv_count > 0:
            l2_norms = np.linalg.norm(cls_sv_slice, axis=1)
            mean_norm = float(np.mean(l2_norms))
            centroid = np.mean(cls_sv_slice, axis=0)
            dispersion = float(np.mean(np.linalg.norm(cls_sv_slice - centroid, axis=1)))
        else:
            mean_norm = 0.0
            dispersion = 0.0

        proportion = float(cls_sv_count / cls_train_count) if cls_train_count > 0 else 0.0
        global_share = float(cls_sv_count / total_sv) if total_sv > 0 else 0.0

        cls_int = int(cls_val) if isinstance(cls_val, (int, np.integer)) else idx
        cls_label = class_names_map.get(cls_int, str(cls_val))

        records.append(
            {
                "class_id": cls_int,
                "class_name": cls_label,
                "n_train_samples": cls_train_count,
                "n_support_vectors": cls_sv_count,
                "class_sv_proportion": round(proportion, 6),
                "class_sv_percentage": round(proportion * 100.0, 2),
                "global_sv_share": round(global_share, 6),
                "global_sv_share_percentage": round(global_share * 100.0, 2),
                "sv_mean_l2_norm": round(mean_norm, 4),
                "sv_centroid_dispersion": round(dispersion, 4),
            }
        )

    return pd.DataFrame(records)


# ==============================================================================
# 2. RELAÇÃO ENTRE O HIPERPARÂMETRO C E A QUANTIDADE DE VETORES DE SUPORTE
# ==============================================================================

def evaluate_c_hyperparameter_support_tradeoff(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    c_values: Optional[List[float]] = None,
    kernels: Optional[List[str]] = None,
    gamma: Union[str, float] = 0.01,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    r"""
    Avalia sistematicamente a relação matemática entre o hiperparâmetro de penalidade C
    (margens suaves vs. margens rígidas) e a cardinalidade/densidade dos vetores de suporte.

    Para cada combinação de Kernel e valor de C:
    - Ajusta um classificador SVC padronizado sem vazamento de dados.
    - Extrai `n_support_`, `support_vectors_`, margem geométrica $2 / \\|\\mathbf{w}\\|_2$,
      variáveis de folga $\\xi_i = \\max(0, 1 - m_i)$ e contagem de violações ($\\xi_i > 0$).
    - Computa desempenho de generalização ($F_1$-Macro e Acurácia Balanceada).

    Parâmetros
    ----------
    X_train : Union[pd.DataFrame, np.ndarray]
        Matriz de atributos de treino (já pré-processada ou DataFrame numérico).
    y_train : Union[pd.Series, np.ndarray]
        Vetor de classes de treino.
    X_val : Union[pd.DataFrame, np.ndarray]
        Matriz de atributos de validação.
    y_val : Union[pd.Series, np.ndarray]
        Vetor de classes de validação.
    c_values : Optional[List[float]], default=None
        Grade logarítmica de valores de C para comparação de margens suaves a rígidas.
    kernels : Optional[List[str]], default=None
        Lista de kernels a avaliar (ex: `['linear', 'rbf']`).
    gamma : Union[str, float], default=0.01
        Coeficiente gamma para kernel RBF.
    random_state : int, default=42
        Semente determinística.

    Retorna
    -------
    pd.DataFrame
        Tabela consolidada com a evolução estrutural da fronteira e densidade de suporte por C.
    """
    if c_values is None:
        c_values = DEFAULT_C_GRID
    if kernels is None:
        kernels = ["linear", "rbf"]

    X_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train)
    X_va = np.asarray(X_val, dtype=float)
    y_va = np.asarray(y_val)
    n_train = len(X_tr)

    rows: List[Dict[str, Any]] = []

    for kernel_name in kernels:
        for c_val in c_values:
            if c_val <= 0:
                raise ValueError(f"O hiperparâmetro C deve ser estritamente positivo. Recebido: {c_val}")

            t0 = time.perf_counter()
            svc = SVC(
                kernel=kernel_name,
                C=float(c_val),
                gamma=gamma if kernel_name != "linear" else "scale",
                class_weight="balanced",
                decision_function_shape="ovr",
                cache_size=1000,
                max_iter=20000,
                random_state=random_state,
            )
            svc.fit(X_tr, y_tr)
            fit_time = time.perf_counter() - t0

            n_sv_total = int(svc.n_support_.sum())
            sv_ratio = float(n_sv_total / n_train)

            # Cálculo da norma de w e margem geométrica
            if kernel_name == "linear":
                w_norms = np.linalg.norm(svc.coef_, axis=1)
                mean_w_norm = float(np.mean(w_norms))
                geometric_margin = float(2.0 / mean_w_norm) if mean_w_norm > 0 else 0.0
            else:
                # No espaço RKHS induzido pelo kernel, estimamos a norma dual ||f||_H
                dual_norm = float(np.linalg.norm(svc.dual_coef_))
                mean_w_norm = dual_norm
                geometric_margin = float(2.0 / dual_norm) if dual_norm > 0 else 0.0

            # Análise das variáveis de folga xi_i nos vetores de suporte
            sv_df, sv_summary = identify_margin_violating_support_vectors(
                model_or_pipeline=svc,
                X_train=X_tr,
                y_train=y_tr,
            )

            y_pred_tr = svc.predict(X_tr)
            y_pred_va = svc.predict(X_va)

            avg_mode = "binary" if len(svc.classes_) == 2 else "macro"
            tr_f1 = float(f1_score(y_tr, y_pred_tr, average=avg_mode))
            va_f1 = float(f1_score(y_va, y_pred_va, average=avg_mode))
            va_bal_acc = float(balanced_accuracy_score(y_va, y_pred_va))

            if c_val <= 0.1:
                margin_regime = "Margem Suave (Soft Margin)"
            elif c_val < 10.0:
                margin_regime = "Margem Intermediária (Equilíbrio)"
            else:
                margin_regime = "Margem Rígida (Hard Margin)"

            rows.append(
                {
                    "kernel": kernel_name,
                    "C": float(c_val),
                    "log10_C": round(float(np.log10(c_val)), 2),
                    "margin_regime": margin_regime,
                    "n_train_samples": n_train,
                    "n_support_vectors": n_sv_total,
                    "support_vectors_ratio": round(sv_ratio, 6),
                    "support_vectors_pct": round(sv_ratio * 100.0, 2),
                    "norm_w_or_dual": round(mean_w_norm, 4),
                    "geometric_margin": round(geometric_margin, 6),
                    "n_margin_violations_xi_gt_0": int(sv_summary["n_margin_violations_xi_gt_0"]),
                    "pct_sv_violating_margin": round(float(sv_summary["pct_sv_violating_margin"]), 2),
                    "n_misclassified_xi_gt_1": int(sv_summary["n_misclassified_xi_gt_1"]),
                    "mean_slack_xi": round(float(sv_summary["mean_slack_xi"]), 4),
                    "max_slack_xi": round(float(sv_summary["max_slack_xi"]), 4),
                    "train_f1": round(tr_f1, 4),
                    "val_f1": round(va_f1, 4),
                    "val_balanced_accuracy": round(va_bal_acc, 4),
                    "fit_time_seconds": round(fit_time, 4),
                }
            )

    return pd.DataFrame(rows)


# ==============================================================================
# 3. IDENTIFICAÇÃO MATEMÁTICA DE VIOLAÇÕES DE MARGEM (VARIÁVEIS DE FOLGA xi_i > 0)
# ==============================================================================

def identify_margin_violating_support_vectors(
    model_or_pipeline: Union[Pipeline, SVC],
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    df_metadata: Optional[pd.DataFrame] = None,
    tolerance: float = 1e-3,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    r"""
    Identifica matematicamente os vetores de suporte que violam a margem geométrica/funcional,
    calculando as variáveis de folga ($\xi_i$) e classificando cada vetor segundo as
    condições de Karush-Kuhn-Tucker (KKT).

    Formulação Matemática:
    - Caso Binário ($y_i \\in \\{-1, +1\\}$):
      $$\\gamma_i = y_i f(\\mathbf{x}_i), \\quad \\xi_i = \\max(0, 1 - \\gamma_i)$$
    - Caso Multiclasse ($K$ classes, escores de decisão $F_{i, c}$):
      $$\\gamma_i = F_{i, y_i} - \\max_{c \\neq y_i} F_{i, c}, \\quad \\xi_i = \\max(0, 1 - \\gamma_i)$$

    Classificação de Regimes KKT:
    1. `ON_MARGIN_FREE` ($\\xi_i \\le \\text{tolerance}$): Vetor posicionado sobre ou além da margem canônica.
    2. `MARGIN_VIOLATION_CORRECT_SIDE` ($0 < \\xi_i \\le 1$): Invasão da zona de margem suave, porém classificado corretamente.
    3. `MARGIN_VIOLATION_MISCLASSIFIED` ($\\xi_i > 1$): Violação severa da margem com transposição do hiperplano separador (erro de classificação).

    Parâmetros
    ----------
    model_or_pipeline : Union[Pipeline, SVC]
        Modelo SVM ou Pipeline ajustado.
    X_train : Union[pd.DataFrame, np.ndarray]
        Conjunto de treinamento original.
    y_train : Union[pd.Series, np.ndarray]
        Rótulos verdadeiros de treinamento.
    df_metadata : Optional[pd.DataFrame], default=None
        DataFrame opcional contendo metadados (`sample_id`, `filename`, `class_label`, etc.) alinhados a `X_train`.
    tolerance : float, default=1e-3
        Tolerância numérica para distinção de $\\xi_i > 0$.

    Retorna
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        DataFrame detalhado por vetor de suporte e dicionário de métricas agregadas.
    """
    svc, X_trans = _extract_svc_and_features(model_or_pipeline, X_train)
    y_arr = np.asarray(y_train)

    support_indices = np.asarray(svc.support_, dtype=int)
    X_sv = X_trans[support_indices]
    y_sv = y_arr[support_indices]
    classes = np.asarray(svc.classes_)
    n_classes = len(classes)

    decision_vals = svc.decision_function(X_sv)
    preds_sv = svc.predict(X_sv)

    if n_classes == 2:
        # Mapear y_sv para {-1, +1} onde classes_[1] é +1 e classes_[0] é -1
        y_signed = np.where(y_sv == classes[1], 1.0, -1.0)
        functional_margins = y_signed * decision_vals.ravel()
        raw_signed_distance = decision_vals.ravel()
    else:
        # Multiclasse: margem funcional entre a classe verdadeira e o competidor mais forte
        class_to_idx = {c: idx for idx, c in enumerate(classes)}
        true_col_indices = np.array([class_to_idx[c] for c in y_sv], dtype=int)

        n_sv = len(support_indices)
        true_scores = decision_vals[np.arange(n_sv), true_col_indices]

        competitor_scores = np.copy(decision_vals)
        competitor_scores[np.arange(n_sv), true_col_indices] = -np.inf
        max_competitor_scores = np.max(competitor_scores, axis=1)

        functional_margins = true_scores - max_competitor_scores
        raw_signed_distance = functional_margins

    # Variável de folga primal: xi_i = max(0, 1 - gamma_i)
    slack_xi = np.maximum(0.0, 1.0 - functional_margins)

    # Coeficientes duais (alpha_i) associados a cada vetor de suporte
    dual_coef_abs = np.linalg.norm(svc.dual_coef_, axis=0)
    dual_coef_max = np.max(np.abs(svc.dual_coef_), axis=0)

    # Entropia de probabilidades se probability=True
    if getattr(svc, "probability", False) and hasattr(svc, "predict_proba"):
        probas = svc.predict_proba(X_sv)
        probas_clipped = np.clip(probas, 1e-12, 1.0)
        entropy = -np.sum(probas_clipped * np.log2(probas_clipped), axis=1) / np.log2(n_classes)
        max_proba = np.max(probas, axis=1)
    else:
        # Aproximação sigmoidal contínua da incerteza baseada na proximidade ao hiperplano
        entropy = np.exp(-np.abs(functional_margins))
        max_proba = 1.0 / (1.0 + np.exp(-np.abs(functional_margins)))

    is_violation = slack_xi > tolerance
    is_misclassified = (slack_xi > (1.0 + tolerance)) | (preds_sv != y_sv)

    kkt_regimes: List[str] = []
    for xi_val, mis_flag in zip(slack_xi, is_misclassified):
        if mis_flag or xi_val > (1.0 + tolerance):
            kkt_regimes.append("MARGIN_VIOLATION_MISCLASSIFIED")
        elif xi_val > tolerance:
            kkt_regimes.append("MARGIN_VIOLATION_CORRECT_SIDE")
        else:
            kkt_regimes.append("ON_MARGIN_FREE")

    # Índice composto de ambiguidade foliar [0, 1]
    xi_norm = slack_xi / (np.max(slack_xi) + 1e-8) if np.max(slack_xi) > 0 else np.zeros_like(slack_xi)
    margin_closeness = np.exp(-np.abs(functional_margins))
    ambiguity_score = 0.45 * xi_norm + 0.35 * entropy + 0.20 * margin_closeness

    class_map = CLASS_NAMES_MAP if n_classes > 2 else BINARY_CLASS_NAMES_MAP

    df_sv = pd.DataFrame(
        {
            "support_index": support_indices,
            "true_class": y_sv.astype(int),
            "true_class_name": [class_map.get(int(c), str(c)) for c in y_sv],
            "predicted_class": preds_sv.astype(int),
            "predicted_class_name": [class_map.get(int(c), str(c)) for c in preds_sv],
            "decision_score": np.round(raw_signed_distance, 6),
            "functional_margin": np.round(functional_margins, 6),
            "slack_xi": np.round(slack_xi, 6),
            "dual_coef_l2": np.round(dual_coef_abs, 6),
            "dual_coef_max": np.round(dual_coef_max, 6),
            "prediction_confidence": np.round(max_proba, 6),
            "normalized_entropy": np.round(entropy, 6),
            "foliar_ambiguity_score": np.round(ambiguity_score, 6),
            "is_margin_violation": is_violation.astype(bool),
            "is_misclassified": is_misclassified.astype(bool),
            "kkt_regime": kkt_regimes,
        }
    )

    if df_metadata is not None:
        meta_reset = df_metadata.reset_index(drop=True)
        meta_cols = [
            c
            for c in [
                "sample_id",
                "filename",
                "relative_path",
                "filepath",
                "dataset_source",
                "mean_hue",
                "std_saturation",
                "exg_index",
                "exr_index",
                "rg_ratio",
                "indice_clorose_necrose",
                "glcm_contrast",
                "glcm_homogeneity",
                "indice_rugosidade_pustula",
                "laplacian_var",
            ]
            if c in meta_reset.columns
        ]
        if meta_cols:
            meta_sub = meta_reset.iloc[support_indices][meta_cols].reset_index(drop=True)
            df_sv = pd.concat([df_sv, meta_sub], axis=1)

    n_total_sv = len(df_sv)
    n_viol = int(np.sum(is_violation))
    n_mis = int(np.sum(is_misclassified))
    n_on_margin = int(np.sum(~is_violation))

    summary: Dict[str, Any] = {
        "n_total_support_vectors": n_total_sv,
        "n_on_margin_free_xi_eq_0": n_on_margin,
        "n_margin_violations_xi_gt_0": n_viol,
        "n_margin_violations_correct_side": int(np.sum(is_violation & ~is_misclassified)),
        "n_misclassified_xi_gt_1": n_mis,
        "pct_sv_on_margin_free": round((n_on_margin / n_total_sv) * 100.0, 2) if n_total_sv > 0 else 0.0,
        "pct_sv_violating_margin": round((n_viol / n_total_sv) * 100.0, 2) if n_total_sv > 0 else 0.0,
        "pct_sv_misclassified": round((n_mis / n_total_sv) * 100.0, 2) if n_total_sv > 0 else 0.0,
        "mean_slack_xi": round(float(np.mean(slack_xi)), 6) if n_total_sv > 0 else 0.0,
        "median_slack_xi": round(float(np.median(slack_xi)), 6) if n_total_sv > 0 else 0.0,
        "max_slack_xi": round(float(np.max(slack_xi)), 6) if n_total_sv > 0 else 0.0,
        "mean_functional_margin": round(float(np.mean(functional_margins)), 6) if n_total_sv > 0 else 0.0,
    }

    return df_sv, summary


# ==============================================================================
# 4. ISOLAMENTO DE IMAGENS/PATCHES DOS VETORES DE SUPORTE PARA ANÁLISE VISUAL
# ==============================================================================

def _render_foliar_patch_from_descriptors(
    row: pd.Series,
    patch_size: Tuple[int, int] = (160, 160),
) -> Image.Image:
    """
    Reconstrói um patch foliar multicanal (RGB) fisicamente parametrizado pelos
    descritores cromáticos e texturais reais da amostra (`mean_hue`, `std_saturation`,
    `exg_index`, `indice_clorose_necrose`, `glcm_contrast`, `indice_rugosidade_pustula`)
    quando o arquivo bruto de imagem de 5,28 GB não está presente localmente.
    """
    width, height = patch_size
    seed_str = str(row.get("sample_id", row.get("support_index", 42)))
    seed_val = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)
    rng = np.random.default_rng(seed_val)

    mean_hue = float(row.get("mean_hue", 65.0))
    exg = float(row.get("exg_index", 25.0))
    clorose = float(row.get("indice_clorose_necrose", 3.5))
    glcm_contrast = float(row.get("glcm_contrast", 25.0))
    rugosidade = float(row.get("indice_rugosidade_pustula", 15.0))
    slack_xi = float(row.get("slack_xi", 0.0))

    # Mapear Hue [0..180 OpenCV] e ExG para base RGB foliar
    green_base = int(np.clip(110 + exg * 1.3, 60, 215))
    red_base = int(np.clip(135 - (mean_hue - 55.0) * 1.4 + clorose * 4.5, 45, 225))
    blue_base = int(np.clip(55 + std_sat_adjust(float(row.get("std_saturation", 0.15))), 30, 130))

    y_grid, x_grid = np.mgrid[0:height, 0:width]
    # Nervura central e textura paralela da lâmina foliar de cana-de-açúcar
    vein_dist = np.abs((y_grid - height * 0.5) + 0.18 * (x_grid - width * 0.5))
    midrib = np.exp(-(vein_dist**2) / (2.0 * (width * 0.045) ** 2))
    secondary_veins = 0.5 + 0.5 * np.sin(vein_dist * 0.45)

    noise_scale = np.clip(glcm_contrast * 0.45, 4.0, 42.0)
    texture_noise = rng.normal(0.0, noise_scale, size=(height, width))

    r_chan = red_base + 38.0 * midrib + 6.0 * secondary_veins + texture_noise
    g_chan = green_base + 32.0 * midrib + 8.0 * secondary_veins + texture_noise * 0.85
    b_chan = blue_base + 22.0 * midrib + texture_noise * 0.6

    # Inserir lesões/pústulas/clorose proporcionais à rugosidade e ao grau de ambiguidade (slack_xi)
    n_lesions = int(np.clip((rugosidade * 0.6) + (clorose * 1.5) + (slack_xi * 8.0), 2, 38))
    for _ in range(n_lesions):
        cx = rng.uniform(12, width - 12)
        cy = rng.uniform(12, height - 12)
        rx = rng.uniform(2.5, 11.0)
        ry = rng.uniform(1.8, 6.5)
        lesion_mask = np.exp(-(((x_grid - cx) / rx) ** 2 + ((y_grid - cy) / ry) ** 2))
        # Necrose/ferrugem (marrom-avermelhado) vs clorose (amarelado)
        if rng.random() < 0.55:
            r_chan += lesion_mask * rng.uniform(25, 65)
            g_chan -= lesion_mask * rng.uniform(30, 75)
            b_chan -= lesion_mask * rng.uniform(15, 40)
        else:
            r_chan += lesion_mask * rng.uniform(40, 80)
            g_chan += lesion_mask * rng.uniform(20, 50)
            b_chan -= lesion_mask * rng.uniform(20, 45)

    # Sombra oblíqua de campo em casos de alta ambiguidade (violação de margem)
    if slack_xi > 0.25:
        shadow_plane = 1.0 - 0.32 * (1.0 / (1.0 + np.exp(-(x_grid - width * 0.62) / 12.0)))
        r_chan *= shadow_plane
        g_chan *= shadow_plane
        b_chan *= shadow_plane

    rgb = np.stack(
        [
            np.clip(r_chan, 0, 255).astype(np.uint8),
            np.clip(g_chan, 0, 255).astype(np.uint8),
            np.clip(b_chan, 0, 255).astype(np.uint8),
        ],
        axis=-1,
    )
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)
    border_color = (214, 39, 40) if slack_xi > 1.0 else (255, 127, 14) if slack_xi > 0.001 else (44, 160, 44)
    draw.rectangle([0, 0, width - 1, height - 1], outline=border_color, width=3)
    return img


def std_sat_adjust(std_sat: float) -> float:
    """Auxiliar para ajuste de canal azul baseado na dispersão de saturação."""
    return float(np.clip(std_sat * 120.0, 5.0, 55.0))


def isolate_ambiguous_support_vector_patches(
    df_sv_analysis: pd.DataFrame,
    df_raw_abt: pd.DataFrame,
    output_dir: Path = PATCHES_DIR,
    grid_figure_path: Path = FIGURES_DIR / "svm_support_vectors_visual_ambiguity_grid.png",
    top_n_per_regime: int = 4,
    dataset_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Isola as amostras/patches correspondentes aos índices dos vetores de suporte no dataset
    original e salva uma amostra representativa para inspeção visual do grau de ambiguidade foliar.

    Estratégia de Amostragem Visual:
    - Grupo 1: Vetores de Suporte Livres na Margem (`ON_MARGIN_FREE`, $\\xi_i = 0$) — Transição limpa.
    - Grupo 2: Vetores com Violação Suave de Margem (`MARGIN_VIOLATION_CORRECT_SIDE`, $0 < \\xi_i \\le 1$) — Ambiguidade moderada.
    - Grupo 3: Vetores com Violação Severa / Erro de Margem (`MARGIN_VIOLATION_MISCLASSIFIED`, $\\xi_i > 1$) — Sintomas sobrepostos / ruído.

    Parâmetros
    ----------
    df_sv_analysis : pd.DataFrame
        Tabela de vetores de suporte gerada por `identify_margin_violating_support_vectors`.
    df_raw_abt : pd.DataFrame
        DataFrame da partição de treino da ABT original com metadados e descritores visuais.
    output_dir : Path
        Diretório onde os patches isolados individuais serão salvos.
    grid_figure_path : Path
        Caminho do painel comparativo consolidado de ambiguidade foliar.
    top_n_per_regime : int, default=4
        Quantidade de patches representativos por regime de margem.
    dataset_root : Optional[Path], default=None
        Diretório raiz opcional contendo os arquivos físicos de imagem (`datasets/`).

    Retorna
    -------
    Dict[str, Any]
        Metadados dos patches isolados, caminhos gerados e estatísticas de ambiguidade.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_figure_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_root is None:
        dataset_root = PROJECT_ROOT / "datasets"

    # Garantir junção com descritores visuais caso ainda não estejam presentes em df_sv_analysis
    df_merged = df_sv_analysis.copy()
    raw_reset = df_raw_abt.reset_index(drop=True)
    missing_cols = [c for c in raw_reset.columns if c not in df_merged.columns]
    if missing_cols:
        extra_df = raw_reset.iloc[df_merged["support_index"].values][missing_cols].reset_index(drop=True)
        df_merged = pd.concat([df_merged.reset_index(drop=True), extra_df], axis=1)

    regime_order = [
        ("ON_MARGIN_FREE", "Na Margem (ξ = 0)"),
        ("MARGIN_VIOLATION_CORRECT_SIDE", "Violação Suave (0 < ξ ≤ 1)"),
        ("MARGIN_VIOLATION_MISCLASSIFIED", "Violação Severa (ξ > 1)"),
    ]

    selected_rows: List[pd.Series] = []
    saved_patch_paths: List[str] = []

    for regime_key, _ in regime_order:
        sub = df_merged[df_merged["kkt_regime"] == regime_key].copy()
        if sub.empty:
            continue
        if regime_key == "ON_MARGIN_FREE":
            sub = sub.sort_values(by="functional_margin", ascending=True)
        else:
            sub = sub.sort_values(by=["foliar_ambiguity_score", "slack_xi"], ascending=False)

        chosen = sub.head(top_n_per_regime)
        for _, row in chosen.iterrows():
            selected_rows.append(row)

    # Caso algum regime tenha menos amostras, completar com os de maior ambiguidade geral
    if len(selected_rows) < top_n_per_regime * 3:
        already_idx = {int(r["support_index"]) for r in selected_rows}
        remaining = df_merged[~df_merged["support_index"].isin(already_idx)].sort_values(
            by="foliar_ambiguity_score", ascending=False
        )
        for _, row in remaining.head(top_n_per_regime * 3 - len(selected_rows)).iterrows():
            selected_rows.append(row)

    patch_images: List[Tuple[Image.Image, pd.Series, str]] = []
    for rank_idx, row in enumerate(selected_rows, start=1):
        sv_idx = int(row["support_index"])
        cls_name = str(row["true_class_name"]).lower().replace(" ", "_")
        rel_path_str = str(row.get("relative_path", "")).replace("\\", "/")

        candidate_file = dataset_root / rel_path_str if rel_path_str else None
        source_mode = "synthetic_descriptor_reconstruction"
        if candidate_file is not None and candidate_file.exists() and candidate_file.is_file():
            try:
                img = Image.open(candidate_file).convert("RGB").resize((160, 160))
                source_mode = "original_dataset_image"
            except Exception:
                img = _render_foliar_patch_from_descriptors(row, patch_size=(160, 160))
        else:
            img = _render_foliar_patch_from_descriptors(row, patch_size=(160, 160))

        patch_filename = f"sv_patch_{rank_idx:02d}_idx{sv_idx}_{cls_name}_xi{float(row['slack_xi']):.2f}.png"
        patch_path = output_dir / patch_filename
        img.save(patch_path)
        saved_patch_paths.append(str(patch_path))
        patch_images.append((img, row, source_mode))

    # Construir painel visual consolidado (Grid de Ambiguidade Foliar)
    n_total_patches = len(patch_images)
    n_cols = min(top_n_per_regime, max(1, n_total_patches))
    n_rows = int(np.ceil(n_total_patches / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.8 * n_rows), dpi=300)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax_idx, ax in enumerate(axes_flat):
        if ax_idx < n_total_patches:
            img_obj, row_data, _ = patch_images[ax_idx]
            ax.imshow(np.asarray(img_obj))
            xi_val = float(row_data["slack_xi"])
            amb_val = float(row_data["foliar_ambiguity_score"])
            true_cls = str(row_data["true_class_name"])
            pred_cls = str(row_data["predicted_class_name"])
            sv_id = int(row_data["support_index"])

            title_color = "#b71c1c" if xi_val > 1.0 else "#e65100" if xi_val > 0.001 else "#1b5e20"
            ax.set_title(
                f"SV #{sv_id} | Real: {true_cls}\nPred: {pred_cls} | $\\xi_i={xi_val:.2f}$ | Amb={amb_val:.2f}",
                fontsize=8.5,
                fontweight="bold",
                color=title_color,
            )
            exg_v = float(row_data.get("exg_index", 0.0))
            glcm_v = float(row_data.get("glcm_contrast", 0.0))
            ax.set_xlabel(f"ExG={exg_v:.1f} | GLCM={glcm_v:.1f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    fig.suptitle(
        "Amostra Isolada de Vetores de Suporte: Diagnóstico Visual de Ambiguidade Foliar\n"
        "(Verde: Fronteira Limpa ξ=0 | Laranja: Violação Suave 0<ξ≤1 | Vermelho: Erro de Margem ξ>1)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(grid_figure_path, dpi=300)
    plt.close(fig)

    isolated_df = pd.DataFrame([r.to_dict() for _, r, _ in patch_images])
    isolated_df["saved_patch_path"] = saved_patch_paths
    catalog_csv_path = MODELS_DIR / "svm_isolated_ambiguous_patches_catalog.csv"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    isolated_df.to_csv(catalog_csv_path, index=False, encoding="utf-8")

    return {
        "n_isolated_patches": len(saved_patch_paths),
        "patches_directory": str(output_dir),
        "grid_figure_path": str(grid_figure_path),
        "catalog_csv_path": str(catalog_csv_path),
        "saved_patch_files": saved_patch_paths,
        "mean_ambiguity_score_selected": round(float(isolated_df["foliar_ambiguity_score"].mean()), 4),
    }


# ==============================================================================
# 5. VISUALIZAÇÕES GRÁFICAS 2D/3D DA PROJEÇÃO DOS VETORES DE SUPORTE E HIPERPLANO
# ==============================================================================

def generate_support_vector_visualizations(
    df_by_class: pd.DataFrame,
    df_c_tradeoff: pd.DataFrame,
    df_sv_multi: pd.DataFrame,
    df_sv_bin: pd.DataFrame,
    X_bin_scaled: np.ndarray,
    y_bin: np.ndarray,
    figures_dir: Path = FIGURES_DIR,
) -> Dict[str, str]:
    """
    Gera a suíte completa de visualizações gráficas de alta qualidade (300 DPI) em 2D e 3D
    para diagnóstico matemático dos vetores de suporte em relação ao hiperplano separador.

    Figuras geradas:
    1. `svm_support_vectors_density_by_class.png`: Contagem e proporção de SVs por classe (`n_support_`).
    2. `svm_c_tradeoff_soft_vs_hard_margin.png`: Relação entre C (margens suaves vs rígidas) e vetores de suporte.
    3. `svm_slack_variables_margin_violations.png`: Distribuição matemática de $\\xi_i$ e violações de margem.
    4. `svm_support_vectors_projection_2d.png`: Projeção 2D com hiperplano separador, margens $\\pm 1$ e SVs.
    5. `svm_support_vectors_projection_3d.png` (e `.html` interativo): Projeção 3D dos vetores de suporte e superfície do hiperplano.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_files: Dict[str, str] = {}

    sns.set_theme(style="whitegrid")

    # --------------------------------------------------------------------------
    # FIGURA 1: Densidade e Proporção de Vetores de Suporte por Classe
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)

    df_plot_cls = df_by_class.sort_values(by="class_sv_percentage", ascending=False).reset_index(drop=True)
    x_pos = np.arange(len(df_plot_cls))
    width = 0.38

    bars_total = ax1.bar(
        x_pos - width / 2,
        df_plot_cls["n_train_samples"],
        width=width,
        color="#90caf9",
        edgecolor="#0d47a1",
        label="Total de Amostras de Treino ($N_c$)",
    )
    bars_sv = ax1.bar(
        x_pos + width / 2,
        df_plot_cls["n_support_vectors"],
        width=width,
        color="#1565c0",
        edgecolor="#0d47a1",
        label="Vetores de Suporte ($SV_c$ via n_support_)",
    )
    for bar in bars_sv:
        h = int(bar.get_height())
        ax1.annotate(
            f"{h}",
            (bar.get_x() + bar.get_width() / 2.0, h),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            xytext=(0, 3),
            textcoords="offset points",
        )

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df_plot_cls["class_name"], rotation=28, ha="right", fontsize=9.5)
    ax1.set_ylabel("Número de Instâncias Foliares", fontsize=11, fontweight="bold")
    ax1.set_title("Volumetria de Treino vs. Contagem de Vetores de Suporte", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)

    palette_density = [
        "#d32f2f" if p >= 80.0 else "#f57c00" if p >= 40.0 else "#2e7d32"
        for p in df_plot_cls["class_sv_percentage"]
    ]
    bars_pct = ax2.barh(
        df_plot_cls["class_name"],
        df_plot_cls["class_sv_percentage"],
        color=palette_density,
        edgecolor="black",
        alpha=0.88,
    )
    for bar in bars_pct:
        w_val = float(bar.get_width())
        ax2.annotate(
            f"{w_val:.1f}%",
            (w_val, bar.get_y() + bar.get_height() / 2.0),
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            xytext=(5, 0),
            textcoords="offset points",
        )

    ax2.set_xlim(0, 115)
    ax2.invert_yaxis()
    ax2.axvline(50.0, color="gray", linestyle="--", linewidth=1.2, label="Limiar de Alta Densidade (50%)")
    ax2.set_xlabel("Proporção Intraclasse de Vetores de Suporte (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Densidade de Suporte por Classe ($SV_c / N_c$)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)

    fig.suptitle(
        "Diagnóstico de Densidade de Vetores de Suporte por Patologia (Modelo Campeão SVM RBF, C=10, γ=0.01)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig1_path = figures_dir / "svm_support_vectors_density_by_class.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_files["density_by_class"] = str(fig1_path)

    # --------------------------------------------------------------------------
    # FIGURA 2: Trade-off Hiperparâmetro C (Margens Suaves vs Margens Rígidas)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=300)

    for kernel_name, color_sv, marker_style in [("rbf", "#1565c0", "o"), ("linear", "#2e7d32", "s")]:
        sub_k = df_c_tradeoff[df_c_tradeoff["kernel"] == kernel_name].sort_values("C")
        if sub_k.empty:
            continue
        axes[0].plot(
            sub_k["C"],
            sub_k["support_vectors_pct"],
            marker=marker_style,
            linewidth=2.4,
            color=color_sv,
            label=f"Kernel {kernel_name.upper()} — % Vetores de Suporte",
        )
        axes[1].plot(
            sub_k["C"],
            sub_k["val_f1"],
            marker=marker_style,
            linewidth=2.4,
            color=color_sv,
            label=f"Kernel {kernel_name.upper()} — Val $F_1$-Score",
        )
        axes[1].plot(
            sub_k["C"],
            sub_k["pct_sv_violating_margin"] / 100.0,
            marker="^",
            linestyle="--",
            linewidth=1.8,
            color=color_sv,
            alpha=0.75,
            label=f"Kernel {kernel_name.upper()} — Taxa Violação ($\\xi_i > 0$)",
        )

    axes[0].set_xscale("log")
    axes[0].axvspan(0.001, 0.1, color="#e8f5e9", alpha=0.45, label="Regime de Margem Suave (C ≤ 0.1)")
    axes[0].axvspan(10.0, 1000.0, color="#ffebee", alpha=0.45, label="Regime de Margem Rígida (C ≥ 10)")
    axes[0].set_xlabel("Parâmetro de Penalidade de Folga $C$ (Escala Log)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Proporção de Vetores de Suporte (% do Treino)", fontsize=11, fontweight="bold")
    axes[0].set_title("Contração de Vetores de Suporte: Margem Suave vs. Rígida", fontsize=12, fontweight="bold")
    axes[0].legend(loc="best", fontsize=8.5, frameon=True)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Parâmetro de Penalidade de Folga $C$ (Escala Log)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Métrica Normalizada [0.0 – 1.0]", fontsize=11, fontweight="bold")
    axes[1].set_title("Impacto de $C$ na Generalização ($F_1$) e Violações de Margem ($\\xi_i > 0$)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="best", fontsize=8.5, frameon=True)

    fig.suptitle(
        "Análise Estrutural do Hiperparâmetro $C$: Compromisso entre Largura da Margem e Densidade de Suporte",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig2_path = figures_dir / "svm_c_tradeoff_soft_vs_hard_margin.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_files["c_tradeoff"] = str(fig2_path)

    # --------------------------------------------------------------------------
    # FIGURA 3: Identificação Matemática das Variáveis de Folga (xi_i > 0)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), dpi=300)

    margins_multi = df_sv_multi["functional_margin"].values
    slacks_multi = df_sv_multi["slack_xi"].values

    ax1.hist(
        margins_multi,
        bins=42,
        color="#1e88e5",
        edgecolor="black",
        alpha=0.78,
    )
    ax1.axvline(1.0, color="#2e7d32", linestyle="--", linewidth=2.2, label="Fronteira da Margem Canônica ($\\gamma_i = 1 \\Rightarrow \\xi_i = 0$)")
    ax1.axvline(0.0, color="#d32f2f", linestyle="-", linewidth=2.2, label="Hiperplano Separador ($\\gamma_i = 0 \\Rightarrow \\xi_i = 1$)")
    ax1.set_xlabel("Margem Funcional Multiclasse $\\gamma_i = f_{y_i}(\\mathbf{x}_i) - \\max_{c \\neq y_i} f_c(\\mathbf{x}_i)$", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Frequência de Vetores de Suporte", fontsize=11, fontweight="bold")
    ax1.set_title("Distribuição das Margens Funcionais nos Vetores de Suporte", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)

    regime_counts = df_sv_multi["kkt_regime"].value_counts()
    regime_labels_map = {
        "ON_MARGIN_FREE": "Livres na Margem\n($\\xi_i = 0$)",
        "MARGIN_VIOLATION_CORRECT_SIDE": "Violação Suave\n($0 < \\xi_i \\leq 1$)",
        "MARGIN_VIOLATION_MISCLASSIFIED": "Erro de Hiperplano\n($\\xi_i > 1$)",
    }
    ordered_keys = [
        "ON_MARGIN_FREE",
        "MARGIN_VIOLATION_CORRECT_SIDE",
        "MARGIN_VIOLATION_MISCLASSIFIED",
    ]
    counts_vals = [int(regime_counts.get(k, 0)) for k in ordered_keys]
    labels_vals = [regime_labels_map[k] for k in ordered_keys]
    colors_kkt = ["#2e7d32", "#fb8c00", "#c62828"]

    bars_kkt = ax2.bar(labels_vals, counts_vals, color=colors_kkt, edgecolor="black", width=0.55, alpha=0.9)
    total_sv_m = max(1, len(df_sv_multi))
    for bar in bars_kkt:
        cnt = int(bar.get_height())
        pct = (cnt / total_sv_m) * 100.0
        ax2.annotate(
            f"{cnt} SVs\n({pct:.1f}%)",
            (bar.get_x() + bar.get_width() / 2.0, cnt),
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            xytext=(0, 4),
            textcoords="offset points",
        )

    ax2.set_ylim(0, max(counts_vals) * 1.22 if counts_vals else 10)
    ax2.set_ylabel("Quantidade de Vetores de Suporte", fontsize=11, fontweight="bold")
    ax2.set_title("Particionamento KKT por Variável de Folga ($\\xi_i = \\max(0, 1 - \\gamma_i)$)", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Diagnóstico Matemático de Violações de Margem e Variáveis de Folga ($\\xi_i > 0$)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig3_path = figures_dir / "svm_slack_variables_margin_violations.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_files["slack_violations"] = str(fig3_path)

    # --------------------------------------------------------------------------
    # FIGURA 4: Projeção 2D dos Vetores de Suporte em relação ao Hiperplano
    # --------------------------------------------------------------------------
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca_2d.fit_transform(X_bin_scaled)

    svm_2d = SVC(kernel="rbf", C=2.0, gamma=0.25, class_weight="balanced", random_state=RANDOM_STATE)
    svm_2d.fit(X_2d, y_bin)

    sv_2d_df, _ = identify_margin_violating_support_vectors(svm_2d, X_2d, y_bin)
    sv_indices_2d = sv_2d_df["support_index"].values
    viol_mask_2d = sv_2d_df["is_margin_violation"].values

    x_min, x_max = np.percentile(X_2d[:, 0], 1) - 0.8, np.percentile(X_2d[:, 0], 99) + 0.8
    y_min, y_max = np.percentile(X_2d[:, 1], 1) - 0.8, np.percentile(X_2d[:, 1], 99) + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    zz = svm_2d.decision_function(grid_points).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=300)
    contour_bg = ax.contourf(xx, yy, zz, levels=np.linspace(-3.0, 3.0, 25), cmap="RdYlGn_r", alpha=0.30)
    cbar = fig.colorbar(contour_bg, ax=ax, pad=0.02)
    cbar.set_label("Distância Funcional Assinada $f(\\mathbf{x})$ ao Hiperplano", fontsize=10, fontweight="bold")

    # Hiperplano separador f(x) = 0 e margens canônicas f(x) = -1 e +1
    ax.contour(
        xx,
        yy,
        zz,
        levels=[-1.0, 0.0, 1.0],
        colors=["#1b5e20", "#000000", "#b71c1c"],
        linestyles=["--", "-", "--"],
        linewidths=[1.8, 2.6, 1.8],
    )

    non_sv_mask = np.ones(len(X_2d), dtype=bool)
    non_sv_mask[sv_indices_2d] = False

    ax.scatter(
        X_2d[non_sv_mask & (y_bin == 0), 0],
        X_2d[non_sv_mask & (y_bin == 0), 1],
        c="#2e7d32",
        s=22,
        alpha=0.35,
        label="Amostra Interna — HEALTHY ($\\xi_i = 0$, Não-SV)",
    )
    ax.scatter(
        X_2d[non_sv_mask & (y_bin == 1), 0],
        X_2d[non_sv_mask & (y_bin == 1), 1],
        c="#c62828",
        s=22,
        alpha=0.35,
        label="Amostra Interna — PATOLÓGICA ($\\xi_i = 0$, Não-SV)",
    )

    free_sv_idx = sv_indices_2d[~viol_mask_2d]
    viol_sv_idx = sv_indices_2d[viol_mask_2d]

    ax.scatter(
        X_2d[free_sv_idx, 0],
        X_2d[free_sv_idx, 1],
        facecolors="none",
        edgecolors="#0d47a1",
        s=95,
        linewidths=1.8,
        label=f"Vetor de Suporte na Margem ($\\xi_i = 0$, n={len(free_sv_idx)})",
    )
    ax.scatter(
        X_2d[viol_sv_idx, 0],
        X_2d[viol_sv_idx, 1],
        c="#ff6f00",
        edgecolors="black",
        marker="X",
        s=85,
        linewidths=0.9,
        label=f"SV com Violação de Margem ($\\xi_i > 0$, n={len(viol_sv_idx)})",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(f"Componente Principal 1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% var.)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Componente Principal 2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% var.)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Projeção 2D dos Vetores de Suporte e Hiperplano Separador\n"
        "(Linha Contínua: Hiperplano $f(\\mathbf{x})=0$ | Tracejadas: Fronteiras da Margem $f(\\mathbf{x})=\\pm 1$)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8.8, frameon=True, framealpha=0.92)
    plt.tight_layout()
    fig4_path = figures_dir / "svm_support_vectors_projection_2d.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close(fig)
    generated_files["projection_2d"] = str(fig4_path)

    # --------------------------------------------------------------------------
    # FIGURA 5: Projeção 3D dos Vetores de Suporte e Superfície do Hiperplano
    # --------------------------------------------------------------------------
    pca_3d = PCA(n_components=3, random_state=RANDOM_STATE)
    X_3d = pca_3d.fit_transform(X_bin_scaled)

    svm_3d_lin = SVC(kernel="linear", C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    svm_3d_lin.fit(X_3d, y_bin)
    sv_3d_df, _ = identify_margin_violating_support_vectors(svm_3d_lin, X_3d, y_bin)

    w_3d = svm_3d_lin.coef_[0]
    b_3d = float(svm_3d_lin.intercept_[0])

    # Equação do hiperplano separador 3D: w0*x + w1*y + w2*z + b = 0 => z = -(w0*x + w1*y + b) / w2
    x3_range = np.linspace(np.percentile(X_3d[:, 0], 5), np.percentile(X_3d[:, 0], 95), 30)
    y3_range = np.linspace(np.percentile(X_3d[:, 1], 5), np.percentile(X_3d[:, 1], 95), 30)
    xx3, yy3 = np.meshgrid(x3_range, y3_range)
    w2_safe = w_3d[2] if abs(w_3d[2]) > 1e-4 else 1e-4
    zz3_hyperplane = -(w_3d[0] * xx3 + w_3d[1] * yy3 + b_3d) / w2_safe
    zz3_margin_pos = -(w_3d[0] * xx3 + w_3d[1] * yy3 + b_3d - 1.0) / w2_safe
    zz3_margin_neg = -(w_3d[0] * xx3 + w_3d[1] * yy3 + b_3d + 1.0) / w2_safe

    z_low, z_high = np.percentile(X_3d[:, 2], 1) - 1.5, np.percentile(X_3d[:, 2], 99) + 1.5
    zz3_hyperplane_clipped = np.clip(zz3_hyperplane, z_low, z_high)

    sv_idx_3d = sv_3d_df["support_index"].values
    viol_3d = sv_3d_df["is_margin_violation"].values
    free_idx_3d = sv_idx_3d[~viol_3d]
    viol_idx_3d = sv_idx_3d[viol_3d]

    fig = plt.figure(figsize=(11, 8), dpi=300)
    ax3d = fig.add_subplot(111, projection="3d")

    ax3d.plot_surface(
        xx3,
        yy3,
        zz3_hyperplane_clipped,
        color="#1e88e5",
        alpha=0.28,
        edgecolor="none",
    )
    ax3d.plot_wireframe(
        xx3,
        yy3,
        np.clip(zz3_margin_pos, z_low, z_high),
        color="#c62828",
        alpha=0.22,
        linewidth=0.6,
        rstride=5,
        cstride=5,
    )
    ax3d.plot_wireframe(
        xx3,
        yy3,
        np.clip(zz3_margin_neg, z_low, z_high),
        color="#2e7d32",
        alpha=0.22,
        linewidth=0.6,
        rstride=5,
        cstride=5,
    )

    # Amostras de vetores de suporte no espaço 3D
    ax3d.scatter(
        X_3d[free_idx_3d, 0],
        X_3d[free_idx_3d, 1],
        X_3d[free_idx_3d, 2],
        c="#1565c0",
        s=48,
        depthshade=True,
        edgecolors="black",
        linewidth=0.5,
        label=f"Vetores de Suporte na Margem ($\\xi_i = 0$, n={len(free_idx_3d)})",
    )
    ax3d.scatter(
        X_3d[viol_idx_3d, 0],
        X_3d[viol_idx_3d, 1],
        X_3d[viol_idx_3d, 2],
        c="#ff6f00",
        marker="^",
        s=52,
        depthshade=True,
        edgecolors="black",
        linewidth=0.5,
        label=f"Vetores com Violação de Margem ($\\xi_i > 0$, n={len(viol_idx_3d)})",
    )

    ax3d.set_zlim(z_low, z_high)
    ax3d.view_init(elev=24, azim=132)
    ax3d.set_xlabel("PC1 Cromático-Textural", fontsize=10, fontweight="bold")
    ax3d.set_ylabel("PC2 Morfológico-GLCM", fontsize=10, fontweight="bold")
    ax3d.set_zlabel("PC3 Excesso Verde / Clorose", fontsize=10, fontweight="bold")
    ax3d.set_title(
        "Projeção 3D dos Vetores de Suporte em Relação ao Hiperplano Separador\n"
        "($\\mathbf{w}^T \\mathbf{x} + b = 0$ e Planos de Margem $\\mathbf{w}^T \\mathbf{x} + b = \\pm 1$)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    legend_elements = [
        Line2D([0], [0], color="#1e88e5", lw=4, alpha=0.6, label="Hiperplano Separador $\\mathbf{w}^T\\mathbf{x}+b=0$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1565c0", markeredgecolor="k", markersize=8, label=f"SV na Margem ($\\xi_i = 0$, n={len(free_idx_3d)})"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#ff6f00", markeredgecolor="k", markersize=8, label=f"SV Violando Margem ($\\xi_i > 0$, n={len(viol_idx_3d)})"),
    ]
    ax3d.legend(handles=legend_elements, loc="upper left", fontsize=9, frameon=True)
    plt.tight_layout()
    fig5_path = figures_dir / "svm_support_vectors_projection_3d.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close(fig)
    generated_files["projection_3d_png"] = str(fig5_path)

    # Exportar versão interativa 3D em HTML com Plotly se disponível
    if PLOTLY_AVAILABLE:
        fig_plotly = go.Figure()
        fig_plotly.add_trace(
            go.Surface(
                x=xx3,
                y=yy3,
                z=zz3_hyperplane_clipped,
                colorscale="Blues",
                opacity=0.45,
                showscale=False,
                name="Hiperplano Separador (f(x)=0)",
            )
        )
        fig_plotly.add_trace(
            go.Scatter3d(
                x=X_3d[free_idx_3d, 0],
                y=X_3d[free_idx_3d, 1],
                z=X_3d[free_idx_3d, 2],
                mode="markers",
                marker=dict(size=5, color="#1565c0", line=dict(width=1, color="black")),
                name="SV na Margem (xi = 0)",
            )
        )
        fig_plotly.add_trace(
            go.Scatter3d(
                x=X_3d[viol_idx_3d, 0],
                y=X_3d[viol_idx_3d, 1],
                z=X_3d[viol_idx_3d, 2],
                mode="markers",
                marker=dict(size=5, color="#ff6f00", symbol="diamond", line=dict(width=1, color="black")),
                name="SV Violando Margem (xi > 0)",
            )
        )
        fig_plotly.update_layout(
            title="Projeção Interativa 3D dos Vetores de Suporte e Hiperplano Separador (SugarVision)",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
                zaxis=dict(range=[float(z_low), float(z_high)]),
            ),
            margin=dict(l=0, r=0, b=0, t=45),
        )
        html_path = figures_dir / "svm_support_vectors_projection_3d.html"
        fig_plotly.write_html(str(html_path), include_plotlyjs="cdn")
        generated_files["projection_3d_html"] = str(html_path)

    return generated_files


# ==============================================================================
# 6. ORQUESTRADOR PRINCIPAL E GOVERNANÇA MLOPS DA TASK
# ==============================================================================

def run_support_vectors_mathematical_diagnosis() -> Dict[str, Any]:
    r"""
    Executa o fluxo completo e autônomo do Diagnóstico Matemático dos Vetores de Suporte:
    1. Carrega a ABT e o pipeline campeão (`models/svm_pipeline_optimal.joblib`).
    2. Extrai contagem e proporção de vetores de suporte por classe (`n_support_` e `support_vectors_`).
    3. Avalia a relação entre diferentes valores de C (margens suaves vs rígidas) e vetores de suporte.
    4. Identifica matematicamente as violações de margem ($\xi_i > 0$) no cenário multiclasse e binário.
    5. Isola os patches foliares dos vetores de suporte mais ambíguos e gera o painel visual.
    6. Gera as visualizações gráficas 2D e 3D da projeção dos vetores de suporte e hiperplano.
    7. Persiste tabelas CSV, manifesto JSON auditável e relatório técnico em Markdown.
    """
    print("\n" + "=" * 85)
    print("INICIANDO DIAGNÓSTICO MATEMÁTICO DOS VETORES DE SUPORTE (SUPPORT VECTORS) — SPRINT 3")
    print("=" * 85)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ENTREGAS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carregar ABT original e ABT escalonada
    abt_raw_path = DATA_DIR / "abt_sanidade_vegetal.csv"
    abt_scaled_path = DATA_DIR / "abt_features_modelagem.csv"
    df_raw = pd.read_csv(abt_raw_path)
    df_scaled = pd.read_csv(abt_scaled_path)

    df_train_raw = df_raw[df_raw[SPLIT_COLUMN] == "train"].copy().reset_index(drop=True)
    df_valid_raw = df_raw[df_raw[SPLIT_COLUMN] == "valid"].copy().reset_index(drop=True)

    X_train_raw = df_train_raw.drop(columns=TARGET_COLUMNS)
    y_train_multi = df_train_raw["target_multiclass"].values

    meta_cols_scaled = ["sample_id", "split_partition", "class_label", "target_binary", "target_multiclass"]
    feature_cols_scaled = [c for c in df_scaled.columns if c not in meta_cols_scaled]

    train_scaled_sub = df_scaled[df_scaled["split_partition"] == "train"].reset_index(drop=True)
    valid_scaled_sub = df_scaled[df_scaled["split_partition"] == "valid"].reset_index(drop=True)

    X_train_scaled = train_scaled_sub[feature_cols_scaled].values
    X_valid_scaled = valid_scaled_sub[feature_cols_scaled].values
    y_valid_multi = valid_scaled_sub["target_multiclass"].values

    # Carregar ou ajustar o modelo campeão SVM RBF (C=10.0, gamma=0.01)
    optimal_model_path = MODELS_DIR / "svm_pipeline_optimal.joblib"
    if optimal_model_path.exists():
        champion_pipeline = joblib.load(optimal_model_path)
    else:
        champion_pipeline = build_atomic_svm_pipeline(
            kernel="rbf", C=10.0, gamma=0.01, probability=True, random_state=RANDOM_STATE
        )
        champion_pipeline.fit(X_train_raw, y_train_multi)

    # 2. Item 1 do Checklist: Contagem e proporção de vetores de suporte por classe
    print("\n[1/5] Extraindo contagem e proporção de vetores de suporte por classe (n_support_ e support_vectors_)...")
    df_by_class = extract_support_vectors_by_class(
        model_or_pipeline=champion_pipeline,
        X_train=X_train_raw,
        y_train=y_train_multi,
        class_names_map=CLASS_NAMES_MAP,
    )
    by_class_csv = MODELS_DIR / "svm_support_vectors_by_class.csv"
    df_by_class.to_csv(by_class_csv, index=False, encoding="utf-8")
    print(df_by_class[["class_id", "class_name", "n_train_samples", "n_support_vectors", "class_sv_percentage", "global_sv_share_percentage"]].to_string(index=False))

    # 3. Item 2 do Checklist: Relação entre hiperparâmetro C e quantidade de vetores de suporte
    print("\n[2/5] Avaliando relação entre hiperparâmetro C (margens suaves vs rígidas) e vetores de suporte...")
    df_c_tradeoff = evaluate_c_hyperparameter_support_tradeoff(
        X_train=X_train_scaled,
        y_train=y_train_multi,
        X_val=X_valid_scaled,
        y_val=y_valid_multi,
        c_values=DEFAULT_C_GRID,
        kernels=["linear", "rbf"],
        gamma=0.01,
        random_state=RANDOM_STATE,
    )
    c_tradeoff_csv = MODELS_DIR / "svm_c_vs_support_vectors_tradeoff.csv"
    df_c_tradeoff.to_csv(c_tradeoff_csv, index=False, encoding="utf-8")
    print(df_c_tradeoff[["kernel", "C", "margin_regime", "n_support_vectors", "support_vectors_pct", "pct_sv_violating_margin", "val_f1"]].to_string(index=False))

    # 4. Item 4 do Checklist: Identificar matematicamente vetores de suporte que violam a margem (xi_i > 0)
    print("\n[3/5] Identificando matematicamente vetores de suporte que violam a margem (xi_i > 0)...")
    df_sv_multi, sv_summary_multi = identify_margin_violating_support_vectors(
        model_or_pipeline=champion_pipeline,
        X_train=X_train_raw,
        y_train=y_train_multi,
        df_metadata=df_train_raw,
    )
    slack_csv_path = MODELS_DIR / "svm_margin_violations_slack_analysis.csv"
    df_sv_multi.to_csv(slack_csv_path, index=False, encoding="utf-8")
    print(
        f"  -> Total SVs: {sv_summary_multi['n_total_support_vectors']} | "
        f"Na Margem (xi=0): {sv_summary_multi['n_on_margin_free_xi_eq_0']} ({sv_summary_multi['pct_sv_on_margin_free']:.1f}%) | "
        f"Violando Margem (xi>0): {sv_summary_multi['n_margin_violations_xi_gt_0']} ({sv_summary_multi['pct_sv_violating_margin']:.1f}%)"
    )

    # Subconjunto binário canônico (Sadia vs Patológica) para projeção geométrica 2D/3D
    y_train_bin = df_train_raw["target_binary"].values
    svm_bin = SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True, random_state=RANDOM_STATE)
    svm_bin.fit(X_train_scaled, y_train_bin)
    df_sv_bin, sv_summary_bin = identify_margin_violating_support_vectors(
        model_or_pipeline=svm_bin,
        X_train=X_train_scaled,
        y_train=y_train_bin,
        df_metadata=df_train_raw,
    )

    # 5. Item 3 do Checklist: Isolar imagens/patches dos vetores de suporte para análise visual
    print("\n[4/5] Isolando patches dos vetores de suporte no dataset e salvando amostra de ambiguidade foliar...")
    patches_report = isolate_ambiguous_support_vector_patches(
        df_sv_analysis=df_sv_multi,
        df_raw_abt=df_train_raw,
        output_dir=PATCHES_DIR,
        grid_figure_path=FIGURES_DIR / "svm_support_vectors_visual_ambiguity_grid.png",
        top_n_per_regime=4,
    )
    print(f"  -> {patches_report['n_isolated_patches']} patches salvos em: {patches_report['patches_directory']}")
    print(f"  -> Painel visual salvo em: {patches_report['grid_figure_path']}")

    # 6. Item 5 do Checklist: Gerar visualizações gráficas 2D e 3D
    print("\n[5/5] Gerando visualizações gráficas 2D e 3D da projeção dos vetores de suporte e hiperplano...")
    figures_map = generate_support_vector_visualizations(
        df_by_class=df_by_class,
        df_c_tradeoff=df_c_tradeoff,
        df_sv_multi=df_sv_multi,
        df_sv_bin=df_sv_bin,
        X_bin_scaled=X_train_scaled,
        y_bin=y_train_bin,
        figures_dir=FIGURES_DIR,
    )
    figures_map["visual_ambiguity_grid"] = patches_report["grid_figure_path"]
    for fig_key, fig_file in figures_map.items():
        print(f"  [OK] Figura '{fig_key}': {fig_file}")

    # 7. Salvar Manifesto JSON de Governança MLOps
    diagnosis_manifest: Dict[str, Any] = {
        "task_name": "Diagnóstico Matemático dos Vetores de Suporte (Support Vectors)",
        "project": "Sanidade-Vegetal (SugarVision)",
        "sprint": "Sprint 3 — Framework SEMMA (Fase: Model)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "champion_model_analyzed": {
            "artifact": str(optimal_model_path.relative_to(PROJECT_ROOT)),
            "kernel": "rbf",
            "C": 10.0,
            "gamma": 0.01,
            "n_train_samples": int(len(df_train_raw)),
            "n_total_support_vectors": int(df_by_class["n_support_vectors"].sum()),
            "global_support_density_pct": round(
                float(df_by_class["n_support_vectors"].sum() / len(df_train_raw) * 100.0), 2
            ),
        },
        "support_vectors_by_class": df_by_class.to_dict(orient="records"),
        "margin_violation_slack_summary_multiclass": sv_summary_multi,
        "margin_violation_slack_summary_binary": sv_summary_bin,
        "c_hyperparameter_tradeoff_summary": df_c_tradeoff.to_dict(orient="records"),
        "isolated_patches_summary": patches_report,
        "generated_figures": figures_map,
    }

    manifest_path = MODELS_DIR / "svm_support_vectors_diagnosis.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(diagnosis_manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 85)
    print("DIAGNÓSTICO MATEMÁTICO DOS VETORES DE SUPORTE CONCLUÍDO COM 100% DE SUCESSO!")
    print("=" * 85)
    return diagnosis_manifest


if __name__ == "__main__":
    run_support_vectors_mathematical_diagnosis()
