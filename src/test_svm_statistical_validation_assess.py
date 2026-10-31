"""
Suíte de Testes Unitários (pytest) — Validação Estatística (Matrizes de Confusão, ROC-AUC, PR e Threshold Tuning)
Arquivo: src/test_svm_statistical_validation_assess.py
Projeto: Sanidade-Vegetal (SugarVision) — Sprint 4 (Fase Assess — Relato Vol. 4)

Cobertura Validada:
1. Cálculo da matriz de confusão absoluta e normalizada por classe (7 patologias e agrupamento 3 classes:
   Saudável, Ferrugem e Outras Patologias).
2. Cálculo das curvas ROC multiclasse One-vs-Rest (AUC Macro, Micro, Weighted e por classe).
3. Construção das curvas Precision-Recall focadas na classe crítica de Ferrugem (`RUST`).
4. Avaliação do comportamento dos limiares de decisão (`evaluate_threshold_tuning_sensitivity`).
5. Geração das figuras em alta resolução em `docs/figures/` e integridade dos artefatos do Relato Vol. 4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from src.svm_statistical_validation_assess import (
    CLASS_NAMES_MAP,
    DOCS_DIR,
    ENTREGAS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    RUST_CLASS_ID,
    compute_confusion_matrices,
    compute_multiclass_roc_ovr,
    compute_precision_recall_curves_focused,
    evaluate_threshold_tuning_sensitivity,
    generate_statistical_validation_figures,
    map_to_agronomic_3classes,
)


# ==============================================================================
# FIXTURES SINTÉTICAS CONTROLADAS PARA VALIDAÇÃO MATEMÁTICA
# ==============================================================================

@pytest.fixture(scope="module")
def synthetic_multiclass_predictions() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cria previsões e matriz de probabilidades sintética bem calibrada para 7 classes."""
    rng = np.random.default_rng(42)
    n_samples = 140
    y_true = np.repeat(np.arange(7), n_samples // 7)

    # Criar matriz de probabilidades estocástica com forte sinal na classe verdadeira
    logits = rng.normal(0.0, 0.6, size=(n_samples, 7))
    for i, c in enumerate(y_true):
        logits[i, c] += 2.6
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    y_proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    y_pred = np.argmax(y_proba, axis=1)

    return y_true, y_pred, y_proba


# ==============================================================================
# 1. TESTES DE MATRIZ DE CONFUSÃO ABSOLUTA E NORMALIZADA
# ==============================================================================

def test_compute_confusion_matrices_absolute_and_normalized(
    synthetic_multiclass_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """
    Valida que a matriz absoluta soma N, a matriz normalizada soma 1.0 por linha,
    e o mapeamento agronômico de 3 classes (Saudável, Ferrugem, Outras) é exato.
    """
    y_true, y_pred, _ = synthetic_multiclass_predictions
    res = compute_confusion_matrices(y_true, y_pred, class_names_map=CLASS_NAMES_MAP)

    cm_abs = res["cm_multiclass_abs"]
    cm_norm = res["cm_multiclass_norm"]
    assert cm_abs.shape == (7, 7)
    assert cm_norm.shape == (7, 7)
    assert int(cm_abs.sum()) == len(y_true)
    np.testing.assert_allclose(cm_norm.sum(axis=1), np.ones(7), atol=1e-5)

    # Validar agrupamento agronômico de 3 classes (0=Saudável, 1=Ferrugem, 2=Outras)
    cm3_abs = res["cm_agronomic_3class_abs"]
    cm3_norm = res["cm_agronomic_3class_norm"]
    assert cm3_abs.shape == (3, 3)
    assert int(cm3_abs.sum()) == len(y_true)
    np.testing.assert_allclose(cm3_norm.sum(axis=1), np.ones(3), atol=1e-5)

    # Verificar mapeamento explícito de IDs
    mapped = map_to_agronomic_3classes(np.array([1, 5, 0, 2, 3, 4, 6]))
    np.testing.assert_array_equal(mapped, np.array([0, 1, 2, 2, 2, 2, 2]))


# ==============================================================================
# 2. TESTES DE CURVAS ROC MULTICLASSE ONE-VS-REST (MACRO E MICRO AUC-ROC)
# ==============================================================================

def test_compute_multiclass_roc_ovr_macro_and_micro(
    synthetic_multiclass_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Garante que o cálculo ROC OvR retorna AUC Macro, Micro e por classe no intervalo [0, 1]."""
    y_true, _, y_proba = synthetic_multiclass_predictions
    roc_res = compute_multiclass_roc_ovr(y_true, y_proba, class_names_map=CLASS_NAMES_MAP)

    assert 0.5 < roc_res["roc_auc_macro"] <= 1.0
    assert 0.5 < roc_res["roc_auc_micro"] <= 1.0
    assert len(roc_res["auc_per_class"]) == 7

    for cls_name, auc_val in roc_res["auc_per_class"].items():
        assert 0.0 <= auc_val <= 1.0, f"AUC inválida para {cls_name}: {auc_val}"
        fpr = roc_res["curves"]["fpr_per_class"][cls_name]
        tpr = roc_res["curves"]["tpr_per_class"][cls_name]
        assert fpr[0] == 0.0 and fpr[-1] == 1.0
        assert tpr[0] == 0.0 and tpr[-1] == 1.0


def test_compute_multiclass_roc_ovr_rejects_mismatched_columns(
    synthetic_multiclass_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Valida que matriz de probabilidades com número incorreto de colunas dispara ValueError."""
    y_true, _, y_proba = synthetic_multiclass_predictions
    with pytest.raises(ValueError, match="Dimensão incompatível"):
        compute_multiclass_roc_ovr(y_true, y_proba[:, :4], class_names_map=CLASS_NAMES_MAP)


# ==============================================================================
# 3. TESTES DE CURVAS PRECISION-RECALL FOCADAS NA FERRUGEM (RUST)
# ==============================================================================

def test_compute_precision_recall_curves_focused_on_rust(
    synthetic_multiclass_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Valida a extração da curva PR e Average Precision focada na classe crítica de Ferrugem."""
    y_true, _, y_proba = synthetic_multiclass_predictions
    pr_res = compute_precision_recall_curves_focused(
        y_true, y_proba, target_class_id=RUST_CLASS_ID, class_names_map=CLASS_NAMES_MAP
    )

    assert pr_res["focus_class_name"] == "RUST"
    assert 0.0 < pr_res["focus_class_metrics"]["average_precision_ap"] <= 1.0
    assert 0.0 < pr_res["focus_class_metrics"]["auc_pr"] <= 1.0
    assert 0.0 < pr_res["ap_macro"] <= 1.0
    assert 0.0 < pr_res["ap_micro"] <= 1.0

    rust_curve = pr_res["per_class_pr"]["RUST"]
    assert len(rust_curve["precision_curve"]) == len(rust_curve["recall_curve"])


# ==============================================================================
# 4. TESTES DE THRESHOLD TUNING E CALIBRAÇÃO DE SENSIBILIDADE
# ==============================================================================

def test_evaluate_threshold_tuning_sensitivity_monotonicity(
    synthetic_multiclass_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """
    Comprova matematicamente que ao elevar o limiar tau de 0.01 a 0.99,
    a Sensibilidade (Recall) é não-crescente e a Especificidade é não-decrescente.
    """
    y_true, _, y_proba = synthetic_multiclass_predictions
    df_thresh, summary = evaluate_threshold_tuning_sensitivity(
        y_true, y_proba, target_class_id=RUST_CLASS_ID
    )

    assert len(df_thresh) == 99
    # Recall deve ser monotonicamente não-crescente com o aumento de tau
    recall_diffs = np.diff(df_thresh["sensitivity_recall"].values)
    assert np.all(recall_diffs <= 1e-9), "Sensibilidade deve ser não-crescente quando o limiar aumenta!"

    # Especificidade deve ser monotonicamente não-decrescente com o aumento de tau
    spec_diffs = np.diff(df_thresh["specificity"].values)
    assert np.all(spec_diffs >= -1e-9), "Especificidade deve ser não-decrescente quando o limiar aumenta!"

    # Verificar que o limiar de máximo F1 realmente atinge o maior F1 da tabela
    assert summary["optimal_f1_threshold"]["f1_score"] == pytest.approx(df_thresh["f1_score"].max())
    assert summary["optimal_f2_agronomic_threshold"]["f2_score_agronomic"] == pytest.approx(
        df_thresh["f2_score_agronomic"].max()
    )


def test_evaluate_threshold_tuning_rejects_out_of_bounds_threshold(
    synthetic_multiclass_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Garante lançamento de ValueError para limiares fora de (0.0, 1.0)."""
    y_true, _, y_proba = synthetic_multiclass_predictions
    with pytest.raises(ValueError, match="intervalo"):
        evaluate_threshold_tuning_sensitivity(
            y_true, y_proba, thresholds=np.array([0.0, 0.5, 1.0])
        )


# ==============================================================================
# 5. TESTES DE PLOTAGEM E INTEGRIDADE DOS ARTEFATOS REAIS DO RELATO VOL. 4
# ==============================================================================

def test_generate_statistical_validation_figures_in_tmp(
    tmp_path: Path,
    synthetic_multiclass_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Valida a geração sem erros de todas as 5 figuras estatísticas em diretório temporário."""
    y_true, y_pred, y_proba = synthetic_multiclass_predictions
    cm_res = compute_confusion_matrices(y_true, y_pred)
    roc_res = compute_multiclass_roc_ovr(y_true, y_proba)
    pr_res = compute_precision_recall_curves_focused(y_true, y_proba, target_class_id=RUST_CLASS_ID)
    df_t_rust, sum_rust = evaluate_threshold_tuning_sensitivity(y_true, y_proba, target_class_id=RUST_CLASS_ID)
    df_t_scald, sum_scald = evaluate_threshold_tuning_sensitivity(y_true, y_proba, target_class_id=2)

    figs = generate_statistical_validation_figures(
        cm_results=cm_res,
        roc_results=roc_res,
        pr_results=pr_res,
        df_thresh_rust=df_t_rust,
        thresh_summary_rust=sum_rust,
        df_thresh_scald=df_t_scald,
        thresh_summary_scald=sum_scald,
        figures_dir=tmp_path / "figs",
    )

    assert len(figs) == 5
    for fig_path_str in figs.values():
        p = Path(fig_path_str)
        assert p.exists() and p.stat().st_size > 20_000


def test_production_sprint4_vol4_artifacts_and_metrics() -> None:
    """
    Valida que a execução sobre o modelo SVM campeão gerou todas as figuras em `docs/figures/`,
    os CSVs e o manifesto JSON com AUC-ROC Macro > 0.91 e isolamento perfeito de Saudável e Ferrugem.
    """
    manifest_path = MODELS_DIR / "svm_statistical_validation_vol4.json"
    report_md = DOCS_DIR / "relatorio_validacao_estatistica_matrizes_roc_pr_vol4.md"
    entrega_md = ENTREGAS_DIR / "README.md"

    for artifact in [manifest_path, report_md, entrega_md]:
        assert artifact.exists(), f"Artefato obrigatório ausente: {artifact}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    val_m = data["validation_partition_metrics"]
    assert val_m["n_samples"] == 617
    assert val_m["roc_auc_macro"] >= 0.91
    assert val_m["roc_auc_micro"] >= 0.91
    assert val_m["rust_pr_metrics"]["average_precision_ap"] >= 0.99

    # Saudável (índice 0) e Ferrugem (índice 1) na matriz agronômica 3x3 devem ter 100% de Recall
    cm3_norm = np.array(val_m["cm_agronomic_3class_norm"])
    assert cm3_norm[0, 0] == pytest.approx(1.0, abs=1e-4)
    assert cm3_norm[1, 1] == pytest.approx(1.0, abs=1e-4)

    expected_figs = [
        FIGURES_DIR / "sprint4_confusion_matrices_multiclass_abs_norm.png",
        FIGURES_DIR / "sprint4_confusion_matrices_agronomic_3class.png",
        FIGURES_DIR / "sprint4_roc_curves_multiclass_ovr.png",
        FIGURES_DIR / "sprint4_pr_curve_rust_and_critical_classes.png",
        FIGURES_DIR / "sprint4_threshold_tuning_sensitivity_calibration.png",
    ]
    for fig_file in expected_figs:
        assert fig_file.exists() and fig_file.stat().st_size > 30_000
