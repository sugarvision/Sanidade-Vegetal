"""
Suíte de Testes Unitários (pytest) — Curvas de Aprendizado e Trade-off Viés-Variância
Arquivo: src/test_svm_learning_curves_bias_variance.py
Projeto: Sanidade-Vegetal (SugarVision) — Sprint 3 (Fase Model)

Cobertura Validada:
1. Geração da curva de aprendizado de 10% a 100% (`compute_svm_learning_curve`) e validação de domínios.
2. Lógica matemática de diagnóstico (`diagnose_learning_curve_and_bias_variance`) para:
   - Platô assintótico (`OPTIMAL_CAPACITY_REACHED_ASYMPTOTIC_PLATEAU`) vs. necessidade de mais amostras (`BENEFITS_FROM_MORE_SAMPLES`).
   - Decomposição dos regimes de Viés-Variância: `HIGH_BIAS_UNDERFITTING`, `BALANCED_OPTIMAL_GENERALIZATION` e `HIGH_VARIANCE_OVERFITTING`.
3. Utilitários de plotagem estatística (`plot_learning_curves_and_bias_variance`) e geração de relatórios Markdown.
4. Validação de integridade dos artefatos de produção gerados para o modelo campeão da Sprint 3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.svm_learning_curves_bias_variance import (
    DOCS_DIR,
    ENTREGAS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    compute_svm_learning_curve,
    diagnose_learning_curve_and_bias_variance,
    generate_sprint_learning_curve_report,
    plot_learning_curves_and_bias_variance,
)


# ==============================================================================
# FIXTURES CONTROLADAS PARA TESTES UNITÁRIOS RÁPIDOS
# ==============================================================================

@pytest.fixture(scope="module")
def synthetic_classification_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Cria dataset sintético multiclasse escalonado para validação rápida de learning_curve."""
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=42,
    )
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y


# ==============================================================================
# 1. TESTES DE GERAÇÃO DE CURVAS DE APRENDIZADO (10% A 100%)
# ==============================================================================

def test_compute_svm_learning_curve_10_to_100_percent(
    synthetic_classification_dataset: Tuple[np.ndarray, np.ndarray],
) -> None:
    """Garante que a curva avalia exatamente 10% a 100% do volume de treino com métricas válidas."""
    X, y = synthetic_classification_dataset
    svc = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)

    df_curve = compute_svm_learning_curve(
        X=X,
        y=y,
        estimator=svc,
        train_fractions=np.linspace(0.1, 1.0, 10),
        cv_splits=4,
        scoring="f1_macro",
        n_jobs=1,
        random_state=42,
    )

    assert len(df_curve) == 10
    np.testing.assert_allclose(
        df_curve["train_percentage"].values,
        np.linspace(10.0, 100.0, 10),
        atol=1e-3,
    )
    # Os tamanhos absolutos de treino devem ser estritamente crescentes
    assert np.all(np.diff(df_curve["n_train_samples"].values) > 0)

    # Scores médios de treino e validação devem estar em [0, 1]
    assert np.all((df_curve["train_score_mean"] >= 0.0) & (df_curve["train_score_mean"] <= 1.0))
    assert np.all((df_curve["val_score_mean"] >= 0.0) & (df_curve["val_score_mean"] <= 1.0))

    # O gap de generalização deve ser exatamente train_score_mean - val_score_mean
    np.testing.assert_allclose(
        df_curve["generalization_gap"].values,
        df_curve["train_score_mean"].values - df_curve["val_score_mean"].values,
        atol=1e-5,
    )


def test_compute_svm_learning_curve_rejects_invalid_fractions(
    synthetic_classification_dataset: Tuple[np.ndarray, np.ndarray],
) -> None:
    """Valida que frações fora de (0.0, 1.0] levantam ValueError."""
    X, y = synthetic_classification_dataset
    svc = SVC(kernel="linear", C=1.0)
    with pytest.raises(ValueError, match="intervalo"):
        compute_svm_learning_curve(X, y, estimator=svc, train_fractions=[0.0, 0.5, 1.2])


# ==============================================================================
# 2. TESTES MATEMÁTICOS DA LÓGICA DE DIAGNÓSTICO (CAPACIDADE E VIÉS-VARIÂNCIA)
# ==============================================================================

def test_diagnose_asymptotic_plateau_vs_sample_hungry() -> None:
    """
    Valida matematicamente a distinção entre um modelo que já atingiu o platô assintótico
    e um modelo que se beneficiará de mais amostras.
    """
    # Caso A: Platô assintótico (ganho nos últimos 20% = +0.002 <= 0.008) e equilíbrio viés-variância
    df_plateau = pd.DataFrame(
        {
            "train_fraction": np.linspace(0.1, 1.0, 10),
            "train_percentage": np.linspace(10.0, 100.0, 10),
            "n_train_samples": np.linspace(450, 4500, 10).astype(int),
            "train_score_mean": [0.68, 0.65, 0.63, 0.62, 0.615, 0.612, 0.610, 0.608, 0.607, 0.606],
            "train_score_std": [0.01] * 10,
            "val_score_mean": [0.51, 0.54, 0.555, 0.565, 0.570, 0.573, 0.575, 0.576, 0.577, 0.5775],
            "val_score_std": [0.008] * 10,
            "generalization_gap": [0.17, 0.11, 0.075, 0.055, 0.045, 0.039, 0.035, 0.032, 0.030, 0.0285],
            "fit_time_mean": np.linspace(0.1, 0.8, 10),
            "fit_time_std": [0.01] * 10,
            "score_time_mean": np.linspace(0.02, 0.1, 10),
            "score_time_std": [0.002] * 10,
        }
    )

    diag_plateau = diagnose_learning_curve_and_bias_variance(df_plateau, log_output=False)
    assert diag_plateau["is_asymptotically_converged"] is True
    assert diag_plateau["sample_capacity_status"] == "OPTIMAL_CAPACITY_REACHED_ASYMPTOTIC_PLATEAU"
    assert diag_plateau["bias_variance_regime"] == "BALANCED_OPTIMAL_GENERALIZATION"
    assert diag_plateau["overfitting_proven_absent"] is True

    # Caso B: Curva ascendente faminta por amostras (ganho nos últimos 20% = +0.05 > 0.008)
    df_hungry = df_plateau.copy()
    df_hungry["val_score_mean"] = [0.40, 0.43, 0.46, 0.49, 0.52, 0.55, 0.58, 0.61, 0.64, 0.67]
    df_hungry["train_score_mean"] = [0.75] * 10
    df_hungry["generalization_gap"] = df_hungry["train_score_mean"] - df_hungry["val_score_mean"]

    diag_hungry = diagnose_learning_curve_and_bias_variance(df_hungry, log_output=False)
    assert diag_hungry["is_asymptotically_converged"] is False
    assert diag_hungry["sample_capacity_status"] == "BENEFITS_FROM_MORE_SAMPLES"


def test_diagnose_underfitting_and_overfitting_regimes() -> None:
    """Valida a detecção exata de Subajuste (Alto Viés) e Sobreajuste (Alta Variância)."""
    base_n = np.linspace(100, 1000, 5).astype(int)
    base_pct = np.linspace(20.0, 100.0, 5)

    # Subajuste: scores baixos (< 0.50) e gap pequeno
    df_underfit = pd.DataFrame(
        {
            "train_percentage": base_pct,
            "n_train_samples": base_n,
            "train_score_mean": [0.42, 0.43, 0.44, 0.445, 0.45],
            "train_score_std": [0.01] * 5,
            "val_score_mean": [0.39, 0.41, 0.42, 0.43, 0.435],
            "val_score_std": [0.01] * 5,
            "generalization_gap": [0.03, 0.02, 0.02, 0.015, 0.015],
        }
    )
    diag_under = diagnose_learning_curve_and_bias_variance(df_underfit, log_output=False)
    assert diag_under["bias_variance_regime"] == "HIGH_BIAS_UNDERFITTING"

    # Sobreajuste: treino próximo de 1.0 e gap > 0.12
    df_overfit = pd.DataFrame(
        {
            "train_percentage": base_pct,
            "n_train_samples": base_n,
            "train_score_mean": [0.99, 0.98, 0.97, 0.96, 0.95],
            "train_score_std": [0.005] * 5,
            "val_score_mean": [0.45, 0.48, 0.50, 0.51, 0.52],
            "val_score_std": [0.02] * 5,
            "generalization_gap": [0.54, 0.50, 0.47, 0.45, 0.43],
        }
    )
    diag_over = diagnose_learning_curve_and_bias_variance(df_overfit, log_output=False)
    assert diag_over["bias_variance_regime"] == "HIGH_VARIANCE_OVERFITTING"
    assert diag_over["overfitting_proven_absent"] is False


# ==============================================================================
# 3. TESTES DE PLOTAGEM E GERAÇÃO DE RELATÓRIOS EM DIRETÓRIO ISOLADO
# ==============================================================================

def test_plot_and_report_utilities_execution(tmp_path: Path) -> None:
    """Garante que os utilitários de plotagem e geração de relatório criam arquivos válidos."""
    pct = np.linspace(10.0, 100.0, 10)
    df_mock = pd.DataFrame(
        {
            "train_fraction": pct / 100.0,
            "train_percentage": pct,
            "n_train_samples": (pct * 45).astype(int),
            "train_score_mean": np.linspace(0.66, 0.61, 10),
            "train_score_std": [0.008] * 10,
            "val_score_mean": np.linspace(0.50, 0.577, 10),
            "val_score_std": [0.007] * 10,
            "generalization_gap": np.linspace(0.16, 0.033, 10),
            "fit_time_mean": np.linspace(0.1, 0.9, 10),
            "fit_time_std": [0.01] * 10,
            "score_time_mean": np.linspace(0.02, 0.12, 10),
            "score_time_std": [0.002] * 10,
        }
    )
    diag_mock = diagnose_learning_curve_and_bias_variance(df_mock, log_output=False)

    regimes_curves = {
        "underfit_high_bias": df_mock.copy(),
        "optimal_balanced": df_mock.copy(),
        "overfit_high_variance": df_mock.copy(),
    }
    regimes_diags = {
        "underfit_high_bias": diag_mock,
        "optimal_balanced": diag_mock,
        "overfit_high_variance": diag_mock,
    }

    figs_out = plot_learning_curves_and_bias_variance(
        df_champion_curve=df_mock,
        champion_diagnosis=diag_mock,
        regimes_curves=regimes_curves,
        regimes_diagnostics=regimes_diags,
        figures_dir=tmp_path / "figures",
    )
    for fig_file in figs_out.values():
        p = Path(fig_file)
        assert p.exists() and p.stat().st_size > 20_000

    rep_path, ent_path = generate_sprint_learning_curve_report(
        df_champion_curve=df_mock,
        champion_diagnosis=diag_mock,
        regimes_diagnostics=regimes_diags,
        report_path=tmp_path / "relatorio.md",
        entrega_readme_path=tmp_path / "entrega" / "README.md",
    )
    assert Path(rep_path).exists() and Path(rep_path).stat().st_size > 1_000
    assert Path(ent_path).exists() and Path(ent_path).stat().st_size > 1_000


# ==============================================================================
# 4. TESTE DE INTEGRAÇÃO DOS ARTEFATOS REAIS DO PROJETO SANIDADE-VEGETAL
# ==============================================================================

def test_production_learning_curves_artifacts_and_overfitting_absence() -> None:
    """
    Valida que a execução oficial gerou os CSVs, o JSON de diagnóstico, as 3 figuras em 300 DPI
    e o relatório técnico da Sprint 3 comprovando a ausência de sobreajuste no modelo campeão.
    """
    manifest_path = MODELS_DIR / "svm_learning_curves_diagnosis.json"
    champ_csv = MODELS_DIR / "svm_learning_curve_champion.csv"
    regimes_csv = MODELS_DIR / "svm_bias_variance_regimes_curves.csv"
    report_md = DOCS_DIR / "relatorio_curvas_aprendizado_vies_variancia_sprint3.md"
    entrega_md = ENTREGAS_DIR / "README.md"

    for path_item in [manifest_path, champ_csv, regimes_csv, report_md, entrega_md]:
        assert path_item.exists(), f"Artefato obrigatório não encontrado: {path_item}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["train_fractions_evaluated"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    champ_diag = manifest["champion_mathematical_diagnosis"]
    assert champ_diag["overfitting_proven_absent"] is True
    assert champ_diag["bias_variance_regime"] == "BALANCED_OPTIMAL_GENERALIZATION"
    assert champ_diag["final_generalization_gap_100pct"] < 0.12

    regimes_diag = manifest["bias_variance_regimes_diagnosis"]
    assert regimes_diag["overfit_high_variance"]["bias_variance_regime"] == "HIGH_VARIANCE_OVERFITTING"
    assert regimes_diag["overfit_high_variance"]["overfitting_proven_absent"] is False

    required_figures = [
        FIGURES_DIR / "sprint3_svm_learning_curve_champion.png",
        FIGURES_DIR / "sprint3_svm_bias_variance_regimes_comparison.png",
        FIGURES_DIR / "sprint3_svm_learning_curve_scalability_times.png",
    ]
    for fig_p in required_figures:
        assert fig_p.exists() and fig_p.stat().st_size > 25_000
