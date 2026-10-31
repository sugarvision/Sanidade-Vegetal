"""
Suíte de Testes Unitários e Matemáticos — Diagnóstico dos Vetores de Suporte (Support Vectors)
Arquivo: src/test_svm_support_vectors_diagnosis.py
Projeto: Sanidade-Vegetal (SugarVision) — Sprint 3 (Fase Model)

Cobertura de Testes (pytest):
1. Extração de contagem e proporção de vetores de suporte por classe (`n_support_` e `support_vectors_`).
2. Consistência algébrica entre `sum(n_support_)`, `support_vectors_.shape[0]` e `len(support_)`.
3. Relação monotônica estrutural entre o hiperparâmetro C (margens suaves vs rígidas) e os vetores de suporte.
4. Identificação matemática exata de violações de margem (variáveis de folga xi_i > 0 e erros xi_i > 1).
5. Isolamento de patches de vetores de suporte e geração do catálogo de ambiguidade foliar.
6. Geração e integridade física das visualizações gráficas 2D e 3D em relação ao hiperplano separador.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.svm_support_vectors_diagnosis import (
    ENTREGAS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    PATCHES_DIR,
    evaluate_c_hyperparameter_support_tradeoff,
    extract_support_vectors_by_class,
    generate_support_vector_visualizations,
    identify_margin_violating_support_vectors,
    isolate_ambiguous_support_vector_patches,
)


# ==============================================================================
# FIXTURES SINTÉTICAS E REAIS PARA VALIDAÇÃO MATEMÁTICA
# ==============================================================================

@pytest.fixture(scope="module")
def synthetic_binary_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gera dataset binário controlado com sobreposição parcial de margem para testes KKT."""
    X, y = make_classification(
        n_samples=240,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=1,
        class_sep=1.15,
        flip_y=0.05,
        random_state=42,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:180])
    y_train = y[:180]
    X_val = scaler.transform(X[180:])
    y_val = y[180:]
    return X_train, y_train, X_val, y_val


@pytest.fixture(scope="module")
def synthetic_multiclass_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """Gera dataset multiclasse com metadados simulando a ABT de Sanidade Vegetal."""
    X, y = make_classification(
        n_samples=210,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.1,
        random_state=42,
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_features = pd.DataFrame(X_scaled, columns=[f"feat_{i}" for i in range(X.shape[1])])
    return df_features, y


# ==============================================================================
# 1. TESTES DE CONTAGEM E PROPORÇÃO POR CLASSE (n_support_ e support_vectors_)
# ==============================================================================

def test_extract_support_vectors_by_class_algebraic_consistency(
    synthetic_multiclass_data: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Valida que a soma de n_support_ coincide com support_vectors_ e as proporções somam 1.0."""
    X_df, y = synthetic_multiclass_data
    pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, random_state=42))])
    pipe.fit(X_df, y)

    svc = pipe.named_steps["svm"]
    df_cls = extract_support_vectors_by_class(
        model_or_pipeline=pipe,
        X_train=X_df,
        y_train=y,
        class_names_map={0: "CLASSE_A", 1: "CLASSE_B", 2: "CLASSE_C"},
    )

    assert len(df_cls) == 3
    assert int(df_cls["n_support_vectors"].sum()) == int(svc.support_vectors_.shape[0])
    assert int(df_cls["n_support_vectors"].sum()) == int(svc.n_support_.sum())
    assert int(df_cls["n_train_samples"].sum()) == len(X_df)

    # Cada proporção intraclasse deve estar estritamente em [0, 1]
    assert np.all((df_cls["class_sv_proportion"] >= 0.0) & (df_cls["class_sv_proportion"] <= 1.0))

    # A soma das participações globais (global_sv_share) deve ser 1.0
    np.testing.assert_allclose(df_cls["global_sv_share"].sum(), 1.0, atol=1e-4)

    # Verificar que a norma L2 média dos vetores de suporte de cada classe é positiva
    assert np.all(df_cls["sv_mean_l2_norm"] > 0.0)


def test_extract_support_vectors_rejects_unfitted_estimator(
    synthetic_binary_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Garante lançamento de ValueError ao tentar extrair vetores de um SVC não ajustado."""
    X_train, y_train, _, _ = synthetic_binary_data
    unfitted_svc = SVC(kernel="rbf", C=1.0)
    with pytest.raises(ValueError, match="não está ajustado"):
        extract_support_vectors_by_class(unfitted_svc, X_train, y_train)


# ==============================================================================
# 2. TESTES DO TRADE-OFF DO HIPERPARÂMETRO C (MARGEM SUAVE VS MARGEM RÍGIDA)
# ==============================================================================

def test_evaluate_c_hyperparameter_support_tradeoff_soft_vs_hard_margin(
    synthetic_binary_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """
    Comprova matematicamente que margens suaves (C pequeno) recrutam mais vetores de suporte
    e produzem maior margem geométrica 2/||w||_2 do que margens rígidas (C grande).
    """
    X_train, y_train, X_val, y_val = synthetic_binary_data
    c_grid = [0.01, 0.1, 1.0, 10.0, 100.0]

    df_tradeoff = evaluate_c_hyperparameter_support_tradeoff(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        c_values=c_grid,
        kernels=["linear", "rbf"],
        gamma="scale",
    )

    assert len(df_tradeoff) == len(c_grid) * 2
    assert set(df_tradeoff["kernel"].unique()) == {"linear", "rbf"}

    # No kernel linear, quando C aumenta de 0.01 (margem suave) para 100.0 (margem rígida):
    # 1. O número de vetores de suporte em C=0.01 deve ser >= ao de C=100.0
    # 2. A margem geométrica 2/||w|| em C=0.01 deve ser > à de C=100.0
    lin_df = df_tradeoff[df_tradeoff["kernel"] == "linear"].sort_values("C")
    sv_soft = int(lin_df.iloc[0]["n_support_vectors"])
    sv_hard = int(lin_df.iloc[-1]["n_support_vectors"])
    margin_soft = float(lin_df.iloc[0]["geometric_margin"])
    margin_hard = float(lin_df.iloc[-1]["geometric_margin"])

    assert sv_soft >= sv_hard, f"Margem suave (SV={sv_soft}) deveria ter >= SVs que margem rígida (SV={sv_hard})"
    assert margin_soft > margin_hard, (
        f"Margem geométrica suave ({margin_soft}) deveria ser maior que margem rígida ({margin_hard})"
    )


def test_evaluate_c_hyperparameter_rejects_non_positive_c(
    synthetic_binary_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Garante que valores inválidos de C <= 0 disparam ValueError."""
    X_train, y_train, X_val, y_val = synthetic_binary_data
    with pytest.raises(ValueError, match="estritamente positivo"):
        evaluate_c_hyperparameter_support_tradeoff(
            X_train, y_train, X_val, y_val, c_values=[0.0, -1.0], kernels=["linear"]
        )


# ==============================================================================
# 3. TESTES MATEMÁTICOS DE VIOLAÇÃO DE MARGEM E VARIÁVEIS DE FOLGA (xi_i > 0)
# ==============================================================================

def test_identify_margin_violating_support_vectors_kkt_conditions(
    synthetic_binary_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """
    Valida a identidade algébrica das variáveis de folga xi_i = max(0, 1 - y_i * f(x_i))
    e o particionamento exato nos 3 regimes KKT.
    """
    X_train, y_train, _, _ = synthetic_binary_data
    svc = SVC(kernel="linear", C=0.5, probability=True, random_state=42)
    svc.fit(X_train, y_train)

    df_sv, summary = identify_margin_violating_support_vectors(
        model_or_pipeline=svc,
        X_train=X_train,
        y_train=y_train,
        tolerance=1e-3,
    )

    assert len(df_sv) == int(svc.n_support_.sum())
    assert summary["n_total_support_vectors"] == len(df_sv)

    # A soma dos 3 regimes KKT deve ser igual ao total de vetores de suporte
    total_partitioned = (
        summary["n_on_margin_free_xi_eq_0"]
        + summary["n_margin_violations_correct_side"]
        + summary["n_misclassified_xi_gt_1"]
    )
    assert total_partitioned == summary["n_total_support_vectors"]

    # Verificar que xi_i = max(0, 1 - functional_margin) para todas as linhas
    expected_xi = np.maximum(0.0, 1.0 - df_sv["functional_margin"].values)
    np.testing.assert_allclose(df_sv["slack_xi"].values, expected_xi, atol=1e-5)

    # Verificar que toda instância marcada como is_margin_violation possui slack_xi > 1e-3
    viol_rows = df_sv[df_sv["is_margin_violation"]]
    assert len(viol_rows) > 0
    assert np.all(viol_rows["slack_xi"].values > 1e-3)

    # Verificar que toda instância com slack_xi > 1.001 é classificada como MARGIN_VIOLATION_MISCLASSIFIED
    severe_rows = df_sv[df_sv["slack_xi"] > 1.001]
    if not severe_rows.empty:
        assert np.all(severe_rows["kkt_regime"] == "MARGIN_VIOLATION_MISCLASSIFIED")


# ==============================================================================
# 4. TESTES DE ISOLAMENTO DE PATCHES E VISUALIZAÇÕES 2D/3D
# ==============================================================================

def test_isolate_ambiguous_support_vector_patches_creates_files(tmp_path: Path) -> None:
    """Valida o isolamento dos índices de suporte e geração dos patches e grid visual."""
    X, y = make_classification(
        n_samples=90,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    svc = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    svc.fit(X, y)

    raw_abt = pd.DataFrame(
        {
            "sample_id": [f"id_{i}" for i in range(90)],
            "filename": [f"leaf_{i}.jpg" for i in range(90)],
            "relative_path": [f"roboflow_sugarcane/train/leaf_{i}.jpg" for i in range(90)],
            "mean_hue": np.linspace(45.0, 85.0, 90),
            "std_saturation": np.linspace(0.08, 0.25, 90),
            "exg_index": np.linspace(15.0, 55.0, 90),
            "indice_clorose_necrose": np.linspace(1.5, 6.0, 90),
            "glcm_contrast": np.linspace(8.0, 35.0, 90),
            "indice_rugosidade_pustula": np.linspace(5.0, 22.0, 90),
        }
    )

    df_sv, _ = identify_margin_violating_support_vectors(svc, X, y, df_metadata=raw_abt)
    patches_out = tmp_path / "patches"
    grid_out = tmp_path / "ambiguity_grid.png"

    report = isolate_ambiguous_support_vector_patches(
        df_sv_analysis=df_sv,
        df_raw_abt=raw_abt,
        output_dir=patches_out,
        grid_figure_path=grid_out,
        top_n_per_regime=2,
    )

    assert report["n_isolated_patches"] > 0
    assert grid_out.exists() and grid_out.stat().st_size > 10_000
    for patch_file in report["saved_patch_files"]:
        p = Path(patch_file)
        assert p.exists() and p.stat().st_size > 500


# ==============================================================================
# 5. TESTES DE INTEGRAÇÃO DOS ARTEFATOS REAIS DO PROJETO SANIDADE-VEGETAL
# ==============================================================================

def test_production_diagnosis_artifacts_and_metrics_integrity() -> None:
    """
    Valida que todos os artefatos reais do diagnóstico matemático da Sprint 3 foram gerados
    e atendem rigorosamente ao checklist da tarefa sobre o modelo campeão do projeto.
    """
    manifest_path = MODELS_DIR / "svm_support_vectors_diagnosis.json"
    by_class_csv = MODELS_DIR / "svm_support_vectors_by_class.csv"
    tradeoff_csv = MODELS_DIR / "svm_c_vs_support_vectors_tradeoff.csv"
    slack_csv = MODELS_DIR / "svm_margin_violations_slack_analysis.csv"
    catalog_csv = MODELS_DIR / "svm_isolated_ambiguous_patches_catalog.csv"

    for artifact in [manifest_path, by_class_csv, tradeoff_csv, slack_csv, catalog_csv]:
        assert artifact.exists(), f"Artefato obrigatório ausente: {artifact}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Validar dados do modelo campeão (5574 amostras de treino, 3447 vetores de suporte)
    champ = manifest["champion_model_analyzed"]
    assert champ["n_train_samples"] == 5574
    assert champ["n_total_support_vectors"] == 3447

    df_by_class = pd.read_csv(by_class_csv)
    assert len(df_by_class) == 7
    assert int(df_by_class["n_support_vectors"].sum()) == 3447
    # HEALTHY e RUST devem apresentar baixíssima densidade de suporte (< 5%), enquanto MOSAIC/YELLOW LEAF > 90%
    healthy_pct = float(df_by_class.loc[df_by_class["class_name"] == "HEALTHY", "class_sv_percentage"].iloc[0])
    rust_pct = float(df_by_class.loc[df_by_class["class_name"] == "RUST", "class_sv_percentage"].iloc[0])
    mosaic_pct = float(df_by_class.loc[df_by_class["class_name"] == "MOSAIC", "class_sv_percentage"].iloc[0])
    assert healthy_pct < 5.0
    assert rust_pct < 5.0
    assert mosaic_pct >= 95.0

    # Validar figuras 2D, 3D e painel visual
    required_figures = [
        FIGURES_DIR / "svm_support_vectors_density_by_class.png",
        FIGURES_DIR / "svm_c_tradeoff_soft_vs_hard_margin.png",
        FIGURES_DIR / "svm_slack_variables_margin_violations.png",
        FIGURES_DIR / "svm_support_vectors_projection_2d.png",
        FIGURES_DIR / "svm_support_vectors_projection_3d.png",
        FIGURES_DIR / "svm_support_vectors_visual_ambiguity_grid.png",
    ]
    for fig_path in required_figures:
        assert fig_path.exists(), f"Figura obrigatória não encontrada: {fig_path}"
        assert fig_path.stat().st_size > 25_000, f"Figura {fig_path.name} corrompida ou vazia"
