#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes Unitários — Auditoria de Erros Críticos e Impacto Agronômico/Econômico (Sprint 4)
=========================================================================================

Valida matematicamente e computacionalmente o módulo `src/svm_agronomic_error_audit.py`:
  1. Parâmetros econômico-agronômicos da cana-de-açúcar (`SugarcaneEconomicParameters`).
  2. Isolamento de instâncias classificadas incorretamente (`isolate_misclassified_test_instances`).
  3. Quantificação do risco agronômico de Falsos Negativos (`quantify_false_negative_agronomic_risk`).
  4. Quantificação do impacto econômico de Falsos Positivos (`quantify_false_positive_economic_impact`).
  5. Matriz de Custo/Perda Agronômica 7x7 (`build_agronomic_cost_matrix`) e avaliação de perda
     esperada + Regra de Decisão de Risco Mínimo de Bayes (`evaluate_expected_agronomic_loss`).
  6. Geração de figuras diagnósticas, relatório automatizado em Markdown e artefatos de produção.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.svm_agronomic_error_audit import (
    HEALTHY_CLASS_ID,
    RED_ROT_CLASS_ID,
    RUST_CLASS_ID,
    SugarcaneEconomicParameters,
    build_agronomic_cost_matrix,
    evaluate_expected_agronomic_loss,
    generate_agronomic_audit_figures,
    generate_automated_agronomic_audit_report,
    isolate_misclassified_test_instances,
    quantify_false_negative_agronomic_risk,
    quantify_false_positive_economic_impact,
)


@pytest.fixture
def synthetic_predictions_df() -> pd.DataFrame:
    """Cria um DataFrame sintético de predições com casos conhecidos de TP, TN, FP e FN."""
    rows = [
        # 1. RUST correto (TP de Ferrugem)
        {
            "sample_id": "IMG_001",
            "split_partition": "test",
            "target_multiclass": 5,
            "y_pred": 5,
            "y_proba_grassy_shoot": 0.01,
            "y_proba_healthy": 0.01,
            "y_proba_leaf_scald": 0.01,
            "y_proba_mosaic": 0.01,
            "y_proba_red_rot": 0.01,
            "y_proba_rust": 0.94,
            "y_proba_yellow_leaf": 0.01,
        },
        # 2. RUST omitido como HEALTHY (FN Crítico de Ferrugem)
        {
            "sample_id": "IMG_002",
            "split_partition": "test",
            "target_multiclass": 5,
            "y_pred": 1,
            "y_proba_grassy_shoot": 0.02,
            "y_proba_healthy": 0.55,
            "y_proba_leaf_scald": 0.02,
            "y_proba_mosaic": 0.02,
            "y_proba_red_rot": 0.02,
            "y_proba_rust": 0.35,
            "y_proba_yellow_leaf": 0.02,
        },
        # 3. HEALTHY classificado como RUST (FP de Fungicida Químico)
        {
            "sample_id": "IMG_003",
            "split_partition": "test",
            "target_multiclass": 1,
            "y_pred": 5,
            "y_proba_grassy_shoot": 0.02,
            "y_proba_healthy": 0.30,
            "y_proba_leaf_scald": 0.02,
            "y_proba_mosaic": 0.02,
            "y_proba_red_rot": 0.02,
            "y_proba_rust": 0.60,
            "y_proba_yellow_leaf": 0.02,
        },
        # 4. RED ROT classificado como YELLOW LEAF (Omissão de Podridão Vermelha)
        {
            "sample_id": "IMG_004",
            "split_partition": "test",
            "target_multiclass": 4,
            "y_pred": 6,
            "y_proba_grassy_shoot": 0.05,
            "y_proba_healthy": 0.05,
            "y_proba_leaf_scald": 0.05,
            "y_proba_mosaic": 0.05,
            "y_proba_red_rot": 0.30,
            "y_proba_rust": 0.05,
            "y_proba_yellow_leaf": 0.45,
        },
        # 5. MOSAIC classificado como YELLOW LEAF (Erro intra-patológico secundário)
        {
            "sample_id": "IMG_005",
            "split_partition": "test",
            "target_multiclass": 3,
            "y_pred": 6,
            "y_proba_grassy_shoot": 0.05,
            "y_proba_healthy": 0.05,
            "y_proba_leaf_scald": 0.05,
            "y_proba_mosaic": 0.35,
            "y_proba_red_rot": 0.05,
            "y_proba_rust": 0.05,
            "y_proba_yellow_leaf": 0.40,
        },
        # 6. HEALTHY correto (TN de Ferrugem)
        {
            "sample_id": "IMG_006",
            "split_partition": "test",
            "target_multiclass": 1,
            "y_pred": 1,
            "y_proba_grassy_shoot": 0.01,
            "y_proba_healthy": 0.94,
            "y_proba_leaf_scald": 0.01,
            "y_proba_mosaic": 0.01,
            "y_proba_red_rot": 0.01,
            "y_proba_rust": 0.01,
            "y_proba_yellow_leaf": 0.01,
        },
        # 7. Instância da partição 'valid' que não deve contaminar o filtro 'test'
        {
            "sample_id": "IMG_007",
            "split_partition": "valid",
            "target_multiclass": 5,
            "y_pred": 1,
            "y_proba_grassy_shoot": 0.05,
            "y_proba_healthy": 0.80,
            "y_proba_leaf_scald": 0.03,
            "y_proba_mosaic": 0.03,
            "y_proba_red_rot": 0.03,
            "y_proba_rust": 0.03,
            "y_proba_yellow_leaf": 0.03,
        },
    ]
    return pd.DataFrame(rows)


def test_sugarcane_economic_parameters_invariants() -> None:
    """Verifica a consistência matemática dos parâmetros econômicos da cana-de-açúcar."""
    params = SugarcaneEconomicParameters()
    assert params.gross_revenue_per_ha_brl == pytest.approx(85.0 * 145.0, rel=1e-4)
    assert params.unit_cost_fp_fungicide_brl == pytest.approx(285.0 + 145.0 + 55.0, rel=1e-4)
    assert params.unit_cost_fn_rust_brl > params.unit_cost_fp_fungicide_brl
    assert params.fn_to_fp_asymmetry_ratio > 20.0


def test_isolate_misclassified_test_instances(
    synthetic_predictions_df: pd.DataFrame,
) -> None:
    """Garante que apenas as instâncias incorretas da partição solicitada sejam isoladas."""
    df_errors, df_transitions = isolate_misclassified_test_instances(
        predictions_df=synthetic_predictions_df,
        partition="test",
    )

    # Na partição 'test' há 6 amostras totais, sendo 4 incorretas (IMG_002, IMG_003, IMG_004, IMG_005)
    assert len(df_errors) == 4
    assert set(df_errors["sample_id"]) == {"IMG_002", "IMG_003", "IMG_004", "IMG_005"}
    assert "IMG_007" not in set(df_errors["sample_id"])

    # Verifica colunas enriquecidas
    expected_cols = {
        "true_class_name",
        "pred_class_name",
        "error_transition_pair",
        "agronomic_error_category",
        "true_class_proba",
        "pred_class_proba",
        "proba_confidence_gap",
    }
    assert expected_cols.issubset(set(df_errors.columns))
    assert (df_errors["proba_confidence_gap"] > 0).all()
    assert int(df_transitions["error_count"].sum()) == 4


def test_quantify_false_negative_agronomic_risk() -> None:
    """Testa o cálculo do risco agronômico de Falso Negativo para Ferrugem e Podridão Vermelha."""
    params = SugarcaneEconomicParameters()
    # 2 amostras de RUST (1 TP, 1 FN) e 1 amostra de RED ROT omitida como YELLOW LEAF (6)
    y_true = np.array([RUST_CLASS_ID, RUST_CLASS_ID, RED_ROT_CLASS_ID, HEALTHY_CLASS_ID])
    y_pred = np.array([RUST_CLASS_ID, HEALTHY_CLASS_ID, 6, HEALTHY_CLASS_ID])

    res = quantify_false_negative_agronomic_risk(y_true, y_pred, params=params)
    assert res["n_rust_foci_evaluated"] == 2
    assert res["rust_true_positives"] == 1
    assert res["rust_false_negatives"] == 1
    assert res["rust_false_negative_rate_pct"] == pytest.approx(50.0)
    assert res["observed_rust_fn_loss_brl"] == pytest.approx(params.unit_cost_fn_rust_brl, rel=1e-3)
    assert res["prevented_rust_catastrophe_value_brl"] == pytest.approx(
        params.unit_cost_fn_rust_brl, rel=1e-3
    )
    assert res["red_rot_false_negatives"] == 1
    assert res["observed_red_rot_fn_loss_brl"] == pytest.approx(
        params.fungal_red_rot_omission_loss_brl, rel=1e-3
    )
    assert len(res["epidemiological_stress_scenarios"]) == 6


def test_quantify_false_positive_economic_impact() -> None:
    """Testa o cálculo de desperdício de fungicida por Falso Positivo e economia frente ao manejo cego."""
    params = SugarcaneEconomicParameters()
    # 1 HEALTHY predito como RUST (FP de fungicida), 1 MOSAIC (3) predito como YELLOW LEAF (6)
    y_true = np.array([HEALTHY_CLASS_ID, 3, RUST_CLASS_ID])
    y_pred = np.array([RUST_CLASS_ID, 6, RUST_CLASS_ID])

    res = quantify_false_positive_economic_impact(y_true, y_pred, params=params)
    assert res["rust_false_positives"] == 1
    assert res["healthy_false_positives_as_diseased"] == 1
    assert res["unnecessary_fungicide_cost_brl"] == pytest.approx(
        params.unit_cost_fp_fungicide_brl, rel=1e-3
    )
    assert res["cross_pathology_mismatches_count"] == 1
    assert res["cross_pathology_management_cost_brl"] == pytest.approx(
        params.bacterial_viral_mismatch_cost_brl, rel=1e-3
    )
    # 3 ha totais vs 2 preditos como RUST -> economia de 1 ha de fungicida
    assert res["chemical_savings_by_svm_triage_brl"] == pytest.approx(
        params.unit_cost_fp_fungicide_brl, rel=1e-3
    )


def test_build_agronomic_cost_matrix_and_bayes_decision() -> None:
    """Valida a Matriz de Custo 7x7 e a Regra de Decisão de Risco Mínimo de Bayes."""
    params = SugarcaneEconomicParameters()
    L, df_cost = build_agronomic_cost_matrix(params)

    assert L.shape == (7, 7)
    assert df_cost.shape == (7, 7)
    # Diagonal principal (acertos) deve ter custo estritamente zero
    assert np.allclose(np.diag(L), 0.0)
    # Assimetria: FN de Ferrugem (L[5, 1]) >> FP de Fungicida (L[1, 5])
    assert L[RUST_CLASS_ID, HEALTHY_CLASS_ID] == pytest.approx(params.unit_cost_fn_rust_brl)
    assert L[HEALTHY_CLASS_ID, RUST_CLASS_ID] == pytest.approx(params.unit_cost_fp_fungicide_brl)
    assert L[RUST_CLASS_ID, HEALTHY_CLASS_ID] > 20.0 * L[HEALTHY_CLASS_ID, RUST_CLASS_ID]

    # Avalia caso onde argmax erra uma amostra de RUST (P(RUST)=0.35, P(HEALTHY)=0.65),
    # mas a Regra de Risco Mínimo de Bayes corrige para RUST devido à assimetria 20,65:1!
    y_true = np.array([RUST_CLASS_ID, HEALTHY_CLASS_ID])
    y_pred_argmax = np.array([HEALTHY_CLASS_ID, HEALTHY_CLASS_ID])
    y_proba = np.array(
        [
            [0.0, 0.65, 0.0, 0.0, 0.0, 0.35, 0.0],  # Amostra 0: argmax = 1 (HEALTHY), mas P(RUST)=0.35
            [0.0, 0.99, 0.0, 0.0, 0.0, 0.01, 0.0],  # Amostra 1: argmax = 1 (HEALTHY)
        ]
    )

    eval_res = evaluate_expected_agronomic_loss(
        y_true=y_true,
        y_pred=y_pred_argmax,
        y_proba=y_proba,
        cost_matrix=L,
        params=params,
    )
    # No argmax, a Amostra 0 sofre perda catastrófica de FN de Ferrugem (R$ 10.017,25)
    assert eval_res["total_agronomic_loss_brl"] == pytest.approx(params.unit_cost_fn_rust_brl, rel=1e-3)
    # Na Regra de Bayes, como 0.35 * 10017.25 > 0.65 * 485.00, Bayes classifica como RUST e zera a perda!
    assert eval_res["bayes_minimum_risk_total_loss_brl"] == pytest.approx(0.0, abs=1e-3)
    assert eval_res["bayes_additional_savings_brl"] == pytest.approx(
        params.unit_cost_fn_rust_brl, rel=1e-3
    )


def test_generate_figures_and_markdown_report_tmp(
    synthetic_predictions_df: pd.DataFrame, tmp_path: Path
) -> None:
    """Verifica a geração ponta a ponta das figuras PNG e dos relatórios Markdown em diretório temporário."""
    params = SugarcaneEconomicParameters()
    L, df_cost = build_agronomic_cost_matrix(params)
    df_err, df_trans = isolate_misclassified_test_instances(
        predictions_df=synthetic_predictions_df, partition="test"
    )
    df_test = synthetic_predictions_df[synthetic_predictions_df["split_partition"] == "test"]
    y_t = df_test["target_multiclass"].to_numpy(dtype=int)
    y_p = df_test["y_pred"].to_numpy(dtype=int)
    proba_cols = [
        "y_proba_grassy_shoot",
        "y_proba_healthy",
        "y_proba_leaf_scald",
        "y_proba_mosaic",
        "y_proba_red_rot",
        "y_proba_rust",
        "y_proba_yellow_leaf",
    ]
    y_prob = df_test[proba_cols].to_numpy(dtype=float)

    fn_audit = quantify_false_negative_agronomic_risk(y_t, y_p, y_prob[:, RUST_CLASS_ID], params)
    fp_audit = quantify_false_positive_economic_impact(y_t, y_p, params)
    loss_eval = evaluate_expected_agronomic_loss(y_t, y_p, y_prob, L, params)

    figs = generate_agronomic_audit_figures(
        df_cost_matrix=df_cost,
        loss_eval_test=loss_eval,
        df_transitions_test=df_trans,
        fn_risk_test=fn_audit,
        params=params,
        figures_dir=tmp_path,
    )
    for fig_path in figs.values():
        p = Path(fig_path)
        assert p.exists()
        assert p.stat().st_size > 15_000

    md_main = tmp_path / "relatorio.md"
    md_entrega = tmp_path / "entrega" / "README.md"
    generate_automated_agronomic_audit_report(
        df_errors_test=df_err,
        df_transitions_test=df_trans,
        fn_risk_test=fn_audit,
        fp_impact_test=fp_audit,
        df_cost_matrix=df_cost,
        loss_eval_test=loss_eval,
        params=params,
        report_path=md_main,
        entrega_readme_path=md_entrega,
    )
    assert md_main.exists() and md_main.stat().st_size > 2_000
    assert md_entrega.exists() and md_entrega.stat().st_size > 2_000


def test_production_agronomic_audit_artifacts_integrity() -> None:
    """Confirma que todos os artefatos de produção da Sprint 4 foram gerados com sucesso e integridade."""
    project_root = Path(__file__).resolve().parent.parent
    json_path = project_root / "models" / "svm_agronomic_error_audit.json"
    misclassified_csv = project_root / "models" / "svm_misclassified_test_instances.csv"
    cost_csv = project_root / "models" / "svm_agronomic_cost_matrix_7x7.csv"
    fig1 = project_root / "docs" / "figures" / "sprint4_agronomic_cost_matrix_and_losses.png"
    fig2 = project_root / "docs" / "figures" / "sprint4_fn_fp_economic_risk_curves.png"
    report_md = project_root / "docs" / "relatorio_auditoria_erros_impacto_agronomico_sprint4.md"

    assert json_path.exists()
    assert misclassified_csv.exists()
    assert cost_csv.exists()
    assert fig1.exists() and fig1.stat().st_size > 50_000
    assert fig2.exists() and fig2.stat().st_size > 50_000
    assert report_md.exists() and report_md.stat().st_size > 4_000

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_summary = data["test_set_error_isolation_summary"]
    assert test_summary["total_test_samples"] == 380
    assert test_summary["misclassified_samples_count"] == 149
    assert data["false_negative_agronomic_risk"]["rust_false_negatives"] == 0
    assert data["false_positive_economic_impact"]["rust_false_positives"] == 0
    assert data["agronomic_cost_matrix_evaluation"]["loss_reduction_vs_inaction_pct"] > 90.0
