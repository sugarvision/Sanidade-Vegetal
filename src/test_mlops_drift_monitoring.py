"""
Suíte de Testes Unitários e de Integração para Monitoramento Contínuo de Drift (MLOps)
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 5 — Framework SEMMA (Fase: Assess & Sustain / MLOps Governance)

Valida:
1. Precisão matemática do cálculo de PSI (Population Stability Index) em distribuições
   idênticas, moderadamente deslocadas, severamente deslocadas, vetores com NaNs e bordas.
2. Precisão matemática do teste bicaudal de Kolmogorov-Smirnov (KS) de duas amostras.
3. Mapeamento completo das fontes de drift no campo (iluminação sazonal, novas câmeras/sensores,
   estádios fenológicos e cultivares).
4. Regras de alerta e gatilhos automatizados de retreinamento e rollback preventivo.
5. Integração ponta a ponta com a ABT oficial e artefatos JSON persistidos.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.mlops_drift_monitoring import (
    CRITICAL_AGRONOMIC_FEATURES,
    FIELD_DRIFT_SOURCES_CATALOG,
    MONITORED_VISUAL_FEATURES,
    DriftMonitoringEngine,
    DriftThresholds,
    calculate_ks_drift,
    calculate_psi,
    classify_psi_severity,
    evaluate_feature_drift,
    evaluate_retraining_triggers,
    export_mlops_drift_artifacts,
    get_field_drift_sources_mapping,
)


# ==============================================================================
# 1. TESTES MATEMÁTICOS DO POPULATION STABILITY INDEX (PSI) — PYTEST
# ==============================================================================


def test_psi_identical_distributions_returns_zero() -> None:
    """Distribuições idênticas devem produzir PSI exatamente igual a zero."""
    rng = np.random.default_rng(123)
    ref = rng.normal(loc=25.0, scale=5.0, size=2000)
    cur = ref.copy()

    psi_q = calculate_psi(ref, cur, bins=10, strategy="quantile")
    psi_u = calculate_psi(ref, cur, bins=10, strategy="uniform")

    assert psi_q == pytest.approx(0.0, abs=1e-10)
    assert psi_u == pytest.approx(0.0, abs=1e-10)


def test_psi_stable_same_distribution_under_threshold() -> None:
    """Duas amostras grandes da mesma distribuição gaussiana devem ter PSI < 0.10 (STABLE)."""
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=50.0, scale=10.0, size=5000)
    cur = rng.normal(loc=50.0, scale=10.0, size=5000)

    psi_val = calculate_psi(ref, cur, bins=10, strategy="quantile")
    assert 0.0 <= psi_val < 0.10
    assert classify_psi_severity(psi_val) == "STABLE"


def test_psi_moderate_shift_triggers_warning_band() -> None:
    """Deslocamento moderado de média (~0.35 desvios-padrão) deve resultar em 0.10 <= PSI < 0.25."""
    rng = np.random.default_rng(7)
    ref = rng.normal(loc=100.0, scale=15.0, size=6000)
    cur = rng.normal(loc=105.5, scale=15.0, size=6000)

    psi_val = calculate_psi(ref, cur, bins=10, strategy="quantile")
    assert 0.10 <= psi_val < 0.25
    assert classify_psi_severity(psi_val) == "WARNING"


def test_psi_severe_shift_triggers_critical_band() -> None:
    """Deslocamento severo de distribuição (> 1 desvio-padrão) deve resultar em PSI >= 0.25."""
    rng = np.random.default_rng(99)
    ref = rng.normal(loc=30.0, scale=5.0, size=4000)
    cur = rng.normal(loc=38.0, scale=6.5, size=4000)

    psi_val = calculate_psi(ref, cur, bins=10, strategy="quantile")
    assert psi_val >= 0.25
    assert classify_psi_severity(psi_val) == "CRITICAL"


def test_psi_known_analytical_two_bin_value() -> None:
    """
    Verifica a exatidão da fórmula matemática do PSI contra o cálculo analítico fechado:
    Para 2 bins com Q = [0.5, 0.5] e P = [0.8, 0.2]:
    PSI = (0.8 - 0.5)*ln(0.8/0.5) + (0.2 - 0.5)*ln(0.2/0.5)
        = 0.3 * ln(1.6) + (-0.3) * ln(0.4) = 0.3 * ln(4.0) ≈ 0.4158883
    """
    ref = np.array([0.0] * 500 + [10.0] * 500)
    cur = np.array([0.0] * 800 + [10.0] * 200)

    psi_val = calculate_psi(ref, cur, bins=2, strategy="uniform")
    expected_psi = 0.3 * np.log(4.0)
    assert psi_val == pytest.approx(expected_psi, rel=1e-4)


def test_psi_handles_nan_and_inf_gracefully() -> None:
    """Valores NaN e Inf devem ser filtrados antes do cálculo sem corromper o resultado."""
    rng = np.random.default_rng(101)
    clean_ref = rng.uniform(0.0, 180.0, size=1000)
    clean_cur = clean_ref.copy()

    ref_with_nan = np.concatenate([clean_ref, [np.nan, np.inf, -np.inf]])
    cur_with_nan = np.concatenate([clean_cur, [np.nan]])

    psi_val = calculate_psi(ref_with_nan, cur_with_nan, bins=10)
    assert psi_val == pytest.approx(0.0, abs=1e-10)


def test_psi_invalid_inputs_raise_errors() -> None:
    """Entradas inválidas (bins < 2, strategy desconhecida, arrays vazios) devem levantar exceções."""
    with pytest.raises(ValueError, match="bins"):
        calculate_psi([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], bins=1)

    with pytest.raises(ValueError, match="Estratégia"):
        calculate_psi([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], strategy="invalid")

    with pytest.raises(ValueError, match="pelo menos 2"):
        calculate_psi([1.0], [1.0, 2.0, 3.0])


# ==============================================================================
# 2. TESTES MATEMÁTICOS DO TESTE DE KOLMOGOROV-SMIRNOV (KS) — PYTEST
# ==============================================================================


def test_ks_identical_arrays_zero_statistic_and_unit_pvalue() -> None:
    """Amostras idênticas devem ter estatística KS D = 0.0, p-value = 1.0 e STABLE."""
    arr = np.linspace(0.0, 100.0, 500)
    res = calculate_ks_drift(arr, arr.copy(), alpha=0.01)

    assert res["ks_statistic"] == pytest.approx(0.0, abs=1e-12)
    assert res["p_value"] == pytest.approx(1.0, abs=1e-12)
    assert res["null_hypothesis_rejected"] is False
    assert res["drift_detected"] is False
    assert res["severity"] == "STABLE"


def test_ks_distinct_distributions_detects_critical_drift() -> None:
    """Amostras com deslocamento substancial devem rejeitar H_0 (p < 0.01) e acusar drift."""
    rng = np.random.default_rng(2026)
    ref = rng.normal(loc=15.0, scale=3.0, size=1000)
    cur = rng.normal(loc=20.0, scale=3.0, size=1000)

    res = calculate_ks_drift(ref, cur, alpha=0.01, min_effect_size=0.10)

    assert res["ks_statistic"] > 0.40
    assert res["p_value"] < 1e-10
    assert res["null_hypothesis_rejected"] is True
    assert res["drift_detected"] is True
    assert res["severity"] == "CRITICAL"
    assert res["n_reference"] == 1000
    assert res["n_current"] == 1000


def test_ks_disjoint_uniform_distributions_maximum_distance_one() -> None:
    """Duas distribuições com suportes disjuntos devem resultar em estatística KS D = 1.0."""
    ref = np.linspace(0.0, 10.0, 300)
    cur = np.linspace(20.0, 30.0, 300)

    res = calculate_ks_drift(ref, cur, alpha=0.01)
    assert res["ks_statistic"] == pytest.approx(1.0, abs=1e-6)
    assert res["drift_detected"] is True
    assert res["severity"] == "CRITICAL"


# ==============================================================================
# 3. TESTES DE FONTES DE DRIFT, GATILHOS DE RETREINAMENTO E INTEGRAÇÃO
# ==============================================================================


class TestMLOpsDriftMonitoringSuite(unittest.TestCase):
    """
    Suíte compatível simultaneamente com `pytest` e `unittest discover`.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.abt_path = cls.base_dir / "data" / "processed" / "abt_sanidade_vegetal.csv"
        cls.config_path = cls.base_dir / "models" / "mlops_drift_monitoring_config.json"
        cls.report_path = cls.base_dir / "models" / "mlops_drift_monitoring_report.json"

    def test_01_field_drift_sources_catalog_completeness(self) -> None:
        """Valida que as 3 fontes obrigatórias de drift no campo estão mapeadas e tipadas."""
        mapping = get_field_drift_sources_mapping()
        self.assertIn("seasonal_illumination", mapping)
        self.assertIn("new_cameras_sensors", mapping)
        self.assertIn("phenological_stages_and_cultivars", mapping)

        for key, item in mapping.items():
            self.assertTrue(len(item["affected_features"]) >= 5, f"Fonte {key} sem features suficientes")
            self.assertTrue(len(item["environmental_causes"]) >= 2)
            self.assertTrue(len(item["physical_mechanism"]) > 30)
            self.assertTrue(len(item["mitigation_and_sustainment_action"]) > 30)

    def test_02_retraining_triggers_nominal_vs_critical(self) -> None:
        """Valida que a lógica de gatilhos separa operação estável de cenários críticos."""
        rng = np.random.default_rng(55)
        n_samples = 800
        ref_df = pd.DataFrame(
            {
                feat: rng.normal(loc=50.0, scale=10.0, size=n_samples)
                for feat in MONITORED_VISUAL_FEATURES
            }
        )
        # Lote nominal sem drift
        cur_nominal = pd.DataFrame(
            {
                feat: rng.normal(loc=50.0, scale=10.0, size=n_samples)
                for feat in MONITORED_VISUAL_FEATURES
            }
        )
        nominal_report = evaluate_feature_drift(ref_df, cur_nominal)
        nominal_decision = evaluate_retraining_triggers(
            nominal_report,
            new_labeled_samples=100,
            days_since_last_train=20,
            current_f1_macro=0.60,
            quarantine_rate=0.01,
        )
        self.assertEqual(nominal_decision["alert_level"], "NORMAL")
        self.assertFalse(nominal_decision["retrain_required"])
        self.assertFalse(nominal_decision["rollback_required"])

        # Lote com drift severo em features agronômicas críticas
        cur_drifted = cur_nominal.copy()
        cur_drifted["exg_index"] += 25.0
        cur_drifted["indice_clorose_necrose"] += 30.0
        cur_drifted["glcm_contrast"] += 35.0
        cur_drifted["laplacian_var"] += 40.0

        drifted_report = evaluate_feature_drift(ref_df, cur_drifted)
        drifted_decision = evaluate_retraining_triggers(
            drifted_report,
            new_labeled_samples=550,
            days_since_last_train=95,
            current_f1_macro=0.45,
            quarantine_rate=0.08,
        )
        self.assertEqual(
            drifted_decision["alert_level"], "EMERGENCY_ROLLBACK_AND_RETRAIN"
        )
        self.assertTrue(drifted_decision["retrain_required"])
        self.assertTrue(drifted_decision["rollback_required"])
        rule_ids = {r["rule_id"] for r in drifted_decision["triggered_rules"]}
        self.assertIn("RULE-PSI-CRITICAL", rule_ids)
        self.assertIn("RULE-KS-CONCEPT-DATA-DRIFT", rule_ids)
        self.assertIn("RULE-DATA-VOLUME-CADENCE", rule_ids)
        self.assertIn("RULE-PHENOLOGICAL-WINDOW-90D", rule_ids)
        self.assertIn("RULE-PERFORMANCE-DEGRADATION-F1", rule_ids)
        self.assertIn("RULE-QUARANTINE-SPIKE", rule_ids)

    def test_03_end_to_end_abt_monitoring_and_exported_artifacts(self) -> None:
        """Valida o motor DriftMonitoringEngine sobre a ABT real e os artefatos JSON gerados."""
        if not self.config_path.exists() or not self.report_path.exists():
            export_mlops_drift_artifacts(self.base_dir)

        self.assertTrue(self.config_path.exists())
        self.assertTrue(self.report_path.exists())

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(len(cfg["monitored_visual_features"]), 17)
        self.assertIn("seasonal_illumination", cfg["field_drift_sources_catalog"])

        with open(self.report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        scenarios = rep["scenarios_evaluated"]
        self.assertEqual(
            scenarios["nominal_operation_stable"]["decision"]["alert_level"], "NORMAL"
        )
        self.assertFalse(
            scenarios["nominal_operation_stable"]["decision"]["retrain_required"]
        )
        self.assertEqual(
            scenarios["seasonal_illumination_shift"]["decision"]["alert_level"],
            "WARNING_MONITORING",
        )
        self.assertTrue(
            scenarios["severe_multivariate_field_drift"]["decision"]["retrain_required"]
        )


if __name__ == "__main__":
    unittest.main()
