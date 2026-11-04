"""
Testes Unitários: Módulo de Inspeção Visual de Casos de Fronteira e Discordância
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 4 — Framework SEMMA (Fase: Assess)
Responsável: Cesar (Lead Técnico & Visão Computacional)
"""

import json
import unittest
from pathlib import Path
import pandas as pd


class TestSVMVisualBoundaryInspection(unittest.TestCase):
    """Valida a inspeção de casos de fronteira e catálogo fotográfico de erros."""

    @classmethod
    def setUpClass(cls):
        cls.margins_csv = Path("models/svm_boundary_geometric_margin_cases.csv")
        cls.summary_csv = Path("models/svm_visual_error_patterns_summary.csv")
        cls.metadata_json = Path("models/svm_visual_boundary_inspection_metadata.json")
        cls.diagnostic_fig = Path("docs/figures/sprint4_visual_boundary_cases_and_limitations.png")

    def test_artifacts_exist_on_disk(self):
        """Verifica a presença dos artefatos em disco."""
        self.assertTrue(self.margins_csv.exists(), "CSV de margens geométricas ausente.")
        self.assertTrue(self.summary_csv.exists(), "CSV de padrões fotográficos ausente.")
        self.assertTrue(self.metadata_json.exists(), "Metadados JSON de inspeção de fronteira ausente.")
        self.assertTrue(self.diagnostic_fig.exists(), "Painel gráfico diagnóstico de fronteira ausente.")

    def test_margins_dataframe_integrity(self):
        """Verifica a integridade dos cálculos de margem geométrica e entropia."""
        df = pd.read_csv(self.margins_csv)
        self.assertEqual(len(df), 380, "Deve haver 380 amostras no teste.")

        expected_cols = [
            "sample_id", "class_label", "y_pred_label",
            "delta_geometric_margin", "shannon_entropy", "is_correct"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Coluna '{col}' ausente no arquivo de margens.")

        # Margem geométrica deve ser não-negativa
        self.assertTrue((df["delta_geometric_margin"] >= 0).all(), "Margens geométricas não podem ser negativas.")

        # Amostras com erro devem ter mediana de margem menor ou comparável à dos corretos
        med_err = df[df["is_correct"] == 0]["delta_geometric_margin"].median()
        med_corr = df[df["is_correct"] == 1]["delta_geometric_margin"].median()
        self.assertLessEqual(med_err, med_corr + 0.1, "Erros deveriam se concentrar em regiões de menor margem.")

    def test_patterns_summary_percentages(self):
        """Verifica se os percentuais de modos de falha fotográfica são consistentes."""
        df_p = pd.read_csv(self.summary_csv)
        self.assertGreater(len(df_p), 3, "Deveria haver ao menos 4 categorias de falha fotográfica.")
        self.assertTrue((df_p["misclassified_samples_count"] >= 0).all())

    def test_metadata_audit_contents(self):
        """Valida os campos de governança no JSON de metadados."""
        with open(self.metadata_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(meta["responsible"], "Cesar (Lead Técnico & Visão Computacional)")
        self.assertIn("top_10_most_ambiguous_test_samples", meta)
        self.assertIn("photographic_failure_modes", meta)


if __name__ == "__main__":
    unittest.main()

