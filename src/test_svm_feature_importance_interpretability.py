"""
Testes Unitários: Módulo de Interpretabilidade e Permutation Feature Importance
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 4 — Framework SEMMA (Fase: Assess)
Responsável: Cesar (Lead Técnico & Visão Computacional)
"""

import json
import unittest
from pathlib import Path
import pandas as pd


class TestSVMFeatureImportanceInterpretability(unittest.TestCase):
    """Valida a integridade matemática e os artefatos de importância de features."""

    @classmethod
    def setUpClass(cls):
        cls.importance_csv = Path("models/svm_permutation_feature_importance.csv")
        cls.groups_csv = Path("models/svm_feature_importance_by_group.csv")
        cls.metadata_json = Path("models/svm_feature_importance_metadata.json")
        cls.ranking_fig = Path("docs/figures/sprint4_permutation_feature_importance_ranking.png")
        cls.groups_fig = Path("docs/figures/sprint4_feature_importance_by_group.png")

    def test_artifacts_exist_on_disk(self):
        """Verifica se todos os artefatos gerados pelo módulo foram salvos fisicamente."""
        self.assertTrue(self.importance_csv.exists(), "CSV de importância de features ausente.")
        self.assertTrue(self.groups_csv.exists(), "CSV de importância por grupos ausente.")
        self.assertTrue(self.metadata_json.exists(), "Metadados JSON de interpretabilidade ausente.")
        self.assertTrue(self.ranking_fig.exists(), "Figura do ranking de features ausente.")
        self.assertTrue(self.groups_fig.exists(), "Figura de importância por grupos ausente.")

    def test_importance_dataframe_structure_and_types(self):
        """Valida o esquema e consistência dos valores de importância de features."""
        df = pd.read_csv(self.importance_csv)
        self.assertGreater(len(df), 20, "Deveria haver ao menos 20 features avaliadas.")

        expected_cols = [
            "feature", "functional_group", "bal_acc_drop_mean",
            "bal_acc_drop_std", "macro_f1_drop_mean", "macro_f1_drop_std"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Coluna obrigatória '{col}' ausente no CSV.")

        # Desvio padrão deve ser não-negativo
        self.assertTrue((df["bal_acc_drop_std"] >= 0).all(), "Desvios padrão não podem ser negativos.")

    def test_groups_aggregation_consistency(self):
        """Verifica se a agregação por famílias funcionais soma coerentemente."""
        df_groups = pd.read_csv(self.groups_csv)
        self.assertIn("functional_group", df_groups.columns)
        self.assertIn("share_of_importance_pct", df_groups.columns)

        # A soma dos shares positivos deve estar normalizada próxima a 100%
        valid_share_sum = df_groups["share_of_importance_pct"].sum()
        self.assertAlmostEqual(valid_share_sum, 100.0, delta=1.0)

    def test_metadata_audit_keys(self):
        """Verifica se os metadados contêm os campos formais de governança."""
        with open(self.metadata_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(meta["responsible"], "Cesar (Lead Técnico & Visão Computacional)")
        self.assertIn("group_comparison_summary", meta)
        self.assertIn("biological_validation_status", meta)
        self.assertEqual(meta["biological_validation_status"], "VALIDATED_WITH_PHYTOPATHOLOGY_LITERATURE")


if __name__ == "__main__":
    unittest.main()

