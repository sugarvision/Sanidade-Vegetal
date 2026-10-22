"""
Módulo de Testes Automatizados para a Rotina de Repeated Stratified K-Fold
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 4
"""

import unittest
import json
from pathlib import Path
import pandas as pd
import numpy as np


class TestRepeatedStratifiedKFoldAudit(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.results_csv = cls.base_dir / "models" / "repeated_kfold_audit_results.csv"
        cls.metadata_json = cls.base_dir / "models" / "repeated_kfold_audit_metadata.json"
        cls.fig_metrics = cls.base_dir / "docs" / "figures" / "repeated_kfold_metrics_dispersion.png"
        cls.fig_strat = cls.base_dir / "docs" / "figures" / "repeated_kfold_class_stratification_audit.png"
        cls.fig_conv = cls.base_dir / "docs" / "figures" / "repeated_kfold_convergence_stability.png"
        
    def test_artifacts_exist(self):
        """Valida a existência física de todos os artefatos gerados."""
        self.assertTrue(self.results_csv.exists(), "CSV de resultados não encontrado!")
        self.assertTrue(self.metadata_json.exists(), "JSON de metadados não encontrado!")
        self.assertTrue(self.fig_metrics.exists(), "Figura de dispersão não encontrada!")
        self.assertTrue(self.fig_strat.exists(), "Figura de estratificação não encontrada!")
        self.assertTrue(self.fig_conv.exists(), "Figura de convergência não encontrada!")

    def test_folds_count_and_topology(self):
        """Valida que exatamente 15 iterações (3 repetições x 5 folds) foram registradas."""
        df = pd.read_csv(self.results_csv)
        self.assertEqual(len(df), 15, "Quantidade de folds divergente de 15!")
        self.assertEqual(sorted(df['repeat_id'].unique().tolist()), [1, 2, 3], "Repetições inválidas!")
        for r in [1, 2, 3]:
            folds_in_r = df[df['repeat_id'] == r]['fold_in_repeat'].tolist()
            self.assertEqual(sorted(folds_in_r), [1, 2, 3, 4, 5], f"Folds da repetição {r} incorretos!")

    def test_stratification_deviations(self):
        """Valida que o desvio de proporção amostral em todos os folds é menor que 0.5%."""
        df = pd.read_csv(self.results_csv)
        max_dev = df['max_val_stratification_dev'].max() * 100.0
        self.assertLess(max_dev, 0.50, f"Desvio de estratificação acima da tolerância: {max_dev:.4f}%!")

    def test_dispersion_limits(self):
        """Valida que os coeficientes de variação (CV%) estão abaixo do teto de robustez (5%)."""
        with open(self.metadata_json, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        cv_f1 = meta['sampling_bias_audit']['coefficient_of_variation_f1_macro']
        cv_bal_acc = meta['sampling_bias_audit']['coefficient_of_variation_balanced_accuracy']
        
        self.assertLess(cv_f1, 5.0, f"CV do F1-Macro acima de 5%: {cv_f1}%!")
        self.assertLess(cv_bal_acc, 5.0, f"CV da Acurácia Balanceada acima de 5%: {cv_bal_acc}%!")

    def test_hypothesis_testing_conclusion(self):
        """Valida que o p-valor da ANOVA é > 0.05 (ausência de viés amostral comprovada)."""
        with open(self.metadata_json, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        p_val_f1 = meta['statistical_hypothesis_tests']['f1_macro_anova']['p_value']
        p_val_bal = meta['statistical_hypothesis_tests']['balanced_accuracy_anova']['p_value']
        
        self.assertGreaterEqual(p_val_f1, 0.05, f"Hipótese nula rejeitada para F1-Macro: p={p_val_f1}")
        self.assertGreaterEqual(p_val_bal, 0.05, f"Hipótese nula rejeitada para Balanced Accuracy: p={p_val_bal}")
        self.assertEqual(meta['sampling_bias_audit']['conclusion'], "SAMPLE_BIAS_REJECTED")


if __name__ == '__main__':
    unittest.main()
