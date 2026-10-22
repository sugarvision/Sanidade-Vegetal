"""
Módulo de Testes Automatizados para a Concepção e Benchmark de Baselines
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 3 — Framework SEMMA (Fase: Model)
"""

import unittest
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib


class TestBaselineModelsBenchmark(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.models_dir = cls.base_dir / "models"
        cls.figures_dir = cls.base_dir / "docs" / "figures"
        cls.data_dir = cls.base_dir / "data" / "processed"
        
        # Arquivos esperados
        cls.dummy_model_path = cls.models_dir / "baseline_dummy_model.joblib"
        cls.logreg_model_path = cls.models_dir / "baseline_logistic_regression.joblib"
        cls.dtree_model_path = cls.models_dir / "baseline_decision_tree.joblib"
        
        cls.results_csv = cls.models_dir / "baseline_benchmark_results.csv"
        cls.per_class_csv = cls.models_dir / "baseline_per_class_metrics.csv"
        cls.stat_sig_csv = cls.models_dir / "baseline_statistical_significance.csv"
        cls.metadata_json = cls.models_dir / "baseline_models_metadata.json"
        
        cls.fig_comparison = cls.figures_dir / "baseline_models_benchmark_comparison.png"
        cls.fig_weights = cls.figures_dir / "baseline_class_weight_impact.png"
        cls.fig_cms = cls.figures_dir / "baseline_confusion_matrices.png"
        cls.fig_tree = cls.figures_dir / "baseline_decision_tree_structure.png"
        cls.tree_rules = cls.figures_dir / "baseline_decision_tree_rules.txt"
        
        # Carga dos dados de teste
        df = pd.read_csv(cls.data_dir / "abt_features_modelagem.csv")
        cls.feature_cols = [c for c in df.columns if c not in ['sample_id', 'split_partition', 'class_label', 'target_binary', 'target_multiclass']]
        cls.X_val = df[df['split_partition'] == 'valid'][cls.feature_cols]
        cls.y_val = df[df['split_partition'] == 'valid']['target_multiclass']

    def test_artifacts_physical_existence(self):
        """Valida que todos os artefatos de modelo, CSVs, JSON e figuras existem no disco."""
        self.assertTrue(self.dummy_model_path.exists(), "Modelo Dummy não encontrado!")
        self.assertTrue(self.logreg_model_path.exists(), "Modelo Regressão Logística não encontrado!")
        self.assertTrue(self.dtree_model_path.exists(), "Modelo Árvore de Decisão não encontrado!")
        
        self.assertTrue(self.results_csv.exists(), "CSV de resultados de benchmark não encontrado!")
        self.assertTrue(self.per_class_csv.exists(), "CSV de métricas por classe não encontrado!")
        self.assertTrue(self.stat_sig_csv.exists(), "CSV de significância estatística não encontrado!")
        self.assertTrue(self.metadata_json.exists(), "JSON de metadados de baselines não encontrado!")
        
        self.assertTrue(self.fig_comparison.exists(), "Gráfico comparativo não encontrado!")
        self.assertTrue(self.fig_weights.exists(), "Gráfico de impacto de class_weight não encontrado!")
        self.assertTrue(self.fig_cms.exists(), "Matrizes de confusão não encontradas!")
        self.assertTrue(self.fig_tree.exists(), "Estrutura da árvore não encontrada!")
        self.assertTrue(self.tree_rules.exists(), "Regras textuais da árvore não encontradas!")

    def test_serialized_models_inference(self):
        """Testa se os modelos serializados em joblib carregam e realizam inferência válida."""
        dummy = joblib.load(self.dummy_model_path)
        logreg = joblib.load(self.logreg_model_path)
        dtree = joblib.load(self.dtree_model_path)
        
        # Testar .predict()
        y_pred_dummy = dummy.predict(self.X_val)
        y_pred_logreg = logreg.predict(self.X_val)
        y_pred_dtree = dtree.predict(self.X_val)
        
        self.assertEqual(len(y_pred_dummy), len(self.X_val), "Predições do Dummy com tamanho incorreto!")
        self.assertEqual(len(y_pred_logreg), len(self.X_val), "Predições da Regressão Logística com tamanho incorreto!")
        self.assertEqual(len(y_pred_dtree), len(self.X_val), "Predições da Árvore com tamanho incorreto!")
        
        # Testar classes preditas
        valid_classes = set(range(7))
        self.assertTrue(set(np.unique(y_pred_dummy)).issubset(valid_classes))
        self.assertTrue(set(np.unique(y_pred_logreg)).issubset(valid_classes))
        self.assertTrue(set(np.unique(y_pred_dtree)).issubset(valid_classes))

    def test_decision_tree_depth_and_interpretability(self):
        """Valida se a árvore de decisão atende à restrição de interpretabilidade (max_depth <= 4)."""
        dtree = joblib.load(self.dtree_model_path)
        self.assertLessEqual(dtree.get_depth(), 4, f"Profundidade da árvore ({dtree.get_depth()}) excede o limite de 4!")
        self.assertLessEqual(dtree.get_n_leaves(), 16, f"Número de folhas ({dtree.get_n_leaves()}) excede o limite de 16!")

    def test_dummy_performance_floor(self):
        """Valida que o classificador dummy fixa o piso estatístico empírico (10% a 25% de acurácia)."""
        df_bench = pd.read_csv(self.results_csv)
        dummy_row = df_bench[df_bench['Modelo'].str.contains("Dummy", case=False)]
        self.assertFalse(dummy_row.empty, "Linha do Dummy não encontrada na tabela de resultados!")
        
        dummy_acc = dummy_row['CV_Acuracia_Media'].values[0]
        dummy_f1 = dummy_row['CV_F1_Macro_Media'].values[0]
        
        self.assertGreater(dummy_acc, 0.10, "Acurácia do dummy excessivamente baixa!")
        self.assertLess(dummy_acc, 0.25, "Acurácia do dummy acima do esperado para chute estratificado!")
        self.assertLess(dummy_f1, 0.20, "F1-Macro do dummy acima do esperado para baseline ingênuo!")

    def test_class_weight_balanced_mitigation(self):
        """Valida que o uso de class_weight='balanced' eleva o recall de classes raras (Grassy Shoot)."""
        df_per_class = pd.read_csv(self.per_class_csv)
        
        # Grassy Shoot (Classe 0)
        gs_unweighted = df_per_class[
            (df_per_class['Modelo'].str.contains("Regressão Logística", case=False)) &
            (df_per_class['Modelo'].str.contains("Sem Ponderação", case=False)) &
            (df_per_class['Classe'] == "GRASSY SHOOT")
        ]['Recall'].values[0]
        
        gs_balanced = df_per_class[
            (df_per_class['Modelo'].str.contains("Regressão Logística", case=False)) &
            (df_per_class['Modelo'].str.contains("balanced", case=False)) &
            (df_per_class['Classe'] == "GRASSY SHOOT")
        ]['Recall'].values[0]
        
        self.assertGreater(gs_balanced, gs_unweighted, "class_weight='balanced' não melhorou o recall de Grassy Shoot!")
        self.assertGreater(gs_balanced, 0.20, f"Recall de Grassy Shoot com balanceamento ({gs_balanced}) ainda insatisfatório!")

    def test_svm_statistical_superiority_hypothesis(self):
        """Valida se o teste de hipótese comprova superioridade estatística do SVM com p < 0.05."""
        df_stat = pd.read_csv(self.stat_sig_csv)
        self.assertFalse(df_stat.empty, "Tabela de significância estatística está vazia!")
        
        # Testar contra Dummy
        row_dummy = df_stat[df_stat['Baseline'].str.contains("Dummy", case=False)]
        self.assertLess(row_dummy['p_value_F1'].values[0], 0.001, "Ganho do SVM frente ao Dummy não significante!")
        
        # Testar contra Árvore rasa balanced
        row_dt = df_stat[df_stat['Baseline'].str.contains("Árvore", case=False) & df_stat['Baseline'].str.contains("balanced", case=False)]
        self.assertLess(row_dt['p_value_F1'].values[0], 0.05, "Ganho do SVM frente à Árvore rasa não significante (p >= 0.05)!")
        
        # Testar contra Regressão Logística balanced
        row_lr = df_stat[df_stat['Baseline'].str.contains("Regressão Logística", case=False) & df_stat['Baseline'].str.contains("balanced", case=False)]
        self.assertLess(row_lr['p_value_F1'].values[0], 0.05, "Ganho do SVM frente à Regressão Logística não significante (p >= 0.05)!")


if __name__ == '__main__':
    unittest.main()
