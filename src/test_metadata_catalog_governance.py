"""
Módulo de Testes Automatizados para Catálogo de Metadados, Dicionário e Rastreabilidade
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 4
"""

import unittest
import json
from pathlib import Path
import pandas as pd


class TestMetadataCatalogGovernance(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.dict_md = cls.base_dir / "Modelagem Inicial e Dicionário de Dados" / "dicionario_de_dados.md"
        cls.dict_csv = cls.base_dir / "Modelagem Inicial e Dicionário de Dados" / "dicionario_de_dados.csv"
        cls.catalog_json = cls.base_dir / "models" / "dataset_lineage_and_metrics_catalog.json"
        cls.preds_csv = cls.base_dir / "data" / "processed" / "svm_predicoes_valid_test.csv"
        cls.abt_csv = cls.base_dir / "data" / "processed" / "abt_sanidade_vegetal.csv"
        cls.raw_meta_csv = cls.base_dir / "data" / "processed" / "metadata_raw_images.csv"
        cls.fig_roc = cls.base_dir / "docs" / "figures" / "svm_multiclass_roc_curves.png"
        cls.fig_comp = cls.base_dir / "docs" / "figures" / "svm_validation_metrics_comparison.png"

    def test_catalog_files_exist(self):
        """Valida a existência física dos arquivos de dicionário, manifesto e figuras."""
        self.assertTrue(self.dict_md.exists(), "dicionario_de_dados.md não encontrado!")
        self.assertTrue(self.dict_csv.exists(), "dicionario_de_dados.csv não encontrado!")
        self.assertTrue(self.catalog_json.exists(), "dataset_lineage_and_metrics_catalog.json não encontrado!")
        self.assertTrue(self.fig_roc.exists(), "Figura de curvas ROC não encontrada!")
        self.assertTrue(self.fig_comp.exists(), "Figura comparativa de métricas não encontrada!")

    def test_metrics_cataloged_in_csv(self):
        """Valida que as métricas formais foram catalogadas na entidade METRICAS_VALIDACAO_SVM no CSV."""
        df_dict = pd.read_csv(self.dict_csv)
        metrics_rows = df_dict[df_dict['entidade'] == 'METRICAS_VALIDACAO_SVM']
        col_names = set(metrics_rows['nome_coluna'].unique())
        
        expected_metrics = {
            'precision_macro', 'precision_weighted', 'recall_macro',
            'recall_weighted', 'f1_score_macro', 'f1_score_weighted',
            'auc_roc_macro', 'auc_roc_weighted', 'specificity_macro',
            'balanced_accuracy', 'accuracy_global'
        }
        self.assertTrue(expected_metrics.issubset(col_names), f"Métricas ausentes no CSV: {expected_metrics - col_names}")

    def test_metrics_documented_in_markdown(self):
        """Valida que as seções de métricas, restrições e linhagem constam no markdown."""
        content = self.dict_md.read_text(encoding='utf-8')
        self.assertIn("9. Catálogo Oficial de Métricas de Avaliação Formal do SVM", content)
        self.assertIn("10. Hiperparâmetros Ótimos Homologados e Restrições Operacionais", content)
        self.assertIn("11. Auditoria e Atestado de Rastreabilidade Ponta a Ponta", content)
        self.assertIn("AUC-ROC (Macro OvR)", content)
        self.assertIn("Specificity / TNR (Macro OvR)", content)

    def test_end_to_end_lineage_traceability(self):
        """Valida 100% de integridade referencial entre predições, ABT e metadados brutos."""
        df_preds = pd.read_csv(self.preds_csv)
        df_abt = pd.read_csv(self.abt_csv)
        df_meta = pd.read_csv(self.raw_meta_csv)
        
        # 1. Preds -> ABT
        self.assertTrue(df_preds['sample_id'].isin(df_abt['sample_id']).all(), "Inconsistência sample_id Preds -> ABT!")
        
        # 2. ABT -> Meta
        self.assertTrue(df_abt['filename'].isin(df_meta['filename']).all(), "Inconsistência filename ABT -> Meta!")
        
        # 3. Calibração de probabilidades
        proba_cols = [c for c in df_preds.columns if c.startswith('y_proba_') and c != 'y_proba_max']
        row_sums = df_preds[proba_cols].sum(axis=1)
        self.assertTrue((abs(row_sums - 1.0) < 1e-4).all(), "Probabilidades não somam 1.0!")

    def test_manifest_mlops_content(self):
        """Valida que o manifesto JSON contém as métricas de validação e teste com AUC > 0.90."""
        with open(self.catalog_json, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            
        self.assertEqual(manifest['lineage_audit']['status'], "APPROVED")
        val_auc = manifest['formal_evaluation_metrics']['valid']['auc_roc_macro']
        test_auc = manifest['formal_evaluation_metrics']['test']['auc_roc_macro']
        self.assertGreater(val_auc, 0.90, f"AUC de validação abaixo de 0.90: {val_auc}")
        self.assertGreater(test_auc, 0.90, f"AUC de teste abaixo de 0.90: {test_auc}")


if __name__ == '__main__':
    unittest.main()
