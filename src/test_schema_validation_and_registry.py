"""
Suíte de Testes Automatizados de Schema Validation, Model Registry e Inferência Segura
Sprint 5 — Governança de MLOps
Projeto: Sanidade-Vegetal (SugarVision)
"""

import unittest
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from src.schema_validator import SchemaValidator, SchemaValidationError, ProductionInferenceService


class TestSchemaValidationAndModelRegistry(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.models_dir = cls.base_dir / "models"
        cls.data_dir = cls.base_dir / "data" / "processed"
        cls.schema_path = cls.models_dir / "model_schema_contract.json"
        cls.registry_path = cls.models_dir / "model_registry.json"
        cls.model_path = cls.models_dir / "sanidade_vegetal_pipeline_final.joblib"
        cls.abt_path = cls.data_dir / "abt_sanidade_vegetal.csv"
        
        cls.validator = SchemaValidator(schema_path=cls.schema_path)

    def test_01_schema_contract_json_structure(self):
        """Valida que o contrato formal de schema possui as 23 features modeladas e regras de domínio."""
        self.assertTrue(self.schema_path.exists(), f"Contrato de schema não encontrado: {self.schema_path}")
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            contract = json.load(f)
            
        self.assertEqual(contract.get("status"), "APPROVED_PRODUCTION_CONTRACT")
        input_contract = contract.get("input_features_contract", {})
        num_features = input_contract.get("numerical_features", {})
        cat_features = input_contract.get("categorical_features", {})
        
        self.assertEqual(len(num_features), 17, "O contrato deve conter exatamente 17 features numéricas!")
        self.assertEqual(len(cat_features), 6, "O contrato deve conter exatamente 6 features categóricas!")
        
        # Validar regras mínimas de domínio em atributos críticos
        self.assertIn("domain_min", num_features["mean_hue"])
        self.assertIn("domain_max", num_features["mean_hue"])
        self.assertIn("allowed_values", cat_features["dataset_source"])

    def test_02_clean_batch_validation_success(self):
        """Valida que lotes reais da base de dados são 100% aprovados sem erros nem quarentena."""
        df_clean = pd.read_csv(self.abt_path, nrows=25)
        valid_df, quarantine_df, violations = self.validator.validate_batch(df_clean, quarantine=True)
        
        self.assertEqual(len(valid_df), 25, "Todas as 25 amostras sadias deveriam ser aprovadas!")
        self.assertEqual(len(quarantine_df), 0, "Nenhuma amostra sadia deve ir para quarentena!")
        self.assertEqual(len(violations), 0, "Lista de violações deve ser vazia para amostras sadias!")

    def test_03_missing_required_columns_rejection(self):
        """Valida que a ausência de colunas contratuais obrigatórias causa rejeição estrutural."""
        df_incomplete = pd.read_csv(self.abt_path, nrows=10)
        df_incomplete = df_incomplete.drop(columns=['mean_hue', 'glcm_contrast'])
        
        valid_df, quarantine_df, violations = self.validator.validate_batch(
            df_incomplete, raise_on_error=False, quarantine=True
        )
        
        self.assertEqual(len(valid_df), 0, "Lote com colunas faltantes não deve gerar amostras válidas!")
        self.assertEqual(len(quarantine_df), 10, "Todas as amostras devem ser isoladas em quarentena!")
        self.assertGreater(len(violations), 0, "Relatório de violações deve detalhar as colunas ausentes!")
        
        # Comportamento com raise_on_error=True
        with self.assertRaises(SchemaValidationError):
            self.validator.validate_batch(df_incomplete, raise_on_error=True, quarantine=False)

    def test_04_out_of_bounds_numerical_rejection(self):
        """Valida que leituras ópticas fora dos limites físicos são detectadas e isoladas."""
        df_sample = pd.read_csv(self.abt_path, nrows=5)
        
        # Corromper propositalmente o índice 2 com saturação impossível (> 1.0)
        # e índice 3 com mean_hue negativo (< 0.0)
        df_corrupted = df_sample.copy()
        df_corrupted.loc[2, 'std_saturation'] = 5.85
        df_corrupted.loc[3, 'mean_hue'] = -15.0
        
        valid_df, quarantine_df, violations = self.validator.validate_batch(df_corrupted, quarantine=True)
        
        self.assertEqual(len(valid_df), 3, "Linhas 0, 1 e 4 devem ser aprovadas!")
        self.assertEqual(len(quarantine_df), 2, "Linhas 2 e 3 devem ser isoladas na quarentena!")
        self.assertIn(2, quarantine_df.index)
        self.assertIn(3, quarantine_df.index)

    def test_05_unknown_categorical_domain_rejection(self):
        """Valida rejeição de amostras com categorias fora do vocabulário homologado."""
        df_sample = pd.read_csv(self.abt_path, nrows=4)
        df_corrupted = df_sample.copy()
        df_corrupted.loc[1, 'dataset_source'] = 'sensor_nao_homologado_camera_x'
        df_corrupted.loc[3, 'extension'] = '.exe'
        
        valid_df, quarantine_df, violations = self.validator.validate_batch(df_corrupted, quarantine=True)
        
        self.assertEqual(len(valid_df), 2, "Linhas 0 e 2 devem ser aprovadas!")
        self.assertEqual(len(quarantine_df), 2, "Linhas 1 e 3 com categorias desconhecidas devem ser isoladas!")

    def test_06_model_registry_json_and_production_alignment(self):
        """Valida a consistência do Model Registry e o alinhamento com o modelo de produção ativo."""
        self.assertTrue(self.registry_path.exists(), f"Model Registry não encontrado: {self.registry_path}")
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
            
        self.assertEqual(registry.get("active_production_version"), "1.0.0")
        self.assertEqual(registry.get("rollback_target_version"), "0.9.0")
        
        # Encontrar modelo de produção
        prod_model = next((m for m in registry["registered_models"] if m["stage"] == "In-Production"), None)
        self.assertIsNotNone(prod_model, "Nenhum modelo com estágio 'In-Production' no Model Registry!")
        self.assertEqual(prod_model["model_version"], "1.0.0")
        
        # Conferir hash do binário físico com o registro
        hasher = hashlib.sha256()
        with open(self.model_path, 'rb') as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        disk_sha = hasher.hexdigest()
        self.assertEqual(disk_sha, prod_model["sha256"], "Hash SHA-256 no Model Registry difere do binário em disco!")

    def test_07_end_to_end_production_service_inference(self):
        """Valida a inferência ponta a ponta no ProductionInferenceService com quarentena resiliente."""
        service = ProductionInferenceService(
            model_path=self.model_path,
            schema_path=self.schema_path
        )
        
        # Lote misto de 6 amostras: 4 normais e 2 corrompidas
        df_batch = pd.read_csv(self.abt_path, nrows=6)
        df_batch.loc[2, 'size_kb'] = -500.0 # Inválido: tamanho negativo
        df_batch.loc[5, 'aspect_ratio'] = 99.9 # Inválido: aspect ratio absurdo
        
        result = service.predict(df_batch)
        
        self.assertEqual(result["status"], "PARTIAL_SUCCESS")
        self.assertEqual(result["n_total_received"], 6)
        self.assertEqual(result["n_valid_processed"], 4)
        self.assertEqual(result["n_quarantined"], 2)
        self.assertEqual(len(result["predictions_discrete"]), 4)
        self.assertEqual(len(result["prediction_probabilities"]), 4)
        
        # Validar probabilidades
        probas = np.array(result["prediction_probabilities"])
        self.assertEqual(probas.shape, (4, 7))
        np.testing.assert_allclose(probas.sum(axis=1), np.ones(4), rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
