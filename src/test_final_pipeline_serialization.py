"""
Suíte de Testes Automatizados de Serialização, Deserialização e Inferência Autônoma
Sprint 5 — Deploy & MLOps
Projeto: Sanidade-Vegetal (SugarVision)
"""

import unittest
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC


class TestFinalPipelineSerialization(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.models_dir = cls.base_dir / "models"
        cls.data_dir = cls.base_dir / "data" / "processed"
        cls.model_path = cls.models_dir / "sanidade_vegetal_pipeline_final.joblib"
        cls.metadata_path = cls.models_dir / "sanidade_vegetal_pipeline_final_metadata.json"
        cls.abt_path = cls.data_dir / "abt_sanidade_vegetal.csv"
        
        # Carregar metadados se existir
        if cls.metadata_path.exists():
            with open(cls.metadata_path, 'r', encoding='utf-8') as f:
                cls.metadata = json.load(f)
        else:
            cls.metadata = {}

    def test_01_model_and_metadata_files_exist(self):
        """Valida a existência física dos artefatos serializados."""
        self.assertTrue(self.model_path.exists(), f"Artefato não encontrado: {self.model_path}")
        self.assertTrue(self.metadata_path.exists(), f"Manifesto não encontrado: {self.metadata_path}")
        self.assertGreater(self.model_path.stat().st_size, 100_000, "Arquivo joblib menor que 100 KB!")

    def test_02_sha256_cryptographic_integrity(self):
        """Valida que o hash SHA-256 do arquivo joblib confere com o manifesto de metadados."""
        hasher = hashlib.sha256()
        with open(self.model_path, 'rb') as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        calculated_sha = hasher.hexdigest()
        
        recorded_sha = self.metadata.get('artifact', {}).get('sha256')
        self.assertEqual(
            calculated_sha,
            recorded_sha,
            f"Hash SHA-256 divergente! Calculado: {calculated_sha} vs Gravado: {recorded_sha}"
        )

    def test_03_clean_environment_deserialization_and_weights_integrity(self):
        """Valida o load em runtime limpo, checando passos e pesos aprendidos do SVM."""
        pipeline = joblib.load(self.model_path)
        
        # 1. Checar tipo e estrutura
        self.assertIsInstance(pipeline, Pipeline, "Objeto des-serializado não é uma Pipeline!")
        self.assertIn('preprocessor', pipeline.named_steps, "Step 'preprocessor' ausente!")
        self.assertIn('svm', pipeline.named_steps, "Step 'svm' ausente!")
        
        preprocessor = pipeline.named_steps['preprocessor']
        svm = pipeline.named_steps['svm']
        
        # 2. Integridade dos transformadores
        self.assertIsInstance(preprocessor, ColumnTransformer, "Preprocessor não é ColumnTransformer!")
        self.assertEqual(preprocessor.remainder, 'drop', "Estratégia remainder não é 'drop'!")
        self.assertIn('num', preprocessor.named_transformers_, "Sub-pipeline 'num' ausente!")
        self.assertIn('cat', preprocessor.named_transformers_, "Sub-pipeline 'cat' ausente!")
        
        # Checar se os transformadores internos preservam parâmetros ajustados
        scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
        self.assertTrue(hasattr(scaler, 'mean_'), "StandardScaler não possui atributos ajustados (mean_)!")
        self.assertTrue(hasattr(scaler, 'scale_'), "StandardScaler não possui atributos ajustados (scale_)!")
        
        onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
        self.assertTrue(hasattr(onehot, 'categories_'), "OneHotEncoder não possui categorias ajustadas!")
        
        # 3. Integridade do Estimador SVM
        self.assertIsInstance(svm, SVC, "Estimador não é SVC!")
        self.assertEqual(svm.kernel, 'poly', "Kernel deve ser 'poly'!")
        self.assertEqual(svm.degree, 3, "Grau deve ser 3!")
        self.assertEqual(svm.C, 0.1, "C deve ser 0.1!")
        self.assertEqual(svm.coef0, 1.0, "Coef0 deve ser 1.0!")
        self.assertTrue(svm.probability, "Estimador deve ter probability=True!")
        
        # Checar pesos e vetores de suporte aprendidos
        self.assertTrue(hasattr(svm, 'support_vectors_'), "SVM não possui vetores de suporte!")
        self.assertTrue(hasattr(svm, 'dual_coef_'), "SVM não possui coeficientes duais!")
        self.assertTrue(hasattr(svm, 'intercept_'), "SVM não possui intercepto!")
        self.assertEqual(len(svm.classes_), 7, "Número de classes do modelo deve ser 7!")
        self.assertGreater(svm.support_vectors_.shape[0], 0, "Quantidade de vetores de suporte é zero!")

    def test_04_autonomous_raw_features_inference_and_extra_columns_rejection(self):
        """Valida inferência autônoma ponta a ponta sem intervenção manual sobre dados brutos."""
        pipeline = joblib.load(self.model_path)
        
        # Carregar lote de teste
        df = pd.read_csv(self.abt_path, nrows=20)
        
        # Simular lote cru com colunas externas e metadados que devem ser ignorados automaticamente
        raw_df = df.copy()
        raw_df['sensor_gps_latitude'] = -22.1234
        raw_df['operator_notes'] = "Folha colhida no talhão C-12"
        raw_df['unmodeled_timestamp'] = "2026-10-23T11:45:00"
        
        # Execução de predição sem intervenção manual
        preds = pipeline.predict(raw_df)
        probas = pipeline.predict_proba(raw_df)
        
        self.assertEqual(len(preds), 20, "Tamanho do vetor de predições divergente!")
        self.assertEqual(probas.shape, (20, 7), "Formato da matriz de probabilidades divergente!")
        
        # Validar que todas as probabilidades somam 1.0
        np.testing.assert_allclose(
            probas.sum(axis=1),
            np.ones(20),
            rtol=1e-5,
            err_msg="As probabilidades preditas não somam 1.0!"
        )
        
        # Validar que as classes preditas estão contidas no domínio [0, 6]
        self.assertTrue(all(p in range(7) for p in preds), "Predição fora do domínio das 7 classes!")

    def test_05_prediction_idempotence(self):
        """Garante que sucessivas invocações sobre o mesmo dado geram exatamente a mesma predição."""
        pipeline = joblib.load(self.model_path)
        df_sample = pd.read_csv(self.abt_path, nrows=10)
        
        preds_1 = pipeline.predict(df_sample)
        probas_1 = pipeline.predict_proba(df_sample)
        
        preds_2 = pipeline.predict(df_sample)
        probas_2 = pipeline.predict_proba(df_sample)
        
        np.testing.assert_array_equal(preds_1, preds_2, err_msg="Predições discretas não foram idempotentes!")
        np.testing.assert_array_equal(probas_1, probas_2, err_msg="Matriz de probabilidades não foi idempotente!")

    def test_06_metadata_contract_completeness(self):
        """Valida que o manifesto de metadados contém todos os campos obrigatórios de governança."""
        required_root_keys = [
            'pipeline_name', 'version', 'project', 'sprint', 'task',
            'responsible', 'timestamp', 'production_status', 'framework',
            'artifact', 'model_architecture', 'input_contract',
            'training_metadata', 'evaluation_metrics'
        ]
        for key in required_root_keys:
            self.assertIn(key, self.metadata, f"Chave obrigatória '{key}' ausente no manifesto de metadados!")
            
        self.assertEqual(self.metadata['version'], "1.0.0")
        self.assertEqual(self.metadata['production_status'], "HOMOLOGATED_PRODUCTION_READY")
        self.assertEqual(len(self.metadata['input_contract']['numerical_features']), 17)
        self.assertEqual(len(self.metadata['input_contract']['all_categorical_features']), 6)


if __name__ == '__main__':
    unittest.main()
