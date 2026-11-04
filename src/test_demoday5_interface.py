"""
Testes Unitários Automatizados da Interface e Validação Operacional — Demoday 5
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 5 — Framework SEMMA (Fase: Assess & Deploy Readiness)
Responsável: Cesar (Lead Técnico & Visão Computacional)
"""

import json
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from src.vision_inference_engine import VisionInferenceEngine


class TestDemoday5InterfaceAndInference(unittest.TestCase):
    """Valida a estabilidade do motor de visão e a integridade da interface do Demoday 5."""

    @classmethod
    def setUpClass(cls):
        cls.engine = VisionInferenceEngine()
        cls.blind_dir = Path("data/demoday5_blind_samples")
        cls.manifest_path = cls.blind_dir / "manifest_teste_cego_demoday5.json"
        cls.app_file = Path("src/app_demoday5_sugarvision.py")

    def test_app_and_manifest_artifacts_exist(self):
        """Verifica a presença dos arquivos da aplicação e do manifesto de teste cego."""
        self.assertTrue(self.app_file.exists(), "Arquivo src/app_demoday5_sugarvision.py ausente!")
        self.assertTrue(self.blind_dir.exists(), "Diretório data/demoday5_blind_samples ausente!")
        self.assertTrue(self.manifest_path.exists(), "Manifesto do teste cego ausente!")

    def test_blind_manifest_structure(self):
        """Valida o esquema e a consistência do manifesto de amostras desafiadoras."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertIn("samples", manifest)
        self.assertEqual(len(manifest["samples"]), 6, "Devem existir 6 amostras no teste cego.")

        for sample in manifest["samples"]:
            self.assertIn("filename", sample)
            self.assertIn("expected_class", sample)
            self.assertIn("agronomic_label", sample)
            self.assertIn("severity_scale", sample)
            # Verifica existência física do arquivo
            sample_path = self.blind_dir / sample["filename"]
            self.assertTrue(sample_path.exists(), f"Amostra física {sample['filename']} não encontrada!")

    def test_feature_extraction_pipeline(self):
        """Verifica se a extração de atributos produz as 27 colunas exatas requeridas pelo modelo."""
        sample_img = Image.new("RGB", (400, 400), color=(50, 140, 45))
        df_feat, rgb_arr, size_kb = self.engine.extract_visual_features(sample_img)

        self.assertEqual(len(df_feat), 1)
        self.assertEqual(len(df_feat.columns), 27)

        expected_features = [
            "exg_index", "exr_index", "rg_ratio", "indice_clorose_necrose",
            "glcm_contrast", "glcm_homogeneity", "laplacian_var", "mean_hue"
        ]
        for feat in expected_features:
            self.assertIn(feat, df_feat.columns, f"Atributo {feat} ausente no dataframe extraído.")

    def test_texture_map_and_lesion_mask_generation(self):
        """Valida se o mapa GLCM e a máscara de segmentação geram imagens PIL RGB válidas."""
        dummy_rgb = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)

        texture_map = self.engine.generate_glcm_texture_map(dummy_rgb)
        self.assertIsInstance(texture_map, Image.Image)
        self.assertEqual(texture_map.size, (300, 300))
        self.assertEqual(texture_map.mode, "RGB")

        lesion_mask = self.engine.generate_lesion_segmentation_mask(dummy_rgb)
        self.assertIsInstance(lesion_mask, Image.Image)
        self.assertEqual(lesion_mask.size, (300, 300))
        self.assertEqual(lesion_mask.mode, "RGB")

    def test_end_to_end_diagnosis(self):
        """Executa o diagnóstico ponta a ponta e valida laudo, probabilidade calibrada e SLA."""
        sample_path = self.blind_dir / "amostra_01_ferrugem_severa_campo.jpg"
        result = self.engine.diagnose_image(sample_path)

        self.assertIn("consolidated_class", result)
        self.assertIn("specific_class", result)
        self.assertIn("confidence_pct", result)
        self.assertIn("delta_margin", result)
        self.assertIn("shannon_entropy", result)
        self.assertIn("sla_status", result)
        self.assertIn("all_probabilities", result)

        # Probabilidade calibrada deve estar no intervalo [0, 100]
        self.assertGreaterEqual(result["confidence_pct"], 0.0)
        self.assertLessEqual(result["confidence_pct"], 100.0)

        # Margem geométrica deve ser não-negativa
        self.assertGreaterEqual(result["delta_margin"], 0.0)

    def test_diagnosis_with_reference_features_rust(self):
        """Valida que amostras da galeria com features de referência diagnosticam Ferrugem com alta certeza."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        sample_rust = manifest["samples"][0]
        sample_path = self.blind_dir / sample_rust["filename"]
        result = self.engine.diagnose_image(sample_path, reference_features=sample_rust["reference_features"])

        self.assertEqual(result["consolidated_class"], "FERRUGEM (RUST)")
        self.assertEqual(result["specific_class"], "RUST")
        self.assertGreater(result["confidence_pct"], 90.0)
        self.assertEqual(result["sla_level"], "APPROVED")


if __name__ == "__main__":
    unittest.main()

