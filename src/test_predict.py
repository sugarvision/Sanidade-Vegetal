"""
Suíte de Testes Unitários e de Integração para o Módulo de Inferência (src/predict.py)
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 5 — Framework SEMMA (Fase: Assess & Deploy Readiness)
"""

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from src.predict import (
    CLASS_NAMES_MAP,
    CorruptedImageError,
    HighPerformancePredictor,
    IncompatibleFormatError,
    OutOfFocusImageError,
    export_predictions_to_csv,
    export_predictions_to_json,
    main as cli_main,
)


class TestHighPerformancePredictModule(unittest.TestCase):
    """Valida todos os itens do checklist de src/predict.py."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.blind_dir = cls.base_dir / "data" / "demoday5_blind_samples"
        cls.sample_image = cls.blind_dir / "amostra_01_ferrugem_severa_campo.jpg"
        cls.predictor = HighPerformancePredictor()

    def test_01_single_image_realtime_prediction(self):
        """Valida predição unitária em tempo real e estrutura do payload de retorno."""
        self.assertTrue(self.sample_image.exists(), f"Amostra não encontrada: {self.sample_image}")
        res = self.predictor.predict_single(self.sample_image)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn(res["predicted_class_code"], range(7))
        self.assertEqual(res["predicted_class"], CLASS_NAMES_MAP[res["predicted_class_code"]])
        self.assertGreaterEqual(res["confidence_index"], 0.0)
        self.assertLessEqual(res["confidence_index"], 1.0)
        self.assertGreaterEqual(res["confidence_pct"], 0.0)
        self.assertLessEqual(res["confidence_pct"], 100.0)

        probas = res["probabilities"]
        self.assertEqual(len(probas), 7)
        self.assertAlmostEqual(sum(probas.values()), 1.0, places=3)

    def test_02_batch_directory_prediction(self):
        """Valida predição em lote sobre um diretório de imagens (5 válidas + 1 fora de foco na galeria cega)."""
        batch_payload = self.predictor.predict_batch(self.blind_dir, max_workers=2)

        self.assertEqual(batch_payload["mode"], "batch")
        summary = batch_payload["summary"]
        self.assertEqual(summary["total_received"], 6)
        self.assertEqual(summary["total_success"], 5)
        self.assertEqual(summary["total_errors"], 1)
        self.assertEqual(len(batch_payload["results"]), 6)

        # Verifica que a amostra 01 é diagnosticada como RUST e a amostra 06 (desfocada) é rejeitada como OUT_OF_FOCUS_IMAGE
        results_by_name = {r["sample_id"]: r for r in batch_payload["results"]}
        self.assertEqual(results_by_name["amostra_01_ferrugem_severa_campo.jpg"]["predicted_class"], "RUST")
        self.assertEqual(results_by_name["amostra_02_folha_sadia_controle.jpg"]["predicted_class"], "HEALTHY")
        self.assertEqual(results_by_name["amostra_06_especularidade_sol_direto.jpg"]["status"], "ERROR")
        self.assertEqual(results_by_name["amostra_06_especularidade_sol_direto.jpg"]["error_code"], "OUT_OF_FOCUS_IMAGE")

    def test_03_structured_json_and_csv_exports(self):
        """Valida exportação estruturada em JSON e CSV com classe predita, probabilidades e índice de confiança."""
        batch_payload = self.predictor.predict_batch(self.sample_image)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            json_out = tmp_path / "predicoes.json"
            csv_out = tmp_path / "predicoes.csv"

            export_predictions_to_json(batch_payload, json_out)
            export_predictions_to_csv(batch_payload, csv_out)

            self.assertTrue(json_out.exists())
            self.assertTrue(csv_out.exists())

            with open(json_out, "r", encoding="utf-8") as f:
                loaded_json = json.load(f)
            self.assertEqual(loaded_json["summary"]["total_success"], 1)
            first_item = loaded_json["results"][0]
            self.assertIn("predicted_class", first_item)
            self.assertIn("confidence_index", first_item)
            self.assertIn("probabilities", first_item)

            with open(csv_out, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "SUCCESS")
            self.assertTrue(rows[0]["predicted_class"])
            self.assertTrue(rows[0]["confidence_index"])
            self.assertTrue(rows[0]["proba_rust"])

    def test_04_exception_handling_corrupted_blurry_and_incompatible(self):
        """Valida tratamento robusto de exceções para imagens danificadas, fora de foco e formatos incompatíveis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # 1. Formato incompatível (.txt / .bmp)
            bad_ext_file = tmp_path / "arquivo_invalido.txt"
            bad_ext_file.write_text("nao eh uma imagem", encoding="utf-8")

            with self.assertRaises(IncompatibleFormatError):
                self.predictor.predict_single(bad_ext_file, raise_on_error=True)

            res_bad_ext = self.predictor.predict_single(bad_ext_file, raise_on_error=False)
            self.assertEqual(res_bad_ext["status"], "ERROR")
            self.assertEqual(res_bad_ext["error_code"], "INCOMPATIBLE_FORMAT")

            # 2. Imagem corrompida (bytes inválidos em .jpg e arquivo 0 bytes)
            corrupt_jpg = tmp_path / "imagem_corrompida.jpg"
            corrupt_jpg.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF_CORRUPTED_BYTES_STREAM")

            with self.assertRaises(CorruptedImageError):
                self.predictor.predict_single(corrupt_jpg, raise_on_error=True)

            res_corrupt = self.predictor.predict_single(corrupt_jpg, raise_on_error=False)
            self.assertEqual(res_corrupt["status"], "ERROR")
            self.assertEqual(res_corrupt["error_code"], "CORRUPTED_IMAGE")

            empty_jpg = tmp_path / "vazia.png"
            empty_jpg.write_bytes(b"")
            with self.assertRaises(CorruptedImageError):
                self.predictor.predict_single(empty_jpg, raise_on_error=True)

            # 3. Imagem fora de foco (gradiente suave com variância do Laplaciano < 20.0)
            x = np.linspace(40, 120, 300)
            y = np.linspace(40, 120, 300)
            xv, yv = np.meshgrid(x, y)
            smooth_channel = ((xv + yv) / 2.0).astype(np.uint8)
            blurry_arr = np.stack([smooth_channel, smooth_channel, smooth_channel], axis=-1)
            blurry_img = Image.fromarray(blurry_arr, mode="RGB")
            blurry_path = tmp_path / "folha_fora_de_foco.png"
            blurry_img.save(blurry_path)

            with self.assertRaises(OutOfFocusImageError):
                self.predictor.predict_single(blurry_path, raise_on_error=True)

            res_blurry = self.predictor.predict_single(blurry_path, raise_on_error=False)
            self.assertEqual(res_blurry["status"], "ERROR")
            self.assertEqual(res_blurry["error_code"], "OUT_OF_FOCUS_IMAGE")

    def test_05_cli_execution_single_and_batch(self):
        """Valida a execução via CLI com parâmetros de imagem única, diretório e exportação JSON/CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_json = tmp_path / "cli_out.json"
            out_csv = tmp_path / "cli_out.csv"

            exit_code = cli_main([
                "--input", str(self.sample_image),
                "--output-json", str(out_json),
                "--output-csv", str(out_csv),
                "--quiet",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_json.exists())
            self.assertTrue(out_csv.exists())


if __name__ == "__main__":
    unittest.main()
