"""
Módulo de Testes Automatizados para Otimização e Paralelização Computacional (Sprint 4)
Projeto: Sanidade-Vegetal (SugarVision)
"""

import unittest
import json
from pathlib import Path
import pandas as pd


class TestComputationalEfficiency(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.models_dir = cls.base_dir / "models"
        cls.sprint_dir = cls.base_dir / "Otimizacao e Paralelizacao Computacional da Validacao Cruzada"
        cls.checkpoint_json = cls.models_dir / "gridsearch_checkpoint.json"
        cls.checkpoint_csv = cls.models_dir / "gridsearch_checkpoint_results.csv"
        cls.metrics_json = cls.models_dir / "computational_efficiency_metrics.json"
        cls.fig_profiling = cls.base_dir / "docs" / "figures" / "computational_efficiency_cpu_ram_profiling.png"
        cls.fig_kernels = cls.base_dir / "docs" / "figures" / "kernel_execution_time_comparison.png"
        cls.fig_timeline = cls.base_dir / "docs" / "figures" / "checkpoint_resilience_timeline.png"

    def test_artifacts_existence(self):
        """Valida que todos os artefatos de governança e figuras foram gerados."""
        self.assertTrue(self.checkpoint_json.exists(), "JSON de checkpoint não encontrado!")
        self.assertTrue(self.checkpoint_csv.exists(), "CSV de checkpoint não encontrado!")
        self.assertTrue(self.metrics_json.exists(), "JSON de métricas de eficiência não encontrado!")
        self.assertTrue(self.fig_profiling.exists(), "Figura de perfilamento CPU/RAM não encontrada!")
        self.assertTrue(self.fig_kernels.exists(), "Figura de comparação de kernels não encontrada!")
        self.assertTrue(self.fig_timeline.exists(), "Figura de timeline de checkpoint não encontrada!")

    def test_checkpoint_content_and_kernels(self):
        """Valida que o checkpoint contém os 27 candidatos, incluindo RBF, Poly e Linear."""
        df = pd.read_csv(self.checkpoint_csv)
        self.assertEqual(len(df), 27, "Quantidade de candidatos divergente de 27!")
        kernels = set(df['kernel'].unique())
        self.assertIn('rbf', kernels, "Kernel RBF ausente no checkpoint!")
        self.assertIn('poly', kernels, "Kernel Poly ausente no checkpoint!")
        self.assertIn('linear', kernels, "Kernel Linear ausente no checkpoint!")
        
        # Verificar que os graus 2 e 3 de Poly foram executados
        poly_degrees = set(df[df['kernel'] == 'poly']['degree'].dropna().astype(int).unique())
        self.assertTrue({2, 3}.issubset(poly_degrees), "Graus 2 e 3 do Poly não encontrados!")

    def test_parallelization_and_telemetry_metrics(self):
        """Valida configuração n_jobs=-1 e métricas de telemetria de hardware."""
        with open(self.metrics_json, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        self.assertEqual(meta['parallelization_profile']['n_jobs'], -1)
        self.assertEqual(meta['parallelization_profile']['backend'], "loky")
        self.assertGreater(meta['parallelization_profile']['overall_throughput_fits_per_sec'], 0.5)
        
        # Telemetria de CPU e RAM
        telem = meta['telemetry_metrics']
        self.assertGreater(telem['peak_cpu_percent'], 50.0, "Pico de CPU abaixo do esperado para n_jobs=-1!")
        self.assertGreater(telem['peak_process_ram_mb'], 100.0, "Consumo de RAM não registrado!")

    def test_resilience_recovery(self):
        """Valida a funcionalidade de recuperação contra falhas."""
        with open(self.checkpoint_json, 'r', encoding='utf-8') as f:
            cp_data = json.load(f)
            
        self.assertEqual(cp_data['total_completed'], 27)
        self.assertEqual(len(cp_data['completed_candidates']), 27)


if __name__ == '__main__':
    unittest.main()
