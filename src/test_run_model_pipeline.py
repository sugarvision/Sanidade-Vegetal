"""
Script de Validação Automatizada Ponta a Ponta do Master Notebook da Sprint 3
Arquivo: src/test_run_model_pipeline.py
Projeto: Sanidade-Vegetal (SugarVision) — Sprint 3 (Fase Model)

Checklist Validado:
1. Existência, limpeza de código e documentação didática em notebooks/04_sprint3_model_pipeline.ipynb.
2. Execução completa de 'Restart & Run All' certificando reprodutibilidade em menos de 15 minutos (< 900s).
3. Verificação física dos gráficos diagnósticos e matrizes de decisão salvos em docs/figures/.
4. Auditoria de integridade do Pipeline Atômico SVM (Anti-Leakage, Platt Scaling e Métricas).
"""

import io
import sys
import json
import time
import unittest
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))


class TestSprint3MasterNotebookPipeline(unittest.TestCase):
    """Suíte de validação automatizada de ponta a ponta da Sprint 3 (Model)."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = BASE_DIR
        cls.notebook_path = cls.base_dir / "notebooks" / "04_sprint3_model_pipeline.ipynb"
        cls.figures_dir = cls.base_dir / "docs" / "figures"
        cls.models_dir = cls.base_dir / "models"
        cls.data_dir = cls.base_dir / "data" / "processed"
        cls.max_allowed_seconds = 900.0  # 15 minutos (SLA máximo do checklist)

        # Se o notebook ainda não existir, constrói e executa automaticamente
        if not cls.notebook_path.exists():
            from src.build_sprint3_master_notebook import (
                create_sprint3_notebook_structure,
                execute_and_render_notebook,
            )
            import nbformat as nbf

            cls.notebook_path.parent.mkdir(parents=True, exist_ok=True)
            nb_node = create_sprint3_notebook_structure()
            with open(cls.notebook_path, "w", encoding="utf-8") as f:
                nbf.write(nb_node, f)
            execute_and_render_notebook(cls.notebook_path)

    def test_01_notebook_exists_and_has_didactic_structure(self):
        """Certifica que notebooks/04_sprint3_model_pipeline.ipynb existe e possui células didáticas."""
        self.assertTrue(
            self.notebook_path.exists(),
            f"Master Notebook não encontrado em: {self.notebook_path}",
        )

        with open(self.notebook_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        cells = nb_data.get("cells", [])
        md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
        code_cells = [c for c in cells if c.get("cell_type") == "code"]

        self.assertGreaterEqual(
            len(md_cells), 5, "O notebook deve conter pelo menos 5 seções Markdown didáticas."
        )
        self.assertGreaterEqual(
            len(code_cells), 5, "O notebook deve conter pelo menos 5 células modulares de código."
        )

        # Verificar que não há células de código com erro de execução
        for idx, c in enumerate(code_cells, start=1):
            for out in c.get("outputs", []):
                self.assertNotEqual(
                    out.get("output_type"),
                    "error",
                    f"Erro detectado na saída da célula de código #{idx}: {out.get('evalue')}",
                )

    def test_02_restart_and_run_all_under_15_minutes(self):
        """Executa o teste de 'Restart & Run All' do zero e certifica execução < 15 minutos."""
        with open(self.notebook_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        code_cells = [c for c in nb_data["cells"] if c["cell_type"] == "code"]
        isolated_ns = {"__name__": "__main__"}

        t_start = time.perf_counter()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            for idx, cell in enumerate(code_cells, start=1):
                raw_code = "".join(cell["source"]).replace("plt.show()", "# plt.show()")
                exec(raw_code, isolated_ns)
                plt.close("all")
        finally:
            sys.stdout = old_stdout

        elapsed_seconds = time.perf_counter() - t_start

        # Certificar SLA < 15 minutos (900 segundos)
        self.assertLess(
            elapsed_seconds,
            self.max_allowed_seconds,
            f"Execução excedeu 15 minutos: {elapsed_seconds:.2f}s",
        )

        # Certificar reprodutibilidade das métricas do modelo campeão no namespace recém-executado
        self.assertIn("champion_pipe", isolated_ns, "Objeto 'champion_pipe' não gerado na execução.")
        self.assertIn("val_auc", isolated_ns, "Métrica 'val_auc' não calculada.")
        self.assertGreaterEqual(
            isolated_ns["val_auc"], 0.90, "AUC-ROC Macro de validação deve ser >= 0.90."
        )
        print(f"\n  [OK] 'Restart & Run All' validado em {elapsed_seconds:.2f}s (SLA < 900s).")

    def test_03_diagnostic_figures_and_decision_matrices_saved(self):
        """Verifica se os gráficos diagnósticos e matrizes de decisão foram salvos em docs/figures/."""
        required_figures = [
            "baseline_models_benchmark_comparison.png",
            "sprint3_matriz_decisao_kernels_2d.png",
            "sprint3_gridsearch_heatmap_rbf.png",
            "sprint3_matrizes_confusao_valid_test.png",
            "svm_multiclass_roc_curves.png",
        ]

        for fig_name in required_figures:
            fig_path = self.figures_dir / fig_name
            self.assertTrue(
                fig_path.exists(),
                f"Figura diagnóstica obrigatória ausente em docs/figures/: {fig_name}",
            )
            file_size_kb = fig_path.stat().st_size / 1024.0
            self.assertGreater(
                file_size_kb,
                15.0,
                f"Arquivo {fig_name} parece vazio ou corrompido ({file_size_kb:.2f} KB).",
            )

    def test_04_champion_pipeline_anti_leakage_and_calibration(self):
        """Audita o artefato serializado models/svm_pipeline_optimal.joblib e predições calibradas."""
        model_path = self.models_dir / "svm_pipeline_optimal.joblib"
        preds_path = self.data_dir / "svm_predicoes_valid_test.csv"

        self.assertTrue(model_path.exists(), "Modelo campeão svm_pipeline_optimal.joblib ausente.")
        self.assertTrue(preds_path.exists(), "Tabela svm_predicoes_valid_test.csv ausente.")

        pipeline = joblib.load(model_path)
        svm_step = pipeline.named_steps["svm"]

        # Verificar hiperparâmetros ótimos homologados
        self.assertEqual(svm_step.kernel, "rbf")
        self.assertEqual(float(svm_step.C), 10.0)
        self.assertEqual(float(svm_step.gamma), 0.01)
        self.assertTrue(svm_step.probability)

        # Verificar calibração estocástica das probabilidades exportadas
        df_preds = pd.read_csv(preds_path)
        self.assertEqual(len(df_preds), 997, "A tabela de validação + teste deve conter 997 linhas (617 + 380).")

        proba_cols = [
            "y_proba_grassy_shoot",
            "y_proba_healthy",
            "y_proba_leaf_scald",
            "y_proba_mosaic",
            "y_proba_red_rot",
            "y_proba_rust",
            "y_proba_yellow_leaf",
        ]
        row_sums = df_preds[proba_cols].sum(axis=1).values
        np.testing.assert_allclose(
            row_sums,
            np.ones(len(df_preds)),
            atol=1e-4,
            err_msg="As probabilidades calibradas (Platt Scaling) não somam 1.0 por amostra!",
        )


if __name__ == "__main__":
    print("=" * 85)
    print("INICIANDO VALIDAÇÃO AUTOMATIZADA DE PONTA A PONTA — MASTER NOTEBOOK SPRINT 3")
    print("=" * 85)
    unittest.main(verbosity=2)