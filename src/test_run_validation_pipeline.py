"""
Suíte de Testes Automatizados Ponta a Ponta do Master Notebook Oficial da Sprint 4 (Fase Assess — SEMMA)
Arquivo: src/test_run_validation_pipeline.py
Projeto: Sanidade-Vegetal (SugarVision) — Sprint 4 (Fase Assess)

Checklist Validado:
1. Criar o notebook oficial da Sprint 4 (notebooks/05_sprint4_assess_validation.ipynb) documentado com células no padrão SEMMA (Assess).
2. Desenvolver o script de testes src/test_run_validation_pipeline.py para validar conformidade dos resultados e métricas de generalização.
3. Garantir execução fluida e reproduzível via "Restart & Run All" sem falhas de biblioteca ou dependências.
4. Automatizar a exportação de todos os relatórios tabulares em formato CSV e Markdown para a pasta docs/.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import unittest
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib_cache"))
(BASE_DIR / ".matplotlib_cache").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_REPORT_STEMS = [
    "sprint4_metricas_globais_generalizacao",
    "sprint4_metricas_por_classe_valid_test",
    "sprint4_matriz_confusao_agronomica_3classes",
    "sprint4_calibracao_limiares_threshold_tuning",
    "sprint4_benchmark_unificado_modelos",
    "sprint4_testes_hipotese_pareados",
    "sprint4_intervalos_confianca_ic95",
    "sprint4_auditoria_repeated_kfold_estabilidade",
    "sprint4_auditoria_transicoes_erros_teste",
    "sprint4_matriz_custo_agronomico_7x7",
    "sprint4_ranking_importancia_features",
    "sprint4_importancia_por_familia_funcional",
    "sprint4_padroes_visuais_falha_fronteira",
    "sprint4_catalogo_relatorios_tabulares",
]


class TestSprint4AssessValidationPipeline(unittest.TestCase):
    """Suíte de validação automatizada de ponta a ponta da Sprint 4 (Fase Assess — SEMMA)."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = BASE_DIR
        cls.notebook_path = cls.base_dir / "notebooks" / "05_sprint4_assess_validation.ipynb"
        cls.docs_dir = cls.base_dir / "docs"
        cls.figures_dir = cls.docs_dir / "figures"
        cls.models_dir = cls.base_dir / "models"
        cls.data_dir = cls.base_dir / "data" / "processed"
        cls.max_allowed_seconds = 900.0  # Limite máximo de 15 minutos para Restart & Run All

        # Se o notebook ainda não existir, constrói e executa automaticamente
        if not cls.notebook_path.exists():
            from src.build_sprint4_assess_notebook import (
                create_sprint4_assess_notebook_structure,
                execute_and_render_notebook,
            )
            import nbformat as nbf

            cls.notebook_path.parent.mkdir(parents=True, exist_ok=True)
            nb_node = create_sprint4_assess_notebook_structure()
            with open(cls.notebook_path, "w", encoding="utf-8") as f:
                nbf.write(nb_node, f)
            execute_and_render_notebook(cls.notebook_path)

    def test_01_notebook_exists_and_follows_semma_assess_standard(self):
        """[Checklist 1] Certifica que notebooks/05_sprint4_assess_validation.ipynb existe e segue o padrão SEMMA (Assess)."""
        self.assertTrue(
            self.notebook_path.exists(),
            f"Notebook oficial da Sprint 4 não encontrado em: {self.notebook_path}",
        )

        with open(self.notebook_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        cells = nb_data.get("cells", [])
        md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
        code_cells = [c for c in cells if c.get("cell_type") == "code"]

        self.assertGreaterEqual(
            len(md_cells), 8, "O notebook deve conter pelo menos 8 células Markdown documentadas no padrão SEMMA (Assess)."
        )
        self.assertGreaterEqual(
            len(code_cells), 8, "O notebook deve conter pelo menos 8 células modulares de código executável."
        )

        # Verificar presença dos cabeçalhos metodológicos SEMMA Assess
        full_md_text = "\n".join("".join(c.get("source", [])) for c in md_cells)
        for required_tag in [
            "SEMMA",
            "ASSESS 1.0",
            "ASSESS 2.0",
            "ASSESS 3.0",
            "ASSESS 4.0",
            "ASSESS 5.0",
            "ASSESS 6.0",
            "ASSESS 7.0",
        ]:
            self.assertIn(
                required_tag,
                full_md_text,
                f"Marcador metodológico obrigatório '{required_tag}' ausente na documentação do notebook.",
            )

        # Certificar que todas as células de código foram executadas e não possuem erros
        for idx, c in enumerate(code_cells, start=1):
            self.assertIsNotNone(
                c.get("execution_count"),
                f"Célula de código #{idx} não possui execution_count (não foi executada).",
            )
            outputs = c.get("outputs", [])
            self.assertGreater(
                len(outputs), 0, f"Célula de código #{idx} não possui saídas renderizadas."
            )
            for out in outputs:
                self.assertNotEqual(
                    out.get("output_type"),
                    "error",
                    f"Erro detectado na saída da célula #{idx}: {out.get('evalue')}",
                )

    def test_02_restart_and_run_all_reproducible_execution(self):
        """[Checklist 2 e 3] Executa 'Restart & Run All' em ambiente isolado sem falhas de biblioteca e valida reprodutibilidade."""
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

        self.assertLess(
            elapsed_seconds,
            self.max_allowed_seconds,
            f"Execução de 'Restart & Run All' excedeu o SLA de 900s: {elapsed_seconds:.2f}s",
        )

        # Verificar objetos fundamentais produzidos em memória
        expected_objects = [
            "df_global_metrics",
            "df_per_class",
            "df_agro_3class",
            "df_thresh_summary",
            "df_bench",
            "df_hypothesis",
            "df_ci95",
            "df_rkf_summary",
            "df_err_transitions",
            "df_cost_matrix",
            "df_perm_imp",
            "df_grp_imp",
            "df_vis_err",
            "df_Catalog_Exports",
        ]
        for obj_name in expected_objects:
            self.assertIn(
                obj_name,
                isolated_ns,
                f"DataFrame '{obj_name}' não foi gerado durante o 'Restart & Run All'.",
            )
            self.assertFalse(
                isolated_ns[obj_name].empty,
                f"DataFrame '{obj_name}' está vazio após a execução.",
            )

    def test_03_generalization_metrics_and_calibration_conformity(self):
        """[Checklist 2] Audita a conformidade matemática das métricas de generalização e da calibração de Platt."""
        # 1. Auditoria da tabela de predições calibradas
        preds_csv = self.data_dir / "svm_predicoes_valid_test.csv"
        self.assertTrue(preds_csv.exists(), "Tabela svm_predicoes_valid_test.csv ausente.")
        df_preds = pd.read_csv(preds_csv)
        self.assertEqual(len(df_preds), 997, "Esperado 997 instâncias (617 valid + 380 test).")

        proba_cols = [
            "y_proba_grassy_shoot",
            "y_proba_healthy",
            "y_proba_leaf_scald",
            "y_proba_mosaic",
            "y_proba_red_rot",
            "y_proba_rust",
            "y_proba_yellow_leaf",
        ]
        probas = df_preds[proba_cols].values
        self.assertTrue((probas >= 0.0).all() and (probas <= 1.0).all(), "Probabilidades fora de [0, 1].")
        np.testing.assert_allclose(
            probas.sum(axis=1),
            np.ones(len(df_preds)),
            atol=1e-4,
            err_msg="As probabilidades calibradas (Platt Scaling) não somam 1.0!",
        )

        # 2. Auditoria das métricas globais exportadas em docs/
        glob_csv = self.docs_dir / "sprint4_metricas_globais_generalizacao.csv"
        self.assertTrue(glob_csv.exists(), "Relatório sprint4_metricas_globais_generalizacao.csv ausente.")
        df_glob = pd.read_csv(glob_csv)
        self.assertEqual(len(df_glob), 3, "Deve conter 3 partições: Validação, Teste Cego e Combinada.")

        val_row = df_glob[df_glob["Particao"].str.contains("Validacao")].iloc[0]
        tst_row = df_glob[df_glob["Particao"].str.contains("Teste Cego")].iloc[0]

        # Conformidade estrita com os patamares homologados na Sprint 4
        self.assertAlmostEqual(float(val_row["AUC_ROC_Macro"]), 0.9148, places=3)
        self.assertAlmostEqual(float(val_row["AUC_ROC_Micro"]), 0.9498, places=3)
        self.assertAlmostEqual(float(val_row["Acuracia_Balanceada"]), 0.6532, places=3)
        self.assertAlmostEqual(float(val_row["F1_Macro"]), 0.5757, places=3)
        self.assertGreaterEqual(float(val_row["Especificidade_Macro"]), 0.93)

        self.assertAlmostEqual(float(tst_row["AUC_ROC_Macro"]), 0.9116, places=3)
        self.assertAlmostEqual(float(tst_row["AUC_ROC_Micro"]), 0.9405, places=3)
        self.assertAlmostEqual(float(tst_row["Acuracia_Balanceada"]), 0.6769, places=3)
        self.assertAlmostEqual(float(tst_row["F1_Macro"]), 0.6054, places=3)
        self.assertGreaterEqual(float(tst_row["Especificidade_Macro"]), 0.93)

        # 3. Auditoria de separabilidade perfeita nas classes críticas HEALTHY e RUST
        per_cls_csv = self.docs_dir / "sprint4_metricas_por_classe_valid_test.csv"
        df_cls = pd.read_csv(per_cls_csv)
        critical_rows = df_cls[df_cls["Patologia"].isin(["HEALTHY", "RUST"])]
        self.assertEqual(len(critical_rows), 4, "Devem existir 4 registros (HEALTHY e RUST em Valid e Test).")
        for _, row in critical_rows.iterrows():
            self.assertEqual(float(row["Sensibilidade_Recall"]), 1.0)
            self.assertEqual(float(row["Especificidade_TNR"]), 1.0)
            self.assertEqual(float(row["Precisao"]), 1.0)
            self.assertEqual(float(row["F1_Score"]), 1.0)
            self.assertEqual(float(row["AUC_ROC_OvR"]), 1.0)
            self.assertEqual(float(row["Average_Precision_AP"]), 1.0)
            self.assertEqual(int(row["FN"]), 0)
            self.assertEqual(int(row["FP"]), 0)

        # 4. Auditoria da Matriz de Confusão Agronômica de 3 Classes
        agro_csv = self.docs_dir / "sprint4_matriz_confusao_agronomica_3classes.csv"
        df_agro = pd.read_csv(agro_csv)
        self.assertEqual(len(df_agro), 6)
        self.assertTrue((df_agro["Sensibilidade_Grupo"] == 1.0).all(), "Sensibilidade no agrupamento agronômico 3 classes deve ser 100%.")

    def test_04_statistical_comparison_and_repeated_kfold_stability(self):
        """[Checklist 2] Valida benchmark unificado, testes de hipótese pareados, IC 95% e Repeated K-Fold."""
        df_bench = pd.read_csv(self.docs_dir / "sprint4_benchmark_unificado_modelos.csv")
        df_hyp = pd.read_csv(self.docs_dir / "sprint4_testes_hipotese_pareados.csv")
        df_ci = pd.read_csv(self.docs_dir / "sprint4_intervalos_confianca_ic95.csv")
        df_rkf = pd.read_csv(self.docs_dir / "sprint4_auditoria_repeated_kfold_estabilidade.csv")

        self.assertEqual(len(df_bench), 10, "O benchmark unificado deve conter exatamente 10 modelos.")
        self.assertEqual(len(df_hyp), 9, "A tabela de testes pareados deve conter 9 comparações.")

        # Validar consistência de IC 95%: IC_Inferior <= Media <= IC_Superior e Margem_Erro_ME > 0
        for _, row in df_ci.iterrows():
            self.assertLessEqual(float(row["IC_Inferior"]), float(row["Media"]))
            self.assertGreaterEqual(float(row["IC_Superior"]), float(row["Media"]))
            self.assertGreater(float(row["Margem_Erro_ME"]), 0.0)

        # Validar estabilidade do RepeatedStratifiedKFold (15 splits): CV de F1-Macro < 2.0%
        global_rkf = df_rkf[df_rkf["Escopo_Avaliacao"].str.contains("Consolidado Global")].iloc[0]
        self.assertEqual(int(global_rkf["N_Splits"]), 15)
        self.assertLess(float(global_rkf["F1_Macro_CV_Pct"]), 2.0)
        self.assertAlmostEqual(float(global_rkf["F1_Macro_Media"]), 0.5761, places=3)

    def test_05_agronomic_error_audit_and_interpretability_conformity(self):
        """[Checklist 2] Valida auditoria de erros no teste (N=149), matriz de custo 7x7 e importância de features."""
        df_err = pd.read_csv(self.docs_dir / "sprint4_auditoria_transicoes_erros_teste.csv")
        df_cost = pd.read_csv(self.docs_dir / "sprint4_matriz_custo_agronomico_7x7.csv")
        df_imp = pd.read_csv(self.docs_dir / "sprint4_ranking_importancia_features.csv")
        df_grp = pd.read_csv(self.docs_dir / "sprint4_importancia_por_familia_funcional.csv")

        # Total de erros no teste cego (N=380) deve somar exatamente 149 e ter 0 omissões de RUST
        self.assertEqual(int(df_err["error_count"].sum()), 149)
        self.assertFalse((df_err["true_class_name"] == "RUST").any(), "Não deve haver nenhum Falso Negativo de RUST.")
        self.assertFalse((df_err["true_class_name"] == "HEALTHY").any(), "Não deve haver nenhum Falso Positivo em folha HEALTHY.")

        # Matriz de custo 7x7 deve ter diagonal zero e penalidade máxima em FN de RUST -> HEALTHY (R$ 10.017,25/ha)
        self.assertEqual(len(df_cost), 7)
        rust_row = df_cost[df_cost["Classe_Real_vs_Predita"] == "RUST"].iloc[0]
        self.assertAlmostEqual(float(rust_row["HEALTHY"]), 10017.25, places=2)
        self.assertEqual(float(rust_row["RUST"]), 0.0)

        # Ranking de importância deve cobrir as 27 features e soma dos shares por grupo = 100%
        self.assertEqual(len(df_imp), 27)
        self.assertAlmostEqual(float(df_grp["share_of_importance_pct"].sum()), 100.0, delta=0.5)

    def test_06_automated_export_of_all_tabular_reports_csv_and_md_in_docs(self):
        """[Checklist 4] Verifica se todos os relatórios tabulares foram exportados em CSV e Markdown para docs/."""
        for stem in EXPECTED_REPORT_STEMS:
            csv_file = self.docs_dir / f"{stem}.csv"
            md_file = self.docs_dir / f"{stem}.md"

            self.assertTrue(csv_file.exists(), f"Relatório CSV obrigatório ausente em docs/: {csv_file.name}")
            self.assertTrue(md_file.exists(), f"Relatório Markdown obrigatório ausente em docs/: {md_file.name}")

            self.assertGreater(csv_file.stat().st_size, 100, f"Arquivo CSV vazio ou truncado: {csv_file.name}")
            self.assertGreater(md_file.stat().st_size, 200, f"Arquivo Markdown vazio ou truncado: {md_file.name}")

            df_check = pd.read_csv(csv_file)
            self.assertGreater(len(df_check), 0, f"Tabela CSV '{csv_file.name}' não possui linhas.")

            md_text = md_file.read_text(encoding="utf-8")
            self.assertIn("| --- |", md_text, f"Tabela pipe Markdown ausente em '{md_file.name}'.")

        # Verificar dossiê consolidado e figuras geradas pelo notebook
        consolidated_md = self.docs_dir / "relatorio_consolidado_tabelas_sprint4_assess.md"
        self.assertTrue(consolidated_md.exists(), "Dossiê consolidado relatorio_consolidado_tabelas_sprint4_assess.md ausente.")

        expected_figs = [
            "sprint4_notebook_assess_discriminacao_multiclasse.png",
            "sprint4_notebook_threshold_tuning_curves.png",
            "sprint4_notebook_ic95_and_rkfold_stability.png",
            "sprint4_notebook_feature_importance_summary.png",
        ]
        for fig_name in expected_figs:
            fig_path = self.figures_dir / fig_name
            self.assertTrue(fig_path.exists(), f"Figura diagnóstica ausente em docs/figures/: {fig_name}")
            self.assertGreater(fig_path.stat().st_size, 15000, f"Figura diagnóstica inválida: {fig_name}")


if __name__ == "__main__":
    print("=" * 90)
    print("INICIANDO SUÍTE DE TESTES AUTOMATIZADOS — SPRINT 4 (NOTEBOOK 05 — FASE ASSESS SEMMA)")
    print("=" * 90)
    unittest.main(verbosity=2)
