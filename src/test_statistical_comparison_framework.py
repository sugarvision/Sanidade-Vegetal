"""
Suíte de Testes Automatizados - Arcabouço de Comparação Estatística (Sprint 4)
Arquivo: src/test_statistical_comparison_framework.py
Projeto: Sanidade-Vegetal (SugarVision)
Responsável Técnico: Luis (Engenheiro Responsável pelo Benchmark e Governança Analítica)

Cobertura Testada dos 4 Itens do Checklist Oficial:
1. Validação da Tabela Unificada de Benchmark (10 modelos das Sprints 3 e 4).
2. Validação dos Testes de Hipótese (Wilcoxon pareado e t-test pareado com p < 0.05).
3. Validação dos Intervalos de Confiança (95%) e Margem de Erro entre folds.
4. Validação da Governança MLOps, Metadados e Visualizações Diagnósticas.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
ENTREGAS_DIR = BASE_DIR / "entregas" / "Estrutura do Arcabouco de Comparacao Estatistica entre Baselines e SVM"


class TestStatisticalComparisonFramework(unittest.TestCase):
    """
    Testes unitários e de integração para homologação do arcabouço estatístico.
    """

    def setUp(self):
        self.unified_csv = MODELS_DIR / "unified_models_benchmark_sprint3_sprint4.csv"
        self.hypothesis_csv = MODELS_DIR / "statistical_hypothesis_tests_summary.csv"
        self.ci_csv = MODELS_DIR / "confidence_intervals_metrics_95.csv"
        self.metadata_json = MODELS_DIR / "statistical_comparison_framework_metadata.json"

    def test_checklist_item1_unified_benchmark_table_integrity(self):
        """
        [Checklist 1] Valida a tabela unificada contendo todos os 10 modelos supervisionados
        avaliados na Sprint 3 e Sprint 4 com métricas de CV, Validação e Teste.
        """
        self.assertTrue(self.unified_csv.exists(), f"Arquivo não encontrado: {self.unified_csv}")
        df = pd.read_csv(self.unified_csv)

        # 1. Total exato de modelos
        self.assertEqual(len(df), 10, f"Esperado exatamente 10 modelos na tabela unificada, encontrado {len(df)}")

        # 2. Verificar presença das principais famílias
        model_names = df["Modelo"].tolist()
        expected_baselines = [
            "Dummy Classifier (Stratified)",
            "Regressão Logística L2 (Sem Ponderação)",
            "Regressão Logística L2 (Ponderada)",
            "Árvore de Decisão rasa (Sem Ponderação)",
            "Árvore de Decisão rasa (Ponderada)",
            "SVM Linear Baseline",
            "SVM RBF Baseline",
            "SVM Polinomial Baseline",
            "SVM RBF Ótimo (C=10, gamma=0.01)",
            "SVM Polinomial Campeão GridSearch"
        ]
        for expected in expected_baselines:
            self.assertIn(expected, model_names, f"Modelo esperado '{expected}' ausente da tabela unificada")

        # 3. Verificar presença das colunas mandatórias
        required_cols = [
            "Ranking_Geral", "Modelo", "Familia", "Sprint",
            "CV_F1_Macro_Media", "CV_F1_Macro_Std", "CV_F1_Macro_Margem_Erro_95", "CV_F1_Macro_IC95",
            "CV_Acuracia_Media", "CV_Acuracia_Std",
            "Valid_Acuracia", "Valid_F1_Macro", "Valid_Latencia_Infer_ms",
            "Test_Acuracia", "Test_F1_Macro", "Test_Latencia_Infer_ms"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Coluna mandatória '{col}' ausente na tabela unificada")

        # 4. Validar domínio das métricas ([0, 1])
        for metric_col in ["CV_F1_Macro_Media", "CV_Acuracia_Media", "Valid_F1_Macro", "Test_F1_Macro"]:
            self.assertTrue((df[metric_col] >= 0.0).all() and (df[metric_col] <= 1.0).all(),
                            f"Métrica {metric_col} contém valores fora do intervalo [0, 1]")

    def test_checklist_item2_hypothesis_testing_significance(self):
        """
        [Checklist 2] Valida a aplicação de testes de hipótese pareados (teste t e Wilcoxon),
        comprovando superioridade estatística (p < 0.05 e Cohen's d > 0) do SVM sobre os baselines.
        """
        self.assertTrue(self.hypothesis_csv.exists(), f"Arquivo não encontrado: {self.hypothesis_csv}")
        df = pd.read_csv(self.hypothesis_csv)

        # 1. Deve comparar contra os outros 9 modelos
        self.assertEqual(len(df), 9, f"Esperado 9 comparações pareadas, encontrado {len(df)}")

        # 2. Rejeição inequívoca de H0 contra baselines ingênuo, lineares e árvores
        critical_baselines = [
            "Dummy Classifier (Stratified)",
            "Regressão Logística L2 (Sem Ponderação)",
            "Regressão Logística L2 (Ponderada)",
            "Árvore de Decisão rasa (Sem Ponderação)",
            "Árvore de Decisão rasa (Ponderada)",
            "SVM Polinomial Baseline"
        ]

        for base in critical_baselines:
            sub = df[df["Modelo_Comparado"] == base]
            self.assertFalse(sub.empty, f"Modelo comparado '{base}' não encontrado nos testes de hipótese")
            row = sub.iloc[0]

            # p-valor do teste t deve ser estatisticamente significante (p < 0.05)
            self.assertLess(row["p_value_F1_t_test"], 0.05,
                            f"p-valor do teste t para '{base}' não é significante: {row['p_value_F1_t_test']}")

            # Decisão formal deve rejeitar H0
            self.assertEqual(row["Decisao_H0_F1"], "Rejeita H0 (Superioridade Comprovada)",
                             f"Decisão de hipótese incorreta para '{base}': {row['Decisao_H0_F1']}")

            # Cohen's d deve indicar efeito positivo de grande magnitude (d > 0.8)
            self.assertGreater(row["Cohen_d_F1"], 0.8,
                               f"Cohen's d para '{base}' menor que 0.8: {row['Cohen_d_F1']}")

            # Ganho absoluto e relativo devem ser positivos
            self.assertGreater(row["Ganho_Abs_F1_Macro"], 0.0)
            self.assertGreater(row["Ganho_Rel_F1_Macro_Pct"], 0.0)

    def test_checklist_item3_confidence_intervals_and_error_margins(self):
        """
        [Checklist 3] Valida a documentação das margens de erro e intervalos de confiança (95%)
        para as principais métricas nos folds de validação cruzada.
        """
        self.assertTrue(self.ci_csv.exists(), f"Arquivo não encontrado: {self.ci_csv}")
        df = pd.read_csv(self.ci_csv)

        # 1. Verificar colunas de cálculo formal
        req_cols = [
            "Modelo", "Familia", "Metrica", "N_Folds", "Graus_Liberdade",
            "Nivel_Confianca", "Valor_Critico_t", "Media", "Desvio_Padrao",
            "Erro_Padrao_SE", "Margem_Erro_ME", "IC_Inferior", "IC_Superior",
            "IC_Formatado", "Media_com_ME_Formatada", "Coef_Variacao_Pct"
        ]
        for col in req_cols:
            self.assertIn(col, df.columns, f"Coluna de IC '{col}' ausente")

        # 2. Validar consistência matemática para cada linha: IC_Inferior <= Media <= IC_Superior
        for _, row in df.iterrows():
            self.assertLessEqual(row["IC_Inferior"], row["Media"],
                                 f"IC_Inferior maior que a Média para {row['Modelo']} - {row['Metrica']}")
            self.assertGreaterEqual(row["IC_Superior"], row["Media"],
                                    f"IC_Superior menor que a Média para {row['Modelo']} - {row['Metrica']}")
            # Margem de erro deve ser positiva
            self.assertGreater(row["Margem_Erro_ME"], 0.0,
                               f"Margem de erro deve ser positiva para {row['Modelo']}")

        # 3. Validar graus de liberdade para 5 folds (df = 4)
        cv5_rows = df[df["N_Folds"] == 5]
        self.assertTrue((cv5_rows["Graus_Liberdade"] == 4).all())
        self.assertTrue((cv5_rows["Valor_Critico_t"] == 2.7764).all())

    def test_checklist_item4_governance_metadata_and_figures(self):
        """
        [Checklist 4] Valida o manifesto MLOps com autoria estrita de Luis e a existência
        das 4 figuras diagnósticas em alta resolução (300 DPI).
        """
        self.assertTrue(self.metadata_json.exists(), f"Manifesto não encontrado: {self.metadata_json}")
        with open(self.metadata_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # 1. Autoria mandatória do Luis
        self.assertIn("Luis", meta.get("responsible", ""), "O responsável técnico deve ser explicitamente Luis")
        self.assertEqual(meta.get("sprint"), "Sprint 4 - Framework SEMMA (Fase: Assess - Relato Vol. 4)")
        self.assertEqual(meta.get("total_models_evaluated"), 10)

        # 2. Existência dos gráficos diagnósticos em docs/figures/
        expected_figures = [
            "models_benchmark_unified_comparison.png",
            "confidence_intervals_error_margins_95.png",
            "statistical_gain_hypothesis_tests.png",
            "pareto_efficiency_latency_vs_f1.png"
        ]
        for fig_name in expected_figures:
            fig_path = FIGURES_DIR / fig_name
            self.assertTrue(fig_path.exists(), f"Gráfico diagnóstico ausente: {fig_path}")
            self.assertGreater(fig_path.stat().st_size, 10000, f"Arquivo de figura muito pequeno: {fig_path}")


if __name__ == "__main__":
    unittest.main()
