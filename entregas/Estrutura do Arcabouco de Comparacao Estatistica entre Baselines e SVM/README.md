# 📊 Entrega Sprint 4: Estrutura do Arcabouço de Comparação Estatística entre Baselines e SVM

**Projeto:** Sanidade Vegetal (SugarVision) — Diagnóstico Fitopatológico em Cana-de-Açúcar  
**Sprint:** 4 — Framework SEMMA (Fase: **Assess** / Avaliação)  
**Cartão / Tarefa:** *Estruturar o arcabouço de comparação estatística entre os modelos baselines e as configurações otimizadas de SVM, avaliando significância estatística e ganho real de generalização.*  
**Responsável Técnico:** **Luis** (Engenheiro Responsável pelo Benchmark e Governança Analítica)  
**Status da Entrega:** ✅ **100% CONCLUÍDO E HOMOLOGADO**

---

## 📌 Checklist de Homologação Concluído

- [x] **Consolidar tabela unificada com os resultados de todos os modelos avaliados na Sprint 3 e Sprint 4.**
  - Portfólio completo com **10 modelos supervisionados** (Dummy, Regressão Logística L2 sem ponderação e ponderada, Árvore de Decisão rasa sem ponderação e ponderada, SVM Linear, SVM RBF baseline, SVM Polinomial baseline, SVM RBF ótimo e SVM Polinomial campeão do Grid Search).
  - Tabela estruturada consolidada em [`../../models/unified_models_benchmark_sprint3_sprint4.csv`](../../models/unified_models_benchmark_sprint3_sprint4.csv) com métricas de Validação Cruzada (5-fold), Holdout de Validação ($N=617$) e Holdout de Teste ($N=380$).

- [x] **Aplicar testes de hipótese estatística (ex: Wilcoxon pareado ou teste t) para comprovar a superioridade do SVM sobre os baselines.**
  - Testes paramétricos (**t-test pareado de Student**) e não-paramétricos (**teste de postos de Wilcoxon**) aplicados fold a fold ($k=5$).
  - Comprovação formal de rejeição de $H_0$ frente ao Dummy Classifier ($p = 0.000$, Cohen's $d = 55.85$), Árvores de Decisão ($p = 0.00017$, Cohen's $d = 6.04$) e Regressão Logística ($p = 0.0107$, Cohen's $d = 2.02$).
  - Resultados salvos em [`../../models/statistical_hypothesis_tests_summary.csv`](../../models/statistical_hypothesis_tests_summary.csv).

- [x] **Documentar a margem de erro e intervalos de confiança (95%) das principais métricas entre os folds.**
  - Cálculo de Erro Padrão ($SE$), Margem de Erro ($ME = t_{\text{crit}} \times SE$) e Intervalos de Confiança a 95% para $F_1\text{-Macro}$, Acurácia, Precisão e Recall.
  - O SVM Campeão atinge $F_1\text{-Macro}$ de **57.86% ± 0.81%** ($IC_{95\%} = [57.04\%, 58.67\%]$ e $CV = 1.13\%$) nos 5 folds de CV, contraindo para **57.61% ± 0.46%** na auditoria estocástica de 15 splits repetidos.
  - Resultados salvos em [`../../models/confidence_intervals_metrics_95.csv`](../../models/confidence_intervals_metrics_95.csv).

- [x] **Redigir a justificativa de governança analítica para adoção do modelo final selecionado.**
  - Documentação detalhada dos motivos de descarte do Dummy (piso nulo), da Árvore rasa (underfitting estrutural e colapso no teste para $49.8\%$) e da Regressão Logística (incapacidade de separar fronteiras espectrais e de textura não-lineares).
  - Homologação do **SVM Polinomial Campeão** para inferência rápida em borda/mobile (latência de $0.14\text{ ms}$) e do **SVM RBF Ótimo** como modelo de referência com discriminação perfeita de Ferrugem ($Recall = 1.000$).
  - Relatório completo em [`./relatorio_arcabouco_comparacao_estatistica_baselines_vs_svm.md`](./relatorio_arcabouco_comparacao_estatistica_baselines_vs_svm.md) e [`../../docs/relatorio_arcabouco_comparacao_estatistica_baselines_vs_svm.md`](../../docs/relatorio_arcabouco_comparacao_estatistica_baselines_vs_svm.md).

---

## 📂 Estrutura de Artefatos Desta Entrega

```text
entregas/Estrutura do Arcabouco de Comparacao Estatistica entre Baselines e SVM/
├── README.md                                                   # Este documento síntese da entrega
├── relatorio_arcabouco_comparacao_estatistica_baselines_vs_svm.md # Relatório técnico completo executivo
├── statistical_comparison_framework_metadata.json              # Manifesto formal de linhagem e metadados MLOps
└── figures/                                                    # Gráficos diagnósticos em alta resolução (300 DPI)
    ├── models_benchmark_unified_comparison.png                 # Comparativo CV vs Valid vs Test dos 10 modelos
    ├── confidence_intervals_error_margins_95.png               # Forest plot com intervalos de confiança e margens de erro
    ├── statistical_gain_hypothesis_tests.png                   # Tamanho de efeito Cohen's d e ganhos estatísticos
    └── pareto_efficiency_latency_vs_f1.png                     # Fronteira de Pareto de latência vs F1-Macro
```

---

## 🧪 Suíte de Testes Automatizados

A integridade do arcabouço é verificada pela suíte [`../../src/test_statistical_comparison_framework.py`](../../src/test_statistical_comparison_framework.py):

```bash
python -m unittest src/test_statistical_comparison_framework.py
```
* **Status:** `Ran 4 tests in 0.031s — OK` (100% de aprovação).
