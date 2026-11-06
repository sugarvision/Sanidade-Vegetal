# 📊 Benchmark Unificado de Modelos Supervisionados — Baselines vs. SVM (Sprints 3 e 4)

**Projeto:** Sanidade-Vegetal (*SugarVision*) | **Sprint:** 4 | **Fase SEMMA:** `Assess 4.0 — Benchmark Unificado`  
**Arquivo CSV Pareado:** `sprint4_benchmark_unificado_modelos.csv` | **Dimensões:** `10 linhas x 27 colunas`

> Tabela comparativa consolidada dos 10 modelos avaliados em Validação Cruzada (5-Fold), Holdout de Validação (N=617), Teste Cego (N=380) e latência de inferência em milissegundos.

## Tabela de Dados Auditada

| Ranking_Geral | Modelo | Familia | Sprint | CV_F1_Macro_Media | CV_F1_Macro_Std | CV_F1_Macro_Margem_Erro_95 | CV_F1_Macro_IC95 | CV_Acuracia_Media | CV_Acuracia_Std | CV_F1_Weighted_Media | CV_Precisao_Macro_Media | CV_Recall_Macro_Media | CV_Coef_Variacao_Pct | Tempo_Treino_CV_Segundos | Valid_Acuracia | Valid_F1_Macro | Valid_F1_Weighted | Valid_Precisao_Macro | Valid_Recall_Macro | Valid_Latencia_Infer_ms | Test_Acuracia | Test_F1_Macro | Test_F1_Weighted | Test_Precisao_Macro | Test_Recall_Macro | Test_Latencia_Infer_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Regressão Logística L2 (Sem Ponderação) | Baseline Linear | Sprint 3 | 0.5595 | 0.0143 | 0.0177 | [54.18%, 57.73%] | 0.6500 | 0.0123 | 0.6117 | 0.6238 | 0.6001 | 2.5500 | 2.4200 | 0.6807 | 0.6240 | 0.6624 | 0.6961 | 0.6557 | 0.0010 | 0.6684 | 0.6468 | 0.6450 | 0.6719 | 0.6779 | 0.0010 |
| 2 | Regressão Logística L2 (Ponderada) | Baseline Linear | Sprint 3 | 0.5690 | 0.0066 | 0.0082 | [56.09%, 57.72%] | 0.6358 | 0.0102 | 0.5888 | 0.5709 | 0.6317 | 1.1600 | 3.0200 | 0.6175 | 0.5837 | 0.5745 | 0.5926 | 0.6528 | 0.0010 | 0.6237 | 0.6285 | 0.5786 | 0.6537 | 0.6907 | 0.0010 |
| 3 | SVM Polinomial Baseline | SVM Polinomial | Sprint 3 | 0.5694 | 0.0058 | 0.0072 | [56.22%, 57.66%] | 0.6394 | 0.0077 | 0.5885 | 0.5978 | 0.6343 | 1.0100 | 3.1300 | 0.6175 | 0.5740 | 0.5635 | 0.6049 | 0.6474 | 0.1240 | 0.6079 | 0.6081 | 0.5549 | 0.6564 | 0.6748 | 0.1230 |
| 4 | SVM Linear Baseline | SVM Linear | Sprint 3 | 0.5709 | 0.0093 | 0.0116 | [55.92%, 58.25%] | 0.6439 | 0.0082 | 0.5879 | 0.5617 | 0.6391 | 1.6400 | 13.5600 | 0.6240 | 0.5742 | 0.5627 | 0.5320 | 0.6520 | 0.1200 | 0.6132 | 0.6063 | 0.5494 | 0.5635 | 0.6791 | 0.1210 |
| 5 | SVM RBF Baseline | SVM RBF | Sprint 3 | 0.5759 | 0.0067 | 0.0084 | [56.75%, 58.42%] | 0.6441 | 0.0097 | 0.5907 | 0.6022 | 0.6429 | 1.1700 | 4.5600 | 0.6159 | 0.5707 | 0.5583 | 0.5301 | 0.6472 | 0.4040 | 0.6079 | 0.6060 | 0.5508 | 0.6332 | 0.6744 | 0.4720 |
| 6 | SVM RBF Ótimo (C=10, gamma=0.01) | SVM RBF Otimizado | Sprint 3/4 | 0.5774 | 0.0086 | 0.0107 | [56.67%, 58.80%] | 0.6453 | 0.0097 | 0.5905 | 0.5904 | 0.6460 | 1.4900 | 4.3600 | 0.6207 | 0.5757 | 0.5624 | 0.5346 | 0.6532 | 0.3340 | 0.6079 | 0.6054 | 0.5475 | 0.5637 | 0.6769 | 0.3350 |
| 7 | SVM Polinomial Campeão GridSearch | SVM Polinomial Otimizado | Sprint 4 | 0.5786 | 0.0065 | 0.0081 | [57.04%, 58.67%] | 0.6453 | 0.0091 | 0.5901 | 0.5773 | 0.6490 | 1.1300 | 2.9800 | 0.6207 | 0.5747 | 0.5620 | 0.5336 | 0.6524 | 0.1520 | 0.6079 | 0.6048 | 0.5473 | 0.5632 | 0.6763 | 0.1230 |
| 8 | Árvore de Decisão rasa (Sem Ponderação) | Baseline Árvore | Sprint 3 | 0.5311 | 0.0132 | 0.0164 | [51.47%, 54.76%] | 0.6290 | 0.0140 | 0.5903 | 0.5572 | 0.5759 | 2.4900 | 0.3000 | 0.6386 | 0.5735 | 0.6147 | 0.5981 | 0.6147 | 0.0010 | 0.6421 | 0.6021 | 0.6034 | 0.5895 | 0.6534 | 0.0010 |
| 9 | Árvore de Decisão rasa (Ponderada) | Baseline Árvore | Sprint 3 | 0.5334 | 0.0252 | 0.0313 | [50.21%, 56.47%] | 0.6057 | 0.0178 | 0.5370 | 0.5369 | 0.6109 | 4.7300 | 0.2500 | 0.5284 | 0.4623 | 0.4392 | 0.4360 | 0.5702 | 0.0020 | 0.5474 | 0.4984 | 0.4592 | 0.4722 | 0.5949 | 0.0010 |
| 10 | Dummy Classifier (Stratified) | Baseline Ingênuo | Sprint 3 | 0.1322 | 0.0021 | 0.0026 | [12.96%, 13.48%] | 0.1640 | 0.0018 | 0.1635 | 0.1322 | 0.1325 | 1.5800 | 0.0700 | 0.1621 | 0.1314 | 0.1605 | 0.1321 | 0.1345 | 0.0010 | 0.1395 | 0.1361 | 0.1424 | 0.1725 | 0.1364 | 0.0020 |
