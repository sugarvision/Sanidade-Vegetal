# 📊 Testes de Hipótese Estatística Pareados: t-Student, Wilcoxon e Tamanho de Efeito (Cohen d)

**Projeto:** Sanidade-Vegetal (*SugarVision*) | **Sprint:** 4 | **Fase SEMMA:** `Assess 4.0 — Testes de Hipótese`  
**Arquivo CSV Pareado:** `sprint4_testes_hipotese_pareados.csv` | **Dimensões:** `9 linhas x 18 colunas`

> Comprovação estatística da superioridade dos modelos SVM otimizados sobre os modelos Baseline (Dummy, Regressão Logística e Árvore de Decisão rasa).

## Tabela de Dados Auditada

| Modelo_Referencia | Modelo_Comparado | Familia | Ganho_Abs_F1_Macro | Ganho_Rel_F1_Macro_Pct | t_stat_F1 | p_value_F1_t_test | w_stat_F1_wilcoxon | p_value_F1_wilcoxon | Cohen_d_F1 | Significancia_F1 | Decisao_H0_F1 | Ganho_Abs_Acuracia | Ganho_Rel_Acuracia_Pct | t_stat_Acuracia | p_value_Acuracia_t_test | Cohen_d_Acuracia | Significancia_Acuracia |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVM Polinomial Campeão GridSearch | Dummy Classifier (Stratified) | Baseline Ingênuo | 0.4464 | 337.7500 | 124.8890 | 0 | 15 | 0.0312 | 55.8520 | Sim (p < 0.01) | Rejeita H0 (Superioridade Comprovada) | 0.4813 | 293.5500 | 105.6900 | 0 | 47.2660 | Sim (p < 0.01) |
| SVM Polinomial Campeão GridSearch | Regressão Logística L2 (Sem Ponderação) | Baseline Linear | 0.0190 | 3.4000 | 4.2460 | 0.0132 | 15 | 0.0312 | 1.8990 | Sim (p < 0.05) | Rejeita H0 (Superioridade Comprovada) | -0.0047 | -0.7200 | -1.8010 | 0.1461 | -0.8050 | Não (p >= 0.05) |
| SVM Polinomial Campeão GridSearch | Regressão Logística L2 (Ponderada) | Baseline Linear | 0.0095 | 1.6700 | 4.5140 | 0.0107 | 15 | 0.0312 | 2.0190 | Sim (p < 0.05) | Rejeita H0 (Superioridade Comprovada) | 0.0095 | 1.5000 | 2.5290 | 0.0647 | 1.1310 | Não (p >= 0.05) |
| SVM Polinomial Campeão GridSearch | Árvore de Decisão rasa (Sem Ponderação) | Baseline Árvore | 0.0474 | 8.9300 | 13.5050 | 0.0002 | 15 | 0.0312 | 6.0390 | Sim (p < 0.01) | Rejeita H0 (Superioridade Comprovada) | 0.0163 | 2.6000 | 5.6350 | 0.0049 | 2.5200 | Sim (p < 0.01) |
| SVM Polinomial Campeão GridSearch | Árvore de Decisão rasa (Ponderada) | Baseline Árvore | 0.0451 | 8.4600 | 4.1450 | 0.0143 | 15 | 0.0312 | 1.8540 | Sim (p < 0.05) | Rejeita H0 (Superioridade Comprovada) | 0.0396 | 6.5500 | 5.7160 | 0.0046 | 2.5560 | Sim (p < 0.01) |
| SVM Polinomial Campeão GridSearch | SVM Linear Baseline | SVM Linear | 0.0077 | 1.3500 | 1.9330 | 0.1254 | 13 | 0.0938 | 0.8640 | Não (p >= 0.05) | Retém H0 (Equivalência Estatística) | 0.0014 | 0.2200 | 1.1150 | 0.3273 | 0.4990 | Não (p >= 0.05) |
| SVM Polinomial Campeão GridSearch | SVM RBF Baseline | SVM RBF | 0.0027 | 0.4600 | 1.3420 | 0.2509 | 11 | 0.2188 | 0.6000 | Não (p >= 0.05) | Retém H0 (Equivalência Estatística) | 0.0013 | 0.2000 | 0.8140 | 0.4613 | 0.3640 | Não (p >= 0.05) |
| SVM Polinomial Campeão GridSearch | SVM Polinomial Baseline | SVM Polinomial | 0.0092 | 1.6100 | 3.0630 | 0.0375 | 15 | 0.0312 | 1.3700 | Sim (p < 0.05) | Rejeita H0 (Superioridade Comprovada) | 0.0059 | 0.9300 | 3.5460 | 0.0239 | 1.5860 | Sim (p < 0.05) |
| SVM Polinomial Campeão GridSearch | SVM RBF Ótimo (C=10, gamma=0.01) | SVM RBF Otimizado | 0.0012 | 0.2100 | 0.6630 | 0.5438 | 9 | 0.4062 | 0.2960 | Não (p >= 0.05) | Retém H0 (Equivalência Estatística) | 0 | 0 | 0 | 0.9999 | 0 | Não (p >= 0.05) |
