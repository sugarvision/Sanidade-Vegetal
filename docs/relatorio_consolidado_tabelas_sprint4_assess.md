# 📑 Dossiê Consolidado de Relatórios Tabulares — Sprint 4 (Fase Assess — SEMMA)

**Projeto:** Sanidade-Vegetal (*SugarVision*)  
**Notebook Gerador:** `notebooks/05_sprint4_assess_validation.ipynb`  
**Total de Relatórios Exportados em Pares (CSV + Markdown):** `14 pares (28 arquivos em docs/)`  

---

## 1. Índice de Relatórios Tabulares Auditados em `docs/`

| id_relatorio | nome_base | secao_semma | titulo_relatorio | arquivo_csv | arquivo_md | n_linhas | n_colunas | tamanho_csv_kb | tamanho_md_kb | sha256_csv_prefix | sha256_md_prefix | status_conformidade |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | sprint4_metricas_globais_generalizacao | Assess 1.0 — Generalização Global | Métricas Globais de Generalização do Estimador Campeão SVM (Sprint 4 — Fase Assess) | docs/sprint4_metricas_globais_generalizacao.csv | docs/sprint4_metricas_globais_generalizacao.md | 3 | 17 | 0.6300 | 1.5000 | 9ebdd7636748085e | 62f1a65a243f923d | APPROVED |
| 2 | sprint4_metricas_por_classe_valid_test | Assess 2.0 — Discriminação Multiclasse | Métricas de Generalização por Classe Fitopatológica (Validação N=617 e Teste Cego N=380) | docs/sprint4_metricas_por_classe_valid_test.csv | docs/sprint4_metricas_por_classe_valid_test.md | 14 | 14 | 1.2400 | 2.2700 | bd7bd072961b9c91 | 8fbd686e32da9526 | APPROVED |
| 3 | sprint4_matriz_confusao_agronomica_3classes | Assess 2.0 — Matriz Agronômica 3 Classes | Matriz de Confusão Agronômica Tripartida: Saudável vs. Ferrugem vs. Outras Patologias | docs/sprint4_matriz_confusao_agronomica_3classes.csv | docs/sprint4_matriz_confusao_agronomica_3classes.md | 6 | 10 | 0.5400 | 1.2800 | 0ae43936a9f3e66c | 9a917abcbc002e76 | APPROVED |
| 4 | sprint4_calibracao_limiares_threshold_tuning | Assess 3.0 — Threshold Tuning | Calibração de Limiares de Decisão (Threshold Tuning) para Sensibilidade Agronômica | docs/sprint4_calibracao_limiares_threshold_tuning.csv | docs/sprint4_calibracao_limiares_threshold_tuning.md | 10 | 13 | 1.0800 | 2.0100 | c713f466a20bd02a | 66f1a35088ffce19 | APPROVED |
| 5 | sprint4_benchmark_unificado_modelos | Assess 4.0 — Benchmark Unificado | Benchmark Unificado de Modelos Supervisionados — Baselines vs. SVM (Sprints 3 e 4) | docs/sprint4_benchmark_unificado_modelos.csv | docs/sprint4_benchmark_unificado_modelos.md | 10 | 27 | 2.6700 | 4.0300 | 49a64209068804bd | 72252e8c24ec09a4 | APPROVED |
| 6 | sprint4_testes_hipotese_pareados | Assess 4.0 — Testes de Hipótese | Testes de Hipótese Estatística Pareados: t-Student, Wilcoxon e Tamanho de Efeito (Cohen d) | docs/sprint4_testes_hipotese_pareados.csv | docs/sprint4_testes_hipotese_pareados.md | 9 | 18 | 2.3300 | 3.3300 | 3efe57c84fa96e39 | 56947c8f99e389ec | APPROVED |
| 7 | sprint4_intervalos_confianca_ic95 | Assess 4.0 — Intervalos de Confiança 95% | Intervalos de Confiança (95%) e Margens de Erro entre Folds de Validação Cruzada | docs/sprint4_intervalos_confianca_ic95.csv | docs/sprint4_intervalos_confianca_ic95.md | 42 | 16 | 6.7100 | 8.7700 | 2dfe5cb22619dd19 | b2e42a3403f6a9be | APPROVED |
| 8 | sprint4_auditoria_repeated_kfold_estabilidade | Assess 4.0 — Estabilidade Repeated K-Fold | Auditoria de Estabilidade Estocástica Repeated Stratified K-Fold (5 Folds x 3 Repetições = 15 Splits) | docs/sprint4_auditoria_repeated_kfold_estabilidade.csv | docs/sprint4_auditoria_repeated_kfold_estabilidade.md | 4 | 14 | 0.6600 | 1.4300 | 3dd7df9fd274fbf6 | 5a7ed1e0763d7d11 | APPROVED |
| 9 | sprint4_auditoria_transicoes_erros_teste | Assess 5.0 — Auditoria de Erros Críticos | Auditoria das 149 Instâncias Classificadas Incorretamente por Par de Transição (Teste Cego N=380) | docs/sprint4_auditoria_transicoes_erros_teste.csv | docs/sprint4_auditoria_transicoes_erros_teste.md | 9 | 9 | 1.1000 | 1.9300 | 9e9eff35c7b17cc6 | 9b8e4c7ec3a541f7 | APPROVED |
| 10 | sprint4_matriz_custo_agronomico_7x7 | Assess 5.0 — Matriz de Custo Agronômico | Matriz de Custo e Perda Agronômica Assimétrica 7x7 (R$/ha) | docs/sprint4_matriz_custo_agronomico_7x7.csv | docs/sprint4_matriz_custo_agronomico_7x7.md | 7 | 8 | 0.4500 | 1.1600 | a495547fb7cafcc6 | 37b4affe805813d8 | APPROVED |
| 11 | sprint4_ranking_importancia_features | Assess 6.0 — Importância de Features | Ranking de Importância de Atributos via Permutation Feature Importance (Teste N=380) | docs/sprint4_ranking_importancia_features.csv | docs/sprint4_ranking_importancia_features.md | 27 | 8 | 3.3000 | 3.0800 | ffbdbb6e5cebd6c1 | f6ba2a4dfb44bf4a | APPROVED |
| 12 | sprint4_importancia_por_familia_funcional | Assess 6.0 — Famílias Funcionais | Importância Agregada por Família Funcional de Descritores Visuais e Metadados | docs/sprint4_importancia_por_familia_funcional.csv | docs/sprint4_importancia_por_familia_funcional.md | 6 | 8 | 0.8600 | 1.2200 | 662cc63fcf2d7c7c | 10d9501bb704dd19 | APPROVED |
| 13 | sprint4_padroes_visuais_falha_fronteira | Assess 6.0 — Limitações Visuais | Taxonomia Quantitativa dos Modos de Falha Fotográfica nos Casos de Fronteira | docs/sprint4_padroes_visuais_falha_fronteira.csv | docs/sprint4_padroes_visuais_falha_fronteira.md | 5 | 3 | 0.3000 | 0.8900 | bd9bac87c6d93455 | aa40e2c85ca79c3c | APPROVED |
| 14 | sprint4_catalogo_relatorios_tabulares | Assess 7.0 — Governança de Relatórios | Catálogo Oficial de Relatórios Tabulares Exportados na Sprint 4 (CSV e Markdown em docs/) | docs/sprint4_catalogo_relatorios_tabulares.csv | docs/sprint4_catalogo_relatorios_tabulares.md | 13 | 13 | 4.2800 | 5.3700 | 2a989fd104485a17 | d29601965ff5adb5 | APPROVED |

---

## 2. Métricas Globais de Generalização (`Valid`, `Test` e `Combined`)

| Particao | N_Amostras | Acuracia_Global | Acuracia_Balanceada | Precisao_Macro | Precisao_Weighted | Recall_Macro | Recall_Weighted | Especificidade_Macro | Especificidade_Weighted | F1_Macro | F1_Weighted | AUC_ROC_Macro | AUC_ROC_Micro | AUC_ROC_Weighted | AP_Macro | AP_Micro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Validacao (N=617) | 617 | 0.6207 | 0.6532 | 0.5346 | 0.5294 | 0.6532 | 0.6207 | 0.9381 | 0.9456 | 0.5757 | 0.5624 | 0.9148 | 0.9498 | 0.9063 | 0.6151 | 0.7713 |
| Teste Cego (N=380) | 380 | 0.6079 | 0.6769 | 0.5637 | 0.5111 | 0.6769 | 0.6079 | 0.9361 | 0.9450 | 0.6054 | 0.5475 | 0.9116 | 0.9405 | 0.8986 | 0.6259 | 0.7341 |
| Avaliacao Externa Combinada (N=997) | 997 | 0.6158 | 0.6687 | 0.5495 | 0.5213 | 0.6687 | 0.6158 | 0.9377 | 0.9480 | 0.5922 | 0.5559 | 0.9147 | 0.9465 | 0.9056 | 0.6200 | 0.7576 |

---

## 3. Matriz de Confusão Agronômica Tripartida (Saudável, Ferrugem e Outras Patologias)

| Particao | Grupo_Real | Suporte_Real | Pred_Saudavel_Abs | Pred_Ferrugem_Abs | Pred_Outras_Abs | Pred_Saudavel_Pct | Pred_Ferrugem_Pct | Pred_Outras_Pct | Sensibilidade_Grupo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Validacao (N=617) | SAUDÁVEL (HEALTHY) | 95 | 95 | 0 | 0 | 100 | 0 | 0 | 1 |
| Validacao (N=617) | FERRUGEM (RUST) | 91 | 0 | 91 | 0 | 0 | 100 | 0 | 1 |
| Validacao (N=617) | OUTRAS PATOLOGIAS | 431 | 0 | 0 | 431 | 0 | 0 | 100 | 1 |
| Teste Cego (N=380) | SAUDÁVEL (HEALTHY) | 46 | 46 | 0 | 0 | 100 | 0 | 0 | 1 |
| Teste Cego (N=380) | FERRUGEM (RUST) | 51 | 0 | 51 | 0 | 0 | 100 | 0 | 1 |
| Teste Cego (N=380) | OUTRAS PATOLOGIAS | 283 | 0 | 0 | 283 | 0 | 0 | 100 | 1 |

---

## 4. Calibração de Limiares de Decisão (*Threshold Tuning*)

| Patologia_Alvo | Criterio_Operacional | Limiar_Tau_Otimo | Sensibilidade_Recall | Especificidade_TNR | Precisao | F1_Score | F2_Agronomico | Indice_Youden_J | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RUST (Ferrugem) | Padrao (tau=0.50) | 0.5000 | 1 | 1 | 1 | 1 | 1 | 1 | 91 | 0 | 0 | 526 |
| RUST (Ferrugem) | Maximo F1-Score | 0.2300 | 1 | 1 | 1 | 1 | 1 | 1 | 91 | 0 | 0 | 526 |
| RUST (Ferrugem) | Maximo F2 Agronomico (beta=2) | 0.2300 | 1 | 1 | 1 | 1 | 1 | 1 | 91 | 0 | 0 | 526 |
| RUST (Ferrugem) | Indice J de Youden | 0.2300 | 1 | 1 | 1 | 1 | 1 | 1 | 91 | 0 | 0 | 526 |
| RUST (Ferrugem) | Triagem Alta Sensibilidade (Recall>=98%) | 0.5400 | 1 | 1 | 1 | 1 | 1 | 1 | 91 | 0 | 0 | 526 |
| LEAF SCALD (Escaldadura) | Padrao (tau=0.50) | 0.5000 | 0.7778 | 0.9368 | 0.5833 | 0.6667 | 0.7292 | 0.7146 | 49 | 35 | 14 | 519 |
| LEAF SCALD (Escaldadura) | Maximo F1-Score | 0.4100 | 0.9048 | 0.9224 | 0.5700 | 0.6994 | 0.8097 | 0.8271 | 57 | 43 | 6 | 511 |
| LEAF SCALD (Escaldadura) | Maximo F2 Agronomico (beta=2) | 0.4100 | 0.9048 | 0.9224 | 0.5700 | 0.6994 | 0.8097 | 0.8271 | 57 | 43 | 6 | 511 |
| LEAF SCALD (Escaldadura) | Indice J de Youden | 0.1200 | 1 | 0.8502 | 0.4315 | 0.6029 | 0.7915 | 0.8502 | 63 | 83 | 0 | 471 |
| LEAF SCALD (Escaldadura) | Triagem Alta Sensibilidade (Recall>=98%) | 0.1300 | 0.9841 | 0.8556 | 0.4366 | 0.6049 | 0.7868 | 0.8397 | 62 | 80 | 1 | 474 |

---

## 5. Estabilidade Estocástica `RepeatedStratifiedKFold (5x3 = 15 Splits)`

| Escopo_Avaliacao | N_Splits | F1_Macro_Media | F1_Macro_Std | F1_Macro_CV_Pct | F1_Macro_IC95_Inf | F1_Macro_IC95_Sup | Balanced_Acc_Media | Balanced_Acc_Std | Balanced_Acc_CV_Pct | Recall_Healthy_Media | Recall_Rust_Media | Vetores_Suporte_Media | Desvio_Max_Estratificacao |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Repeticao 1 (5 Folds) | 5 | 0.5774 | 0.0086 | 1.4900 | 0.5667 | 0.5880 | 0.6460 | 0.0142 | 2.1900 | 1 | 0.9934 | 2782 | 0.0007 |
| Repeticao 2 (5 Folds) | 5 | 0.5768 | 0.0076 | 1.3200 | 0.5673 | 0.5863 | 0.6454 | 0.0127 | 1.9700 | 1 | 0.9902 | 2778 | 0.0007 |
| Repeticao 3 (5 Folds) | 5 | 0.5742 | 0.0099 | 1.7200 | 0.5619 | 0.5865 | 0.6424 | 0.0118 | 1.8400 | 1 | 0.9891 | 2774.2000 | 0.0007 |
| Consolidado Global (15 Splits) | 15 | 0.5761 | 0.0082 | 1.4300 | 0.5716 | 0.5807 | 0.6446 | 0.0121 | 1.8700 | 1 | 0.9909 | 2778.1000 | 0.0007 |
