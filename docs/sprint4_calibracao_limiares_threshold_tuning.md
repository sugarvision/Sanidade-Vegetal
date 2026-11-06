# 📊 Calibração de Limiares de Decisão (Threshold Tuning) para Sensibilidade Agronômica

**Projeto:** Sanidade-Vegetal (*SugarVision*) | **Sprint:** 4 | **Fase SEMMA:** `Assess 3.0 — Threshold Tuning`  
**Arquivo CSV Pareado:** `sprint4_calibracao_limiares_threshold_tuning.csv` | **Dimensões:** `10 linhas x 13 colunas`

> Pontos ótimos de operação (tau) avaliados para RUST (Ferrugem) e LEAF SCALD (Escaldadura) segundo critérios Padrão (0.50), Máximo F1, Máximo F2 Agronômico, Índice J de Youden e Triagem de Alta Sensibilidade (Recall >= 98%).

## Tabela de Dados Auditada

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
