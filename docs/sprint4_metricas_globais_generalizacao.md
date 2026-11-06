# 📊 Métricas Globais de Generalização do Estimador Campeão SVM (Sprint 4 — Fase Assess)

**Projeto:** Sanidade-Vegetal (*SugarVision*) | **Sprint:** 4 | **Fase SEMMA:** `Assess 1.0 — Generalização Global`  
**Arquivo CSV Pareado:** `sprint4_metricas_globais_generalizacao.csv` | **Dimensões:** `3 linhas x 17 colunas`

> Consolidação formal de Acurácia, Acurácia Balanceada, Precisão, Recall, Especificidade, F1-Score, AUC-ROC (Macro/Micro/Weighted) e Average Precision nas partições de Validação Independente (N=617), Teste Cego (N=380) e Avaliação Combinada (N=997).

## Tabela de Dados Auditada

| Particao | N_Amostras | Acuracia_Global | Acuracia_Balanceada | Precisao_Macro | Precisao_Weighted | Recall_Macro | Recall_Weighted | Especificidade_Macro | Especificidade_Weighted | F1_Macro | F1_Weighted | AUC_ROC_Macro | AUC_ROC_Micro | AUC_ROC_Weighted | AP_Macro | AP_Micro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Validacao (N=617) | 617 | 0.6207 | 0.6532 | 0.5346 | 0.5294 | 0.6532 | 0.6207 | 0.9381 | 0.9456 | 0.5757 | 0.5624 | 0.9148 | 0.9498 | 0.9063 | 0.6151 | 0.7713 |
| Teste Cego (N=380) | 380 | 0.6079 | 0.6769 | 0.5637 | 0.5111 | 0.6769 | 0.6079 | 0.9361 | 0.9450 | 0.6054 | 0.5475 | 0.9116 | 0.9405 | 0.8986 | 0.6259 | 0.7341 |
| Avaliacao Externa Combinada (N=997) | 997 | 0.6158 | 0.6687 | 0.5495 | 0.5213 | 0.6687 | 0.6158 | 0.9377 | 0.9480 | 0.5922 | 0.5559 | 0.9147 | 0.9465 | 0.9056 | 0.6200 | 0.7576 |
