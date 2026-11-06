# 📊 Métricas de Generalização por Classe Fitopatológica (Validação N=617 e Teste Cego N=380)

**Projeto:** Sanidade-Vegetal (*SugarVision*) | **Sprint:** 4 | **Fase SEMMA:** `Assess 2.0 — Discriminação Multiclasse`  
**Arquivo CSV Pareado:** `sprint4_metricas_por_classe_valid_test.csv` | **Dimensões:** `14 linhas x 14 colunas`

> Detalhamento granular de Suporte, TP, FP, FN, TN, Sensibilidade (Recall), Especificidade (TNR), Precisão, F1-Score, AUC-ROC One-vs-Rest e Average Precision (AP) por patologia.

## Tabela de Dados Auditada

| Particao | Class_ID | Patologia | Suporte | TP | FP | FN | TN | Sensibilidade_Recall | Especificidade_TNR | Precisao | F1_Score | AUC_ROC_OvR | Average_Precision_AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Validacao (N=617) | 0 | GRASSY SHOOT | 20 | 6 | 51 | 14 | 546 | 0.3000 | 0.9146 | 0.1053 | 0.1558 | 0.8827 | 0.1482 |
| Validacao (N=617) | 1 | HEALTHY | 95 | 95 | 0 | 0 | 522 | 1 | 1 | 1 | 1 | 1 | 1 |
| Validacao (N=617) | 2 | LEAF SCALD | 63 | 55 | 41 | 8 | 513 | 0.8730 | 0.9260 | 0.5729 | 0.6918 | 0.9525 | 0.6018 |
| Validacao (N=617) | 3 | MOSAIC | 154 | 0 | 1 | 154 | 462 | 0 | 0.9978 | 0 | 0 | 0.8168 | 0.5413 |
| Validacao (N=617) | 4 | RED ROT | 94 | 62 | 32 | 32 | 491 | 0.6596 | 0.9388 | 0.6596 | 0.6596 | 0.9057 | 0.5921 |
| Validacao (N=617) | 5 | RUST | 91 | 91 | 0 | 0 | 526 | 1 | 1 | 1 | 1 | 1 | 1 |
| Validacao (N=617) | 6 | YELLOW LEAF | 100 | 74 | 109 | 26 | 408 | 0.7400 | 0.7892 | 0.4044 | 0.5230 | 0.8458 | 0.4222 |
| Teste Cego (N=380) | 0 | GRASSY SHOOT | 36 | 18 | 47 | 18 | 297 | 0.5000 | 0.8634 | 0.2769 | 0.3564 | 0.8526 | 0.3113 |
| Teste Cego (N=380) | 1 | HEALTHY | 46 | 46 | 0 | 0 | 334 | 1 | 1 | 1 | 1 | 1 | 1 |
| Teste Cego (N=380) | 2 | LEAF SCALD | 63 | 51 | 37 | 12 | 280 | 0.8095 | 0.8833 | 0.5795 | 0.6755 | 0.9162 | 0.5710 |
| Teste Cego (N=380) | 3 | MOSAIC | 93 | 0 | 1 | 93 | 286 | 0 | 0.9965 | 0 | 0 | 0.7790 | 0.4541 |
| Teste Cego (N=380) | 4 | RED ROT | 49 | 35 | 14 | 14 | 317 | 0.7143 | 0.9577 | 0.7143 | 0.7143 | 0.9417 | 0.6647 |
| Teste Cego (N=380) | 5 | RUST | 51 | 51 | 0 | 0 | 329 | 1 | 1 | 1 | 1 | 1 | 1 |
| Teste Cego (N=380) | 6 | YELLOW LEAF | 42 | 30 | 50 | 12 | 288 | 0.7143 | 0.8521 | 0.3750 | 0.4918 | 0.8917 | 0.3804 |
