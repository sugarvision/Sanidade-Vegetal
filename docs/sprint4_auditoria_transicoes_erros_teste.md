# 📊 Auditoria das 149 Instâncias Classificadas Incorretamente por Par de Transição (Teste Cego N=380)

**Projeto:** Sanidade-Vegetal (*SugarVision*) | **Sprint:** 4 | **Fase SEMMA:** `Assess 5.0 — Auditoria de Erros Críticos`  
**Arquivo CSV Pareado:** `sprint4_auditoria_transicoes_erros_teste.csv` | **Dimensões:** `9 linhas x 9 colunas`

> Decomposição exata dos pares de erro (Classe Real -> Classe Predita), categoria agronômica de impacto, probabilidades médias e participação percentual no total de erros.

## Tabela de Dados Auditada

| true_class_name | pred_class_name | error_transition_pair | agronomic_error_category | error_count | mean_pred_proba | mean_true_proba | mean_proba_gap | share_of_total_errors_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOSAIC | YELLOW LEAF | MOSAIC -> YELLOW LEAF | CROSS_PATHOLOGY_TREATMENT_MISMATCH | 36 | 0.4694 | 0.3555 | 0.1139 | 24.1600 |
| MOSAIC | GRASSY SHOOT | MOSAIC -> GRASSY SHOOT | CROSS_PATHOLOGY_TREATMENT_MISMATCH | 35 | 0.2796 | 0.3905 | -0.1109 | 23.4900 |
| MOSAIC | LEAF SCALD | MOSAIC -> LEAF SCALD | CROSS_PATHOLOGY_TREATMENT_MISMATCH | 19 | 0.5113 | 0.2968 | 0.2145 | 12.7500 |
| GRASSY SHOOT | LEAF SCALD | GRASSY SHOOT -> LEAF SCALD | CROSS_PATHOLOGY_TREATMENT_MISMATCH | 18 | 0.5139 | 0.1762 | 0.3377 | 12.0800 |
| RED ROT | YELLOW LEAF | RED ROT -> YELLOW LEAF | FUNGAL_OMISSION_FN_RED_ROT | 14 | 0.4619 | 0.1641 | 0.2978 | 9.4000 |
| LEAF SCALD | GRASSY SHOOT | LEAF SCALD -> GRASSY SHOOT | CROSS_PATHOLOGY_TREATMENT_MISMATCH | 12 | 0.2781 | 0.2567 | 0.0214 | 8.0500 |
| YELLOW LEAF | RED ROT | YELLOW LEAF -> RED ROT | CROSS_PATHOLOGY_TREATMENT_MISMATCH | 11 | 0.4613 | 0.3163 | 0.1450 | 7.3800 |
| MOSAIC | RED ROT | MOSAIC -> RED ROT | CROSS_PATHOLOGY_TREATMENT_MISMATCH | 3 | 0.4589 | 0.2283 | 0.2307 | 2.0100 |
| YELLOW LEAF | MOSAIC | YELLOW LEAF -> MOSAIC | CROSS_PATHOLOGY_TREATMENT_MISMATCH | 1 | 0.3763 | 0.3109 | 0.0654 | 0.6700 |
