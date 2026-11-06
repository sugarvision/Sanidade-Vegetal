# 📊 Auditoria de Estabilidade Estocástica Repeated Stratified K-Fold (5 Folds x 3 Repetições = 15 Splits)

**Projeto:** Sanidade-Vegetal (*SugarVision*) | **Sprint:** 4 | **Fase SEMMA:** `Assess 4.0 — Estabilidade Repeated K-Fold`  
**Arquivo CSV Pareado:** `sprint4_auditoria_repeated_kfold_estabilidade.csv` | **Dimensões:** `4 linhas x 14 colunas`

> Análise de dispersão entre repetições comprovando invariância ao particionamento amostral (CV de F1-Macro = 1,43%, ANOVA p = 0,8334 > 0,05).

## Tabela de Dados Auditada

| Escopo_Avaliacao | N_Splits | F1_Macro_Media | F1_Macro_Std | F1_Macro_CV_Pct | F1_Macro_IC95_Inf | F1_Macro_IC95_Sup | Balanced_Acc_Media | Balanced_Acc_Std | Balanced_Acc_CV_Pct | Recall_Healthy_Media | Recall_Rust_Media | Vetores_Suporte_Media | Desvio_Max_Estratificacao |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Repeticao 1 (5 Folds) | 5 | 0.5774 | 0.0086 | 1.4900 | 0.5667 | 0.5880 | 0.6460 | 0.0142 | 2.1900 | 1 | 0.9934 | 2782 | 0.0007 |
| Repeticao 2 (5 Folds) | 5 | 0.5768 | 0.0076 | 1.3200 | 0.5673 | 0.5863 | 0.6454 | 0.0127 | 1.9700 | 1 | 0.9902 | 2778 | 0.0007 |
| Repeticao 3 (5 Folds) | 5 | 0.5742 | 0.0099 | 1.7200 | 0.5619 | 0.5865 | 0.6424 | 0.0118 | 1.8400 | 1 | 0.9891 | 2774.2000 | 0.0007 |
| Consolidado Global (15 Splits) | 15 | 0.5761 | 0.0082 | 1.4300 | 0.5716 | 0.5807 | 0.6446 | 0.0121 | 1.8700 | 1 | 0.9909 | 2778.1000 | 0.0007 |
