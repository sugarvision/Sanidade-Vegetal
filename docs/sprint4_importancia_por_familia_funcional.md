# 📊 Importância Agregada por Família Funcional de Descritores Visuais e Metadados

**Projeto:** Sanidade-Vegetal (*SugarVision*) | **Sprint:** 4 | **Fase SEMMA:** `Assess 6.0 — Famílias Funcionais`  
**Arquivo CSV Pareado:** `sprint4_importancia_por_familia_funcional.csv` | **Dimensões:** `6 linhas x 8 colunas`

> Comparação quantitativa entre Discretização & Categóricas, Metadados Geométricos, Cor & Índices Espectrais (ExR, ExG, R/G, ICN), Bordas (Laplaciano) e Textura Haralick (GLCM).

## Tabela de Dados Auditada

| functional_group | n_features | bal_acc_drop_sum | bal_acc_drop_mean | bal_acc_drop_std | macro_f1_drop_sum | macro_f1_drop_mean | share_of_importance_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Discretização & Categóricas | 6 | 0.1443 | 0.0241 | 0.0047 | 0.0684 | 0.0114 | 49.7898 |
| Metadados Geométricos | 7 | 0.1139 | 0.0163 | 0.0064 | 0.0996 | 0.0142 | 39.2778 |
| Cor & Índices Espectrais | 7 | 0.0241 | 0.0034 | 0.0036 | 0.0209 | 0.0030 | 8.2991 |
| Bordas & Foco (Laplaciano) | 1 | 0.0076 | 0.0076 | 0.0032 | 0.0058 | 0.0058 | 2.6334 |
| Outros | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Textura Haralick (GLCM) | 5 | -0.0025 | -0.0005 | 0.0028 | 3.61e-05 | 7.22e-06 | 0 |
