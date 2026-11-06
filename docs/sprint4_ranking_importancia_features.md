# 📊 Ranking de Importância de Atributos via Permutation Feature Importance (Teste N=380)

**Projeto:** Sanidade-Vegetal (*SugarVision*) | **Sprint:** 4 | **Fase SEMMA:** `Assess 6.0 — Importância de Features`  
**Arquivo CSV Pareado:** `sprint4_ranking_importancia_features.csv` | **Dimensões:** `27 linhas x 8 colunas`

> Queda média e desvio padrão em Acurácia Balanceada e F1-Macro após 15 permutações estocásticas para cada uma das 27 features do pipeline.

## Tabela de Dados Auditada

| feature | functional_group | bal_acc_drop_mean | bal_acc_drop_std | macro_f1_drop_mean | macro_f1_drop_std | relative_importance_score | rank_bal_acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| size_kb_bin | Discretização & Categóricas | 0.0741 | 0.0103 | 0.0680 | 0.0083 | 0.0741 | 1 |
| dataset_source | Discretização & Categóricas | 0.0604 | 0.0115 | 0.0141 | 0.0118 | 0.0604 | 2 |
| size_kb | Metadados Geométricos | 0.0570 | 0.0154 | 0.0317 | 0.0144 | 0.0570 | 3 |
| total_pixels | Metadados Geométricos | 0.0216 | 0.0116 | 0.0262 | 0.0094 | 0.0216 | 4 |
| resolution_mp | Metadados Geométricos | 0.0190 | 0.0108 | 0.0244 | 0.0092 | 0.0190 | 5 |
| aspect_ratio | Metadados Geométricos | 0.0162 | 0.0072 | 0.0173 | 0.0050 | 0.0162 | 6 |
| extension | Discretização & Categóricas | 0.0098 | 0.0030 | -0.0144 | 0.0038 | 0.0098 | 7 |
| exr_index | Cor & Índices Espectrais | 0.0090 | 0.0057 | 0.0083 | 0.0048 | 0.0090 | 8 |
| laplacian_var | Bordas & Foco (Laplaciano) | 0.0076 | 0.0032 | 0.0058 | 0.0023 | 0.0076 | 9 |
| indice_clorose_necrose | Cor & Índices Espectrais | 0.0041 | 0.0022 | 0.0023 | 0.0023 | 0.0041 | 10 |
| rg_ratio | Cor & Índices Espectrais | 0.0040 | 0.0044 | 0.0038 | 0.0035 | 0.0040 | 11 |
| std_saturation | Cor & Índices Espectrais | 0.0038 | 0.0033 | 0.0041 | 0.0027 | 0.0038 | 12 |
| aspect_ratio_bin | Discretização & Categóricas | 0.0032 | 0.0016 | 0.0024 | 0.0012 | 0.0032 | 13 |
| mean_hue | Cor & Índices Espectrais | 0.0029 | 0.0032 | 0.0025 | 0.0030 | 0.0029 | 14 |
| exg_index | Cor & Índices Espectrais | 0.0012 | 0.0025 | 0.0009 | 0.0023 | 0.0012 | 15 |
| channels | Metadados Geométricos | 0 | 0 | 0 | 0 | 0 | 16 |
| height | Metadados Geométricos | 0 | 0 | 0 | 0 | 0 | 17 |
| color_mode | Outros | 0 | 0 | 0 | 0 | 0 | 18 |
| width | Metadados Geométricos | 0 | 0 | 0 | 0 | 0 | 19 |
| resolution_bin | Discretização & Categóricas | 0 | 0 | 0.0001 | 0 | 0 | 20 |
| indice_rugosidade_pustula | Textura Haralick (GLCM) | -4.45e-05 | 0.0022 | 0.0006 | 0.0016 | 0 | 21 |
| glcm_homogeneity | Textura Haralick (GLCM) | -0.0003 | 0.0030 | 0.0002 | 0.0020 | 0 | 22 |
| glcm_energy | Textura Haralick (GLCM) | -0.0005 | 0.0025 | 0.0003 | 0.0017 | 0 | 23 |
| glcm_contrast | Textura Haralick (GLCM) | -0.0005 | 0.0044 | -0.0005 | 0.0030 | 0 | 24 |
| hue_dispersion | Cor & Índices Espectrais | -0.0010 | 0.0035 | -0.0010 | 0.0032 | 0 | 25 |
| glcm_dissimilarity | Textura Haralick (GLCM) | -0.0011 | 0.0020 | -0.0006 | 0.0017 | 0 | 26 |
| laplacian_var_bin | Discretização & Categóricas | -0.0031 | 0.0020 | -0.0018 | 0.0014 | 0 | 27 |
