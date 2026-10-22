# 📓 Notebooks do Projeto Sanidade-Vegetal (SugarVision)

**Sprint 1 — Framework SEMMA (Fases Ativas: Sample & Explore)**

---

## 📂 Catálogo de Notebooks

| Arquivo | Descrição | Responsável Principal | Fases SEMMA |
| :--- | :--- | :--- | :---: |
| **[`01A_ingestao_dados_brutos.ipynb`](./01A_ingestao_dados_brutos.ipynb)** | Notebook de ingestão inicial dos arquivos de imagem brutos, validação física de integridade e catálogo de metadados. | Elisa (Engenharia) / Guilherme | `Sample` |
| **[`01B_construcao_abt_agricola.ipynb`](./01B_construcao_abt_agricola.ipynb)** | Construção da Analytical Base Table com engenharia inicial de atributos agronômicos. | Equipe de Dados | `Sample` & `Modify` |
| **[`02A_sprint1_master_pipeline_reprodutivel.ipynb`](./02A_sprint1_master_pipeline_reprodutivel.ipynb)** | **Master Pipeline Reprodutível da Sprint 1:** Execução unificada de ponta a ponta (Ingestão $\rightarrow$ Amostragem Estratificada $\rightarrow$ Auditoria de Qualidade $\rightarrow$ EDA Univariada/Multivariada $\rightarrow$ Consolidação da ABT). | Cesar / Equipe | `Sample` & `Explore` |
| **[`03A_eda_univariada_sprint1.ipynb`](./03A_eda_univariada_sprint1.ipynb)** | Análise Exploratória Univariada de distribuições, assimetrias e dispersão. | Equipe de Análise | `Explore` |
| **[`04A_eda_multivariada_sprint1.ipynb`](./04A_eda_multivariada_sprint1.ipynb)** | Análise Exploratória Multivariada, correlações e separabilidade inicial. | Equipe de Análise | `Explore` |
| **[`05_estudo_teorico_e_aplicado_svm.ipynb`](./05_estudo_teorico_e_aplicado_svm.ipynb)** | Estudo teórico e experimental do Support Vector Machine e funções de Kernel. | Pesquisa IA | `Model` |
| **[`06_pipeline_atomico_columntransformer.ipynb`](./06_pipeline_atomico_columntransformer.ipynb)** | **Pipeline Atômico de Pré-processamento e Anti-Leakage (Sprint 2):** Fluxo unificado no Scikit-Learn com `ColumnTransformer`, imputação por mediana, normalização z-score, codificação One-Hot, validação estrita anti-leakage e serialização MLOps com `joblib`. | Elisa (`EA`) | `Modify` |
| **[`07_consolidacao_exportacao_abt_final.ipynb`](./07_consolidacao_exportacao_abt_final.ipynb)** | **Consolidação e Exportação da ABT Final (Sprint 2):** Validação de 0 NaNs, persistência em Apache Parquet colunar e CSV, benchmark de desempenho I/O, hashes SHA-256 e teste de prontidão com `GridSearchCV`. | Elisa (`EA`) | `Modify` |
| **[`08_pipeline_atomico_svm_anti_leakage.ipynb`](./08_pipeline_atomico_svm_anti_leakage.ipynb)** | Pipeline Atômico de Treinamento SVM com blindagem contra Data Leakage via CV interna. | Squad Modelagem | `Model` |
| **[`09_grid_search_exaustivo_svm.ipynb`](./09_grid_search_exaustivo_svm.ipynb)** | Grid Search exaustivo de hiperparâmetros SVM (C, gamma, degree) com validação cruzada. | Squad Modelagem | `Model` & `Assess` |
| **[`dossie_fase_modify_semma.ipynb`](./dossie_fase_modify_semma.ipynb)** | **Dossiê e Inventário Interativo da Fase Modify:** Notebook executável com todo o catálogo, documentação, inventário de scripts e smoke tests pronto para abrir no Google Colab. | Squad 2 | `Modify` |

---

## 🚀 Como Executar o Master Pipeline Reprodutível

1. Certifique-se de que o arquivo `data/processed/metadata_raw_images.csv` está presente.
2. Abra o notebook `02A_sprint1_master_pipeline_reprodutivel.ipynb` no VS Code ou Jupyter Lab.
3. Clique em **Restart & Run All**.
4. Os gráficos diagnósticos serão salvos automaticamente em `docs/figures/` e a Analytical Base Table será consolidada em `data/processed/abt_sanidade_vegetal.csv`.
