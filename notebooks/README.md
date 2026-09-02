# 📓 Notebooks do Projeto Sanidade-Vegetal (SugarVision)

**Sprint 1 — Framework SEMMA (Fases Ativas: Sample & Explore)**

---

## 📂 Catálogo de Notebooks

| Arquivo | Descrição | Responsável Principal | Fases SEMMA |
| :--- | :--- | :--- | :---: |
| **[`01_ingestao_dados_brutos.ipynb`](./01_ingestao_dados_brutos.ipynb)** | Notebook de ingestão inicial dos arquivos de imagem brutos, validação física de integridade e catálogo de metadados. | Elisa (Engenharia) / Guilherme | `Sample` |
| **[`02_sprint1_master_pipeline_reprodutivel.ipynb`](./02_sprint1_master_pipeline_reprodutivel.ipynb)** | **Master Pipeline Reprodutível da Sprint 1:** Execução unificada de ponta a ponta (Ingestão $\rightarrow$ Amostragem Estratificada $\rightarrow$ Auditoria de Qualidade $\rightarrow$ EDA Univariada/Multivariada $\rightarrow$ Consolidação da ABT). | Cesar / Equipe | `Sample` & `Explore` |

---

## 🚀 Como Executar o Master Pipeline Reprodutível

1. Certifique-se de que o arquivo `data/processed/metadata_raw_images.csv` está presente.
2. Abra o notebook `02_sprint1_master_pipeline_reprodutivel.ipynb` no VS Code ou Jupyter Lab.
3. Clique em **Restart & Run All**.
4. Os gráficos diagnósticos serão salvos automaticamente em `docs/figures/` e a Analytical Base Table será consolidada em `data/processed/abt_sanidade_vegetal.csv`.
