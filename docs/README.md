# 📚 Documentação Técnica do Projeto — Sanidade Vegetal (SugarVision)

**Framework Metodológico:** SEMMA (*Sample, Explore, Modify, Model, Assess*)  
**Sprint:** 1 — Setup, Sample & Explore  

---

## 📂 Índice Geral de Entregas e Documentos Técnicos

### 1. Definição do Escopo e Domínio Agronômico (Cesar)
- **[`../Definição do Escopo do Problema/README.md`](../Definição%20do%20Escopo%20do%20Problema/README.md)**: Resumo executivo e cobertura do checklist.
- **[`../Definição do Escopo do Problema/definicao_do_escopo_sanidade_vegetal.md`](../Definição%20do%20Escopo%20do%20Problema/definicao_do_escopo_sanidade_vegetal.md)**: Delimitação técnica do problema (binário vs multiclasse), classes fitopatológicas e impacto agronômico.
- **[`../Definição do Escopo do Problema/hipoteses_iniciais_e_metadados.md`](../Definição%20do%20Escopo%20do%20Problema/hipoteses_iniciais_e_metadados.md)**: Formulação de 5 hipóteses agronômicas/computacionais e matriz de variáveis de interesse.

### 2. Exploração de Imagens e Padrões Visuais (Cesar)
- **[`../Exploração de Imagens e Padrões Visuais/README.md`](../Explora%C3%A7%C3%A3o%20de%20Imagens%20e%20Padr%C3%B5es%20Visuais/README.md)**: Visão geral da caracterização visual e cobertura do checklist.
- **[`../Exploração de Imagens e Padrões Visuais/analise_padroes_visuais_e_desafios.md`](../Explora%C3%A7%C3%A3o%20de%20Imagens%20e%20Padr%C3%B5es%20Visuais/analise_padroes_visuais_e_desafios.md)**: Estudo aprofundado dos padrões de textura (GLCM), cor (HSV/RGB), desafios fotométricos (iluminação, sombras, fundo) e disparidade de resolução.
- **[`./figures/`](./figures/)**: Gráficos gerados para diagnóstico e análise exploratória visual.

### 3. Modelagem de Dados e Dicionário (Heitor)
- **[`../Modelagem Inicial e Dicionário de Dados/dicionario_de_dados.md`](../Modelagem%20Inicial%20e%20Dicion%C3%A1rio%20de%20Dados/dicionario_de_dados.md)**: Dicionário completo de dados e tipagem.
- **[`../Modelagem Inicial e Dicionário de Dados/modelagem_conceitual_entidades.md`](../Modelagem%20Inicial%20e%20Dicion%C3%A1rio%20de%20Dados/modelagem_conceitual_entidades.md)**: Modelagem relacional e conceitual das entidades.

### 4. Estratégia de Amostragem (Heitor)
- **[`../Definição da Estratégia de Amostragem/estrategia_de_amostragem.md`](../Defini%C3%A7%C3%A3o%20da%20Estrat%C3%A9gia%20de%20Amostragem/estrategia_de_amostragem.md)**: Metodologia de particionamento estratificado e mitigação de leakage.
- **[`../Definição da Estratégia de Amostragem/distribuicao_e_balanceamento.md`](../Defini%C3%A7%C3%A3o%20da%20Estrat%C3%A9gia%20de%20Amostragem/distribuicao_e_balanceamento.md)**: Estudo de desbalanceamento e pesos de classe.

### 5. Pipelines e Execução Técnica (Cesar & Equipe - Sprint 1)
- **[`../notebooks/02_sprint1_master_pipeline_reprodutivel.ipynb`](../notebooks/02_sprint1_master_pipeline_reprodutivel.ipynb)**: Notebook mestre da Sprint 1 executável de ponta a ponta sem falhas.
- **[`../data/processed/abt_sanidade_vegetal.csv`](../data/processed/abt_sanidade_vegetal.csv)**: Tabela analítica consolidada preliminar (6.571 registros x 16 atributos).

### 🏆 Relatórios Executivos da Sprint Review (Sprint 1)
- **[`./Sprint_Review_Sprint_1_Sanidade_Vegetal.docx`](./Sprint_Review_Sprint_1_Sanidade_Vegetal.docx)**: Relatório executivo completo em formato Microsoft Word (`.docx`), com tabelas estritas, caixas de destaque e imagens diagnósticas incorporadas.
- **[`./Sprint_Review_Sprint_1_Sanidade_Vegetal.pdf`](./Sprint_Review_Sprint_1_Sanidade_Vegetal.pdf)**: Relatório executivo completo em formato PDF pronto para distribuição e apresentação.

---

## 📂 Entregas da Sprint 2 — Framework SEMMA (Fase: Modify)

### 6. Engenharia de Atributos e Descritores Visuais (Cesar)
- **[`../Engenharia de Atributos e Descritores Visuais/README.md`](../Engenharia%20de%20Atributos%20e%20Descritores%20Visuais/README.md)**: Resumo executivo da entrega e cobertura dos checklists das tarefas do Cesar.
- **[`../Engenharia de Atributos e Descritores Visuais/engenharia_descritores_cromaticos_e_indices.md`](../Engenharia%20de%20Atributos%20e%20Descritores%20Visuais/engenharia_descritores_cromaticos_e_indices.md)**: Documentação da Tarefa 1 (Espaços HSV/CIELAB, índices $ExG, ExR$, razões espectrais $R/G$ e testes estatísticos de separabilidade).
- **[`../Engenharia de Atributos e Descritores Visuais/extracao_textura_glcm_e_rugosidade.md`](../Engenharia%20de%20Atributos%20e%20Descritores%20Visuais/extracao_textura_glcm_e_rugosidade.md)**: Documentação da Tarefa 2 (Matriz GLCM multidirecional, propriedades de Haralick, invariância à rotação e validação da hipótese de rugosidade foliar).
- **[`../src/feature_engineering_visual.py`](../src/feature_engineering_visual.py)**: Módulo Python executável de transformação de features visuais e geração de figuras diagnósticas.
- **[`./figures/sprint2_analise_cromaticas_hsv_exg.png`](./figures/sprint2_analise_cromaticas_hsv_exg.png)**: Diagnóstico visual dos índices cromáticos e espectrais.
- **[`./figures/sprint2_analise_texturas_glcm_haralick.png`](./figures/sprint2_analise_texturas_glcm_haralick.png)**: Diagnóstico visual dos descritores texturais de Haralick.

### 7. Construção do Pipeline Atômico com ColumnTransformer e Anti-Leakage (Elisa)
- **[`../Pipeline Atômico de Pré-processamento e Anti-Leakage/README.md`](../Pipeline%20At%C3%B4mico%20de%20Pr%C3%A9-processamento%20e%20Anti-Leakage/README.md)**: Resumo executivo da entrega e cobertura de 100% do checklist da Sprint 2.
- **[`../Pipeline Atômico de Pré-processamento e Anti-Leakage/pipeline_atomico_columntransformer.md`](../Pipeline%20At%C3%B4mico%20de%20Pr%C3%A9-processamento%20e%20Anti-Leakage/pipeline_atomico_columntransformer.md)**: Documentação técnica completa da arquitetura atômica, sub-pipelines numérico/categórico, auditoria matemática anti-leakage e governança MLOps.
- **[`../src/atomic_pipeline_preprocessing.py`](../src/atomic_pipeline_preprocessing.py)**: Módulo Python executável de pré-processamento atômico, testes de conformidade e serialização.
- **[`../notebooks/06_pipeline_atomico_columntransformer.ipynb`](../notebooks/06_pipeline_atomico_columntransformer.ipynb)**: Notebook executável auditável de ponta a ponta com diagrama interativo e testes.
- **[`../models/preprocessor_pipeline.joblib`](../models/preprocessor_pipeline.joblib)**: Pipeline de pré-processamento serializado e comprimido (2,58 KB) pronto para a Sprint 3.
- **[`../models/preprocessor_metadata.json`](../models/preprocessor_metadata.json)**: Manifesto de governança MLOps com hash SHA-256 e schema das 35 features resultantes.
- **[`../data/processed/abt_features_modelagem.parquet`](../data/processed/abt_features_modelagem.parquet)**: Tabela Analítica Base consolidada com 6.571 registros e zero valores ausentes.

### 8. Consolidação e Exportação da ABT Final (Elisa)
- **[`../Consolidação e Exportação da ABT Final/README.md`](../Consolida%C3%A7%C3%A3o%20e%20Exporta%C3%A7%C3%A3o%20da%20ABT%20Final/README.md)**: Resumo executivo da entrega e cobertura de 100% do checklist da Sprint 2.
- **[`../Consolidação e Exportação da ABT Final/consolidacao_exportacao_abt_final.md`](../Consolida%C3%A7%C3%A3o%20e%20Exporta%C3%A7%C3%A3o%20da%20ABT%20Final/consolidacao_exportacao_abt_final.md)**: Documentação técnica detalhada, auditoria estrita de 0 NaNs, benchmark de I/O (Parquet vs CSV) e contrato de integração com GridSearchCV.
- **[`../src/export_final_modeling_abt.py`](../src/export_final_modeling_abt.py)**: Script executável de consolidação, auditoria automatizada, hashes SHA-256 e smoke test com GridSearchCV.
- **[`../notebooks/07_consolidacao_exportacao_abt_final.ipynb`](../notebooks/07_consolidacao_exportacao_abt_final.ipynb)**: Notebook executável demonstrando benchmark de leitura, integridade dos alvos e prontidão para a Sprint 3.
- **[`../data/processed/abt_features_modelagem.parquet`](../data/processed/abt_features_modelagem.parquet)**: Tabela Analítica Base final em Apache Parquet (916,92 KB, Snappy, 6.571 x 40).
- **[`../data/processed/abt_features_modelagem.csv`](../data/processed/abt_features_modelagem.csv)**: Espelho em texto plano CSV para interoperabilidade e auditoria (2.989,66 KB).
- **[`../data/processed/abt_features_modelagem_manifest.json`](../data/processed/abt_features_modelagem_manifest.json)**: Manifesto de integridade com hashes criptográficos SHA-256 e estatísticas.

