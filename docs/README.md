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

