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

### 5. Pipelines e Execução Técnica (Cesar & Equipe)
- **[`../notebooks/02_sprint1_master_pipeline_reprodutivel.ipynb`](../notebooks/02_sprint1_master_pipeline_reprodutivel.ipynb)**: Notebook mestre da Sprint 1 executável de ponta a ponta sem falhas.
- **[`../data/processed/abt_sanidade_vegetal.csv`](../data/processed/abt_sanidade_vegetal.csv)**: Tabela analítica consolidada (6.571 registros x 16 atributos).
