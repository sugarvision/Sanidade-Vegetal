# Entrega: Engenharia de Atributos e Descritores Visuais (Fase Modify)
**Projeto:** Sanidade-Vegetal (SugarVision)  
**Sprint:** 2 — Framework SEMMA (Fase: Modify)  
**Responsável:** Cesar (Lead Técnico & Visão Computacional)  
**Data:** Setembro de 2026  

---

## 📌 Visão Geral da Entrega

Esta pasta formaliza as atividades e artefatos de **Engenharia de Descritores Cromáticos, Índices Espectrais, Extração de Textura Haralick (GLCM) e Discretização de Variáveis Contínuas (Binning Estatístico)** de responsabilidade do membro **Cesar** para a **Sprint 2 (Fase Modify)**.

O objetivo central foi converter o sinal bruto das imagens em um conjunto representativo, discriminatório e padronizado de atributos matemáticos e faixas discretas capazes de diferenciar tecidos sadios de folhas acometidas por **Ferrugem (*Puccinia spp.*)** e outras patologias foliares da cana-de-açúcar, preparando a Tabela Analítica Base (ABT) para a modelagem supervisionada com **Support Vector Machines (SVM)** na Sprint 3.

---

## 📂 Arquivos Desta Entrega

1. **[`engenharia_descritores_cromaticos_e_indices.md`](./engenharia_descritores_cromaticos_e_indices.md)**: Documentação técnica e agronômica da **Tarefa 1**, abordando conversão de espaços de cor (HSV / CIELAB), formulações de índices de vegetação ($ExG, ExR$), razões espectrais ($R/G$) e testes de significância estatística.
2. **[`extracao_textura_glcm_e_rugosidade.md`](./extracao_textura_glcm_e_rugosidade.md)**: Documentação técnica da **Tarefa 2**, detalhando a formulação da Matriz de Co-ocorrência em Níveis de Cinza (GLCM), propriedades de Haralick (*Contraste, Dissimilaridade, Homogeneidade, Energia*), invariância à rotação e validação da hipótese de rugosidade foliar provocada por pústulas.
3. **[`discretizacao_variaveis_continuas_binning.md`](./discretizacao_variaveis_continuas_binning.md)**: Documentação técnica da **Tarefa 3**, cobrindo a avaliação de assimetria de metadados (`size_kb`, `resolution_mp`, `aspect_ratio`, `laplacian_var`), configuração e comparação empírica do `KBinsDiscretizer` (*Uniform* vs. *Quantile* vs. *KMeans*), protocolo anti-leakage e tabela explicativa dos intervalos de corte.
4. **Módulos Executáveis:**
   - [`../src/feature_engineering_visual.py`](../src/feature_engineering_visual.py): Transformação de cor, índices e texturas Haralick.
   - [`../src/discretizacao_binning.py`](../src/discretizacao_binning.py): Discretização estatística por quantis e k-means com isolamento estrito no treino.
5. **Gráficos Diagnósticos Gerados em `docs/figures/`:**
   - [`../docs/figures/sprint2_analise_cromaticas_hsv_exg.png`](../docs/figures/sprint2_analise_cromaticas_hsv_exg.png)
   - [`../docs/figures/sprint2_analise_texturas_glcm_haralick.png`](../docs/figures/sprint2_analise_texturas_glcm_haralick.png)
   - [`../docs/figures/sprint2_distribuicao_assimetria_metadados.png`](../docs/figures/sprint2_distribuicao_assimetria_metadados.png)
   - [`../docs/figures/sprint2_comparacao_binning_uniform_vs_quantile.png`](../docs/figures/sprint2_comparacao_binning_uniform_vs_quantile.png)

---

## ✅ Cobertura do Checklist das Tarefas do Cesar (Sprint 2)

### 📋 Tarefa 1: Engenharia de Descritores Cromáticos e Índices Foliares ($ExG$, HSV, LAB)

| Item do Checklist | Status | Onde Encontrar |
| :--- | :---: | :--- |
| **Implementar conversão de canais (RGB $\rightarrow$ HSV e LAB)** | Concluído | Seção 1 de [`engenharia_descritores_cromaticos_e_indices.md`](./engenharia_descritores_cromaticos_e_indices.md) |
| **Extrair métricas estatísticas por canal (médias e desvios de Matiz/Saturação)** | Concluído | Seção 2 de [`engenharia_descritores_cromaticos_e_indices.md`](./engenharia_descritores_cromaticos_e_indices.md) |
| **Implementar fórmula do Índice de Excesso de Verde ($ExG = 2G - R - B$)** | Concluído | Seção 3 de [`engenharia_descritores_cromaticos_e_indices.md`](./engenharia_descritores_cromaticos_e_indices.md) |
| **Calcular razões espectrais $R/G$ e $G/B$ para necrose/clorose** | Concluído | Seção 3 de [`engenharia_descritores_cromaticos_e_indices.md`](./engenharia_descritores_cromaticos_e_indices.md) |
| **Validar aderência agronômica das métricas calculadas** | Concluído | Seção 4 de [`engenharia_descritores_cromaticos_e_indices.md`](./engenharia_descritores_cromaticos_e_indices.md) |
| **Gerar gráficos de dispersão comparativos entre classes** | Concluído | Gráficos salvos em `docs/figures/` e Seção 5 |

---

### 📋 Tarefa 2: Extração de Textura Haralick via GLCM (Gray-Level Co-occurrence Matrix)

| Item do Checklist | Status | Onde Encontrar |
| :--- | :---: | :--- |
| **Converter imagens para escala de cinza e quantizar intensidade** | Concluído | Seção 1 de [`extracao_textura_glcm_e_rugosidade.md`](./extracao_textura_glcm_e_rugosidade.md) |
| **Calcular matriz GLCM em 4 direções ($0^\circ, 45^\circ, 90^\circ, 135^\circ$)** | Concluído | Seção 2 de [`extracao_textura_glcm_e_rugosidade.md`](./extracao_textura_glcm_e_rugosidade.md) |
| **Extrair descritores de Haralick (Contraste, Dissimilaridade, Homogeneidade, Energia)** | Concluído | Seção 3 de [`extracao_textura_glcm_e_rugosidade.md`](./extracao_textura_glcm_e_rugosidade.md) |
| **Calcular média direcional para invariância à rotação da folha** | Concluído | Seção 3.2 de [`extracao_textura_glcm_e_rugosidade.md`](./extracao_textura_glcm_e_rugosidade.md) |
| **Testar estatisticamente a hipótese de aumento de contraste em folhas doentes** | Concluído | Seção 4 de [`extracao_textura_glcm_e_rugosidade.md`](./extracao_textura_glcm_e_rugosidade.md) |
| **Disponibilizar módulo de extração para o pipeline mestre** | Concluído | [`src/feature_engineering_visual.py`](../src/feature_engineering_visual.py) |

---

### 📋 Tarefa 3: Discretização de Variáveis Contínuas (Binning Estatístico)

| Item do Checklist do Cartão | Status | Onde Encontrar | Detalhes da Implementação |
| :--- | :---: | :--- | :--- |
| **1. Avaliar a distribuição de variáveis contínuas de metadados e identificar assimetrias acentuadas** | Concluído | Seção 2 de [`discretizacao_variaveis_continuas_binning.md`](./discretizacao_variaveis_continuas_binning.md) | Skewness e curtose calculadas para `size_kb` ($Skew = 2,74$), `resolution_mp` ($Skew = 2,55$), `aspect_ratio` ($Skew = 2,10$) e `laplacian_var` ($Skew = 1,61$). |
| **2. Configurar o transformador `KBinsDiscretizer` testando estratégias baseadas em quantis (`quantile`) e uniforme (`uniform`)** | Concluído | Seção 3 de [`discretizacao_variaveis_continuas_binning.md`](./discretizacao_variaveis_continuas_binning.md) | Comparação formal comprovando o colapso do método uniforme (91% das amostras no Bin 0) e a superioridade de quantis e k-means com 100% de equilíbrio de entropia. |
| **3. Determinar o número ideal de intervalos (bins) para cada variável selecionada** | Concluído | Seção 4 de [`discretizacao_variaveis_continuas_binning.md`](./discretizacao_variaveis_continuas_binning.md) | Selecionados $k=4$ (Quartis) para `size_kb` e `laplacian_var`, e $k=3$ (Tiers de resolução e proporção) para `resolution_mp` e `aspect_ratio`. |
| **4. Garantir que o cálculo dos limites dos bins (`fit`) ocorra estritamente no conjunto de treino** | Concluído | Seção 5 de [`discretizacao_variaveis_continuas_binning.md`](./discretizacao_variaveis_continuas_binning.md) e script | `discretizer.fit(X_train)` executado exclusivamente nas 5.574 amostras de treino, com `transform()` aplicado em validação e teste sem vazamento de dados. |
| **5. Gerar tabela explicativa com os intervalos de corte gerados e suas interpretações** | Concluído | Seção 6 de [`discretizacao_variaveis_continuas_binning.md`](./discretizacao_variaveis_continuas_binning.md) | Tabela completa com limites numéricos exatos, rótulos textuais explicativos e justificativas agronômicas e computacionais. |
