# Entrega: Engenharia de Atributos e Descritores Visuais (Fase Modify)
**Projeto:** Sanidade-Vegetal (SugarVision)  
**Sprint:** 2 — Framework SEMMA (Fase: Modify)  
**Responsável:** Cesar (Lead Técnico & Visão Computacional)  
**Data:** Setembro de 2026  

---

## 📌 Visão Geral da Entrega

Esta entrega formaliza as atividades e artefatos de **Engenharia de Descritores Cromáticos, Índices Espectrais e Extração de Textura Haralick (GLCM)** de responsabilidade do membro **Cesar** para a **Sprint 2 (Fase Modify)**.

O objetivo central foi converter o sinal bruto das imagens em um conjunto representativo, discriminatório e padronizado de atributos matemáticos capazes de diferenciar tecidos sadios de folhas acometidas por **Ferrugem (*Puccinia spp.*)** e outras patologias foliares da cana-de-açúcar, preparando a Tabela Analítica Base (ABT) para a modelagem com **Support Vector Machines (SVM)** na Sprint 3.

---

## 📂 Arquivos Desta Entrega

1. **[`engenharia_descritores_cromaticos_e_indices.md`](./engenharia_descritores_cromaticos_e_indices.md)**: Documentação técnica e agronômica da **Tarefa 1**, abordando conversão de espaços de cor (HSV / CIELAB), formulações de índices de vegetação ($ExG, ExR$), razões espectrais ($R/G$) e testes de significância estatística.
2. **[`extracao_textura_glcm_e_rugosidade.md`](./extracao_textura_glcm_e_rugosidade.md)**: Documentação técnica da **Tarefa 2**, detalhando a formulação da Matriz de Co-ocorrência em Níveis de Cinza (GLCM), propriedades de Haralick (*Contraste, Dissimilaridade, Homogeneidade, Energia*), invariância à rotação e validação da hipótese de rugosidade foliar provocada por pústulas.
3. **[`../src/feature_engineering_visual.py`](../src/feature_engineering_visual.py)**: Módulo Python executável contendo as funções de transformação, cálculo de índices compostos e geração automatizada de gráficos diagnósticos.
4. **Gráficos Diagnósticos:**
   - [`../docs/figures/sprint2_analise_cromaticas_hsv_exg.png`](../docs/figures/sprint2_analise_cromaticas_hsv_exg.png)
   - [`../docs/figures/sprint2_analise_texturas_glcm_haralick.png`](../docs/figures/sprint2_analise_texturas_glcm_haralick.png)

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
