# Entrega: Modelagem de Dados e Dicionário de Variáveis (SugarVision)

**Projeto:** Sanidade-Vegetal (SugarVision)  
**Sprints:** Sprint 1 (Modelagem Inicial) & Sprint 2 (Framework SEMMA — Atualização do Dicionário de Dados)  
**Responsável:** Cesar (Lead Técnico & Visão Computacional) e Equipe de Ciência de Dados  
**Versão:** 2.0.0 (Setembro de 2026)  

---

## 📌 Visão Geral da Entrega

Esta pasta consolida a estruturação formal, governança e modelagem analítica dos dados para o projeto **Sanidade-Vegetal (SugarVision)**, cujo escopo é o diagnóstico automatizado e classificação inteligente de patologias foliares na cultura de cana-de-açúcar (*Saccharum officinarum*).

Na **Sprint 1**, estruturou-se a modelagem conceitual das entidades de campo (`TALHAO`, `PLANTA`, `COLETA_DATA`, `IMAGEM`, `DIAGNOSTICO`).  
Na **Sprint 2 (Fase Modify)**, realizou-se a **Atualização Completa do Dicionário de Dados e da Tabela Analítica Base (ABT)**, incorporando formalmente todas as novas variáveis numéricas (espaço HSV, índices de vegetação $ExG/ExR$, razões espectrais, descritores de textura Haralick via GLCM e métricas de qualidade de imagem), com especificação matemática, tipagem estatística e analítica, intervalos teóricos, regras de validação e auditoria no padrão `snake_case`.

---

## 📂 Arquivos Desta Entrega

1. **[`dicionario_de_dados.md`](./dicionario_de_dados.md)**: Documentação formal completa do catálogo de variáveis (Versão 2.0.0), contendo taxonomia analítica, formulações matemáticas em LaTeX, unidades de medida, intervalos teóricos, regras de validação contra vazamento de dados e auditoria de nomenclatura.
2. **[`dicionario_de_dados.csv`](./dicionario_de_dados.csv)**: Dicionário tabular em formato CSV consumível por ferramentas de catálogo de metadados, governança e pipelines de machine learning (inclui colunas: `entidade`, `nome_coluna`, `tipo_primitivo`, `tipo_estatistico`, `papel_analitico`, `formula_calculo`, `unidade_medida`, `intervalo_teorico`, `descricao`, `dominio_exemplo`, `permite_nulo`).
3. **[`modelagem_conceitual_entidades.md`](./modelagem_conceitual_entidades.md)**: Modelagem relacional e conceitual das entidades do ecossistema agrícola com diagramas Mermaid atualizados (MER/DER) e mapeamento para a Tabela Analítica Base (ABT).
4. **Artefato de Dados Relacionado:**
   - [`../data/processed/abt_sanidade_vegetal.csv`](../data/processed/abt_sanidade_vegetal.csv): Tabela Analítica Base (ABT) consolidada com 6.571 registros e 28 colunas em estrito padrão `snake_case`.

---

## ✅ Cobertura do Checklist do Cartão: "Atualização do Dicionário de Dados" (Sprint 2)

| Item do Checklist do Cartão | Status | Onde Encontrar | Detalhes da Implementação |
| :--- | :---: | :--- | :--- |
| **1. Catalogar todas as novas variáveis numéricas (cores, índices, texturas) e metadados gerados** | Concluído | Seção 4 de [`dicionario_de_dados.md`](./dicionario_de_dados.md) e [`dicionario_de_dados.csv`](./dicionario_de_dados.csv) | Catalogadas 48 variáveis, incluindo as 28 colunas da ABT (`mean_hue`, `std_saturation`, `exg_index`, `exr_index`, `rg_ratio`, `indice_clorose_necrose`, `hue_dispersion`, `glcm_contrast`, `glcm_dissimilarity`, `glcm_homogeneity`, `glcm_energy`, `indice_rugosidade_pustula`, `laplacian_var`, dimensões e metadados). |
| **2. Definir o papel analítico de cada coluna (Target, Feature Preditiva Contínua, Discreta, Identificador)** | Concluído | Seção 3 e 4 de [`dicionario_de_dados.md`](./dicionario_de_dados.md) e [`dicionario_de_dados.csv`](./dicionario_de_dados.csv) | Todas as colunas possuem classificação formal padronizada em: `Identificador`, `Target`, `Feature Preditiva Contínua`, `Feature Preditiva Discreta` ou `Metadado Técnico`. |
| **3. Descrever detalhadamente a fórmula, unidade de medida e intervalo teórico de cada atributo** | Concluído | Seção 4 e 5 de [`dicionario_de_dados.md`](./dicionario_de_dados.md) e [`dicionario_de_dados.csv`](./dicionario_de_dados.csv) | Equações matemáticas expressas em LaTeX, fundamentação biofísica, unidades (graus, px, KB, adimensional) e intervalos teóricos e práticos documentados. |
| **4. Atualizar os arquivos formais `dicionario_de_dados.md` e `dicionario_de_dados.csv`** | Concluído | [`dicionario_de_dados.md`](./dicionario_de_dados.md) e [`dicionario_de_dados.csv`](./dicionario_de_dados.csv) | Ambos os artefatos foram totalmente reformulados e sincronizados para a versão 2.0.0. |
| **5. Auditar a conformidade de nomes de colunas seguindo o padrão snake_case** | Concluído | Seção 7 de [`dicionario_de_dados.md`](./dicionario_de_dados.md) | Auditoria automatizada executada via regex `^[a-z][a-z0-9_]*$`, confirmando 100% de conformidade de todas as variáveis do catálogo e da ABT. |

---

## 🔄 Rastreabilidade com a Sprint 1

| Item Original da Sprint 1 | Status | Atualização na Sprint 2 |
| :--- | :---: | :--- |
| **Listar colunas ou atributos principais** | Mantido | Expandido de 46 para 48 atributos detalhados. |
| **Definir tipo de cada variável** | Mantido | Tipagem primitiva e estatística enriquecida com a taxonomia analítica. |
| **Identificar variáveis alvo (targets)** | Mantido | Mantida a tríade de targets: multiclasse (`class_label`), binário (`target_binary`) e regressão (`target_grau_severidade_pct`). |
| **Identificar variáveis candidatas a features** | Aprimorado | Features candidatas agora foram matematicamente extraídas, testadas com p-valor $< 10^{-15}$ e consolidadas na ABT. |
| **Registrar possíveis entidades** | Mantido | Entidades `TALHAO`, `PLANTA`, `COLETA_DATA`, `IMAGEM`, `DIAGNOSTICO` e `FEATURES_EXTRACTED` sincronizadas. |
