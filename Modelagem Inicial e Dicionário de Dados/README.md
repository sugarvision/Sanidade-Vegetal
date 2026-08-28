# Entrega: Modelagem Inicial e Dicionário de Dados
**Projeto:** Sanidade-Vegetal (SugarVision)  
**Sprint:** 1  
**Responsável:** Equipe de Ciência de Dados  
**Data:** 28 de Agosto de 2026  

---

## 📌 Visão Geral da Entrega

Esta entrega consolida a estruturação formal dos dados para o projeto **Sanidade-Vegetal (SugarVision)**, focado no diagnóstico e classificação inteligente de patologias foliares na cultura de cana-de-açúcar (*Saccharum officinarum*).

O objetivo é padronizar e documentar a modelagem conceitual, entidades de domínio agronômico e computacional, dicionário de variáveis com tipagem estatística e analítica, identificação de targets (variáveis alvo) e features preditivas para pipelines de Machine Learning e Visão Computacional.

---

## 📂 Arquivos Desta Entrega

1. **[`dicionario_de_dados.md`](./dicionario_de_dados.md)**: Dicionário detalhado de dados em Markdown, contendo entidades, atributos, tipos, descrições, papéis no modelo e regras de validação.
2. **[`dicionario_de_dados.csv`](./dicionario_de_dados.csv)**: Dicionário tabular em formato CSV para consumo automatizado em pipelines e governança de dados.
3. **[`modelagem_conceitual_entidades.md`](./modelagem_conceitual_entidades.md)**: Modelagem relacional e conceitual das entidades do ecossistema (*Talhão, Planta, Coleta/Data, Imagem, Diagnóstico, Features*), incluindo diagramas Mermaid (DER) e cardinalidades.

---

## ✅ Cobertura do Checklist da Task

| Item do Checklist | Status | Onde Encontrar |
| :--- | :---: | :--- |
| **Listar colunas ou atributos principais** | Concluído | [`dicionario_de_dados.md`](./dicionario_de_dados.md) e [`dicionario_de_dados.csv`](./dicionario_de_dados.csv) |
| **Definir tipo de cada variável** | Concluído | Colunas *Tipo de Dado* e *Tipo Estatístico* |
| **Identificar variáveis alvo (targets)** | Concluído | Seção 3 de [`dicionario_de_dados.md`](./dicionario_de_dados.md) |
| **Identificar variáveis candidatas a features** | Concluído | Seção 4 de [`dicionario_de_dados.md`](./dicionario_de_dados.md) |
| **Registrar possíveis entidades (Imagem, Talhão, Planta, Data)** | Concluído | [`modelagem_conceitual_entidades.md`](./modelagem_conceitual_entidades.md) |
| **Criar dicionário de dados em Markdown ou CSV** | Concluído | Ambos os formatos disponíveis (`.md` e `.csv`) |
