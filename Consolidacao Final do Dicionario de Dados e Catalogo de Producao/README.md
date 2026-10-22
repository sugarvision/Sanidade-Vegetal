# 📦 Consolidação Final do Dicionário de Dados e Catálogo de Produção

**Sprint 5 — Governança MLOps, Rastreabilidade SEMMA e Congelamento de Código**  
**Projeto:** Sanidade-Vegetal (SugarVision)  
**Responsável Técnico:** Heitor Partezani (Engenheiro de Machine Learning & IA SVM)  
**Versão Oficial:** 3.0.0 (Release de Produção)  

---

## 📌 Visão Geral

Este diretório contém os entregáveis formais da tarefa de fechamento e auditoria da Sprint 5:
1. **Dicionário de Dados Auditado (v3.0.0):** Especificação completa de todas as entidades de domínio, atributos da ABT, saídas preditivas e matriz de artefatos de produção.
2. **Catálogo Mestre de Produção:** Registro unificado dos artefatos do projeto vinculados ao framework SEMMA (*Sample, Explore, Modify, Model, Assess*).
3. **Auditoria de Integridade Documental:** Validação automatizada de 45 arquivos Markdown e 236 links com **0 links quebrados** antes do congelamento final de código (*Code Freeze*).

---

## 📂 Conteúdo do Diretório

| Arquivo | Descrição e Finalidade |
| :--- | :--- |
| **[`relatorio_consolidacao_dicionario_e_catalogo_producao.md`](./relatorio_consolidacao_dicionario_e_catalogo_producao.md)** | Relatório técnico completo de consolidação final, cobrindo os 4 itens do checklist, matriz SEMMA e auditoria de links. |
| **[`dicionario_de_dados.md`](./dicionario_de_dados.md)** | Cópia sincronizada do Dicionário de Dados oficial na versão 3.0.0 com a Seção 12 completa de artefatos. |
| **[`dicionario_de_dados.csv`](./dicionario_de_dados.csv)** | Tabela tabular auditada contendo todas as variáveis e a nova entidade `ARTEFATO_PRODUCAO`. |
| **[`production_artifacts_catalog.json`](./production_artifacts_catalog.json)** | Manifesto MLOps estruturado contendo a auditoria das 5 fases SEMMA e a integridade de links. |

---

## 🚀 Como Reexecutar a Auditoria

Para reexecutar a auditoria automatizada de componentes SEMMA e integridade de links documentais:

```powershell
# Execução direta do script validador
python src/validate_documentation_and_links.py
```

Para rodar a suíte de testes automatizados de governança:

```powershell
python -m unittest src/test_production_catalog_semma.py
```
