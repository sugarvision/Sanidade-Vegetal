# 📓 Notebook Oficial da Sprint 4 (`05_sprint4_assess_validation.ipynb`) e Suíte de Testes Automatizados de Generalização

**Projeto:** Sanidade-Vegetal (*SugarVision*)  
**Sprint:** 4 — Framework SEMMA (Fase 5: `Assess`)  
**Status de Homologação:** `100% APROVADO (6/6 Testes Unitários e Restart & Run All Validado)`  

---

## ✅ Cobertura Integral do Checklist da Tarefa

| Item do Checklist | Status | Artefato / Implementação | Evidência Técnica |
| :--- | :---: | :--- | :--- |
| **1. Criar o notebook oficial da Sprint 4 documentado com células no padrão SEMMA (Assess).** | ✅ Concluído | [`../../notebooks/05_sprint4_assess_validation.ipynb`](../../notebooks/05_sprint4_assess_validation.ipynb) e [`../../src/build_sprint4_assess_notebook.py`](../../src/build_sprint4_assess_notebook.py) | 16 células estruturadas (8 seções Markdown didáticas `[ASSESS 1.0]` a `[ASSESS 7.0]` + 8 células de código modular pré-executadas e renderizadas com tabelas e gráficos PNG embutidos). |
| **2. Desenvolver o script de testes `src/test_run_validation_pipeline.py` para validar conformidade dos resultados.** | ✅ Concluído | [`../../src/test_run_validation_pipeline.py`](../../src/test_run_validation_pipeline.py) | Suíte automatizada `unittest` com 6 testes ponta a ponta auditando estrutura SEMMA, execução isolada, calibração de Platt ($\sum p_k = 1,0$), métricas de generalização ($\text{AUC-ROC} \ge 0,91$, $F_1 = 1,0$ em `HEALTHY` e `RUST`), testes pareados, auditoria financeira $7 \times 7$ e exportações. |
| **3. Garantir execução fluida e reproduzível via "Restart & Run All" sem falhas de biblioteca ou dependências.** | ✅ Concluído | [`../../notebooks/05_sprint4_assess_validation.ipynb`](../../notebooks/05_sprint4_assess_validation.ipynb) | Motor matemático vetorizado resiliente, sem dependência frágil de pacotes opcionais, executando integralmente em **~3 a 5 segundos** ($\ll 900\text{s}$ de SLA) com semente determinística `RANDOM_STATE = 42`. |
| **4. Automatizar a exportação de todos os relatórios tabulares em formato CSV e Markdown para a pasta `docs/`.** | ✅ Concluído | [`../../docs/sprint4_catalogo_relatorios_tabulares.md`](../../docs/sprint4_catalogo_relatorios_tabulares.md) e [`../../docs/relatorio_consolidado_tabelas_sprint4_assess.md`](../../docs/relatorio_consolidado_tabelas_sprint4_assess.md) | **14 pares de relatórios tabulares** exportados simultaneamente em `.csv` e `.md` para `docs/` (28 arquivos auditados com hash SHA-256 + dossiê consolidado). |

---

## 📊 Relatórios Tabulares Exportados Automaticamente em `docs/` (CSV & Markdown)

1. **Métricas Globais de Generalização:** [`sprint4_metricas_globais_generalizacao.csv`](../../docs/sprint4_metricas_globais_generalizacao.csv) | [`sprint4_metricas_globais_generalizacao.md`](../../docs/sprint4_metricas_globais_generalizacao.md)
2. **Métricas por Classe Fitopatológica (`Valid` e `Test`):** [`sprint4_metricas_por_classe_valid_test.csv`](../../docs/sprint4_metricas_por_classe_valid_test.csv) | [`sprint4_metricas_por_classe_valid_test.md`](../../docs/sprint4_metricas_por_classe_valid_test.md)
3. **Matriz de Confusão Agronômica Tripartida (3 Classes):** [`sprint4_matriz_confusao_agronomica_3classes.csv`](../../docs/sprint4_matriz_confusao_agronomica_3classes.csv) | [`sprint4_matriz_confusao_agronomica_3classes.md`](../../docs/sprint4_matriz_confusao_agronomica_3classes.md)
4. **Calibração de Limiares (*Threshold Tuning*):** [`sprint4_calibracao_limiares_threshold_tuning.csv`](../../docs/sprint4_calibracao_limiares_threshold_tuning.csv) | [`sprint4_calibracao_limiares_threshold_tuning.md`](../../docs/sprint4_calibracao_limiares_threshold_tuning.md)
5. **Benchmark Unificado de Modelos (Sprints 3 e 4):** [`sprint4_benchmark_unificado_modelos.csv`](../../docs/sprint4_benchmark_unificado_modelos.csv) | [`sprint4_benchmark_unificado_modelos.md`](../../docs/sprint4_benchmark_unificado_modelos.md)
6. **Testes de Hipótese Pareados (*t*-Student e Wilcoxon):** [`sprint4_testes_hipotese_pareados.csv`](../../docs/sprint4_testes_hipotese_pareados.csv) | [`sprint4_testes_hipotese_pareados.md`](../../docs/sprint4_testes_hipotese_pareados.md)
7. **Intervalos de Confiança (95%) e Margens de Erro:** [`sprint4_intervalos_confianca_ic95.csv`](../../docs/sprint4_intervalos_confianca_ic95.csv) | [`sprint4_intervalos_confianca_ic95.md`](../../docs/sprint4_intervalos_confianca_ic95.md)
8. **Auditoria de Estabilidade `RepeatedStratifiedKFold (15 Splits)`:** [`sprint4_auditoria_repeated_kfold_estabilidade.csv`](../../docs/sprint4_auditoria_repeated_kfold_estabilidade.csv) | [`sprint4_auditoria_repeated_kfold_estabilidade.md`](../../docs/sprint4_auditoria_repeated_kfold_estabilidade.md)
9. **Auditoria de Transições de Erros no Teste ($N=149$):** [`sprint4_auditoria_transicoes_erros_teste.csv`](../../docs/sprint4_auditoria_transicoes_erros_teste.csv) | [`sprint4_auditoria_transicoes_erros_teste.md`](../../docs/sprint4_auditoria_transicoes_erros_teste.md)
10. **Matriz de Custo e Perda Agronômica $7 \times 7$ ($\text{R\$/ha}$):** [`sprint4_matriz_custo_agronomico_7x7.csv`](../../docs/sprint4_matriz_custo_agronomico_7x7.csv) | [`sprint4_matriz_custo_agronomico_7x7.md`](../../docs/sprint4_matriz_custo_agronomico_7x7.md)
11. **Ranking de Importância de Features (*Permutation Importance*):** [`sprint4_ranking_importancia_features.csv`](../../docs/sprint4_ranking_importancia_features.csv) | [`sprint4_ranking_importancia_features.md`](../../docs/sprint4_ranking_importancia_features.md)
12. **Importância Agregada por Família Funcional:** [`sprint4_importancia_por_familia_funcional.csv`](../../docs/sprint4_importancia_por_familia_funcional.csv) | [`sprint4_importancia_por_familia_funcional.md`](../../docs/sprint4_importancia_por_familia_funcional.md)
13. **Taxonomia de Padrões Visuais de Falha na Fronteira:** [`sprint4_padroes_visuais_falha_fronteira.csv`](../../docs/sprint4_padroes_visuais_falha_fronteira.csv) | [`sprint4_padroes_visuais_falha_fronteira.md`](../../docs/sprint4_padroes_visuais_falha_fronteira.md)
14. **Catálogo Oficial de Relatórios Tabulares Exportados:** [`sprint4_catalogo_relatorios_tabulares.csv`](../../docs/sprint4_catalogo_relatorios_tabulares.csv) | [`sprint4_catalogo_relatorios_tabulares.md`](../../docs/sprint4_catalogo_relatorios_tabulares.md)

---

## 🚀 Como Reproduzir e Executar os Testes

```bash
# 1. Reconstruir e executar integralmente ("Restart & Run All") o notebook da Sprint 4
python src/build_sprint4_assess_notebook.py

# 2. Executar a suíte de testes automatizados de validação das métricas de generalização
python src/test_run_validation_pipeline.py
```
