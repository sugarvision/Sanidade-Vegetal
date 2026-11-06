# 📊 Matriz de Confusão Agronômica Tripartida: Saudável vs. Ferrugem vs. Outras Patologias

**Projeto:** Sanidade-Vegetal (*SugarVision*) | **Sprint:** 4 | **Fase SEMMA:** `Assess 2.0 — Matriz Agronômica 3 Classes`  
**Arquivo CSV Pareado:** `sprint4_matriz_confusao_agronomica_3classes.csv` | **Dimensões:** `6 linhas x 10 colunas`

> Auditoria de separabilidade absoluta e normalizada no agrupamento fitossanitário de 3 classes, atestando 100% de sensibilidade e zero confusão cruzada para Folha Sadia (HEALTHY) e Ferrugem (RUST).

## Tabela de Dados Auditada

| Particao | Grupo_Real | Suporte_Real | Pred_Saudavel_Abs | Pred_Ferrugem_Abs | Pred_Outras_Abs | Pred_Saudavel_Pct | Pred_Ferrugem_Pct | Pred_Outras_Pct | Sensibilidade_Grupo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Validacao (N=617) | SAUDÁVEL (HEALTHY) | 95 | 95 | 0 | 0 | 100 | 0 | 0 | 1 |
| Validacao (N=617) | FERRUGEM (RUST) | 91 | 0 | 91 | 0 | 0 | 100 | 0 | 1 |
| Validacao (N=617) | OUTRAS PATOLOGIAS | 431 | 0 | 0 | 431 | 0 | 0 | 100 | 1 |
| Teste Cego (N=380) | SAUDÁVEL (HEALTHY) | 46 | 46 | 0 | 0 | 100 | 0 | 0 | 1 |
| Teste Cego (N=380) | FERRUGEM (RUST) | 51 | 0 | 51 | 0 | 0 | 100 | 0 | 1 |
| Teste Cego (N=380) | OUTRAS PATOLOGIAS | 283 | 0 | 0 | 283 | 0 | 0 | 100 | 1 |
