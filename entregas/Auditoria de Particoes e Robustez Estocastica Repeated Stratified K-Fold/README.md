# Auditoria de Partições e Robustez Estocástica (Repeated Stratified K-Fold)

## 📌 Contexto da Tarefa (Sprint 4)
Este módulo é parte das entregas da **Sprint 4** do projeto **Sanidade-Vegetal (SugarVision)**, focado na comprovação matemática de que o desempenho preditivo do modelo **SVM com Kernel RBF** não é fruto de viés amostral (*sampling bias* ou *partition luck*).

### 🎯 Objetivos Centrais
1. Executar bateria de validação cruzada com **Repeated Stratified K-Fold** ($5 \text{ splits} \times 3 \text{ repetições} = 15 \text{ folds}$), garantindo isolamento total anti-leakage.
2. Mensurar e mapear a **dispersão estatística** das métricas de $F_1$-Score (Macro e Weighted) e **Acurácia Balanceada**.
3. Auditar a **robustez da convergência do estimador SVM** frente a diferentes permutações das amostras, analisando estabilidade dos vetores de suporte ($N_{SV}$), tempo de fit e executando testes de hipótese inferenciais (ANOVA e Kruskal-Wallis).
4. Documentar de forma completa e rastreável os resultados e formalizações matemáticas no relatório oficial da sprint.

---

## 📋 Resumo das Evidências Obtidas

- **Particionamento:** `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)`
- **Amostras Auditadas:** 5.574 instâncias distribuídas em 7 patologias de cana-de-açúcar.
- **Divergência Máxima de Estratificação:** $\le 0,07\%$ frente à população global (estratificação estritamente perfeita).
- **Dispersão do $F_1$-Macro:** Média $= 0,5761 \pm 0,0082$ | **$CV = 1,43\%$** | $IC_{95\%} = [0,5716 ; 0,5807]$.
- **Dispersão da Acurácia Balanceada:** Média $= 64,46\% \pm 1,21\%$ | **$CV = 1,87\%$** | $IC_{95\%} = [63,79\% ; 65,13\%]$.
- **Acurácia Global:** Média $= 64,48\% \pm 0,79\%$ | **$CV = 1,22\%$**.
- **Convergência SVM:** Média $= 2.778,1 \pm 16,4$ Vetores de Suporte (**$CV = 0,59\%$**), tempo médio $= 0,70$s por fold.
- **Teste ANOVA One-Way:**
  - $F_1$-Macro: $F = 0,1850$, $p = 0,8334$ ($p \gg 0,05 \implies$ não rejeita $H_0$).
  - Acurácia Balanceada: $F = 0,1130$, $p = 0,8941$ ($p \gg 0,05 \implies$ não rejeita $H_0$).
- **Conclusão:** **Ausência total de viés amostral**. O estimador é robusto e invariante a permutações.

---

## 📁 Estrutura de Arquivos da Entrega

```text
├── Auditoria de Particoes e Robustez Estocastica Repeated Stratified K-Fold/
│   ├── README.md                                             # Este resumo executivo
│   ├── relatorio_auditoria_particoes_robustez_estocastica.md # Relatório técnico completo da sprint
│   ├── repeated_kfold_audit_results.csv                      # Tabela com as 15 iterações e métricas
│   ├── repeated_kfold_audit_metadata.json                    # Manifesto MLOps estruturado
│   └── figures/
│       ├── repeated_kfold_metrics_dispersion.png             # Boxplots de dispersão de métricas
│       ├── repeated_kfold_class_stratification_audit.png     # Auditoria de estratificação classe a classe
│       └── repeated_kfold_convergence_stability.png         # Estabilidade de vetores de suporte e tempo
├── src/
│   └── repeated_stratified_kfold_audit.py                    # Módulo Python executável de auditoria
└── models/
    ├── repeated_kfold_audit_results.csv                      # Cópia para governança de modelos
    └── repeated_kfold_audit_metadata.json                    # Cópia para governança de modelos
```

---

## 🚀 Como Reproduzir os Experimentos

Para reexecutar a auditoria e regerar todas as tabelas, métricas e gráficos:

```bash
python src/repeated_stratified_kfold_audit.py
```
