# Catálogo Formal de Metadados, Dicionário de Dados e Saídas Calibradas

## 📌 Contexto da Tarefa (Sprint 4)
Módulo da **Sprint 4** do projeto **Sanidade-Vegetal (SugarVision)**, focado na atualização do catálogo formal de metadados, expansão do dicionário de dados (`dicionario_de_dados.md` e `dicionario_de_dados.csv`), inclusão da entidade de métricas de avaliação formal do modelo SVM e atestado de linhagem e rastreabilidade ponta a ponta (*Data Lineage*).

---

## 🎯 Checklist da Tarefa e Resultados

- [x] **Atualizar os arquivos `dicionario_de_dados.md` e `dicionario_de_dados.csv` com as métricas de validação**:
  - Dicionário de dados atualizado para a versão **2.2.0**.
  - No `dicionario_de_dados.csv`, criada a entidade analítica **`METRICAS_VALIDACAO_SVM`**, catalogando 11 métricas formais com fórmulas, tipos, unidades e valores observados.
  - No `dicionario_de_dados.md`, adicionadas as Seções 9 (Catálogo de Métricas), 10 (Hiperparâmetros Ótimos e Restrições) e 11 (Auditoria e Linhagem).
- [x] **Catalogar as métricas de avaliação formal: Precision, Recall, F1-Score (macro/weighted), AUC-ROC e Specificity**:
  - Todas as 5 famílias calculadas estritamente nas partições de validação independente ($N=617$) e teste cego final ($N=380$).
  - **AUC-ROC Macro (OvR):** **0,9148** (validação) e **0,9116** (teste).
  - **Specificity Macro (OvR):** **0,9381** (validação) e **0,9361** (teste).
  - **$F_1$-Score Macro:** **0,5757** (validação) e **0,6054** (teste).
  - **Separabilidade Perfeita (100% de acerto e AUC=1.000)** comprovada para as patologias críticas **HEALTHY** e **RUST (Ferrugem da Cana)**.
- [x] **Documentar os hiperparâmetros ótimos homologados e suas restrições operacionais**:
  - Estimador campeão oficial: SVM com Kernel RBF ($C=10.0, \gamma=0.01$, `class_weight='balanced'`, `probability=True`, `StandardScaler`).
  - Matriz de 5 restrições operacionais documentada (dimensionalidade, calibração Platt Scaling, limites de z-score, latência de 0,12 ms/amostra e compatibilidade de versionamento).
- [x] **Auditar e atestar a rastreabilidade completa entre os dados brutos e os resultados de validação**:
  - Auditoria automatizada via `src/audit_metadata_lineage.py` com status **APPROVED**.
  - 100% de integridade referencial: 997 registros de predições mapeiam 1:1 para a ABT via `sample_id`, e 100% dos arquivos da ABT rastreiam até os metadados brutos via `filename`.

---

## 📁 Estrutura de Arquivos

```text
├── Catalogo Formal de Metadados e Dicionario com Saidas Calibradas/
│   ├── README.md                                             # Este resumo executivo
│   ├── relatorio_governanca_catalogo_metricas_calibradas.md  # Relatório técnico completo
│   ├── dicionario_de_dados.md                                # Cópia sincronizada do dicionário v2.2.0
│   ├── dicionario_de_dados.csv                               # Cópia sincronizada do CSV com métricas
│   ├── dataset_lineage_and_metrics_catalog.json              # Manifesto formal de governança MLOps
│   └── figures/
│       ├── svm_multiclass_roc_curves.png                     # Curvas ROC One-vs-Rest
│       └── svm_validation_metrics_comparison.png             # Comparativo das métricas de avaliação
├── Modelagem Inicial e Dicionário de Dados/
│   ├── dicionario_de_dados.md                                # Dicionário oficial atualizado (v2.2.0)
│   └── dicionario_de_dados.csv                               # CSV oficial atualizado com métricas
├── src/
│   ├── audit_metadata_lineage.py                             # Script de cálculo e auditoria de linhagem
│   └── test_metadata_catalog_governance.py                   # Testes automatizados de governança
└── models/
    └── dataset_lineage_and_metrics_catalog.json              # Registro unificado de linhagem e métricas
```

---

## 🚀 Como Executar

Para rodar a auditoria de linhagem e regerar o catálogo de métricas e figuras:

```bash
python src/audit_metadata_lineage.py
```

Para rodar a suíte de testes automatizados:

```bash
python -m unittest src/test_metadata_catalog_governance.py
```
