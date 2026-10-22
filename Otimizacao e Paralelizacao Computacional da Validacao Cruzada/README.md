# Otimização e Paralelização Computacional da Validação Cruzada (Grid Search)

## 📌 Contexto da Tarefa (Sprint 4)
Módulo da **Sprint 4** do projeto **Sanidade-Vegetal (SugarVision)**, focado na otimização de desempenho de hardware, paralelização massiva (`n_jobs=-1`), controle rigoroso de memória RAM e tolerância a falhas na execução da malha de validação cruzada do **Support Vector Machine (SVM)** com kernels pesados (RBF e Polinomial).

---

## 🎯 Checklist da Tarefa e Resultados

- [x] **Configurar execução paralela no Scikit-Learn utilizando `n_jobs=-1` com controle de consumo de memória RAM**:
  - Utilização de todos os 12 núcleos lógicos da CPU via backend `loky`.
  - Controle de serialização IPC com `max_nbytes='10M'` e barreira de segurança de 85% de RAM do host.
  - Pico de consumo de RAM do processo + subprocessos restrito a **2.346,3 MB** (sem vazamento).
- [x] **Monitorar tempo total de execução e pico de uso de CPU durante a validação cruzada dos kernels pesados (RBF e Poly)**:
  - Telemetria assíncrona contínua (694 amostras a cada 100ms).
  - **Pico de CPU de 100,0%** (utilização multi-core exaustiva) e média de 78,4%.
  - 135 fits concluídos em apenas **66,78 segundos** (Throughput de **2,02 fits/s**).
- [x] **Implementar rotina de checkpoint para salvar os resultados intermediários do GridSearch contra interrupções**:
  - Persistência incremental atômica a cada candidato em JSON e CSV.
  - Recuperação instantânea comprovada (*Cache Hit de 100%* com **0,00s** de reprocessamento em caso de reinício).
- [x] **Gerar relatório de eficiência computacional e perfilamento de tempo para a Sprint 4**:
  - Relatório formal detalhado em `relatorio_eficiencia_computacional_paralelizacao.md`.
  - 3 figuras analíticas de alta resolução em `figures/` e `docs/figures/`.
  - Manifesto JSON de governança MLOps em `computational_efficiency_metrics.json`.

---

## 📁 Estrutura de Arquivos da Entrega

```text
├── Otimizacao e Paralelizacao Computacional da Validacao Cruzada/
│   ├── README.md                                             # Este resumo executivo
│   ├── relatorio_eficiencia_computacional_paralelizacao.md   # Relatório técnico completo
│   ├── gridsearch_checkpoint_results.csv                     # Tabela de resultados dos 27 candidatos
│   ├── computational_efficiency_metrics.json                 # Manifesto de telemetria de hardware
│   └── figures/
│       ├── computational_efficiency_cpu_ram_profiling.png    # Telemetria contínua de CPU e RAM
│       ├── kernel_execution_time_comparison.png              # Comparativo de tempo Linear vs RBF vs Poly
│       └── checkpoint_resilience_timeline.png                # Linha do tempo de persistência incremental
├── src/
│   ├── resilient_parallel_grid_search.py                     # Módulo executável de paralelização e checkpoints
│   └── test_parallel_efficiency.py                           # Suite de testes unitários automatizados (4/4 OK)
└── models/
    ├── gridsearch_checkpoint.json                            # Checkpoint transacional atômico
    ├── gridsearch_checkpoint_results.csv                     # Resultados consolidados
    └── computational_efficiency_metrics.json                 # Registro unificado de eficiência
```

---

## 🚀 Como Executar

Para rodar a busca paralela com monitoramento de telemetria e checkpoints incrementais:

```bash
python src/resilient_parallel_grid_search.py
```

Para rodar a suíte de testes unitários automatizados:

```bash
python -m unittest src/test_parallel_efficiency.py
```
