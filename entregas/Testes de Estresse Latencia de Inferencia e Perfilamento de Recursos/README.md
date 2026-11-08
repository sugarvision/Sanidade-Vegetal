# 📦 Entrega Sprint 5 — Testes de Estresse, Medição de Latência de Inferência e Perfilamento de Recursos (CPU/RAM)

**Projeto:** Sanidade-Vegetal (*SugarVision AI*)  
**Fase SEMMA:** `Assess` (Prontidão Operacional de Campo, Benchmarking de Latência e Teste de Carga)  
**Status:** ✅ **100% CONCLUÍDO, TESTADO E HOMOLOGADO**

---

## ✅ Checklist da Atividade e Evidências Auditadas

- [x] **1. Desenvolver script de benchmarking `src/test_inference_latency.py` para medir tempo de resposta por imagem (ms):**
  - Implementado e integrado à suíte `unittest` do projeto ([`../../src/test_inference_latency.py`](../../src/test_inference_latency.py)).
  - Latência média ponta a ponta aferida em **233.41 ms** (Mediana P50 = **214.65 ms**, P95 = **328.30 ms**).
- [x] **2. Realizar teste de carga processando lote de 1.000 imagens consecutivas e registrar estabilidade de memória RAM:**
  - Lote de **1.000 imagens consecutivas** processado com **100% de sucesso (0 erros)** em **235.22 segundos** (**4.25 img/s**).
  - Memória RAM RSS permaneceu estritamente estável: inicial de **163.28 MB**, final de **160.80 MB** ($\Delta\text{RAM} = -2.47\text{ MB}$, desvio padrão de $\pm 11.16\text{ MB}$), comprovando ausência de *memory leak*.
- [x] **3. Comparar o tempo de extração de features versus o tempo de decisão do SVM na etapa de inferência:**
  - **Extração de Features Visuais (Resize + HSV + $ExG/ExR$ + GLCM Haralick):** **209.38 ms** (**89.71%** do tempo total).
  - **Decisão do Pipeline SVM (`ColumnTransformer` + `SVC` + Platt):** **24.02 ms** (**10.29%** do tempo total; sendo **8.00 ms** apenas no estimador SVM).
  - Em inferência matricial vetorizada (lote de 1.000), a decisão do SVM leva apenas **0.4605 ms/imagem**.
- [x] **4. Consolidar gráficos e tabelas de perfilamento de desempenho para o Demoday e relatório final:**
  - Tabelas exportadas em [`../../docs/sprint5_benchmark_latencia_estresse_campo.csv`](../../docs/sprint5_benchmark_latencia_estresse_campo.csv) e [`../../docs/sprint5_benchmark_latencia_estresse_campo.md`](../../docs/sprint5_benchmark_latencia_estresse_campo.md).
  - Relatório técnico detalhado em [`../../docs/relatorio_testes_estresse_latencia_recursos_campo.md`](../../docs/relatorio_testes_estresse_latencia_recursos_campo.md).
  - 3 figuras de alta resolução (300 DPI) salvas em [`../../docs/figures/`](../../docs/figures/) e [`./figures/`](./figures/).

---

## 🚀 Como Executar o Benchmarking e a Suíte de Testes

```bash
# Executar o benchmarking completo de 1.000 imagens e regenerar gráficos e relatórios:
python src/test_inference_latency.py

# Executar como suíte de testes automatizados (unittest):
python -m unittest src/test_inference_latency.py -v
```
