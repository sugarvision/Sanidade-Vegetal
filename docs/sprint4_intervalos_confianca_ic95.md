# 📊 Intervalos de Confiança (95%) e Margens de Erro entre Folds de Validação Cruzada

**Projeto:** Sanidade-Vegetal (*SugarVision*) | **Sprint:** 4 | **Fase SEMMA:** `Assess 4.0 — Intervalos de Confiança 95%`  
**Arquivo CSV Pareado:** `sprint4_intervalos_confianca_ic95.csv` | **Dimensões:** `42 linhas x 16 colunas`

> Documentação formal de Média, Desvio Padrão, Erro Padrão (SE), Margem de Erro bicaudal a 95% e Coeficiente de Variação para todos os modelos.

## Tabela de Dados Auditada

| Modelo | Familia | Metrica | N_Folds | Graus_Liberdade | Nivel_Confianca | Valor_Critico_t | Media | Desvio_Padrao | Erro_Padrao_SE | Margem_Erro_ME | IC_Inferior | IC_Superior | IC_Formatado | Media_com_ME_Formatada | Coef_Variacao_Pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy Classifier (Stratified) | Baseline Ingênuo | F1_Macro | 5 | 4 | 95% | 2.7764 | 0.1322 | 0.0021 | 0.0009 | 0.0026 | 0.1296 | 0.1348 | [12.96%, 13.48%] | 13.22% ± 0.26% | 1.5800 |
| Dummy Classifier (Stratified) | Baseline Ingênuo | Acurácia | 5 | 4 | 95% | 2.7764 | 0.1640 | 0.0018 | 0.0008 | 0.0022 | 0.1617 | 0.1662 | [16.17%, 16.62%] | 16.40% ± 0.22% | 1.1000 |
| Dummy Classifier (Stratified) | Baseline Ingênuo | Precisão_Macro | 5 | 4 | 95% | 2.7764 | 0.1322 | 0.0020 | 0.0009 | 0.0024 | 0.1298 | 0.1346 | [12.98%, 13.46%] | 13.22% ± 0.24% | 1.4800 |
| Dummy Classifier (Stratified) | Baseline Ingênuo | Recall_Macro | 5 | 4 | 95% | 2.7764 | 0.1325 | 0.0022 | 0.0010 | 0.0028 | 0.1297 | 0.1352 | [12.97%, 13.52%] | 13.25% ± 0.28% | 1.6800 |
| Regressão Logística L2 (Sem Ponderação) | Baseline Linear | F1_Macro | 5 | 4 | 95% | 2.7764 | 0.5595 | 0.0143 | 0.0064 | 0.0177 | 0.5418 | 0.5773 | [54.18%, 57.73%] | 55.95% ± 1.77% | 2.5500 |
| Regressão Logística L2 (Sem Ponderação) | Baseline Linear | Acurácia | 5 | 4 | 95% | 2.7764 | 0.6500 | 0.0123 | 0.0055 | 0.0153 | 0.6347 | 0.6653 | [63.47%, 66.53%] | 65.00% ± 1.53% | 1.9000 |
| Regressão Logística L2 (Sem Ponderação) | Baseline Linear | Precisão_Macro | 5 | 4 | 95% | 2.7764 | 0.6238 | 0.0446 | 0.0199 | 0.0553 | 0.5684 | 0.6791 | [56.84%, 67.91%] | 62.38% ± 5.53% | 7.1400 |
| Regressão Logística L2 (Sem Ponderação) | Baseline Linear | Recall_Macro | 5 | 4 | 95% | 2.7764 | 0.6001 | 0.0104 | 0.0047 | 0.0129 | 0.5872 | 0.6130 | [58.72%, 61.30%] | 60.01% ± 1.29% | 1.7300 |
| Regressão Logística L2 (Ponderada) | Baseline Linear | F1_Macro | 5 | 4 | 95% | 2.7764 | 0.5690 | 0.0066 | 0.0029 | 0.0082 | 0.5609 | 0.5772 | [56.09%, 57.72%] | 56.90% ± 0.82% | 1.1600 |
| Regressão Logística L2 (Ponderada) | Baseline Linear | Acurácia | 5 | 4 | 95% | 2.7764 | 0.6358 | 0.0102 | 0.0045 | 0.0126 | 0.6232 | 0.6484 | [62.32%, 64.84%] | 63.58% ± 1.26% | 1.6000 |
| Regressão Logística L2 (Ponderada) | Baseline Linear | Precisão_Macro | 5 | 4 | 95% | 2.7764 | 0.5709 | 0.0280 | 0.0125 | 0.0348 | 0.5360 | 0.6057 | [53.60%, 60.57%] | 57.09% ± 3.48% | 4.9100 |
| Regressão Logística L2 (Ponderada) | Baseline Linear | Recall_Macro | 5 | 4 | 95% | 2.7764 | 0.6317 | 0.0089 | 0.0040 | 0.0111 | 0.6206 | 0.6428 | [62.06%, 64.28%] | 63.17% ± 1.11% | 1.4100 |
| Árvore de Decisão rasa (Sem Ponderação) | Baseline Árvore | F1_Macro | 5 | 4 | 95% | 2.7764 | 0.5311 | 0.0132 | 0.0059 | 0.0164 | 0.5147 | 0.5476 | [51.47%, 54.76%] | 53.11% ± 1.64% | 2.4900 |
| Árvore de Decisão rasa (Sem Ponderação) | Baseline Árvore | Acurácia | 5 | 4 | 95% | 2.7764 | 0.6290 | 0.0140 | 0.0063 | 0.0174 | 0.6116 | 0.6464 | [61.16%, 64.64%] | 62.90% ± 1.74% | 2.2300 |
| Árvore de Decisão rasa (Sem Ponderação) | Baseline Árvore | Precisão_Macro | 5 | 4 | 95% | 2.7764 | 0.5572 | 0.0134 | 0.0060 | 0.0166 | 0.5406 | 0.5738 | [54.06%, 57.38%] | 55.72% ± 1.66% | 2.4000 |
| Árvore de Decisão rasa (Sem Ponderação) | Baseline Árvore | Recall_Macro | 5 | 4 | 95% | 2.7764 | 0.5759 | 0.0085 | 0.0038 | 0.0106 | 0.5653 | 0.5864 | [56.53%, 58.64%] | 57.59% ± 1.06% | 1.4800 |
| Árvore de Decisão rasa (Ponderada) | Baseline Árvore | F1_Macro | 5 | 4 | 95% | 2.7764 | 0.5334 | 0.0252 | 0.0113 | 0.0313 | 0.5021 | 0.5647 | [50.21%, 56.47%] | 53.34% ± 3.13% | 4.7300 |
| Árvore de Decisão rasa (Ponderada) | Baseline Árvore | Acurácia | 5 | 4 | 95% | 2.7764 | 0.6057 | 0.0178 | 0.0079 | 0.0220 | 0.5836 | 0.6277 | [58.36%, 62.77%] | 60.57% ± 2.20% | 2.9300 |
| Árvore de Decisão rasa (Ponderada) | Baseline Árvore | Precisão_Macro | 5 | 4 | 95% | 2.7764 | 0.5369 | 0.0430 | 0.0192 | 0.0534 | 0.4835 | 0.5903 | [48.35%, 59.03%] | 53.69% ± 5.34% | 8.0100 |
| Árvore de Decisão rasa (Ponderada) | Baseline Árvore | Recall_Macro | 5 | 4 | 95% | 2.7764 | 0.6109 | 0.0137 | 0.0061 | 0.0170 | 0.5939 | 0.6279 | [59.39%, 62.79%] | 61.09% ± 1.70% | 2.2400 |
| SVM Linear Baseline | SVM Linear | F1_Macro | 5 | 4 | 95% | 2.7764 | 0.5709 | 0.0093 | 0.0042 | 0.0116 | 0.5592 | 0.5825 | [55.92%, 58.25%] | 57.09% ± 1.16% | 1.6400 |
| SVM Linear Baseline | SVM Linear | Acurácia | 5 | 4 | 95% | 2.7764 | 0.6439 | 0.0082 | 0.0037 | 0.0101 | 0.6337 | 0.6540 | [63.37%, 65.40%] | 64.39% ± 1.01% | 1.2700 |
| SVM Linear Baseline | SVM Linear | Precisão_Macro | 5 | 4 | 95% | 2.7764 | 0.5617 | 0.0678 | 0.0303 | 0.0841 | 0.4775 | 0.6458 | [47.75%, 64.58%] | 56.17% ± 8.41% | 12.0700 |
| SVM Linear Baseline | SVM Linear | Recall_Macro | 5 | 4 | 95% | 2.7764 | 0.6391 | 0.0123 | 0.0055 | 0.0153 | 0.6239 | 0.6544 | [62.39%, 65.44%] | 63.91% ± 1.53% | 1.9200 |
| SVM RBF Baseline | SVM RBF | F1_Macro | 5 | 4 | 95% | 2.7764 | 0.5759 | 0.0067 | 0.0030 | 0.0084 | 0.5675 | 0.5842 | [56.75%, 58.42%] | 57.59% ± 0.84% | 1.1700 |
| SVM RBF Baseline | SVM RBF | Acurácia | 5 | 4 | 95% | 2.7764 | 0.6441 | 0.0097 | 0.0043 | 0.0121 | 0.6320 | 0.6561 | [63.20%, 65.61%] | 64.41% ± 1.21% | 1.5100 |
| SVM RBF Baseline | SVM RBF | Precisão_Macro | 5 | 4 | 95% | 2.7764 | 0.6022 | 0.0588 | 0.0263 | 0.0731 | 0.5292 | 0.6753 | [52.92%, 67.53%] | 60.22% ± 7.31% | 9.7700 |
| SVM RBF Baseline | SVM RBF | Recall_Macro | 5 | 4 | 95% | 2.7764 | 0.6429 | 0.0109 | 0.0049 | 0.0135 | 0.6294 | 0.6564 | [62.94%, 65.64%] | 64.29% ± 1.35% | 1.6900 |
| SVM Polinomial Baseline | SVM Polinomial | F1_Macro | 5 | 4 | 95% | 2.7764 | 0.5694 | 0.0058 | 0.0026 | 0.0072 | 0.5622 | 0.5766 | [56.22%, 57.66%] | 56.94% ± 0.72% | 1.0100 |
| SVM Polinomial Baseline | SVM Polinomial | Acurácia | 5 | 4 | 95% | 2.7764 | 0.6394 | 0.0077 | 0.0034 | 0.0095 | 0.6299 | 0.6489 | [62.99%, 64.89%] | 63.94% ± 0.95% | 1.2000 |
| SVM Polinomial Baseline | SVM Polinomial | Precisão_Macro | 5 | 4 | 95% | 2.7764 | 0.5978 | 0.0482 | 0.0216 | 0.0599 | 0.5379 | 0.6577 | [53.79%, 65.77%] | 59.78% ± 5.99% | 8.0700 |
| SVM Polinomial Baseline | SVM Polinomial | Recall_Macro | 5 | 4 | 95% | 2.7764 | 0.6343 | 0.0082 | 0.0037 | 0.0102 | 0.6241 | 0.6446 | [62.41%, 64.46%] | 63.43% ± 1.02% | 1.3000 |
| SVM RBF Ótimo (C=10, gamma=0.01) | SVM RBF Otimizado | F1_Macro | 5 | 4 | 95% | 2.7764 | 0.5774 | 0.0086 | 0.0038 | 0.0107 | 0.5667 | 0.5880 | [56.67%, 58.80%] | 57.74% ± 1.07% | 1.4900 |
| SVM RBF Ótimo (C=10, gamma=0.01) | SVM RBF Otimizado | Acurácia | 5 | 4 | 95% | 2.7764 | 0.6453 | 0.0097 | 0.0043 | 0.0120 | 0.6333 | 0.6573 | [63.33%, 65.73%] | 64.53% ± 1.20% | 1.5000 |
| SVM RBF Ótimo (C=10, gamma=0.01) | SVM RBF Otimizado | Precisão_Macro | 5 | 4 | 95% | 2.7764 | 0.5904 | 0.0612 | 0.0274 | 0.0760 | 0.5144 | 0.6664 | [51.44%, 66.64%] | 59.04% ± 7.60% | 10.3700 |
| SVM RBF Ótimo (C=10, gamma=0.01) | SVM RBF Otimizado | Recall_Macro | 5 | 4 | 95% | 2.7764 | 0.6460 | 0.0142 | 0.0063 | 0.0176 | 0.6284 | 0.6636 | [62.84%, 66.36%] | 64.60% ± 1.76% | 2.1900 |
| SVM Polinomial Campeão GridSearch | SVM Polinomial Otimizado | F1_Macro | 5 | 4 | 95% | 2.7764 | 0.5786 | 0.0065 | 0.0029 | 0.0081 | 0.5704 | 0.5867 | [57.04%, 58.67%] | 57.86% ± 0.81% | 1.1300 |
| SVM Polinomial Campeão GridSearch | SVM Polinomial Otimizado | Acurácia | 5 | 4 | 95% | 2.7764 | 0.6453 | 0.0091 | 0.0041 | 0.0113 | 0.6340 | 0.6566 | [63.40%, 65.66%] | 64.53% ± 1.13% | 1.4100 |
| SVM Polinomial Campeão GridSearch | SVM Polinomial Otimizado | Precisão_Macro | 5 | 4 | 95% | 2.7764 | 0.5773 | 0.0670 | 0.0300 | 0.0832 | 0.4942 | 0.6605 | [49.42%, 66.05%] | 57.73% ± 8.32% | 11.6000 |
| SVM Polinomial Campeão GridSearch | SVM Polinomial Otimizado | Recall_Macro | 5 | 4 | 95% | 2.7764 | 0.6490 | 0.0113 | 0.0050 | 0.0140 | 0.6350 | 0.6630 | [63.50%, 66.30%] | 64.90% ± 1.40% | 1.7300 |
| SVM Campeão (Repeated Stratified K-Fold 10x5) | Auditoria Estocástica | F1_Macro | 15 | 14 | 95% | 2.1448 | 0.5761 | 0.0082 | 0.0021 | 0.0046 | 0.5716 | 0.5807 | [57.16%, 58.07%] | 57.61% ± 0.46% | 1.4300 |
| SVM Campeão (Repeated Stratified K-Fold 10x5) | Auditoria Estocástica | Acurácia | 15 | 14 | 95% | 2.1448 | 0.6448 | 0.0079 | 0.0020 | 0.0044 | 0.6405 | 0.6492 | [64.05%, 64.92%] | 64.48% ± 0.44% | 1.2200 |
