# 📐 Diagnóstico Matemático dos Vetores de Suporte (Support Vectors) — Sprint 3

**Projeto:** Sanidade-Vegetal (SugarVision)  
**Fase SEMMA:** Model (Sprint 3)  
**Módulo Principal:** [`src/svm_support_vectors_diagnosis.py`](../../src/svm_support_vectors_diagnosis.py)  
**Suíte de Testes:** [`src/test_svm_support_vectors_diagnosis.py`](../../src/test_svm_support_vectors_diagnosis.py)

---

## 1. Contagem e Proporção de Vetores de Suporte por Classe (`n_support_` e `support_vectors_`)

No estimador campeão (`models/svm_pipeline_optimal.joblib`, Kernel RBF, $C=10.0, \gamma=0.01$, ajustado sobre $N_{\text{train}} = 5.574$ instâncias), o número total de vetores de suporte extraído de `support_vectors_` é de **3.447 amostras** (**61,84%** do conjunto de treino):

| ID | Classe Fitopatológica | Amostras no Treino ($N_c$) | Vetores de Suporte ($SV_c$) | Densidade Intraclasse ($SV_c / N_c$) | Participação Global ($SV_c / SV_{\text{total}}$) | Norma Média $\|\mathbf{x}_{sv}\|_2$ |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: |
| 0 | **GRASSY SHOOT** | 150 | 140 | **93,33%** | 4,06% | 4,2930 |
| 1 | **HEALTHY** | 1.005 | 18 | **1,79%** | 0,52% | 4,4046 |
| 2 | **LEAF SCALD** | 313 | 249 | **79,55%** | 7,22% | 3,9622 |
| 3 | **MOSAIC** | 1.010 | 1.010 | **100,00%** | 29,30% | 3,4537 |
| 4 | **RED ROT** | 1.125 | 979 | **87,02%** | 28,40% | 3,6684 |
| 5 | **RUST** | 915 | 39 | **4,26%** | 1,13% | 4,6294 |
| 6 | **YELLOW LEAF** | 1.056 | 1.012 | **95,83%** | 29,36% | 3,6236 |

> **Interpretação Fitopatológica:**  
> As classes **HEALTHY** ($1,79\%$) e **RUST** ($4,26\%$) ocupam regiões altamente separáveis no espaço RKHS, exigindo uma fração mínima de vetores de suporte para sustentar a margem máxima ($F_1 = 1,0000$). Em contrapartida, patologias com clorose difusa e necroses secundárias sobrepostas (**MOSAIC**, **YELLOW LEAF**, **GRASSY SHOOT** e **RED ROT**) exibem densidades de suporte entre $87\%$ e $100\%$.

---

## 2. Trade-off do Hiperparâmetro $C$: Margens Suaves vs. Margens Rígidas

A avaliação sistemática de $C \in [0.01, 0.1, 1.0, 10.0, 100.0]$ demonstra o compromisso estrutural entre largura da margem e recrutamento de vetores de suporte:

| Kernel | Parâmetro $C$ | Regime de Margem | Total SVs | % SVs no Treino | % SVs com $\xi_i > 0$ | Val $F_1$-Macro | Val Balanced Acc |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **RBF** | 0,01 | Margem Suave (*Soft*) | 5.112 | 91,71% | 50,80% | 0,4467 | 0,6316 |
| **RBF** | 0,10 | Margem Suave (*Soft*) | 4.002 | 71,80% | 59,20% | 0,4767 | 0,6231 |
| **RBF** | 1,00 | Margem Intermediária | 3.574 | 64,12% | 57,75% | 0,5717 | 0,6494 |
| **RBF** | **10,00** | **Margem Rígida (Campeão)** | **3.447** | **61,84%** | **59,01%** | **0,5757** | **0,6532** |
| **RBF** | 100,00 | Margem Rígida (*Hard*) | 3.469 | 62,24% | 66,65% | 0,5761 | 0,6431 |
| **Linear** | 0,01 | Margem Suave (*Soft*) | 3.787 | 67,94% | 56,51% | 0,5714 | 0,6473 |
| **Linear** | 1,00 | Margem Intermediária | 3.645 | 65,39% | 59,70% | 0,5742 | 0,6484 |
| **Linear** | 100,00 | Margem Rígida (*Hard*) | 3.592 | 64,44% | 87,81% | 0,5125 | 0,5759 |

---

## 3. Identificação Matemática de Violações de Margem ($\xi_i > 0$)

A variável de folga primal é definida por $\xi_i = \max(0, 1 - \gamma_i)$, onde $\gamma_i$ representa a margem funcional assinada da instância de suporte $i$:
- **Vetores de Suporte Livres na Margem ($\xi_i = 0$):** **1.413 instâncias (40,99%)** situadas na fronteira canônica sem violação.
- **Vetores de Suporte com Violação de Margem ($\xi_i > 0$):** **2.034 instâncias (59,01%)**, subdivididas entre invasões suaves no lado correto do hiperplano ($0 < \xi_i \le 1$: **491 SVs**) e transposições severas de fronteira ($\xi_i > 1$: **1.543 SVs**).

---

## 4. Artefatos e Visualizações Geradas

- **Tabelas e Manifestos (`models/`):**
  - `models/svm_support_vectors_by_class.csv`
  - `models/svm_c_vs_support_vectors_tradeoff.csv`
  - `models/svm_margin_violations_slack_analysis.csv`
  - `models/svm_isolated_ambiguous_patches_catalog.csv`
  - `models/svm_support_vectors_diagnosis.json`
- **Visualizações 2D/3D e Amostras Visuais (`docs/figures/`):**
  - `docs/figures/svm_support_vectors_density_by_class.png`
  - `docs/figures/svm_c_tradeoff_soft_vs_hard_margin.png`
  - `docs/figures/svm_slack_variables_margin_violations.png`
  - `docs/figures/svm_support_vectors_projection_2d.png`
  - `docs/figures/svm_support_vectors_projection_3d.png` & `svm_support_vectors_projection_3d.html`
  - `docs/figures/svm_support_vectors_visual_ambiguity_grid.png` & `docs/figures/support_vectors_patches/`
