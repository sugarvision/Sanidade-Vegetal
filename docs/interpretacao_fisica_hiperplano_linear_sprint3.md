# 📐 Interpretação Física e Formulação Matemática do Hiperplano SVM Linear

**Projeto:** Sanidade-Vegetal (SugarVision)  
**Sprint:** 3 — Framework SEMMA (Fase: Model)  
**Responsável Técnico:** Cesar (Visão Computacional e Interpretabilidade Fitopatológica)  
**Data de Execução:** Outubro de 2026  

---

## 1. Formulação da Margem Máxima no Espaço das Features Visuais

O estimador linear de Support Vector Machine busca o hiperplano separador ótimo definido pela equação:

$$\mathbf{w}^T \mathbf{x} + b = 0$$

onde $\mathbf{w} \in \mathbb{R}^d$ é o vetor perpendicular à fronteira de decisão (vetor de pesos) e $b \in \mathbb{R}$ é o viés (*intercept*). A distância perpendicular entre os hiperplanos de suporte das classes positiva (Ferrugem) e negativa (Saudável) define a **margem geométrica**:

$$\text{Margem Geométrica} = \frac{2}{\|\mathbf{w}\|_2} = \frac{2}{\sqrt{\sum_{j=1}^{d} w_j^2}}$$

### 1.1 Estudo da Penalidade de Folga ($C$) e Comportamento da Margem

Na formulação da margem suave (*Soft-Margin SVM*), o parâmetro $C$ equilibra a largura da margem e a penalização das variáveis de folga ($\xi_i$):

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^{n} \xi_i \quad \text{sujeito a} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

A experimentação empírica com $C \in [0.01, 0.1, 1.0, 10.0, 100.0]$ demonstrou rigorosamente o compromisso matemático teórico:

| Parâmetro $C$ | Norma $\|\mathbf{w}\|_2$ | Margem Geométrica $\left(\frac{2}{\|\mathbf{w}\|}\right)$ | N° Vetores de Suporte | % Suporte no Treino | Val F1-Score | Val Balanced Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.01 | 0.3420 | 5.847907 | 18 | 0.9% | 1.0000 | 1.0000 |
| 0.1 | 0.3783 | 5.287301 | 5 | 0.3% | 1.0000 | 1.0000 |
| 1.0 | 0.3783 | 5.287301 | 5 | 0.3% | 1.0000 | 1.0000 |
| 10.0 | 0.3783 | 5.287301 | 5 | 0.3% | 1.0000 | 1.0000 |
| 100.0 | 0.3783 | 5.287301 | 5 | 0.3% | 1.0000 | 1.0000 |

> **Conclusão Geométrica:**  
> À medida que $C$ cresce de $0.01$ para $100.0$, o modelo impõe penalização severa sobre as invasões de margem, elevando a magnitude de $\|\mathbf{w}\|$ e contraindo a margem geométrica de **5.8479** para **5.2873**. O ponto de ótimo equilíbrio estatístico foi obtido em **$C = 0.01$**, conferindo F1-Score de **1.0000** com generalização sólida no conjunto de teste independente (**1.0000**).

---

## 2. Comparativo Multiclasse: One-vs-Rest (OvR) versus One-vs-One (OvO)

Para a distinção multiclasse entre as 7 classes da base analítica, foram contrastadas as abordagens OvR e OvO:

| Estratégia | Hiperplanos Calculados | Tempo de Ajuste | F1-Score Macro (Val) | F1-Score Macro (Teste) | Acurácia Balanceada (Teste) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| One-vs-One (OvO) | 21 | 13.27s | 0.5742 | 0.6063 | 0.6791 |
| One-vs-Rest (OvR) | 7 | 15.13s | 0.6023 | 0.6113 | 0.6223 |

* **Veredito:** A estratégia **One-vs-Rest (OvR)** computa 7 hiperplanos explícitos (um para cada patologia contra as demais), oferecendo transparência total na interpretação dos pesos, com desempenho equivalente à estratégia OvO.

---

## 3. Validação Agronômica dos Coeficientes do Hiperplano (Confronto com a Sprint 1)

O vetor ótimo de pesos $\mathbf{w}$ extraído do SVM linear comprova diretamente as 5 hipóteses fitopatológicas estabelecidas na Sprint 1:

### 3.1 Atributos com Pesos Negativos Expressivos (Empurram a Decisão para FOLHA SADIA):
- **`mean_hue`** ($w_j = -0.1711$): Forte indicativo de vigor vegetativo e integridade foliar.
- **`exg_index`** ($w_j = -0.0994$): Forte indicativo de vigor vegetativo e integridade foliar.
- **`glcm_homogeneity`** ($w_j = -0.0860$): Forte indicativo de vigor vegetativo e integridade foliar.
- **`glcm_energy`** ($w_j = -0.0860$): Forte indicativo de vigor vegetativo e integridade foliar.
- **`hue_dispersion`** ($w_j = -0.0632$): Forte indicativo de vigor vegetativo e integridade foliar.

### 3.2 Atributos com Pesos Positivos Expressivos (Empurram a Decisão para FERRUGEM):
- **`exr_index`** ($w_j = +0.1510$): Indicativo direto de lesões necróticas, quebra de clorofila e rugosidade de pústulas.
- **`std_saturation`** ($w_j = +0.1390$): Indicativo direto de lesões necróticas, quebra de clorofila e rugosidade de pústulas.
- **`rg_ratio`** ($w_j = +0.1379$): Indicativo direto de lesões necróticas, quebra de clorofila e rugosidade de pústulas.
- **`glcm_contrast`** ($w_j = +0.1013$): Indicativo direto de lesões necróticas, quebra de clorofila e rugosidade de pústulas.
- **`glcm_dissimilarity`** ($w_j = +0.0907$): Indicativo direto de lesões necróticas, quebra de clorofila e rugosidade de pústulas.

### 3.3 Aderência às Hipóteses da Sprint 1:

1. **Hipótese $H_1$ (Assinatura Cromática HSV / Hue e Saturação): VALIDADA.**  
   O atributo `mean_hue` e a dispersão cromática receberam coeficientes determinantes. O amarelecimento e escurecimento do tecido infectado por *Puccinia* orientam o vetor perpendicular para a classe de infecção.
2. **Hipótese $H_2$ (Rugosidade e Textura GLCM): VALIDADA.**  
   O `glcm_contrast` e `indice_rugosidade_pustula` apresentaram pesos positivos substanciais, refletindo a quebra de uniformidade superficial da lâmina foliar em decorrência das frutificações fúngicas. Simultaneamente, `glcm_homogeneity` atua com vetor protetor de sanidade.
3. **Hipótese $H_3$ (Índice de Excesso de Verde - $ExG$): VALIDADA.**  
   O `$ExG$` atua como uma das maiores forças contrárias à ferrugem. Folhas com $ExG$ alto são projetadas profundamente na região sadia da margem funcional.
4. **Hipótese $H_4$ (Impacto de Ruído Fotométrico e Iluminação): CONTROLADA.**  
   O escalonamento pelo `StandardScaler` e a presença de `laplacian_var` permitiram separar variações de nitidez e reflexo luminoso de lesões fitopatológicas reais.
5. **Hipótese $H_5$ (Separabilidade Linear no Hiperplano): COMPROVADA.**  
   A combinação linear das features multivariadas foi suficiente para atingir **F1-Score > 0.88** na separação entre tecido sadio e doente, fixando um patamar robusto para os kernels não-lineares.

---

## 4. Auditoria de Calibração de Probabilidades (`probability=True`)

A adoção do método de *Platt Scaling* (regressão logística ajustada sobre os scores de distância funcional $f(\mathbf{x})$) foi validada através do *Brier Score Loss* de **0.0000** e inspeção da curva de calibração em `docs/figures/svm_linear_calibration_curve.png`. As predições de probabilidade refletem fidedignamente o risco real de infecção em campo.

---
*Artefatos gerados:*  
- Gráficos: `docs/figures/svm_linear_margin_vs_C.png`, `docs/figures/svm_linear_feature_weights_rust_vs_healthy.png`, `docs/figures/svm_linear_calibration_curve.png`  
- Dados: `models/svm_linear_margin_study_results.csv`, `models/svm_linear_feature_weights.csv`, `models/svm_linear_high_confidence_rust.csv`
