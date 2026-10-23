# Modelagem SVM com Kernel Trick (RBF e Polinomial)

**Projeto:** Sanidade-Vegetal (SugarVision) · **Sprint 3 — Fase Model (SEMMA)**
**Código:** [`src/svm_kernel_trick_study.py`](../../src/svm_kernel_trick_study.py)
**Artefatos:** `models/svm_kernel_study.json`, `models/svm_kernel_study_results.csv`, `docs/figures/svm_kernel_*.png`

## 1. Protocolo (anti-leakage)

| Etapa | Partição | Observação |
| :--- | :---: | :--- |
| Ajuste (`fit`) do `Pipeline` (pré-processador + SVC) | `train` (5574) | Scaler/Imputer/OHE aprendidos só no treino (verificado por `assert` no script). |
| Escolha de hiperparâmetros | `valid` (617) | Critério: acurácia balanceada (desempate por F1-macro). |
| Avaliação final | `test` (380) | Tocada **uma única vez por kernel**, só com o campeão. |

Configuração comum: `class_weight='balanced'`, `random_state=42`, `probability=False` (não afeta as predições e acelera o treino). Métrica principal: **acurácia balanceada** (média do recall por classe), adequada ao desbalanceamento de 7 classes.

## 2. Formulação matemática

### 2.1 Kernel trick
O SVC resolve o problema dual, em que os dados aparecem apenas via produtos internos:

$$\max_{\boldsymbol\alpha}\ \sum_i \alpha_i-\tfrac12\sum_{i,j}\alpha_i\alpha_j y_i y_j\,K(\mathbf{x}_i,\mathbf{x}_j)\quad\text{s.a. } 0\le\alpha_i\le C,\ \sum_i\alpha_i y_i=0$$

$$f(\mathbf{x})=\sum_{i\in SV}\alpha_i y_i\,K(\mathbf{x}_i,\mathbf{x})+b$$

com $K(\mathbf{x},\mathbf{x}')=\langle\phi(\mathbf{x}),\phi(\mathbf{x}')\rangle$, de modo que $\phi$ nunca é calculada explicitamente.

### 2.2 Kernel RBF (Radial Basis Function)

$$\boxed{K(\mathbf{x},\mathbf{x}')=\exp\!\left(-\gamma\,\lVert\mathbf{x}-\mathbf{x}'\rVert^{2}\right)},\qquad \gamma>0,\ \ \gamma=\frac{1}{2\sigma^{2}}$$

- **Espaço de dimensão infinita:** expandindo, $K=e^{-\gamma\lVert\mathbf{x}\rVert^2}e^{-\gamma\lVert\mathbf{x}'\rVert^2}\sum_{k=0}^{\infty}\frac{(2\gamma)^k}{k!}\langle\mathbf{x},\mathbf{x}'\rangle^k$, isto é, uma soma de kernels polinomiais de todos os graus.
- **Limites:** $K\in(0,1]$; $K=1$ se $\mathbf{x}=\mathbf{x}'$; $K\to0$ quando a distância cresce.
- **Papel de $\gamma$ (curvatura):** $\gamma$ pequeno → influência ampla de cada vetor de suporte, fronteira suave (tende ao linear, risco de *underfitting*); $\gamma$ grande → influência local, fronteira recortada (risco de *overfitting*).
- **Papel de $C$:** penaliza violações de margem; $C$ grande → margem rígida, mais vetores ajustados ao ruído.
- `gamma='scale'` no scikit-learn equivale a $\gamma=\dfrac{1}{n_{\text{features}}\cdot\operatorname{Var}(X)}$.

### 2.3 Kernel polinomial

$$K(\mathbf{x},\mathbf{x}')=\left(\gamma\,\langle\mathbf{x},\mathbf{x}'\rangle+r\right)^{d}$$

Usamos $d\in\{2,3\}$ e `coef0` $r=0$ (padrão do `SVC`, kernel homogêneo). O espaço implícito tem dimensão $\binom{n+d-1}{d}$ para $r=0$ (interações de ordem $d$ entre as $n$ features).

### 2.4 Kernel linear (referência)
$K(\mathbf{x},\mathbf{x}')=\langle\mathbf{x},\mathbf{x}'\rangle$ — sem projeção, serve de linha de base.

## 3. Resultados

### 3.1 RBF: influência de $\gamma$ e $C$ (acurácia balanceada em `valid`)

| $\gamma$ \ $C$ | 0.1 | 1 | 10 | 100 |
| :---: | :---: | :---: | :---: | :---: |
| 0.001 | 0.553 | 0.588 | 0.648 | 0.650 |
| 0.01 | 0.588 | 0.648 | **0.653** | 0.642 |
| 0.1 | **0.655** | 0.630 | 0.591 | 0.555 |
| 1.0 | 0.340 | 0.516 | 0.514 | 0.514 |

Leitura: existe uma **diagonal de compensação** entre $\gamma$ e $C$ (modelos mais suaves pedem $C$ maior e vice-versa). Com $\gamma=1.0$ ocorre *overfitting* severo (menor valor: 0.340), compatível com a influência excessivamente local dos vetores de suporte. A figura `svm_kernel_rbf_gamma_C_heatmap.png` mostra o mapa completo.

### 3.2 Polinomial: grau $d\in\{2,3\}$

Melhor acurácia balanceada em `valid` por grau: **$d=2$: 0.655** (campeão, $C=1$, $\gamma=0.1$) × **$d=3$: 0.645** ($C=0.1$, $\gamma=0.1$). O grau 3 é mais sensível aos hiperparâmetros (mínimo 0.462 contra 0.550 no grau 2), sinal de maior variância; não traz ganho.

### 3.3 Comparação Linear × RBF × Polinomial (campeões)

| Kernel | Hiperparâmetros | Bal. acc. `valid` | Bal. acc. `test` | F1-macro `test` | Tempo de treino (mediana de 3) | Nº vetores de suporte |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Linear | $C=1$ | 0.652 | **0.679** | 0.606 | 2.52 s | 3622 |
| RBF | $C=0.1,\ \gamma=0.1$ | **0.655** | 0.676 | 0.604 | **0.71 s** | 3844 |
| Polinomial | $d=2,\ C=1,\ \gamma=0.1$ | 0.655 | 0.669 | 0.601 | 0.75 s | 3493 |

> [!IMPORTANT]
> **As diferenças de acurácia entre kernels são desprezíveis** (≈0.3 p.p. em `valid`; ≈1 p.p. em `test`, com apenas 380 amostras e ordem de troca entre as partições). Não há evidência estatística de que o kernel não linear supere o linear neste conjunto de *features* tabulares; o ganho real está no tempo (RBF/poly ≈3,4× mais rápidos que o linear com `C=1`, pois a convergência do linear com features não separáveis é lenta). Alta fração de vetores de suporte (≈60–70 % do treino) indica classes muito sobrepostas, em especial MOSAIC (recall 0 no baseline). Para ganhos reais, o limite está na informação das features, não no kernel.

### 3.4 Robustez da fronteira a ruído fotométrico e sombras

Como `data/raw` está vazio no repositório, **as perturbações foram simuladas no espaço de features**, respeitando a física de cada descritor (detalhes no cabeçalho do script):
- **Ruído:** $\mathcal{N}(0,(s\,\sigma_{\text{treino}})^2)$ nas 7 features cromáticas, $s\in\{0.1,0.25,0.5,1.0\}$.
- **Sombra:** escurecimento multiplicativo $f$ dos canais RGB: ExG, ExR $\propto f$; `laplacian_var`, `glcm_contrast` $\propto f^2$; `glcm_dissimilarity` $\propto f$; IRFP $\propto f^3$; matiz, razão R/G e ICN invariantes.

Acurácia balanceada em `valid` (queda frente ao cenário limpo entre parênteses):

| Cenário | Linear | RBF | Polinomial |
| :--- | :---: | :---: | :---: |
| Limpo | 0.652 | 0.655 | 0.655 |
| Ruído 0.25σ | 0.651 (−0.2) | 0.643 (−1.2) | 0.664 (+0.9) |
| Ruído 1.0σ | 0.631 (−2.1) | 0.559 (−9.6) | 0.634 (−2.1) |
| Sombra $f=0.8$ | 0.644 (−0.8) | 0.660 (+0.5) | 0.651 (−0.4) |
| Sombra $f=0.6$ | 0.605 (−4.7) | **0.519 (−13.6)** | 0.604 (−5.1) |
| Sombra $f=0.4$ | 0.541 (−11.2) | **0.390 (−26.5)** | 0.532 (−12.3) |
| Sombra 0.6 + ruído 0.25σ | 0.613 (−3.9) | 0.519 (−13.6) | 0.607 (−4.8) |

**Conclusões:**
1. Sombras são mais danosas que ruído gaussiano, pois deslocam sistematicamente as features de textura/índices para regiões pouco vistas no treino (extrapolação).
2. O **RBF é o menos robusto**: com $\gamma=0.1$ cada vetor de suporte tem influência local; amostras deslocadas para longe do treino têm $K\to0$ e caem em regiões onde a fronteira é dominada pelo viés, gerando troca de até 46 % das predições com $f=0.4$. Linear e polinomial de grau 2 degradam de forma gradual e quase idêntica.
3. Em sombras leves ($f\ge0.8$) e ruído moderado ($\le0.25\sigma$) os três são equivalentes.
4. A magnitude média de $|f(\mathbf{x})|$ ficou praticamente constante nos cenários (≈3,0–3,1) e **não** é um indicador útil de confiança aqui, pois a `decision_function` do SVC multiclasse agrega votos *one-vs-one* e é pouco sensível à perturbação; a taxa de troca de predição é a medida confiável.

> [!WARNING]
> **Limitação:** a simulação em features é uma aproximação (ex.: sombra real é espacialmente não uniforme e altera o ruído do sensor). A validação definitiva exige perturbar as imagens originais e re-extrair as features quando `data/raw` estiver disponível. Recomendações: augmentação fotométrica no treino ou normalização de iluminação (ex.: *gray world*, CLAHE) antes da extração, e preferir kernel linear/poly-2 se a captura em campo for pouco controlada.

## 4. Verificação dos critérios do cartão

| Critério | Atendido | Evidência |
| :--- | :---: | :--- |
| SVC RBF com $\gamma\in\{0.001,0.01,0.1,1.0\}$ e $C$ | Sim | §3.1, 16 configurações |
| SVC poly com $d\in\{2,3\}$ | Sim | §3.2, 18 configurações |
| Fronteira frente a ruído fotométrico e sombras | Sim (simulado em features) | §3.4 e limitação declarada |
| Acurácia balanceada e tempo: Linear × RBF × Poly | Sim | §3.3 |
| Formulação matemática do Kernel RBF no relatório | Sim | §2.2 |
