# 🔬 Hipóteses Iniciais de Pesquisa e Metadados Fitopatológicos

**Projeto:** Sanidade-Vegetal (SugarVision)  
**Sprint:** 1 — Setup, Sample & Explore (Framework SEMMA)  
**Responsável:** Cesar (Domínio, Escopo e Visão Computacional)  
**Data:** 02 de Setembro de 2026  

---

## 1. Formulação das Hipóteses Iniciais

Para orientar a Análise Exploratória de Dados (EDA) conduzida pelo Marvin e a extração de atributos para a Tabela Analítica (ABT), foram estabelecidas 5 hipóteses agronômicas e computacionais:

### 📌 Hipótese 1 ($H_1$): Assinatura Cromática da Ferrugem no Espaço HSV
* **Enunciado:** Amostras com ferrugem (*Puccinia spp.*) apresentam um deslocamento estatisticamente significante do Matiz médio ($\mu_{\text{Hue}}$) em direção à faixa espectral de 10° a 30° (tons castanho/laranja/vermelho), acompanhado por um aumento expressivo no desvio-padrão da Saturação ($\sigma_{\text{Sat}}$) devido ao contraste entre o limbo sadio e a lesão.
* **Métrica de Teste:** Teste de Mann-Whitney / Kruskal-Wallis entre as distribuições de $\mu_{\text{Hue}}$ e $\sigma_{\text{Sat}}$ nas classes `HEALTHY` vs `RUST`.
* **Critério de Validação:** $p\text{-valor} < 0.01$ e tamanho de efeito (*Cohen's d*) $> 0.8$.

### 📌 Hipótese 2 ($H_2$): Rugosidade e Dissimilaridade de Textura via GLCM
* **Enunciado:** As pústulas de ferrugem rompem a continuidade da cutícula cerosa da folha, elevando substancialmente o **Contraste de Haralick** e a **Dissimilaridade** da Matriz de Co-ocorrência em Níveis de Cinza (GLCM), enquanto reduzem a **Homogeneidade** em comparação a folhas saudáveis.
* **Métrica de Teste:** Coeficiente de Correlação de Spearman e ganho de informação (*Mutual Information*) das features GLCM em relação ao target binário.
* **Critério de Validação:** Aumento de pelo menos $40\%$ no contraste médio em amostras com ferrugem severa.

### 📌 Hipótese 3 ($H_3$): Quebra do Índice de Excesso de Verde (Excess Green - ExG)
* **Enunciado:** O índice de vegetação baseado em RGB $ExG = 2G - R - B$ apresenta correlação negativa com a densidade de lesões de ferrugem, atuando como um filtro linear preliminar eficaz para a triagem binária de sanidade.
* **Métrica de Teste:** Análise de curva ROC-AUC utilizando unicamente $ExG$ como classificador univariado.
* **Critério de Validação:** ROC-AUC $\ge 0.80$ para separabilidade binária básica.

### 📌 Hipótese 4 ($H_4$): Impacto da Iluminação e Sombras no Fundo da Imagem
* **Enunciado:** Imagens capturadas em campo sob iluminação natural não controlada (presença de sombras duras, reflexo especular solar na lâmina foliar ou fundo com solo/erva daninha) introduzem ruído nos canais de luminância ($L^*$ no CIELab e $V$ no HSV), reduzindo a acurácia de classificadores que dependam exclusivamente de médias globais de intensidade.
* **Métrica de Teste:** Correlação entre o desvio de iluminação ($\sigma_V$) e a dispersão dos centróides das classes.
* **Critério de Validação:** Necessidade comprovada de segmentação prévia da lâmina foliar (*Otsu* / *Masking*) ou normalização de histograma (CLAHE).

### 📌 Hipótese 5 ($H_5$): Separabilidade Linear no Hiperplano SVM
* **Enunciado:** A combinação linear de 6 descritores multivariados (média de Hue, razão $R/G$, contraste GLCM, homogeneidade GLCM, densidade de bordas Canny e $ExG$) é suficiente para atingir uma acurácia balanceada $\ge 85\%$ e F1-Score $\ge 0.82$ em um classificador SVM linear/RBF sem a necessidade inicial de redes neurais profundas.
* **Métrica de Teste:** Validação cruzada estratificada em 5 folds na ABT consolidada.

---

## 2. Matriz de Variáveis de Interesse para a ABT

| Categoria | Nome da Variável | Tipo de Dado | Papel | Justificativa Fitopatológica / Computacional |
| :--- | :--- | :--- | :--- | :--- |
| **Identificação** | `sample_id` | String | Chave Primária | Identificador exclusivo e determinístico da imagem. |
| **Metadados** | `dataset_source` | Categórico | Controle | Rastreabilidade da fonte (*Roboflow* vs *Mendeley*). |
| **Metadados** | `split_partition` | Categórico | Partição | Divisão de amostragem (`train`, `valid`, `test`). |
| **Target Binário** | `target_binary` | Binário (0/1) | Alvo Principal | `0` = Saudável, `1` = Ferrugem. |
| **Target Multiclasse**| `target_multiclass` | Categórico (0 a 6)| Alvo Secundário | Diagnóstico diferencial completo das 7 patologias. |
| **Cor** | `mean_hue` | Contínuo | Feature | Captura o tom amarelado/alaranjado das pústulas. |
| **Cor** | `std_saturation` | Contínuo | Feature | Dispersão da saturação gerada por lesões pontuais. |
| **Cor** | `exg_index` | Contínuo | Feature | Índice de vegetação saudável ($2G - R - B$). |
| **Cor** | `exr_index` | Contínuo | Feature | Índice de excesso de vermelho ($1.4R - G$). |
| **Cor** | `ratio_rg` | Contínuo | Feature | Razão canal Vermelho / Verde. |
| **Textura** | `glcm_contrast` | Contínuo | Feature | Rugosidade e quebra de gradiente na pústula. |
| **Textura** | `glcm_homogeneity` | Contínuo | Feature | Uniformidade do limbo da folha sadia. |
| **Textura** | `glcm_energy` | Contínuo | Feature | Repetibilidade de padrões de cinza. |
| **Morfologia** | `laplacian_variance`| Contínuo | Feature/Qualidade | Nitidez da imagem e presença de microestruturas. |
| **Morfologia** | `canny_edge_density`| Contínuo | Feature | Densidade de contornos de lesões. |
