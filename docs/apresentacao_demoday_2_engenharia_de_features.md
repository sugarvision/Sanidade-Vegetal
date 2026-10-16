# 📊 Roteiro Completo de Apresentação (Slides): Demoday 2 — Engenharia de Features

**Evento:** Demoday 2 — Apresentação Técnica & Defesa da Solução  
**Projeto:** Sanidade Vegetal AI (SugarVision) — Diagnóstico Fitossanitário Inteligente em Cana-de-Açúcar (*Saccharum officinarum*)  
**Etapa SEMMA:** **Modify** (Transformação, Engenharia de Features e Pré-processamento)  
**Squad:** Squad 2 | FATEC Rio Claro 2026  
**Duração Alvo:** 10 minutos cravados  
**Arquivo PowerPoint Gerado:** [`docs/Apresentacao_Demoday_2_Engenharia_de_Features.pptx`](../docs/Apresentacao_Demoday_2_Engenharia_de_Features.pptx)  
**Notebooks e Scripts de Suporte:**
- [`notebooks/06_pipeline_atomico_columntransformer.ipynb`](../notebooks/06_pipeline_atomico_columntransformer.ipynb)
- [`notebooks/07_consolidacao_exportacao_abt_final.ipynb`](../notebooks/07_consolidacao_exportacao_abt_final.ipynb)
- [`src/atomic_pipeline_preprocessing.py`](../src/atomic_pipeline_preprocessing.py)
- [`src/feature_engineering_visual.py`](../src/feature_engineering_visual.py)
- [`src/export_final_modeling_abt.py`](../src/export_final_modeling_abt.py)

---

## 🧭 Visão Geral do Tempo e Estrutura dos Slides (10 Minutos)

```mermaid
gantt
    title Cronograma dos 9 Slides da Apresentação do Demoday 2 (10 Minutos)
    dateFormat  m
    axisFormat %M min
    section Roteiro Demoday 2 (Fase Modify)
    Slide 1: Capa & Contexto do Desafio Fitossanitário          :s1, 0, 1m
    Slide 2: O Que e Como — Arquitetura da Fase Modify          :s2, 1, 2m
    Slide 3: Descritores Cromáticos & Índices Espectrais        :s3, 2, 3m15s
    Slide 4: Textura GLCM Haralick & Rugosidade de Pústulas    :s4, 3m15s, 4m30s
    Slide 5: Defesa Técnica de Discretização & Bins             :s5, 4m30s, 5m45s
    Slide 6: Pipeline Atômico & Imputações com ColumnTransformer:s6, 5m45s, 7m
    Slide 7: Prova Matemática de Ausência de Data Leakage       :s7, 7m, 8m15s
    Slide 8: Apresentação da ABT Final & Benchmark Parquet vs CSV:s8, 8m15s, 9m15s
    Slide 9: Conclusão, DoD e Prontidão para a Fase Model (SVM) :s9, 9m15s, 10m
```

---

## 📑 Slide 1: Abertura, Identificação e Contextualização do Demoday 2
* **Tempo:** 1 minuto (00:00 - 01:00)
* **Objetivo:** Posicionar a Squad 2 no ciclo SEMMA, reforçar o impacto agroindustrial da cana-de-açúcar e alinhar a pauta da etapa Modify.

### Conteúdo do Slide:
* **Tag Superior:** SQUAD 2 | DEMODAY 2 • ETAPA MODIFY (SEMMA)
* **Título Principal:** Engenharia de Features & Consolidação da ABT Final
* **Subtítulo:** Defesa Técnica de Transformações, Imputações e Garantia Matemática Anti-Leakage para o SugarVision
* **Cards Informativos:**
  * **Problema Agroindustrial:** Ferrugem (*Puccinia spp.*) e 6 patologias destruidoras de clorofila com perdas de 30% a 70% em TCH e ATR.
  * **Transição SEMMA:** Saída da Sprint 1 (6.571 imagens cruas auditadas) ➔ Entrada na Sprint 2 (Engenharia de Features e Pipeline Atômico).
  * **Resultado Chave da Entrega:** ABT Final com 6.571 linhas, 40 colunas, 35 features preditivas normalizadas, 0 NaNs e 0% de leakage.

### 🎙️ Script de Fala (Speaker Notes):
> *"Bom dia, professor Julio e colegas avaliadores! Sejam muito bem-vindos à apresentação do Demoday 2 da Squad 2.*
> *Hoje apresentamos a conclusão da Etapa Modify do framework SEMMA no projeto SugarVision.*
> *Na Sprint 1, consolidamos um repositório auditado com 6.571 imagens de folhas de cana-de-açúcar. O desafio agora foi transformar esse mar de pixels e metadados em uma Tabela Analítica Base (ABT) robusta, rica em sinais agronômicos e matematicamente blindada contra vazamento de dados.*
> *Ao longo dos próximos 9 minutos, vamos defender rigorosamente o 'O Que' e o 'Como' de cada descritor criado, os critérios de imputação e transformação, a prova inequívoca de ausência de Data Leakage e a entrega da nossa base final homologada em Apache Parquet para os modelos de Support Vector Machines."*

---

## 📑 Slide 2: Visão Geral Estratégica — O Que e Como da Etapa Modify
* **Tempo:** 1 minuto (01:00 - 02:00)
* **Objetivo:** Expor o mapa conceitual do pipeline de ponta a ponta ("O Que" foi concebido e "Como" foi implementado).

### Conteúdo do Slide:
* **Tag Superior:** SQUAD 2 | VISÃO GERAL • FRAMEWORK SEMMA
* **Título Principal:** "O Que" Construímos e "Como" Executamos na Fase Modify
* **Fluxograma Arquitetural:**
  * **Entrada:** 6.571 imagens com 35 colunas brutas e segregação determinística em Train (84,8%), Valid (9,4%) e Test (5,8%).
  * **Engenharia Visual ("O Que"):** 7 descritores cromáticos (HSV/ExG/ExR) + 5 descritores texturais (GLCM Haralick) + 1 índice de rugosidade + 4 metadados discretizados.
  * **Pipeline Atômico ("Como"):** `ColumnTransformer` desacoplado no Scikit-Learn com imputação de mediana + padronização Z-score nas contínuas, e imputação de moda + One-Hot Encoding nas categóricas.
  * **Saída Homologada:** `abt_features_modelagem` serializada em Parquet (916 KB) e CSV (2,98 MB) com hashes SHA-256 e zero valores nulos.

### 🎙️ Script de Fala (Speaker Notes):
> *"Para estruturar nossa defesa, dividimos a engenharia em duas dimensões fundamentais: 'O Que' foi feito e 'Como' foi feito.*
> *'O Que' nós construímos? Sintetizamos o conhecimento fitopatológico em 35 atributos numéricos de alto poder discriminatório. Extraímos a resposta espectral da quebra de clorofila, quantificamos a rugosidade microscópica das pústulas de ferrugem via matriz de coocorrência em níveis de cinza e categorizamos os metadados fotográficos em faixas estatísticas.*
> *E 'Como' nós executamos? Criamos uma arquitetura de Pipeline Atômico com ColumnTransformer. Em vez de scripts soltos ou pré-processamentos manuais que poluem o ciclo de vida do modelo, todo o aprendizado de parâmetros estatísticos foi encapsulado e rigidamente restrito à partição de treino, garantindo isolamento total para validação e teste cego."*

---

## 📑 Slide 3: Engenharia Cromática — Espaços de Cor e Índices Espectrais
* **Tempo:** 1 minuto e 15 segundos (02:00 - 03:15)
* **Objetivo:** Defender a base matemática e os testes estatísticos dos atributos cromáticos.
* **Imagem de Suporte:** [`docs/figures/sprint2_analise_cromaticas_hsv_exg.png`](../docs/figures/sprint2_analise_cromaticas_hsv_exg.png)

### Conteúdo do Slide:
* **Tag Superior:** SQUAD 2 | DESCRITORES CROMÁTICOS • TAREFA 1 (CESAR)
* **Título Principal:** Índices Espectrais e Desacoplamento Luminoso no Espaço HSV
* **Equações Fundamentais:**
  * **Índice de Excesso de Verde:** $ExG = 2G - R - B$ (quantifica vigor clorofiliano).
  * **Índice de Excesso de Vermelho:** $ExR = 1.4R - G$ (quantifica necrose e esporulação).
  * **Razão Espectral:** $R/G = R / (G + \epsilon)$ e Razão Normalizada $(R - G) / (R + G + \epsilon)$.
  * **Matiz Circular e Dispersão:** $\mu_{\text{Hue}}$ e $\sigma_{\text{Sat}}$ no espaço HSV.
* **Tabela de Evidência Estatística (6.571 Imagens):**

| Atributo Espectral | Folha Sadia (Média) | Folha Ferrugem (Média) | Teste Mann-Whitney | Cohen's d (Tamanho Efeito) |
| :--- | :---: | :---: | :---: | :---: |
| **Excesso de Verde ($ExG$)** | **+44,82** | **-18,45** | **$p < 10^{-15}$** | **6,346 (Extremamente Alto)** |
| **Excesso de Vermelho ($ExR$)** | **-18,10** | **+24,30** | **$p < 10^{-15}$** | **5,812 (Extremamente Alto)** |
| **Matiz Médio ($\mu_{\text{Hue}}$)** | **76,5° (Verde)** | **24,2° (Laranja)** | **$p < 10^{-15}$** | **7,410 (Extremamente Alto)** |

### 🎙️ Script de Fala (Speaker Notes):
> *"Entrando nos detalhes técnicos da Tarefa 1, o maior desafio no campo é a variação de iluminação e sombras. Câmeras comuns fornecem RGB misturado com brilho.*
> *Para isolar a biologia da folha, migramos para o espaço HSV, onde o Matiz separa a cor pura, e calculamos índices espectrais clássicos da agricultura de precisão.*
> *Vejam o índice ExG: ele duplica o canal verde e subtrai o vermelho e o azul. Uma folha sadia tem ExG médio positivo em +44,82. Quando o fungo da ferrugem ataca e destrói os cloroplastos, o ExG despenca para -18,45.*
> *O teste de Mann-Whitney resultou em p-valor menor que 10 elevado a menos 15, e o Cohen's d ultrapassou 6.3! Na estatística, um Cohen's d acima de 0.8 já é considerado efeito grande; atingir mais de 6.0 comprova que esses índices oferecem um sinal discriminatório brutal para o classificador."*

---

## 📑 Slide 4: Textura Haralick via GLCM e Análise de Rugosidade
* **Tempo:** 1 minuto e 15 segundos (03:15 - 04:30)
* **Objetivo:** Demonstrar a extração de textura, invariância rotacional e detecção das pústulas de ferrugem.
* **Imagem de Suporte:** [`docs/figures/sprint2_analise_texturas_glcm_haralick.png`](../docs/figures/sprint2_analise_texturas_glcm_haralick.png)

### Conteúdo do Slide:
* **Tag Superior:** SQUAD 2 | DESCRITORES TEXTURAIS • TAREFA 2 (CESAR)
* **Título Principal:** Matriz GLCM Multidirecional e Detecção de Pústulas Urediais
* **Fundamentação Matemática:**
  * **GLCM Normalizada:** $p(i, j \mid d=1, \theta)$ para $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$.
  * **Invariância Rotacional:** $\bar{F} = \frac{1}{4} \sum_{\theta} F_\theta$ (independente de a folha estar em pé, deitada ou inclinada).
  * **Contraste de Haralick:** $\sum |i - j|^2 \cdot p(i, j)$ (sensível à quebra da cutícula foliar).
  * **Homogeneidade:** $\sum \frac{p(i, j)}{1 + |i - j|^2}$ (suavidade da lâmina foliar sadia).
  * **Índice de Rugosidade Foliar de Pústula (IRFP):** $\frac{\text{Contraste} \times \text{Dissimilaridade}}{\text{Homogeneidade} + \epsilon}$.
* **Evidência:** O contraste em folhas com ferrugem é **1,72x maior** e a homogeneidade cai expressivamente ($p < 10^{-15}$, Cohen's d = 3,42).

### 🎙️ Script de Fala (Speaker Notes):
> *"A cor, por si só, não basta: folhas velhas ou sob estresse hídrico também amarelam. Como diferenciar isso de uma ferrugem ativa? Pela textura e rugosidade!*
> *O fungo Puccinia rompe a cutícula da folha com pústulas salientes e ásperas. Para quantificar essa micromorfologia, construímos a Matriz de Co-ocorrência em Níveis de Cinza (GLCM).*
> *Um ponto crucial de engenharia: fotos de campo nunca estão na mesma orientação. Para garantir invariância à rotação, calculamos a GLCM nas 4 direções canônicas — 0, 45, 90 e 135 graus — e consolidamos pela média direcional.*
> *Criamos ainda uma feature proprietária: o Índice de Rugosidade Foliar de Pústula (IRFP), que combina o salto de contraste e a perda de homogeneidade. O resultado comprovou nossa hipótese H2: folhas doentes têm textura significativamente mais rugosa, validada estatisticamente."*

---

## 📑 Slide 5: Defesa Técnica de Discretização e Metadados em Faixas
* **Tempo:** 1 minuto e 15 segundos (04:30 - 05:45)
* **Objetivo:** Justificar a discretização de metadados ópticos em bins e a escolha do One-Hot Encoding em vez de codificação ordinal.
* **Imagem de Suporte:** [`docs/figures/sprint2_comparacao_binning_uniform_vs_quantile.png`](../docs/figures/sprint2_comparacao_binning_uniform_vs_quantile.png)

### Conteúdo do Slide:
* **Tag Superior:** SQUAD 2 | METADADOS & DISCRETIZAÇÃO • SPRINT 2
* **Título Principal:** Discretização Estatística de Metadados e Justificativa do One-Hot Encoding
* **Estratégias de Binning Utilizadas:**
  * **Peso do Arquivo (`size_kb_bin`):** 4 quartis empíricos (*Muito Leve, Leve, Moderado, Pesado*).
  * **Nitidez e Foco (`laplacian_var_bin`):** 4 quartis via operador Laplaciano (*Fraco, Moderado, Alto, Ultra*).
  * **Resolução (`resolution_bin`):** 3 tiers via KMeans (*Baixa 640px, Média HD, Alta DSLR*).
  * **Aspect Ratio (`aspect_ratio_bin`):** 3 clusters de enquadramento (*Retrato, Quadrado, Paisagem*).
* **Defesa Técnica Crucial para Algoritmos de Margem (SVM):**
  * Codificar bins como inteiros ordinais ($0, 1, 2, 3$) forçaria o SVM a assumir uma falsa relação linear métrica (ex: que o bin 3 é três vezes "maior" que o bin 1).
  * A codificação via `OneHotEncoder(handle_unknown='ignore')` converte cada faixa em uma dimensão ortogonal binária independente, permitindo ao hiperplano do SVM encontrar fronteiras não-lineares sem impor monotonicidade espúria.

### 🎙️ Script de Fala (Speaker Notes):
> *"O nosso dataset unificou imagens de duas fontes distintas: Roboflow e Mendeley Data. Isso gerou uma tremenda disparidade em resoluções, nitidez de foco e peso dos arquivos.*
> *Para evitar que o modelo sofra com assimetrias severas ou outliers de câmeras distintas, discretizamos esses metadados ópticos em faixas estatísticas.*
> *Utilizamos quartis e clusterização KMeans para segmentar peso de arquivo, variância do Laplaciano para nitidez, e proporção de aspecto.*
> *E aqui reside uma defesa técnica essencial: por que aplicamos One-Hot Encoding nessas faixas em vez de deixá-las como números ordinais? Porque nosso modelo alvo é o SVM! Se passássemos números de 0 a 3, o SVM assumiria erroneamente uma monotonicidade linear contínua. Ao binarizar em One-Hot, damos liberdade geométrica total para que o modelo aprenda pesos independentes para cada faixa de resolução ou foco."*

---

## 📑 Slide 6: Arquitetura do Pipeline Atômico com ColumnTransformer
* **Tempo:** 1 minuto e 15 segundos (05:45 - 07:00)
* **Objetivo:** Apresentar a engenharia de software e MLOps por trás das transformações e imputações.

### Conteúdo do Slide:
* **Tag Superior:** SQUAD 2 | ARQUITETURA MLOPS • PIPELINE ATÔMICO (ELISA)
* **Título Principal:** Sub-Pipelines Especializados e Imputação Determinística
* **Diagrama do ColumnTransformer:**
  * **Sub-Pipeline Numérico (17 Features Contínuas):**
    * `SimpleImputer(strategy='median')`: Imputação robusta contra assimetrias e outliers.
    * `StandardScaler(with_mean=True, with_std=True)`: Padronização Z-score ($\mu=0, \sigma=1$).
  * **Sub-Pipeline Categórico (6 Features Nominais e Bins):**
    * `SimpleImputer(strategy='most_frequent')`: Imputação categórica pela moda.
    * `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`: Expansão em 18 colunas dummy sem quebra em categorias não vistas.
  * **Camada de Filtro (`remainder='drop'`):** Expurgo automático de 8 colunas de identificadores, caminhos e dimensões redundantes (`sample_id`, `filepath`, `width`, `height`).
* **Serialização e Artefatos:** Objeto exportado como `models/preprocessor_pipeline.joblib` (2,58 KB) e schema governado em `models/preprocessor_metadata.json`.

### 🎙️ Script de Fala (Speaker Notes):
> *"No slide 6, entramos na engenharia do nosso Pipeline Atômico implementado pela Elisa com o ColumnTransformer do Scikit-Learn.*
> *Estruturamos dois sub-pipelines perfeitamente desacoplados:*
> *No fluxo numérico, cobrindo 17 features, tratamos eventuais ausências pela mediana — que é imune a distorções por valores extremos — e padronizamos por Z-score com média zero e variância unitária, o que é mandatório para a estabilidade numérica dos kernels do SVM.*
> *No fluxo categórico, cobrindo as fontes, extensões e faixas discretizadas, imputamos pela moda e expandimos com One-Hot Encoder, ativando handle_unknown='ignore' para garantir que nenhuma categoria imprevista em produção derrube a inferência.*
> *Além disso, com remainder='drop', blindamos a entrada do modelo contra colunas textuais e redundantes. O pipeline inteiro foi serializado em joblib com apenas 2.5 KB de pegada de memória."*

---

## 📑 Slide 7: Comprovação Matemática de Ausência de Data Leakage
* **Tempo:** 1 minuto e 15 segundos (07:00 - 08:15)
* **Objetivo:** Provar formalmente a blindagem contra vazamento de dados com auditoria de parâmetros e testes adversariais.

### Conteúdo do Slide:
* **Tag Superior:** SQUAD 2 | GOVERNANÇA & QA • PRINCÍPIO ANTI-LEAKAGE
* **Título Principal:** Prova Rigorosa de Isolamento Amostral e Imutabilidade
* **A Regra de Ouro do Fit:**
  * $\mathbf{X}_{\text{train}}$ (5.574 amostras, 84,8%): **ÚNICO** conjunto onde `preprocessor.fit_transform()` é executado.
  * $\mathbf{X}_{\text{valid}}$ (617) e $\mathbf{X}_{\text{test}}$ (380): Recebem estritamente `preprocessor.transform()`.
* **Auditoria Matemática Realizada:**
  * As médias ($\mu_{\text{train}}$) e desvios ($\sigma_{\text{train}}$) foram inspecionados: as distribuições de validação e teste possuem $\mu \neq 0$ e $\sigma \neq 1$, o que **prova matematicamente que os dados de teste não participaram do cálculo da escala**!
* **Teste Adversarial de Perturbação:**
  * Injetamos ruído intencional e outliers simulados no conjunto de teste: os parâmetros aprendidos no objeto `preprocessor` permaneceram **100% idênticos bit a bit** (Zero Vazamento / 0% Leakage).

### 🎙️ Script de Fala (Speaker Notes):
> *"Chegamos ao ponto mais crítico de qualquer auditoria de Machine Learning: a comprovação de ausência de Data Leakage.*
> *Muitos projetos cometem o erro fatal de aplicar imputação ou StandardScaler no dataset inteiro antes de separar o teste. Isso contamina a média e o desvio padrão globais, criando métricas falsamente infladas que falham no campo.*
> *No SugarVision, seguimos a Regra de Ouro do Fit. A separação foi determinística na Sprint 1. O método fit_transform foi executado exclusivamente nas 5.574 amostras de treino.*
> *Como provamos isso matematicamente? Quando transformamos o conjunto de validação e o de teste cego, as médias calculadas neles não são exatamente zero e o desvio não é exatamente um. Se fossem, haveria vazamento!*
> *Além disso, rodamos um teste de estresse adversarial alterando os valores de teste: os atributos internos do pipeline mantiveram seus hashes e parâmetros inalterados. Garantimos zero por cento de contaminação."*

---

## 📑 Slide 8: Apresentação da ABT Final e Benchmark Parquet vs CSV
* **Tempo:** 1 minuto (08:15 - 09:15)
* **Objetivo:** Apresentar a tabela analítica consolidada (`abt_features_modelagem`) e justificar o ganho de eficiência com Apache Parquet.

### Conteúdo do Slide:
* **Tag Superior:** SQUAD 2 | ABT FINAL & BENCHMARK I/O • TAREFA 3 (ELISA)
* **Título Principal:** Consolidação da `abt_features_modelagem` e Auditoria de Qualidade
* **Métricas Formais da Base Analítica:**
  * **Volumetria:** 6.571 instâncias $\times$ 40 colunas (35 features preditivas + 5 metadados de controle).
  * **Auditoria de Nulos:** **0 NaNs (100% íntegra)** em todas as colunas e partições.
  * **Particionamento Preservado:** 5.574 Treino (84,8%), 617 Validação (9,4%), 380 Teste Cego (5,8%).
  * **Alvos Supervisionados:** Binário (1.146 Sadias / 5.425 Doentes) e Multiclasse (7 classes balanceadas nos splits).
* **Tabela de Benchmark de Performance de I/O:**

| Dimensão de Análise | Apache Parquet (`.parquet`) | Texto Plano (`.csv`) | Impacto Prático |
| :--- | :---: | :---: | :--- |
| **Tamanho em Disco** | **916,92 KB (0,90 MB)** | **2.989,66 KB (2,92 MB)** | **3,26x mais compacto (-69,3%)** |
| **Tempo de Carga (I/O)** | **36,19 ms** | **51,01 ms** | **1,41x mais rápido em loops CV** |
| **Preservação de Schema** | Estrita (int64, float64, string) | Inferência textual (risco de parsing) | **Tipagem nativa sem conversão** |

### 🎙️ Script de Fala (Speaker Notes):
> *"Apresentamos agora a materialização física da nossa entrega: a ABT Final abt_features_modelagem.*
> *Executamos uma auditoria automatizada em 100% dos registros. São exatamente 6.571 linhas e 40 colunas, com ZERO valores nulos. Mantivemos os identificadores UUID, a marcação determinística de split e os targets binário e multiclasse.*
> *Para garantir governança moderna de dados, exportamos a base em dois formatos: CSV para interoperabilidade e Apache Parquet com compressão Snappy como padrão oficial de treinamento.*
> *Vejam o benchmark: o Parquet reduziu o tamanho em disco em mais de 3 vezes — ocupando menos de 1 Megabyte — e acelerou a leitura em 41%. Mais importante ainda: ele preserva os tipos de dados float64 e int64 de forma nativa, eliminando erros silenciosos de parsing durante a validação cruzada.*
> *Ambos os arquivos tiveram seus hashes criptográficos SHA-256 registrados em manifesto."*

---

## 📑 Slide 9: Conclusão, Definition of Done e Prontidão para o SVM
* **Tempo:** 45 segundos (09:15 - 10:00)
* **Objetivo:** Resumir a entrega do Demoday 2, demonstrar o smoke test bem-sucedido e fazer o handoff perfeito para a Sprint 3 (Model).

### Conteúdo do Slide:
* **Tag Superior:** SQUAD 2 | CONCLUSÃO & PRONTIDÃO • SPRINT REVIEW
* **Título Principal:** 100% da Fase Modify Concluída e Prontidão para Modelagem
* **Checklist Definition of Done (DoD):**
  * [x] Extração de atributos cromáticos e texturais estatisticamente validados ($p < 10^{-15}$).
  * [x] Pipeline Atômico `ColumnTransformer` serializado (`preprocessor_pipeline.joblib`).
  * [x] Zero Data Leakage comprovado matematicamente e auditado em código.
  * [x] Base `abt_features_modelagem` gerada em Parquet e CSV com 0 NaNs.
  * [x] Manifestos SHA-256 e contratos de consumo validados para a Sprint 3.
* **Smoke Test de Integração:** Executado com `GridSearchCV` e `SVC(kernel='rbf')` em 0,05 segundos, obtendo convergência estável e abrindo caminho para o ajuste fino dos hiperparâmetros $C$ e $\gamma$.

### 🎙️ Script de Fala (Speaker Notes):
> *"Para finalizar nossa apresentação do Demoday 2, revisamos nosso Definition of Done: cumprimos 100% das metas da etapa Modify.*
> *Entregamos descritores com validação agronômica e física, um pipeline atômico robusto, ausência total de vazamento de dados e uma ABT final compacta, rápida e auditada.*
> *Para selar a entrega, realizamos um teste de fumaça conectando a ABT Parquet diretamente a um GridSearchCV com SVM RBF: o treinamento executou em fração de segundo sem qualquer inconsistência.*
> *Estamos 100% preparados para a Fase Model na Sprint 3. Agradecemos a atenção da banca e abrimos para perguntas e considerações!"*
