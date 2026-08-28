# 📖 Dicionário de Dados — Sanidade Vegetal (SugarVision)

**Projeto:** Classificação e Diagnóstico Inteligente de Patologias em Cana-de-Açúcar  
**Fase:** Modelagem Inicial e Governança de Dados (Sprint 1)  
**Versão:** 1.0.0  

---

## 1. Contextualização e Estrutura de Dados no Repositório

O projeto **Sanidade-Vegetal (SugarVision)** processa imagens e metadados de lavouras de cana-de-açúcar (*Saccharum officinarum*) para identificar, classificar e quantificar a severidade de fitopatologias foliares.

### 📚 Fontes Oficiais de Dados e Validação Externa:
1. **Mendeley Data — *Sugarcane Diseases* (DOI: [10.17632/rzh99cj2rj.1](https://doi.org/10.17632/rzh99cj2rj.1)):**
   * **Autores:** Ijaz Kalhoro, Rafaqat Hussain e Hidayatullah Shaikh.
   * **Características:** Amostras fotográficas RGB de alta resolução coletadas em condições reais de campo com variações de iluminação natural, orientação foliar, estágios fenológicos e níveis de severidade.
   * **Patologias Cobertas:** *Red Rot* (Podridão Vermelha - *Colletotrichum falcatum*), *Smut* (Carvão - *Sporisorium scitamineum*), *Leaf Scald* (Escaldadura das Folhas - *Xanthomonas albilineans*) e amostras sadias (*Healthy*).
2. **Roboflow Universe — *Sugarcane Disease Classification*:**
   * **Mantenedor:** Asad Unvar ([Roboflow Universe](https://universe.roboflow.com/asad-unvar/sugarcane-disease-classification)).
   * **Características:** Dataset particionado em `train/`, `valid/` e `test/`, com anotações e segmentações/bounding boxes para visão computacional.
   * **Patologias Cobertas:** *Red Rot*, *Mosaic Virus*, *Rust* (Ferrugem), *Yellow Leaf* e *Healthy*.

### 📂 Alocação dos Dados no Repositório:
* **`data/raw/`**: Armazena os pacotes brutos extraídos das duas fontes (`datasets/` ou `data/raw/`).
* **`data/processed/`**: Imagens redimensionadas (ex.: 640×640 px), tensores normalizados, máscaras e arquivos de metadados particionados.
* **`notebooks/`**: Análises exploratórias, validação cruzada estratificada por talhão/fonte e prototipagem.
* **`src/`**: Pipelines de ingestão, engenharia de features e treinamento de modelos de Visão Computacional.

---

## 2. Visão Geral das Entidades de Dados

| Entidade | Descrição | Granularidade |
| :--- | :--- | :--- |
| **`TALHAO`** | Unidade de gestão agrícola onde a cana está plantada (área geográfica delimitada). | Nível de Gestão/Área |
| **`PLANTA`** | Indivíduo/touceira vegetal monitorado dentro de um talhão específico. | Nível de Espécime |
| **`COLETA_DATA`** | Evento temporal de monitoramento, inspeção visual ou captura de imagens. | Nível Temporal / Sessão |
| **`IMAGEM`** | Registro visual capturado (RGB/Multiespectral) de folhas ou colmos. | Nível de Arquivo / Registro |
| **`DIAGNOSTICO`** | Anotação da condição fitossanitária (rótulo/target agronômico). | Nível de Avaliação |
| **`FEATURES_EXTRACTED`** | Representações numéricas e tensores derivados para os modelos de ML/DL. | Nível de Feature Vector |

---

## 3. Identificação das Variáveis Alvo (Targets)

As variáveis alvo foram definidas para suportar três tipos de tarefas preditivas:

1. **Classificação Multiclasse (Primária):**
   * **Nome da Coluna:** `target_doenca_classe`
   * **Tipo:** Categórica Nominal (String / Integer Encoded)
   * **Classes Típicas:**
     * `0: Saudavel` (Healthy)
     * `1: Podridao_Vermelha` (Red Rot - *Colletotrichum falcatum*)
     * `2: Mosaico` (Mosaic Virus - *Sugarcane Mosaic Virus*)
     * `3: Ferrugem_Marrom` (Brown Rust - *Puccinia melanocephala*)
     * `4: Ferrugem_Alaranjada` (Orange Rust - *Puccinia kuehnii*)
     * `5: Mancha_Amarela` (Yellow Leaf - *Sugarcane Yellow Leaf Virus*)
     * `6: Carvao` (Smut - *Sporisorium scitamineum*)

2. **Detecção Binária de Anomalia (Secundária):**
   * **Nome da Coluna:** `target_is_doente`
   * **Tipo:** Booleana / Binária (`0: Não / Sadia`, `1: Sim / Patológica`)
   * **Uso:** Triagem rápida em borda (*Edge Computing* / Mobile).

3. **Grau de Severidade da Lesão (Regressão / Ordinal):**
   * **Nome da Coluna:** `target_grau_severidade_pct` / `target_escala_diagramatica`
   * **Tipo:** Numérica Contínua ($[0.0, 100.0]\%$) ou Categórica Ordinal (`Grau 1` a `Grau 5`).
   * **Uso:** Estimativa de área foliar afetada para tomada de decisão sobre aplicação de defensivos.

---

## 4. Identificação das Variáveis Candidatas a Features

As variáveis de entrada do modelo dividem-se em 4 grandes grupos:

### A. Tensores e Features Visuais Diretas (Deep Learning)
* `raw_image_tensor`: Matriz tridimensional $(H \times W \times C)$ da imagem normalizada $[0, 1]$ ou padronizada via ImageNet stats.
* `cnn_embeddings`: Vetores densos de features extraídos de backbones convolucionais (ex.: EfficientNet, ResNet, ConvNeXt) ou Vision Transformers (ViT).

### B. Features Morfológicas e de Textura (Computer Vision Clássica)
* `leaf_lesion_area_ratio`: Razão da área de lesão detectada sobre a área foliar total visível.
* `glcm_contrast`, `glcm_homogeneity`, `glcm_energy`, `glcm_correlation`: Métricas de textura por Matriz de Co-ocorrência em Níveis de Cinza (GLCM).
* `color_histogram_hsv`: Distribuição dos canais Matiz (H), Saturação (S) e Valor (V) para diferenciar necrose foliar de clorose.
* `excess_green_index` ($ExG = 2G - R - B$): Índice de vegetação visível para segmentação de biomassa verde vs. tecido necrosado.

### C. Metadados de Imagem e Aquisição
* `image_resolution_width`, `image_resolution_height`: Dimensões originais da imagem para controle de distorção de escala.
* `lighting_condition`: Iluminação ambiente (`Luz Direta`, `Sombra`, `Nublado`, `Flash Artificial`).
* `capture_device_type`: Dispositivo de captura (`Smartphone`, `Câmera RGB Alta Resolução`, `Drone/VANT`).

### D. Variáveis Contextuais e Agronômicas (Fusão Multimodal)
* `dias_apos_plantio` / `estadio_fenologico`: Idade da planta e estágio de desenvolvimento.
* `variedade_cultivar`: Variedade genética da cana (ex.: RB867515, CTC4, SP80-3280), pois cada cultivar apresenta suscetibilidades diferentes.
* `regiao_uf` / `historico_chuva_ultimos_7d`: Variáveis climáticas associadas ao ciclo biológico de fungos e vírus.

---

## 5. Dicionário de Dados Completo

| Entidade | Nome da Coluna | Tipo Primitivo | Tipo Estatístico | Papel no ML | Descrição / Significado | Domínio / Valores Exemplo | Permite Nulo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **`TALHAO`** | `talhao_id` | `VARCHAR(64)` | Identificador | Metadado / ID | Identificador único do talhão ou gleba agrícola | `"TAL-2026-N04"`, `"GLEBA-B12"` | Não |
| **`TALHAO`** | `fazenda_nome` | `VARCHAR(128)` | Categórica Nominal | Metadado | Nome da propriedade rural ou usina | `"Usina Santa Adélia"`, `"Fazenda Boa Vista"` | Não |
| **`TALHAO`** | `latitude` | `FLOAT` | Numérica Contínua | Contexto Espacial | Coordenada geográfica de latitude do talhão | `-21.1767` | Sim |
| **`TALHAO`** | `longitude` | `FLOAT` | Numérica Contínua | Contexto Espacial | Coordenada geográfica de longitude do talhão | `-47.8208` | Sim |
| **`TALHAO`** | `variedade_cultivar` | `VARCHAR(64)` | Categórica Nominal | Feature Contextual | Código da variedade genética da cana-de-açúcar | `"RB867515"`, `"CTC4"`, `"SP80-3280"` | Sim |
| **`TALHAO`** | `tipo_solo` | `VARCHAR(64)` | Categórica Nominal | Feature Contextual | Classificação do solo predominante | `"Latossolo Vermelho"`, `"Argissolo"` | Sim |
| **`PLANTA`** | `planta_id` | `VARCHAR(64)` | Identificador | Metadado / ID | Identificador da planta/touceira dentro do talhão | `"PLT-00482"`, `"P-12"` | Não |
| **`PLANTA`** | `posicao_linha` | `INTEGER` | Numérica Discreta | Metadado Espacial | Número da linha de plantio no talhão | `12`, `45` | Sim |
| **`PLANTA`** | `estadio_fenologico` | `VARCHAR(32)` | Categórica Ordinal | Feature Contextual | Estágio de maturação fenológica | `"Brotação"`, `"Perfilhamento"`, `"Crescimento Intenso"`, `"Maturação"` | Sim |
| **`PLANTA`** | `idade_dias` | `INTEGER` | Numérica Discreta | Feature Contextual | Idade da planta em dias desde o corte/plantio | `45`, `120`, `270` | Sim |
| **`COLETA_DATA`** | `coleta_id` | `VARCHAR(64)` | Identificador | Metadado / ID | Identificador único da sessão/amostragem de campo | `"COL-2026-08-28-01"` | Não |
| **`COLETA_DATA`** | `data_hora_captura` | `DATETIME` | Temporal | Metadado Temporal | Data e horário exatos em que o registro foi realizado | `"2026-08-28T09:30:00Z"` | Não |
| **`COLETA_DATA`** | `temperatura_celsius` | `FLOAT` | Numérica Contínua | Feature Contextual | Temperatura ambiente no momento da coleta (°C) | `28.5`, `32.0` | Sim |
| **`COLETA_DATA`** | `umidade_relativa_pct`| `FLOAT` | Numérica Contínua | Feature Contextual | Umidade relativa do ar no momento da coleta (%) | `65.2`, `82.0` | Sim |
| **`COLETA_DATA`** | `responsavel_coleta` | `VARCHAR(128)` | Categórica Nominal | Metadado | Nome do agrônomo, técnico ou agente de amostragem | `"João Silva"`, `"Equipe Campo 02"` | Sim |
| **`IMAGEM`** | `imagem_id` | `VARCHAR(64)` | Identificador | Metadado / Chave | Identificador exclusivo do arquivo de imagem | `"IMG_ROBOFLOW_00291"`, `"MENDEL_RR_104"` | Não |
| **`IMAGEM`** | `file_path_relativo` | `VARCHAR(255)` | Texto / URI | Metadado | Caminho do arquivo relativo à raiz do repositório (`data/raw/` ou `data/processed/`) | `"data/raw/roboflow/train/img01.jpg"`, `"data/processed/images/folha_042.png"` | Não |
| **`IMAGEM`** | `data_source` | `VARCHAR(64)` | Categórica Nominal | Metadado / Split | Fonte de origem dos dados brutos | `"Roboflow"`, `"Mendeley_Data"`, `"Coleta_Propria"` | Não |
| **`IMAGEM`** | `split_dataset` | `VARCHAR(16)` | Categórica Nominal | Controle de Treino | Partição de Machine Learning para validação | `"train"`, `"valid"`, `"test"` | Não |
| **`IMAGEM`** | `image_width_px` | `INTEGER` | Numérica Discreta | Metadado Técnico | Largura da imagem em pixels | `640`, `1920`, `4000` | Não |
| **`IMAGEM`** | `image_height_px` | `INTEGER` | Numérica Discreta | Metadado Técnico | Altura da imagem em pixels | `640`, `1080`, `3000` | Não |
| **`IMAGEM`** | `num_channels` | `INTEGER` | Numérica Discreta | Metadado Técnico | Número de canais de cor da imagem | `3` (RGB), `1` (Grayscale), `4` (RGBA) | Não |
| **`IMAGEM`** | `file_extension` | `VARCHAR(8)` | Categórica Nominal | Metadado Técnico | Formato de compressão/armazenamento da imagem | `"jpg"`, `"jpeg"`, `"png"` | Não |
| **`IMAGEM`** | `condicao_iluminacao` | `VARCHAR(32)` | Categórica Nominal | Feature de Qualidade | Condição de luminosidade no enquadramento | `"Luz Natural Direta"`, `"Sombra"`, `"Difusa"` | Sim |
| **`IMAGEM`** | `angulo_captura` | `VARCHAR(32)` | Categórica Nominal | Feature de Qualidade | Perspectiva do registro fotográfico | `"Folha Superior"`, `"Face Adaxial"`, `"Face Abaxial"`, `"Visão Geral"` | Sim |
| **`DIAGNOSTICO`** | `target_doenca_classe` | `VARCHAR(64)` | Categórica Nominal | **Variável Alvo Principal** | Diagnóstico patológico da doença foliar | `"Saudavel"`, `"Podridao_Vermelha"`, `"Mosaico"`, `"Ferrugem_Marrom"`, `"Mancha_Amarela"` | Não |
| **`DIAGNOSTICO`** | `target_doenca_cod` | `INTEGER` | Categórica Discreta (ID) | **Target Codificado** | Rótulo numérico para funções de perda (Cross-Entropy) | `0, 1, 2, 3, 4, 5` | Não |
| **`DIAGNOSTICO`** | `target_is_doente` | `BOOLEAN` | Binária / Booleana | **Variável Alvo Binária** | Indicador de presença de qualquer fitopatologia | `True` (1), `False` (0) | Não |
| **`DIAGNOSTICO`** | `target_grau_severidade_pct` | `FLOAT` | Numérica Contínua | **Target Secundário (Regressão)** | Porcentagem da superfície foliar com lesões | `0.0` a `100.0` | Sim |
| **`DIAGNOSTICO`** | `confianca_anotacao` | `FLOAT` | Numérica Contínua | Peso de Amostra | Nível de concordância do especialista anotador ($0.0 - 1.0$) | `0.95`, `1.00` | Sim |
| **`DIAGNOSTICO`** | `tem_sintoma_visivel`| `BOOLEAN` | Binária | Validação de Rótulo | Se a imagem apresenta sintoma visual claro | `True`, `False` | Não |
| **`FEATURES_EXTRACTED`**| `exg_mean` | `FLOAT` | Numérica Contínua | Feature Candidata | Média do índice Excess Green ($2G - R - B$) | `-15.4`, `42.8` | Sim |
| **`FEATURES_EXTRACTED`**| `hsv_hue_mean` | `FLOAT` | Numérica Contínua | Feature Candidata | Média do matiz de cores na região de interesse foliar | `35.2` (amarelado/marrom), `85.0` (verde) | Sim |
| **`FEATURES_EXTRACTED`**| `glcm_contrast` | `FLOAT` | Numérica Contínua | Feature Candidata | Grau de contraste de textura entre pixels vizinhos | `0.12`, `1.84` | Sim |
| **`FEATURES_EXTRACTED`**| `glcm_homogeneity` | `FLOAT` | Numérica Contínua | Feature Candidata | Medida de homogeneidade da textura da folha | `0.88` (lisa/sadia), `0.35` (pústulas/rugosa) | Sim |
| **`FEATURES_EXTRACTED`**| `lesion_bbox_count` | `INTEGER` | Numérica Discreta | Feature Candidata | Quantidade de focos de infecção / bounding boxes | `0`, `5`, `23` | Sim |
| **`FEATURES_EXTRACTED`**| `embedding_vector` | `ARRAY[FLOAT]` | Vetor / Tensor | Feature de Deep Learning | Vetor de características (ex.: 512 ou 1024 dimensões) | `[-0.12, 0.44, ...]` | Sim |

---

## 6. Validação e Regras de Negócio para Ciência de Dados

1. **Integridade de Rótulos:** Toda imagem associada a `target_doenca_classe = 'Saudavel'` deve obrigatoriamente ter `target_is_doente = False` e `target_grau_severidade_pct = 0.0`.
2. **Prevenção de Vazamento de Dados (Data Leakage):** Amostras de imagens capturadas da mesma planta ou no mesmo talhão na mesma data devem pertencer exclusivamente à mesma partição (`train`, `valid` ou `test`) para evitar contaminação de dependência espacial/temporal no modelo.
3. **Consistência de Cores:** Imagens devem ser validadas quanto ao espaço de cor (RGB nativo), rejeitando ou corrigindo canais invertidos (BGR originado de OpenCV sem conversão).
