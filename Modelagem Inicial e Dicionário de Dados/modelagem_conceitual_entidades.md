# 🏗️ Modelagem Conceitual de Entidades e Relacionamentos

**Projeto:** Sanidade-Vegetal (SugarVision)  
**Sprint:** 1 — Modelagem Inicial e Dicionário de Dados  

---

## 1. Visão Geral da Arquitetura de Dados

A modelagem de dados do ecossistema **SugarVision** foi desenhada para conectar o contexto agronômico de campo com a esteira de Visão Computacional e Aprendizado Profundo (*Deep Learning*).

As quatro entidades nucleares solicitadas pelo checklist — **`TALHAO`**, **`PLANTA`**, **`COLETA_DATA`** e **`IMAGEM`** — integram-se de forma consistente às entidades de inferência e aprendizado de máquina (**`DIAGNOSTICO`** e **`FEATURES_EXTRACTED`**).

---

## 2. Diagrama Entidade-Relacionamento (MER / DER)

```mermaid
erDiagram
    TALHAO ||--o{ PLANTA : "contem (1:N)"
    TALHAO ||--o{ COLETA_DATA : "recebe_visita (1:N)"
    PLANTA ||--o{ IMAGEM : "e_fotografada (1:N)"
    COLETA_DATA ||--o{ IMAGEM : "registra (1:N)"
    IMAGEM ||--|| DIAGNOSTICO : "possui_rotulo (1:1)"
    IMAGEM ||--o| FEATURES_EXTRACTED : "gera_vetores (1:1)"

    TALHAO {
        string talhao_id PK
        string fazenda_nome
        float latitude
        float longitude
        string variedade_cultivar
        string tipo_solo
    }

    PLANTA {
        string planta_id PK
        string talhao_id FK
        int posicao_linha
        string estadio_fenologico
        int idade_dias
    }

    COLETA_DATA {
        string coleta_id PK
        string talhao_id FK
        datetime data_hora_captura
        float temperatura_celsius
        float umidade_relativa_pct
        string responsavel_coleta
    }

    IMAGEM {
        string imagem_id PK
        string coleta_id FK
        string planta_id FK
        string file_path_relativo
        string data_source
        string split_dataset
        int image_width_px
        int image_height_px
        int num_channels
        string file_extension
        string condicao_iluminacao
        string angulo_captura
    }

    DIAGNOSTICO {
        string diagnostico_id PK
        string imagem_id FK
        string target_doenca_classe "TARGET MULTICLASSE"
        int target_doenca_cod
        boolean target_is_doente "TARGET BINARIO"
        float target_grau_severidade_pct "TARGET REGRESSAO"
        float confianca_anotacao
        boolean tem_sintoma_visivel
    }

    FEATURES_EXTRACTED {
        string feature_id PK
        string imagem_id FK
        float exg_mean
        float hsv_hue_mean
        float glcm_contrast
        float glcm_homogeneity
        int lesion_bbox_count
        array embedding_vector "TENSOR EMBEDDING"
    }
```

---

## 3. Descrição Detalhada das Entidades e Cardinalidades

### 1. `TALHAO` (Área de Cultivo / Gleba)
* **Conceito:** Representa a unidade produtiva e geográfica delimitada dentro de uma fazenda ou usina canavieira.
* **Cardinalidade:** 
  * $1:N$ com `PLANTA` (um talhão contém múltiplas plantas/linhas).
  * $1:N$ com `COLETA_DATA` (um talhão passa por múltiplas campanhas de amostragem ao longo da safra).
* **Importância em Ciência de Dados:** Permite controlar o efeito de lote (*batch effect*), solo e variedade genética (*cultivar*), além de ser a unidade ideal para agrupamento (*GroupKFold*) para evitar vazamento de dados geográficos no treino de modelos.

### 2. `PLANTA` (Indivíduo Vegetal / Touceira)
* **Conceito:** A planta específica de cana monitorada no talhão.
* **Cardinalidade:** 
  * $N:1$ com `TALHAO`.
  * $1:N$ com `IMAGEM` (uma mesma planta pode ter fotos de diferentes folhas, faces adaxial/abaxial e diferentes ângulos).
* **Importância em Ciência de Dados:** Permite agregação temporal da evolução da sanidade da mesma planta ao longo dos dias.

### 3. `COLETA_DATA` (Evento Temporal de Monitoramento)
* **Conceito:** O registro da visita a campo ou voo de inspeção onde as fotos foram capturadas.
* **Cardinalidade:** 
  * $1:N$ com `IMAGEM` (uma única coleta produz centenas de registros fotográficos).
* **Importância em Ciência de Dados:** Agrupa condições microclimáticas (temperatura, umidade) que afetam tanto o desenvolvimento biológico de fungos/vírus quanto a qualidade visual da imagem (iluminação, reflexos de orvalho).

### 4. `IMAGEM` (Artefato Visual / Tensor Bruto)
* **Conceito:** O arquivo de imagem capturado (JPEG/PNG), seus metadados de aquisição e partição de treino.
* **Cardinalidade:**
  * $N:1$ com `COLETA_DATA` e $N:1$ com `PLANTA`.
  * $1:1$ com `DIAGNOSTICO`.
  * $1:1$ (ou $1:N$ por patch) com `FEATURES_EXTRACTED`.
* **Importância em Ciência de Dados:** Entrada principal (*input*) dos modelos de Visão Computacional (CNNs / Vision Transformers).

### 5. `DIAGNOSTICO` (Ground Truth / Rótulo de Treinamento)
* **Conceito:** O laudo fitopatológico emitido por especialistas ou extraído dos datasets benchmark (Roboflow e Mendeley Data).
* **Alvos Primários:**
  * Doença classificada (`Saudavel`, `Podridao_Vermelha`, `Mosaico`, `Ferrugem_Marrom`, `Mancha_Amarela`, `Carvao`).
  * Indicador binário (`target_is_doente`).
  * Percentual de severidade da área foliar lesionada.

### 6. `FEATURES_EXTRACTED` (Espaço de Características para ML)
* **Conceito:** Vetor numérico derivado para treinamento de modelos híbridos (Visão + Tabular) ou modelos leves (*LightGBM / XGBoost / SVM* sobre embeddings e métricas GLCM/HSV).

---

## 4. Fluxo de Dados e Armazenamento no Repositório

```mermaid
flowchart LR
    A["📂 data/raw/<br/>(Imagens Brutas: Roboflow, Mendeley, Campo)"] --> B["⚙️ Pré-processamento & Augmentation<br/>(Resize, Normalização, Split)"]
    B --> C["📂 data/processed/<br/>(Imagens padronizadas, Máscaras e CSVs)"]
    C --> D1["🧠 Deep Learning Backbone<br/>(ResNet / EfficientNet / ViT)"]
    C --> D2["🔬 Engenharia de Features<br/>(GLCM, ExG, HSV)"]
    
    D1 --> E["🏷️ Loss Function & Otimizador<br/>(Cross-Entropy / Focal Loss)"]
    D2 --> E
    
    E --> F["📊 Output do Modelo:<br/>1. Classe da Patologia (Multiclasse)<br/>2. Detecção de Anomalia (Binário)<br/>3. % Severidade Foliar"]
```
