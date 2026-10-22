# 🌾 SugarVision — Sanidade Vegetal
### Classificação e Diagnóstico Inteligente de Fitopatologias em Cana-de-Açúcar (*Saccharum officinarum*)

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange?logo=scikit-learn)
![Framework](https://img.shields.io/badge/Framework-SEMMA%20(SAS)-darkgreen)
![Status](https://img.shields.io/badge/Sprint-5%20%7C%20Homologado-brightgreen)
![Tests](https://img.shields.io/badge/Tests-14%2F14%20Passing-success)
![License](https://img.shields.io/badge/License-MIT%20%2B%20CC%20BY%204.0-lightgrey)

---

## 📌 1. Visão Geral do Projeto

O **SugarVision** é uma solução de Visão Computacional e Inteligência Artificial desenvolvida para automatizar a identificação, classificação e diagnóstico fitopatológico precoce na cultura da cana-de-açúcar. 

O sistema foi desenhado para resolver o desafio crítico de detecção da **Ferrugem da Cana-de-Açúcar (*Puccinia melanocephala* e *Puccinia kuehnii*)**, além de discriminar com alta precisão tecidos sadios e outras 5 patologias foliares com sintomas visuais sobrepostos:
1. **HEALTHY** (Folhas Sadias / Ausência de Patologia)
2. **RUST** (Ferrugem da Cana-de-Açúcar)
3. **LEAF SCALD** (Escaldadura das Folhas — *Xanthomonas albilineans*)
4. **RED ROT** (Podridão Vermelha — *Colletotrichum falcatum*)
5. **MOSAIC** (Vírus do Mosaico da Cana — SCMV)
6. **YELLOW LEAF** (Vírus do Amarelecimento Foliar — SCYLV)
7. **GRASSY SHOOT** (Complexo Fitoplasma de Enfezamento)

---

## 🏗️ 2. Arquitetura da Solução (Framework SEMMA)

O ciclo de vida do projeto foi estruturado segundo as 5 fases do **Framework SEMMA**:

```mermaid
flowchart LR
    S["1. SAMPLE\n(Amostragem)\nEstratificação\nAnti-Leakage"] --> E["2. EXPLORE\n(Exploração)\nEDA Univariada\ne Multivariada"]
    E --> M["3. MODIFY\n(Engenharia)\nDescritores HSV,\nGLCM & Bins"]
    M --> MO["4. MODEL\n(Modelagem)\nSVM RBF Ótimo\n(C=10, γ=0.01)"]
    MO --> A["5. ASSESS\n(Avaliação)\nRepeated CV,\nAUC > 0.91 & Deploy"]
```

1. **Sample (Sprint 1):** Divisão determinística estratificada ($70\%$ treino, $15\%$ validação, $15\%$ teste cego), balanceamento e blindagem de talhão.
2. **Explore (Sprint 1):** Análise exploratória de qualidade ótica, resolução, variância laplaciana e histogramas cromáticos.
3. **Modify (Sprint 2):** Engenharia de atributos visuais: índices de excesso verde/vermelho (ExG, ExR), razão R/G, dispersão de matiz HSV, textura Haralick GLCM (contraste, homogeneidade, energia) e discretização estatística por quantis e k-means.
4. **Model (Sprint 3):** Pipeline atômico integrado (`StandardScaler` + `SVC`), estudo exaustivo de Kernel Trick (Linear, Polinomial e RBF) e otimização por Grid Search.
5. **Assess (Sprints 4 e 5):** Auditoria com **Repeated Stratified K-Fold** ($15$ folds), comprovação de ausência de viés amostral, paralelização com controle de RAM (`n_jobs=-1`), persistência incremental de checkpoints, catálogo formal de metadados, congelamento de dependências e governança de deploy.

---

## 📂 3. Estrutura Canônica de Pastas do Repositório

```text
Sanidade-Vegetal/
├── data/
│   ├── raw/                                           # Imagens e anotações originais (Roboflow e Mendeley)
│   └── processed/
│       ├── abt_sanidade_vegetal.csv                   # Tabela Analítica Base (6.571 instâncias x 35 atributos)
│       ├── metadata_raw_images.csv                    # Metadados físicos de extração (13.142 registros)
│       └── svm_predicoes_valid_test.csv               # Tabela de registros analíticos e predições calibradas
├── docs/
│   ├── figures/                                       # Gráficos analíticos e curvas de diagnóstico (300 DPI)
│   ├── relatorio_conformidade_licencas_e_permissoes.md# Auditoria jurídica de licenças de código e dados
│   └── relatorios_sprints/                            # Dossiês executivos e relatos técnicos das entregas
├── models/
│   ├── svm_pipeline_optimal.joblib                    # Estimador SVM campeão serializado para produção
│   ├── gridsearch_checkpoint.json                     # Checkpoint transacional atômico tolerante a falhas
│   ├── gridsearch_checkpoint_results.csv              # Resultados comparativos da malha de hiperparâmetros
│   ├── dataset_lineage_and_metrics_catalog.json       # Manifesto MLOps de rastreabilidade e métricas
│   ├── computational_efficiency_metrics.json          # Telemetria de CPU e memória RAM
│   ├── repeated_kfold_audit_metadata.json             # Estatísticas de dispersão estocástica (15 folds)
│   └── repository_structure_audit.json                # Atestado de conformidade estrutural
├── notebooks/                                         # Cadernos Jupyter interativos de cada fase SEMMA
├── src/
│   ├── atomic_svm_pipeline.py                         # Construtor do pipeline atômico anti-leakage
│   ├── repeated_stratified_kfold_audit.py             # Validador de robustez estocástica (15 splits)
│   ├── resilient_parallel_grid_search.py              # Grid Search paralelo (n_jobs=-1) com telemetria
│   ├── audit_metadata_lineage.py                      # Auditoria de linhagem e cálculo de métricas formais
│   ├── audit_repository_structure.py                  # Auditor de pastas canônicas e integridade
│   ├── test_repeated_kfold_audit.py                   # Suite de testes de validação cruzada
│   ├── test_parallel_efficiency.py                    # Suite de testes de eficiência computacional
│   └── test_metadata_catalog_governance.py            # Suite de testes de governança e integridade
├── Modelagem Inicial e Dicionário de Dados/
│   ├── dicionario_de_dados.md                         # Dicionário de dados técnico detalhado (v2.2.0)
│   └── dicionario_de_dados.csv                        # Catálogo tabular completo com entidade de métricas
├── requirements.txt                                   # Congelamento estrito de dependências
├── LICENSE                                            # Licença MIT de código aberto
└── README.md                                          # Este guia didático e portal de reprodução
```

---

## 🚀 4. Guia Didático: Instalação e Reprodução Passo a Passo

Siga o roteiro abaixo para reproduzir 100% do projeto em qualquer ambiente (Windows, Linux ou macOS):

### Passo 1: Clonar o Repositório Git
Abra o terminal e execute:
```bash
git clone https://github.com/heitorvittip/Sanidade-Vegetal.git
cd Sanidade-Vegetal
```

### Passo 2: Configurar o Ambiente Virtual Python
Recomenda-se utilizar Python $\ge 3.10$:
```bash
# No Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# No Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Passo 3: Instalar as Dependências Estritas
Instale o ecossistema com versões congeladas para garantir reprodutibilidade idêntica:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Passo 4: Estruturação dos Conjuntos de Dados (Datasets)
Caso deseje extrair as features novamente a partir dos dados brutos:
1. Crie os diretórios executando:
   ```bash
   python setup_datasets.py
   ```
2. Baixe os pacotes oficiais nos portais de origem:
   - **Roboflow Universe:** [Sugarcane Disease Classification](https://universe.roboflow.com/asad-unvar/sugarcane-disease-classification) (extrair em `datasets/roboflow_sugarcane/`).
   - **Mendeley Data:** [Sugarcane Diseases](https://data.mendeley.com/datasets/rzh99cj2rj/1) (extrair em `datasets/mendeley_data/`).
> *Nota: A base processada consolidada já se encontra versionada em `data/processed/abt_sanidade_vegetal.csv`, permitindo executar todos os pipelines de modelagem imediatamente.*

### Passo 5: Executar a Validação de Robustez Estocástica (Repeated K-Fold)
Executa 15 folds de validação cruzada estratificada repetida para comprovar a ausência de viés amostral:
```bash
python src/repeated_stratified_kfold_audit.py
```

### Passo 6: Executar a Malha Paralela com Monitoramento de Hardware e Checkpoint
Executa a busca em grade com `n_jobs=-1` em paralelo, telemetria contínua de CPU e RAM e tolerância a falhas:
```bash
python src/resilient_parallel_grid_search.py
```

### Passo 7: Auditar a Linhagem de Dados e Atualizar o Catálogo de Métricas
Audita a integridade referencial ponta a ponta e consolida as métricas estatísticas:
```bash
python src/audit_metadata_lineage.py
```

### Passo 8: Executar a Suíte Completa de Testes Automatizados
Valida formalmente a integridade de todas as entregas do projeto:
```bash
python -m unittest discover -s src -p "test_*.py"
```
*(Todos os 14 testes devem retornar `OK` em menos de 1 segundo).*

---

## 📊 5. Métricas Oficiais do Estimador Campeão Homologado

O estimador oficial é um **SVM com Kernel RBF** ($C=10.0, \gamma=0.01$, `class_weight='balanced'`, `probability=True` com calibração Platt):

| Família de Métrica | Partição de Validação ($N=617$) | Partição de Teste Cego ($N=380$) | Repeated CV (15 Folds) | Interpretação |
| :--- | :---: | :---: | :---: | :--- |
| **Acurácia Global** | **62,07%** | **60,79%** | $64,48\% \pm 0,79\%$ | Proporção de laudos perfeitamente concordantes. |
| **Acurácia Balanceada** | **65,32%** | **67,69%** | $64,46\% \pm 1,21\%$ | Média não ponderada de acertos (imune ao desbalanceamento). |
| **$F_1$-Score (Macro)** | **0,5757** | **0,6054** | $\mathbf{0,5761 \pm 0,0082}$ | Métrica principal de seleção de modelo ($CV = 1,43\%$). |
| **$F_1$-Score (Weighted)** | **0,5624** | **0,5475** | $0,5904 \pm 0,0080$ | Média harmônica ponderada pelo inventário foliar. |
| **AUC-ROC (Macro OvR)** | **0,9148** | **0,9116** | - | **Excepcional capacidade discriminativa (> 0.91)**. |
| **Specificity (Macro OvR)**| **0,9381** | **0,9361** | - | **Taxa de verdadeiros negativos > 93,6%** (sem falsos alarmes). |
| **HEALTHY (Sadia)** | **$F_1 = 1,0000$** | **$F_1 = 1,0000$** | - | **Separabilidade Perfeita (100% de Acerto)**. |
| **RUST (Ferrugem)** | **$F_1 = 1,0000$** | **$F_1 = 1,0000$** | - | **Separabilidade Perfeita (100% de Acerto)**. |

---

## 💻 6. Exemplo de Inferência Programática (Deploy)

Para utilizar o modelo campeão homologado para classificar novas amostras em ambiente de produção:

```python
import joblib
import pandas as pd
from pathlib import Path

# 1. Carregar pipeline atômico serializado (Pré-processador + SVM RBF Calibrado)
model_path = Path("models/svm_pipeline_optimal.joblib")
pipeline = joblib.load(model_path)

# 2. Amostras de entrada (devem conter as features preditivas da ABT)
# Exemplo com dados da partição de teste:
df = pd.read_csv("data/processed/abt_sanidade_vegetal.csv")
X_new = df[df['split_partition'] == 'test'].drop(
    columns=['sample_id', 'filepath', 'filename', 'relative_path',
             'dataset_source', 'split_partition', 'class_label',
             'target_binary', 'target_multiclass', 'extension']
).iloc[:5]

# 3. Predição da Classe e Probabilidades Calibradas
pred_codes = pipeline.predict(X_new)
pred_probas = pipeline.predict_proba(X_new)

class_names = ['GRASSY SHOOT', 'HEALTHY', 'LEAF SCALD', 'MOSAIC', 'RED ROT', 'RUST', 'YELLOW LEAF']

for i, code in enumerate(pred_codes):
    confidence = pred_probas[i, code] * 100.0
    print(f"Amostra #{i+1}: Diagnóstico = {class_names[code]:<15} | Confiança = {confidence:5.2f}%")
```

---

## ⚖️ 7. Conformidade de Licenças e Permissões de Dados

O projeto **SugarVision** foi desenvolvido sob rigorosa conformidade de propriedade intelectual e governança de dados abertos:

1. **Licenciamento do Código-Fonte:**  
   O código deste repositório está licenciado sob a **Licença MIT** ([LICENSE](LICENSE)), garantindo liberdade de uso acadêmico e comercial.
2. **Conjuntos de Dados Externos:**  
   - **Roboflow Universe (*Sugarcane Disease Classification*):** Licenciado sob **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Autoria: Asad Unvar.
   - **Mendeley Data (*Sugarcane Diseases*):** Licenciado sob **Creative Commons Attribution 4.0 International (CC BY 4.0)** (DOI: [10.17632/rzh99cj2rj.1](https://doi.org/10.17632/rzh99cj2rj.1)). Autoria: Kalhoro et al.
   - Ambas as licenças permitem expressamente a transformação, adaptação e uso acadêmico/comercial mediante a citação e atribuição dos autores originais.
3. **Bibliotecas de Código Aberto:**  
   Todas as dependências congeladas em `requirements.txt` (`scikit-learn`, `numpy`, `pandas`, `scipy`, `matplotlib`, `joblib`, `psutil`, `Pillow`, `opencv`) possuem licenças permissivas compatíveis (**BSD 3-Clause, Apache 2.0 ou MIT**), ausentes de restrições copyleft virais.
   Consulte o relatório formal em [`docs/relatorio_conformidade_licencas_e_permissoes.md`](docs/relatorio_conformidade_licencas_e_permissoes.md).

---

## 👥 Equipe e Responsáveis Técnicos
- **Heitor Partezani (HP)** — Engenharia de Machine Learning, IA SVM & MLOps
- **Gabriel Ramos (GR)** — Eficiência Computacional e Infraestrutura
- **Cesar** — Lead Técnico & Visão Computacional
- **Elisa** — Engenharia de Dados & Governança SEMMA
