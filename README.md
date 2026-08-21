# Sanidade-Vegetal (SugarVision)

Projeto desenvolvido para a classificação e o diagnóstico inteligente de patologias foliares em cana-de-açúcar utilizando técnicas de Aprendizado de Máquina e Visão Computacional.

---

## 📂 Conjuntos de Dados (Datasets)

Devido ao volume total dos conjuntos de dados (**5,28 GB no total**), os arquivos de imagens não são versionados diretamente no repositório Git.

### 1. Preparação Automática do Diretório

Para criar a estrutura inicial de pastas localmente, execute o script auxiliar:

```bash
python setup_datasets.py
```

O script criará o diretório base `datasets/` no seu ambiente de trabalho.

---

### 2. Instruções de Download e Extração

Baixe manualmente os pacotes nos links oficiais abaixo e descompacte-os nas respectivas pastas de destino:

#### **A. Dataset do Roboflow**
* **URL:** [Roboflow Universe - Sugarcane Disease Classification](https://universe.roboflow.com/asad-unvar/sugarcane-disease-classification)
* **Ação:** Baixe o dataset como arquivo ZIP e extraia o conteúdo em:
  ```text
  datasets/roboflow_sugarcane/
  ```

#### **B. Dataset do Mendeley Data**
* **URL:** [Mendeley Data - Sugarcane Diseases](https://data.mendeley.com/datasets/rzh99cj2rj/1)
* **Ação:** Baixe os arquivos disponibilizados e extraia-os dentro de:
  ```text
  datasets/mendeley_data/
  ```

---

### 3. Estrutura Final do Diretório de Dados

Após o download e a extração completa dos arquivos, a estrutura da pasta `datasets/` deve estar configurada da seguinte forma:

```text
datasets/
├── roboflow_sugarcane/
│   ├── train/
│   ├── valid/
│   └── test/
└── mendeley_data/
    └── [arquivos_do_dataset]/
```

---

## 🚀 Como Executar o Projeto

1. Clone o repositório em sua máquina:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd Sanidade-Vegetal
   ```
2. Prepare os diretórios de dados:
   ```bash
   python setup_datasets.py
   ```
3. Siga as instruções acima para alocar as bases em `datasets/roboflow_sugarcane/` e `datasets/mendeley_data/`.
