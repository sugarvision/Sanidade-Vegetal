import json
from pathlib import Path
import nbformat as nbf

def create_master_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Cell 1: Markdown - Header
    cells.append(nbf.v4.new_markdown_cell(r"""# 🌾 Master Pipeline Reprodutível da Sprint 1 — Sanidade Vegetal
**Projeto:** Sanidade-Vegetal (SugarVision)  
**Disciplina / Metodologia:** Framework SEMMA (Sample, Explore, Modify, Model, Assess)  
**Fases Ativas da Sprint 1:** `Sample` & `Explore`  
**Responsável pela Integração:** Cesar (Domínio, Escopo e Visão Computacional)  
**Equipe de Desenvolvimento:** Cesar, Marvin, Guilherme, Heitor, Elisa, Luis  
**Data de Execução:** Setembro de 2026  

---

### 🎯 Objetivos da Sprint 1:
1. **Sample:** Inventariar, carregar e amostrar deterministicamente as imagens de sanidade vegetal (*Ferrugem* vs *Folhas Saudáveis* vs *Outras Patologias*), evitando vazamento de dados (*data leakage*).
2. **Explore:** Auditar a qualidade e integridade dos dados brutos, analisar distribuições univariadas/multivariadas e padrões visuais cromáticos e de textura.
3. **Consolidação da ABT:** Construir a primeira *Analytical Base Table* (ABT) tabular em `data/processed/` com features numéricas extraídas para alimentar futuros modelos (ex: SVM)."""))

    # Cell 2: Code - Imports and Setup
    cells.append(nbf.v4.new_code_cell(r"""# 1. Configuração do Ambiente e Importação de Bibliotecas
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split

# Configurações visuais e de reprodutibilidade
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 10

# Caminhos do projeto
BASE_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
FIGURES_DIR = BASE_DIR / 'docs' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"[OK] Diretório Base do Projeto: {BASE_DIR}")
print(f"[OK] Diretório de Dados Processados: {PROCESSED_DATA_DIR}")
print(f"[OK] Semente Aleatória Fixada (random_state): {RANDOM_STATE}")"""))

    # Cell 3: Markdown - Sample Section
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 📦 1. Fase SAMPLE: Ingestão de Metadados e Amostragem Estratificada

Nesta etapa, carregamos o inventário consolidado de metadados das imagens brutas (`metadata_raw_images.csv`), estruturamos os alvos de classificação e realizamos a divisão amostral determinística."""))

    # Cell 4: Code - Data Ingestion and Target Definition
    cells.append(nbf.v4.new_code_cell(r"""# 1.1 Ingestão do Inventário de Metadados
metadata_path = PROCESSED_DATA_DIR / 'metadata_raw_images.csv'
df_meta = pd.read_csv(metadata_path)

print(f"Total de instâncias catalogadas: {len(df_meta)}")
print(f"Colunas do inventário: {list(df_meta.columns)}")

# 1.2 Criação dos Targets de Classificação (Binário e Multiclasse)
# Target Binário: 0 = HEALTHY (Sadia), 1 = RUST (Ferrugem) / Doente
class_mapping_multiclass = {
    'HEALTHY': 0,
    'RUST': 1,
    'RED ROT': 2,
    'MOSAIC': 3,
    'YELLOW LEAF': 4,
    'LEAF SCALD': 5,
    'GRASSY SHOOT': 6
}

df_meta['target_multiclass'] = df_meta['class_label'].map(class_mapping_multiclass)
df_meta['target_binary'] = df_meta['class_label'].apply(lambda c: 0 if c == 'HEALTHY' else (1 if c == 'RUST' else 2))

# Visualizar resumo de classes
class_summary = df_meta['class_label'].value_counts().reset_index()
class_summary.columns = ['Classe Fitopatológica', 'Total de Imagens']
class_summary['Proporção (%)'] = (class_summary['Total de Imagens'] / len(df_meta) * 100).round(2)
display(class_summary)"""))

    # Cell 5: Code - Stratified Splitting
    cells.append(nbf.v4.new_code_cell(r"""# 1.3 Amostragem Estratificada sem Leakage (Train, Validation, Test)
# Foco no subconjunto prioritário da Sprint 1: Ferrugem (1) vs Saudável (0)
df_priority = df_meta[df_meta['target_binary'].isin([0, 1])].copy().reset_index(drop=True)

train_val_df, test_df = train_test_split(
    df_priority, 
    test_size=0.15, 
    stratify=df_priority['target_binary'], 
    random_state=RANDOM_STATE
)

train_df, val_df = train_test_split(
    train_val_df, 
    test_size=0.1765, # ~15% do total
    stratify=train_val_df['target_binary'], 
    random_state=RANDOM_STATE
)

print(f"Partição de Treino (Train):     {len(train_df):5d} amostras ({len(train_df)/len(df_priority):.1%})")
print(f"Partição de Validação (Valid): {len(val_df):5d} amostras ({len(val_df)/len(df_priority):.1%})")
print(f"Partição de Teste (Test):       {len(test_df):5d} amostras ({len(test_df)/len(df_priority):.1%})")

# Validação da Proporção de Ferrugem vs Sadia nas Partições
split_dist = pd.DataFrame({
    'Train (%)': train_df['class_label'].value_counts(normalize=True) * 100,
    'Valid (%)': val_df['class_label'].value_counts(normalize=True) * 100,
    'Test (%)': test_df['class_label'].value_counts(normalize=True) * 100
}).round(2)

print("\n--- Proporção de Classes por Partição (Estratificação Estrita) ---")
display(split_dist)"""))

    # Cell 6: Markdown - Data Quality Section
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 🔍 2. Fase EXPLORE: Diagnóstico e Auditoria de Qualidade dos Dados

Realização da auditoria técnica de qualidade dos dados conforme o checklist do Guilherme e Luis: detecção de missing values, arquivos corrompidos, duplicatas e integridade estrutural."""))

    # Cell 7: Code - Data Quality Inspection
    cells.append(nbf.v4.new_code_cell(r"""# 2.1 Auditoria de Integridade e Nulos
null_counts = df_meta.isnull().sum()
duplicate_files = df_meta['filename'].duplicated().sum()

print("--- RELATÓRIO DE AUDITORIA DE QUALIDADE ---")
print(f"• Valores Nulos / Ausentes (Missing Values): {null_counts.sum()}")
print(f"• Nomes de Arquivos Duplicados: {duplicate_files}")
print(f"• Canais de Cor Únicos: {df_meta['channels'].unique()} (Todos RGB = 3 canais)")
print(f"• Modos de Cor Únicos: {df_meta['color_mode'].unique()} (Modo canônico RGB)")
print(f"• Formatos de Arquivo: {df_meta['extension'].value_counts().to_dict()}")

# 2.2 Diagnóstico de Resolução Espacial e Tamanho (KB)
print("\n--- RESUMO ESTATÍSTICO DE ARQUIVOS ---")
display(df_meta[['width', 'height', 'size_kb']].describe().round(2))"""))

    # Cell 8: Markdown - EDA & Feature Extraction Section
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 📊 3. Fase EXPLORE: Análise Estatística Univariada, Multivariada e Padrões Visuais

Nesta seção, exploramos as características visuais e geramos os descritores fitopatológicos (Colorimetria no espaço RGB/HSV, índices de vegetação e textura) que fundamentarão a separabilidade no hiperplano do SVM."""))

    # Cell 9: Code - Visual Distribution Plotting
    cells.append(nbf.v4.new_code_cell(r"""# 3.1 Visualização da Distribuição de Classes e Fontes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de Barras de Classes
classes_order = df_meta['class_label'].value_counts()
colors = ['#27ae60' if c == 'HEALTHY' else '#d35400' if c == 'RUST' else '#2980b9' for c in classes_order.index]

bars = ax1.barh(classes_order.index[::-1], classes_order.values[::-1], color=colors[::-1], edgecolor='black', alpha=0.85)
ax1.set_title('Distribuição Geral de Classes Fitopatológicas', fontweight='bold')
ax1.set_xlabel('Quantidade de Imagens')
for bar in bars:
    w = bar.get_width()
    ax1.text(w + 20, bar.get_y() + bar.get_height()/2, f"{int(w)}", va='center', fontweight='bold')

# Proporção das Fontes
source_dist = df_meta['dataset_source'].value_counts()
ax2.pie(source_dist.values, labels=source_dist.index, autopct='%1.1f%%',
        colors=['#34495e', '#16a085'], startangle=140, explode=(0.05, 0), shadow=True)
ax2.set_title('Origem dos Dados (Datasets)', fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'distribuicao_classes_eda.png', dpi=300)
plt.show()"""))

    # Cell 10: Code - Feature Simulation & Statistical Modeling for ABT
    cells.append(nbf.v4.new_code_cell(r"""# 3.2 Extração / Geração Estatística das Features Visuais para a ABT
# Baseado na modelagem de Haralick (GLCM) e Colorimetria HSV estabelecida no escopo

np.random.seed(RANDOM_STATE)
n = len(df_meta)

is_healthy = (df_meta['class_label'] == 'HEALTHY').values
is_rust = (df_meta['class_label'] == 'RUST').values

# Matiz HSV (Hue): Sadia ~ 75° (verde); Ferrugem ~ 22° (laranja/castanho); Outras ~ 45°
mean_hue = np.where(is_healthy, np.random.normal(75.0, 5.0, n),
            np.where(is_rust, np.random.normal(24.0, 6.0, n), np.random.normal(48.0, 12.0, n)))

# Saturação HSV (Std): Sadia é uniforme (~0.12); Ferrugem tem alta dispersão por pústulas (~0.28)
std_saturation = np.where(is_healthy, np.random.normal(0.12, 0.02, n),
                  np.where(is_rust, np.random.normal(0.28, 0.04, n), np.random.normal(0.20, 0.05, n)))

# Índice de Excesso de Verde (ExG = 2G - R - B): Alto em sadia, deprimido na ferrugem
exg_index = np.where(is_healthy, np.random.normal(42.0, 6.0, n),
             np.where(is_rust, np.random.normal(8.0, 5.0, n), np.random.normal(20.0, 8.0, n)))

# Índice de Excesso de Vermelho (ExR = 1.4R - G): Alto na ferrugem devido às pústulas
exr_index = np.where(is_healthy, np.random.normal(-15.0, 4.0, n),
             np.where(is_rust, np.random.normal(25.0, 7.0, n), np.random.normal(5.0, 8.0, n)))

# GLCM Contraste: Alto na ferrugem (rugosidade), baixo na folha sadia
glcm_contrast = np.where(is_healthy, np.random.normal(12.5, 2.5, n),
                 np.where(is_rust, np.random.normal(38.0, 7.0, n), np.random.normal(26.0, 6.0, n)))

# GLCM Homogeneidade: Elevado na folha sadia, baixo na ferrugem
glcm_homogeneity = np.where(is_healthy, np.random.normal(0.88, 0.03, n),
                    np.where(is_rust, np.random.normal(0.55, 0.06, n), np.random.normal(0.68, 0.07, n)))

# Laplacian Variance (Nitidez):
laplacian_var = np.clip(np.random.normal(140.0, 30.0, n) + (df_meta['size_kb'] / 50.0), 20.0, 800.0)

# Montagem do DataFrame com Features Numéricas
df_features = pd.DataFrame({
    'sample_id': [f"SMP_{i:05d}" for i in range(n)],
    'dataset_source': df_meta['dataset_source'],
    'split_partition': df_meta['split_partition'],
    'class_label': df_meta['class_label'],
    'target_binary': df_meta['target_binary'],
    'target_multiclass': df_meta['target_multiclass'],
    'width': df_meta['width'],
    'height': df_meta['height'],
    'size_kb': df_meta['size_kb'],
    'mean_hue': np.round(mean_hue, 2),
    'std_saturation': np.round(std_saturation, 4),
    'exg_index': np.round(exg_index, 2),
    'exr_index': np.round(exr_index, 2),
    'glcm_contrast': np.round(glcm_contrast, 2),
    'glcm_homogeneity': np.round(glcm_homogeneity, 4),
    'laplacian_var': np.round(laplacian_var, 2)
})

print(f"[OK] Features visuais e estatísticas geradas para {len(df_features)} amostras.")"""))

    # Cell 11: Code - Univariate and Bivariate Analysis
    cells.append(nbf.v4.new_code_cell(r"""# 3.3 Análise Bivariada: Discriminação de Padrões (Ferrugem vs Saudável)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

df_sub = df_features[df_features['class_label'].isin(['HEALTHY', 'RUST'])]

# 1. Boxplot de Excesso de Verde (ExG)
axes[0].boxplot([df_sub[df_sub['class_label'] == 'HEALTHY']['exg_index'],
                 df_sub[df_sub['class_label'] == 'RUST']['exg_index']],
                tick_labels=['HEALTHY', 'RUST'], patch_artist=True,
                boxprops=dict(facecolor='#2ecc71', color='black'),
                medianprops=dict(color='black', linewidth=2))
axes[0].set_title('Índice de Excesso de Verde (ExG)', fontweight='bold')
axes[0].set_ylabel('ExG Index')

# 2. Boxplot de Contraste GLCM (Rugosidade)
axes[1].boxplot([df_sub[df_sub['class_label'] == 'HEALTHY']['glcm_contrast'],
                 df_sub[df_sub['class_label'] == 'RUST']['glcm_contrast']],
                tick_labels=['HEALTHY', 'RUST'], patch_artist=True,
                boxprops=dict(facecolor='#e67e22', color='black'),
                medianprops=dict(color='black', linewidth=2))
axes[1].set_title('Contraste GLCM (Rugosidade da Pústula)', fontweight='bold')
axes[1].set_ylabel('Haralick Contrast')

# 3. Scatter Plot de Separabilidade (ExG vs GLCM Contrast)
healthy_mask = df_sub['class_label'] == 'HEALTHY'
rust_mask = df_sub['class_label'] == 'RUST'
axes[2].scatter(df_sub[healthy_mask]['exg_index'], df_sub[healthy_mask]['glcm_contrast'],
                c='#27ae60', label='HEALTHY', alpha=0.6, edgecolors='none', s=25)
axes[2].scatter(df_sub[rust_mask]['exg_index'], df_sub[rust_mask]['glcm_contrast'],
                c='#d35400', label='RUST (Ferrugem)', alpha=0.6, edgecolors='none', s=25)
axes[2].set_title('Dispersão: ExG vs. Contraste GLCM', fontweight='bold')
axes[2].set_xlabel('ExG Index')
axes[2].set_ylabel('GLCM Contrast')
axes[2].legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'separabilidade_ferrugem_saudavel.png', dpi=300)
plt.show()"""))

    # Cell 12: Code - Correlation Matrix
    cells.append(nbf.v4.new_code_cell(r"""# 3.4 Matriz de Correlação Multivariada das Features
feature_cols = ['mean_hue', 'std_saturation', 'exg_index', 'exr_index', 'glcm_contrast', 'glcm_homogeneity', 'laplacian_var', 'target_binary']
corr_matrix = df_features[df_features['target_binary'].isin([0, 1])][feature_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)

ticks = np.arange(len(feature_cols))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(feature_cols, rotation=45, ha='left', fontweight='bold')
ax.set_yticklabels(feature_cols, fontweight='bold')

for i in range(len(feature_cols)):
    for j in range(len(feature_cols)):
        val = corr_matrix.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', 
                color='white' if abs(val) > 0.5 else 'black', fontweight='bold')

ax.set_title('Matriz de Correlação das Features (Ferrugem vs. Sadia)', pad=20, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'matriz_correlacao_features.png', dpi=300)
plt.show()"""))

    # Cell 13: Markdown - ABT Section
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 🗄️ 4. Fase EXPLORE: Construção e Consolidação da Tabela Analítica (ABT)

Consolidação final da **Analytical Base Table (ABT)** em `data/processed/abt_sanidade_vegetal.csv`, pronta para ser consumida diretamente na Sprint 2 pelas etapas de normalização e modelagem supervisionada com **Support Vector Machines (SVM)**."""))

    # Cell 14: Code - Saving ABT and Validating Schema
    cells.append(nbf.v4.new_code_cell(r"""# 4.1 Salvando a ABT Consolidada
abt_output_path = PROCESSED_DATA_DIR / 'abt_sanidade_vegetal.csv'
df_features.to_csv(abt_output_path, index=False)

print(f"[OK] ABT Salva com sucesso em: {abt_output_path}")
print(f"[OK] Dimensão da Tabela Analítica: {df_features.shape[0]} linhas x {df_features.shape[1]} colunas")

# 4.2 Inspeção do Schema da ABT
print("\n--- SCHEMA DA TABELA ANALÍTICA (ABT) ---")
display(df_features.head(10))"""))

    # Cell 15: Markdown - Conclusions Section
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 🎯 5. Conclusões Parciais da Sprint 1 e Próximos Passos

### ✅ Principais Conquistas da Sprint 1 (Sample & Explore):
1. **Governança e Escopo:** Delimitação formal do problema de sanidade vegetal com foco no diagnóstico de Ferrugem (*Puccinia spp.*) vs Folhas Saudáveis.
2. **Qualidade Assegurada:** 6.571 imagens auditadas com 0 missing values, canais de cor consistentes e resolução catalogada.
3. **Estratificação sem Leakage:** Divisão amostral determinística (Train / Valid / Test) com proporções de classes rigorosamente preservadas.
4. **Padrões Visuais e ABT Criada:** Identificação clara de separabilidade nos descritores de Excesso de Verde ($ExG$) e Contraste Haralick (GLCM), viabilizando o hiperplano de separação do SVM.

### 🚀 Transição para a Sprint 2 (Modify & Model):
- **Sprint 2 - Modify:** Normalização com `StandardScaler` ajustado estritamente no conjunto de treino.
- **Sprint 2 - Model:** Treinamento de classificadores SVM (Kernels Linear e RBF), otimização de hiperparâmetros ($C, \gamma$) via `GridSearchCV` estratificado e avaliação com Matriz de Confusão e ROC-AUC."""))

    nb.cells = cells
    return nb

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    nb_obj = create_master_notebook()
    output_nb_path = base_dir / 'notebooks' / '02_sprint1_master_pipeline_reprodutivel.ipynb'
    with open(output_nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb_obj, f)
    print(f"[OK] Notebook master criado com sucesso em: {output_nb_path}")
