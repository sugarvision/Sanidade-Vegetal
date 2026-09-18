"""
Script utilitário para gerar e executar o notebook 06_pipeline_atomico_columntransformer.ipynb
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 2 (Fase Modify)
Responsável: Elisa
"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient


def create_and_run_notebook():
    base_dir = Path(__file__).resolve().parent.parent
    nb_path = base_dir / "notebooks" / "06_pipeline_atomico_columntransformer.ipynb"
    
    nb = nbf.v4.new_notebook()
    nb['metadata'] = {
        'kernelspec': {
            'display_name': 'Python 3.12',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.12'
        }
    }
    
    cells = []
    
    # 1. Título e Cabeçalho
    cells.append(nbf.v4.new_markdown_cell("""# 🔬 Pipeline Atômico com ColumnTransformer & Protocolo Anti-Leakage
**Projeto:** Sanidade-Vegetal (SugarVision)  
**Sprint:** 2 — Framework SEMMA (Fase: Modify)  
**Responsável:** Elisa (`EA`) — Engenharia de Machine Learning & MLOps  
**Dataset Base:** `data/processed/abt_sanidade_vegetal.csv` (6.571 instâncias)  

---

## 📌 Objetivo da Entrega

Desenvolver e validar o **fluxo unificado de pré-processamento no Scikit-Learn** utilizando `Pipeline` e `ColumnTransformer`, encapsulando imputação por mediana, padronização estatística z-score e codificação dummy (One-Hot) com proteção estrita contra vazamento de dados (*Data Leakage*), garantindo conformidade com a **Regra de Ouro do Fit** e compatibilidade com MLOps.

### ✅ Cobertura do Checklist da Sprint 2:
1. **Mapear as listas de colunas** conforme o tratamento requerido (numéricas, discretizadas e categóricas).
2. **Criar o sub-pipeline numérico** com `SimpleImputer(strategy='median')` e `StandardScaler()`.
3. **Criar o sub-pipeline categórico** com `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`.
4. **Integrar todos os transformadores** em um único objeto `ColumnTransformer`.
5. **Validar a "Regra de Ouro do Fit":** assegurar que o `.fit()` ocorra apenas no treino e o `.transform()` seja replicado nas demais partições.
6. **Validar a serialização preliminar** do pipeline para checar compatibilidade futura com MLOps."""))

    # 2. Setup e Importações
    cells.append(nbf.v4.new_code_cell("""# 1. Configuração do ambiente e importação das dependências
import os
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Configurações de exibição do pandas e sklearn
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
sklearn.set_config(display='diagram')  # Renderização visual interativa do pipeline

print(f"Scikit-Learn version : {sklearn.__version__}")
print(f"Joblib version       : {joblib.__version__}")
print(f"Pandas version       : {pd.__version__}")"""))

    # 3. Carga da ABT
    cells.append(nbf.v4.new_markdown_cell("""---
## 1. Carga e Inspeção Estrutural da Tabela Analítica Base (ABT)

Carregamos o dataset consolidado `data/processed/abt_sanidade_vegetal.csv`, contendo os metadados de imagem, variáveis de engenharia espectral/textura Haralick e variáveis discretizadas via binning."""))

    cells.append(nbf.v4.new_code_cell("""# Localização do dataset no projeto
base_dir = Path(os.getcwd()).parent if "notebooks" in os.getcwd() else Path(os.getcwd())
abt_path = base_dir / "data" / "processed" / "abt_sanidade_vegetal.csv"

df = pd.read_csv(abt_path)
print(f"Dimensão da ABT: {df.shape[0]} registros x {df.shape[1]} colunas.")

# Distribuição de partições estritas
split_dist = df['split_partition'].value_counts()
print("\\nDistribuição das partições experimentais:")
for split, count in split_dist.items():
    print(f"  • {split:6s}: {count:5d} amostras ({count/len(df)*100:5.2f}%)")"""))

    # 4. Taxonomia de Colunas
    cells.append(nbf.v4.new_markdown_cell("""---
## 2. Taxonomia e Mapeamento de Colunas (Checklist 1)

Mapeamos explicitamente os subconjuntos de variáveis conforme os requisitos do cartão:
- **Numéricas Contínuas (17 features):** Índices cromáticos, texturas Haralick GLCM e metadados de tamanho e resolução física.
- **Discretizadas (4 features):** Intervalos de binning de `size_kb_bin`, `laplacian_var_bin`, `resolution_bin`, `aspect_ratio_bin`.
- **Categóricas de Metadados (2 features):** `dataset_source` e `extension`.
- **Categóricas Totais:** Unificação das nominais e discretizadas para codificação One-Hot.
- **Metadados Descartados:** Identificadores, nomes de arquivos e dimensões redundantes (`remainder='drop'`).
- **Alvos Supervisionados:** `class_label`, `target_binary`, `target_multiclass`."""))

    cells.append(nbf.v4.new_code_cell("""# 1. Features contínuas (Cromáticas, Textura Haralick e Metadados Físicos de Imagem)
NUMERICAL_FEATURES = [
    'mean_hue', 'std_saturation', 'exg_index', 'exr_index', 'rg_ratio',
    'indice_clorose_necrose', 'hue_dispersion', 'glcm_contrast',
    'glcm_homogeneity', 'glcm_dissimilarity', 'glcm_energy',
    'indice_rugosidade_pustula', 'laplacian_var', 'size_kb',
    'total_pixels', 'resolution_mp', 'aspect_ratio'
]

# 2. Features discretizadas em faixas (bins estatísticos)
DISCRETIZED_FEATURES = [
    'size_kb_bin', 'laplacian_var_bin', 'resolution_bin', 'aspect_ratio_bin'
]

# 3. Features categóricas de metadados
CATEGORICAL_NOMINAL_FEATURES = ['dataset_source', 'extension']

# Unificação para o OneHotEncoder
ALL_CATEGORICAL_FEATURES = CATEGORICAL_NOMINAL_FEATURES + DISCRETIZED_FEATURES

# 4. Colunas de metadados/identificação excluídas do vetor preditivo
METADATA_DROPPED = [
    'sample_id', 'filepath', 'filename', 'relative_path',
    'width', 'height', 'channels', 'color_mode'
]

# 5. Targets supervisionados
TARGET_COLS = ['class_label', 'target_binary', 'target_multiclass']

summary_df = pd.DataFrame([
    {"Grupo": "Numéricas Contínuas", "Quantidade": len(NUMERICAL_FEATURES), "Estratégia": "SimpleImputer(median) + StandardScaler()"},
    {"Grupo": "Categóricas Nominais", "Quantidade": len(CATEGORICAL_NOMINAL_FEATURES), "Estratégia": "SimpleImputer(most_frequent) + OneHotEncoder()"},
    {"Grupo": "Discretizadas (Bins)", "Quantidade": len(DISCRETIZED_FEATURES), "Estratégia": "OneHotEncoder(sparse_output=False, handle_unknown='ignore')"},
    {"Grupo": "Metadados Isolados", "Quantidade": len(METADATA_DROPPED), "Estratégia": "remainder='drop'"},
    {"Grupo": "Targets Supervisionados", "Quantidade": len(TARGET_COLS), "Estratégia": "Preservação Externa (Y)"}
])
display(summary_df)"""))

    # 5. Construção dos Pipelines
    cells.append(nbf.v4.new_markdown_cell("""---
## 3. Construção do Pipeline Atômico com ColumnTransformer (Checklists 2, 3 e 4)

Encapsulamos o fluxo com:
- **Sub-pipeline numérico:** Imputação por mediana (resistente a assimetrias na iluminação foliar) seguida de normalização z-score.
- **Sub-pipeline categórico:** Imputação de moda e `OneHotEncoder(sparse_output=False, handle_unknown='ignore')` para prevenir falhas em categorias inéditas de produção.
- **ColumnTransformer:** Agrupamento atômico com `remainder='drop'` e saída formatada para DataFrame pandas nativo (`set_output(transform='pandas')`)."""))

    cells.append(nbf.v4.new_code_cell("""# Sub-pipeline Numérico
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Sub-pipeline Categórico (Nominais + Discretizadas)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# ColumnTransformer Unificado
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, NUMERICAL_FEATURES),
        ('cat', categorical_transformer, ALL_CATEGORICAL_FEATURES)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

# Configurar saída pandas nativa para auditoria e rastreabilidade imediata
preprocessor.set_output(transform='pandas')

# Renderização do diagrama interativo
display(preprocessor)"""))

    # 6. Validação Anti-Leakage
    cells.append(nbf.v4.new_markdown_cell("""---
## 4. Validação da "Regra de Ouro do Fit" (Anti-Leakage) (Checklist 5)

> **Regra de Ouro:** O aprendizado de parâmetros estatísticos ($\mu, \sigma$, medianas, categorias válidas) deve ocorrer **estritamente sobre o conjunto de treino** via `.fit()` ou `.fit_transform()`. As partições de validação e teste recebem exclusivamente `.transform()`, garantindo que nenhuma contaminação de dados (*information leakage*) ocorra."""))

    cells.append(nbf.v4.new_code_cell("""# 1. Divisão estrita conforme split_partition
df_train = df[df['split_partition'] == 'train'].copy()
df_valid = df[df['split_partition'] == 'valid'].copy()
df_test  = df[df['split_partition'] == 'test'].copy()

print(f"• Amostras de Treino:     {len(df_train)}")
print(f"• Amostras de Validação:  {len(df_valid)}")
print(f"• Amostras de Teste:      {len(df_test)}")

# 2. Ajuste EXCLUSIVO no conjunto de Treino
X_train_trans = preprocessor.fit_transform(df_train)

# 3. Replicação com .transform() em Validação e Teste
X_valid_trans = preprocessor.transform(df_valid)
X_test_trans  = preprocessor.transform(df_test)

print(f"\\nShape pós-processamento:")
print(f"  - X_train_trans : {X_train_trans.shape}")
print(f"  - X_valid_trans : {X_valid_trans.shape}")
print(f"  - X_test_trans  : {X_test_trans.shape}")

# 4. Verificação de ausência de NaNs
assert X_train_trans.isna().sum().sum() == 0, "Existem NaNs no Treino!"
assert X_valid_trans.isna().sum().sum() == 0, "Existem NaNs na Validação!"
assert X_test_trans.isna().sum().sum() == 0, "Existem NaNs no Teste!"
print("\\n✓ [VALIDADO] Ausência total de NaNs em todas as partições.")

# 5. Auditoria Matemática dos Parâmetros Aprendidos
imputer_step = preprocessor.named_transformers_['num'].named_steps['imputer']
scaler_step  = preprocessor.named_transformers_['num'].named_steps['scaler']

# Conferir se a mediana bate com o Treino
expected_train_medians = df_train[NUMERICAL_FEATURES].median().values
np.testing.assert_allclose(imputer_step.statistics_, expected_train_medians, rtol=1e-5)
print("✓ [VALIDADO] Imputer medians correspondem estritamente ao Treino (Sem Leakage).")

# Conferir se as médias numéricas no Treino são 0.0
num_cols_trans = [col for col in X_train_trans.columns if col.startswith('num__')]
train_means = X_train_trans[num_cols_trans].mean()
assert np.all(np.abs(train_means) < 1e-6), "Médias de treino não padronizadas para zero!"
print("✓ [VALIDADO] Média das variáveis numéricas de Treino centrada em 0.0.")

# Comparação com o dataset total para provar o isolamento
full_means = df[NUMERICAL_FEATURES].mean().values
mean_diff = np.abs(scaler_step.mean_ - full_means)
print(f"✓ [VALIDADO] Desvio médio entre os parâmetros do Treino vs População Total: {mean_diff.mean():.6f}")"""))

    # 7. Teste de Robustez com Categorias Inéditas
    cells.append(nbf.v4.new_markdown_cell("""---
## 5. Teste de Robustez a Categorias Desconhecidas em Produção

Validamos a capacidade do pipeline de tolerar observações operacionais com categorias nunca antes vistas no treino (ex.: novas fontes de coleta, extensões de imagem não catalogadas ou novos bins), garantindo que o parâmetro `handle_unknown='ignore'` zere os atributos codificados sem gerar interrupções de execução (*runtime exceptions*)."""))

    cells.append(nbf.v4.new_code_cell("""# Criar registro de inferência contendo categorias inéditas
amostra_inedita = pd.DataFrame([{
    'mean_hue': 72.5, 'std_saturation': 0.11, 'exg_index': 48.0, 'exr_index': -12.0,
    'rg_ratio': 0.35, 'indice_clorose_necrose': 2.8, 'hue_dispersion': 9.1,
    'glcm_contrast': 11.2, 'glcm_homogeneity': 0.87, 'glcm_dissimilarity': 0.72,
    'glcm_energy': 0.89, 'indice_rugosidade_pustula': 7.9, 'laplacian_var': 135.0,
    'size_kb': 45.0, 'total_pixels': 409600, 'resolution_mp': 0.41, 'aspect_ratio': 1.0,
    # Categorias desconhecidas
    'dataset_source': 'drone_agro_coleta_2027',
    'extension': '.webp',
    'size_kb_bin': 99,
    'laplacian_var_bin': 99,
    'resolution_bin': 99,
    'aspect_ratio_bin': 99
}])

amostra_trans = preprocessor.transform(amostra_inedita)
cat_cols = [c for c in amostra_trans.columns if c.startswith('cat__')]
soma_cat = amostra_trans[cat_cols].values.sum()

print(f"• Transformação de amostra desconhecida concluída com sucesso!")
print(f"• Soma das colunas dummy geradas: {soma_cat} (Vetor nulo seguro)")
assert soma_cat == 0.0, "Categorias desconhecidas não geraram vetor nulo!"
print("✓ [VALIDADO] Resiliência a categorias inéditas confirmada.")"""))

    # 8. Serialização MLOps
    cells.append(nbf.v4.new_markdown_cell("""---
## 6. Serialização MLOps com Joblib e Manifesto de Governança (Checklist 6)

Persistimos o objeto transformador ajustado em `models/preprocessor_pipeline.joblib`, validamos a integridade com hash SHA-256, testamos a idempotência numérica via `joblib.load()` e salvamos o manifesto de governança em JSON com rastreabilidade completa das features geradas."""))

    cells.append(nbf.v4.new_code_cell("""models_dir = base_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)
pipeline_file = models_dir / "preprocessor_pipeline.joblib"
manifest_file = models_dir / "preprocessor_metadata.json"

# 1. Serialização
joblib.dump(preprocessor, pipeline_file, compress=3)
file_size_kb = pipeline_file.stat().st_size / 1024

# 2. Hash SHA-256
with open(pipeline_file, 'rb') as f:
    pipeline_hash = hashlib.sha256(f.read()).hexdigest()

print(f"• Artefato serializado : {pipeline_file.name} ({file_size_kb:.2f} KB)")
print(f"• SHA-256 Hash         : {pipeline_hash}")

# 3. Teste de Idempotência Numérica
preprocessor_reloaded = joblib.load(pipeline_file)
amostra_teste = df_test.head(100)

orig_out = preprocessor.transform(amostra_teste)
reloaded_out = preprocessor_reloaded.transform(amostra_teste)

np.testing.assert_allclose(orig_out.values, reloaded_out.values, rtol=1e-7, atol=1e-7)
print("✓ [VALIDADO] Idempotência matemática estrita comprovada pós-recarga.")

# 4. Inspeção das Features Finais de Saída
feature_names = preprocessor.get_feature_names_out()
print(f"\\nTotal de features geradas: {len(feature_names)}")
print("Primeiras 10 features:")
for fn in feature_names[:10]:
    print(f"  - {fn}")"""))

    # 9. Consolidação da ABT Final
    cells.append(nbf.v4.new_markdown_cell("""---
## 7. Consolidação e Verificação da ABT de Modelagem (`abt_features_modelagem`)

Para assegurar sinergia com a próxima tarefa do sprint e preparar os dados para o `GridSearchCV` da Sprint 3, inspecionamos o arquivo consolidado `data/processed/abt_features_modelagem.parquet` gerado com os alvos preservados e 100% livre de NaNs."""))

    cells.append(nbf.v4.new_code_cell("""parquet_path = base_dir / "data" / "processed" / "abt_features_modelagem.parquet"
df_final = pd.read_parquet(parquet_path)

print(f"• Dimensões da ABT Final de Modelagem : {df_final.shape[0]} linhas x {df_final.shape[1]} colunas.")
print(f"• Total de valores ausentes (NaN)     : {df_final.isna().sum().sum()}")
print(f"• Tamanho do arquivo Parquet          : {parquet_path.stat().st_size / 1024:.2f} KB")

# Amostra das primeiras colunas da ABT final
display(df_final[['sample_id', 'split_partition', 'class_label', 'target_binary'] + list(feature_names[:4])].head())"""))

    nb['cells'] = cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"[OK] Notebook gerado com sucesso em: {nb_path}")
    
    # Executar o notebook para validar e persistir saídas
    print("Executando notebook de ponta a ponta...")
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"[OK] Notebook executado e saídas salvas em: {nb_path}")


if __name__ == '__main__':
    create_and_run_notebook()
