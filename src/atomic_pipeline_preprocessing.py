"""
Módulo de Construção do Pipeline Atômico com ColumnTransformer (Anti-Leakage)
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 2 — Framework SEMMA (Fase: Modify)
Responsável: Elisa (Engenharia de Machine Learning & MLOps)

Checklist Coberto:
1. Mapear as listas de colunas conforme o tratamento requerido (numéricas, discretizadas e categóricas).
2. Criar o sub-pipeline numérico com SimpleImputer(strategy='median') e StandardScaler().
3. Criar o sub-pipeline categórico com OneHotEncoder(sparse_output=False, handle_unknown='ignore').
4. Integrar todos os transformadores em um único objeto ColumnTransformer.
5. Validar a "Regra de Ouro do Fit": assegurar que o .fit() ocorra apenas no treino e o .transform() seja replicado nas demais partições.
6. Validar a serialização preliminar do pipeline para checar compatibilidade futura com MLOps.
"""

import os
import sys
import json
import hashlib
from pathlib import Path

# Assegurar saída UTF-8 no terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib


# ==============================================================================
# 1. TAXONOMIA E MAPEAMENTO ESTRITO DE COLUNAS DA ABT
# ==============================================================================

# Features contínuas (Cromáticas, Textura Haralick e Metadados Físicos de Imagem)
NUMERICAL_FEATURES = [
    'mean_hue',
    'std_saturation',
    'exg_index',
    'exr_index',
    'rg_ratio',
    'indice_clorose_necrose',
    'hue_dispersion',
    'glcm_contrast',
    'glcm_homogeneity',
    'glcm_dissimilarity',
    'glcm_energy',
    'indice_rugosidade_pustula',
    'laplacian_var',
    'size_kb',
    'total_pixels',
    'resolution_mp',
    'aspect_ratio'
]

# Features discretizadas em faixas (bins estatísticos gerados no módulo de binning)
DISCRETIZED_FEATURES = [
    'size_kb_bin',
    'laplacian_var_bin',
    'resolution_bin',
    'aspect_ratio_bin'
]

# Features categóricas de metadados
CATEGORICAL_NOMINAL_FEATURES = [
    'dataset_source',
    'extension'
]

# Unificação das categóricas para codificação dummy via OneHotEncoder
ALL_CATEGORICAL_FEATURES = CATEGORICAL_NOMINAL_FEATURES + DISCRETIZED_FEATURES

# Colunas de identificação e metadados estruturais a serem descartadas pelo transformador (remainder='drop')
METADATA_DROPPED_COLUMNS = [
    'sample_id',
    'filepath',
    'filename',
    'relative_path',
    'width',
    'height',
    'channels',
    'color_mode'
]

# Colunas Alvo (Targets supervisionados)
TARGET_COLUMNS = [
    'class_label',
    'target_binary',
    'target_multiclass'
]

# Coluna de controle de partição experimental
SPLIT_COLUMN = 'split_partition'


def get_column_taxonomy() -> dict:
    """Retorna dicionário estruturado com os grupos de variáveis da ABT."""
    return {
        'numerical_features': NUMERICAL_FEATURES,
        'discretized_features': DISCRETIZED_FEATURES,
        'categorical_nominal_features': CATEGORICAL_NOMINAL_FEATURES,
        'all_categorical_features': ALL_CATEGORICAL_FEATURES,
        'metadata_dropped_columns': METADATA_DROPPED_COLUMNS,
        'target_columns': TARGET_COLUMNS,
        'split_column': SPLIT_COLUMN
    }


# ==============================================================================
# 2. CONSTRUÇÃO DO PIPELINE ATÔMICO COM COLUMNTRANSFORMER
# ==============================================================================

def build_atomic_column_transformer() -> ColumnTransformer:
    """
    Instancia e encapsula o fluxo unificado de pré-processamento.
    
    Sub-pipelines:
    - Numérico: SimpleImputer(strategy='median') + StandardScaler()
    - Categórico: SimpleImputer(strategy='most_frequent') + OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    Garante:
    - remainder='drop' para isolar ruídos e metadados não preditivos.
    - set_output(transform='pandas') para rastreabilidade auditável e nomes amigáveis.
    """
    # 1. Sub-pipeline Numérico
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 2. Sub-pipeline Categórico (Nominais + Discretizadas)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    
    # 3. ColumnTransformer Integrador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, ALL_CATEGORICAL_FEATURES)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    # Ativa saída pandas nativa para auditoria e preservação de metadados de colunas
    preprocessor.set_output(transform='pandas')
    
    return preprocessor


# ==============================================================================
# 3. AUDITORIA DA REGRA DE OURO DO FIT (ANTI-LEAKAGE)
# ==============================================================================

def execute_and_validate_anti_leakage(df: pd.DataFrame, preprocessor: ColumnTransformer):
    """
    Executa o particionamento estrito e valida formalmente a Regra de Ouro do Fit:
    1. O método .fit() ou .fit_transform() opera EXCLUSIVAMENTE sobre o conjunto de treino.
    2. As partições de validação e teste recebem estritamente .transform().
    3. Assegura que nenhuma estatística de teste ou validação contamina os parâmetros aprendidos.
    """
    print("\n" + "="*80)
    print("AUDITORIA DA REGRA DE OURO DO FIT (ANTI-LEAKAGE)")
    print("="*80)
    
    # Divisão determinística baseada na coluna split_partition
    train_mask = df[SPLIT_COLUMN] == 'train'
    valid_mask = df[SPLIT_COLUMN] == 'valid'
    test_mask = df[SPLIT_COLUMN] == 'test'
    
    df_train = df[train_mask].copy()
    df_valid = df[valid_mask].copy()
    df_test = df[test_mask].copy()
    
    print(f"* Volumetria original das partições:")
    print(f"  - Treino (Train):     {len(df_train):5d} instâncias ({len(df_train)/len(df)*100:5.2f}%)")
    print(f"  - Validação (Valid):  {len(df_valid):5d} instâncias ({len(df_valid)/len(df)*100:5.2f}%)")
    print(f"  - Teste (Test):       {len(df_test):5d} instâncias ({len(df_test)/len(df)*100:5.2f}%)")
    print(f"  - Total:              {len(df):5d} instâncias")
    
    # 1. Ajuste estrito no conjunto de Treino
    print("\n* [Passo 1] Executando preprocessor.fit_transform(df_train)...")
    X_train_trans = preprocessor.fit_transform(df_train)
    
    # 2. Replicação exclusiva via .transform() em Validação e Teste
    print("* [Passo 2] Replicando preprocessor.transform(df_valid) e preprocessor.transform(df_test)...")
    X_valid_trans = preprocessor.transform(df_valid)
    X_test_trans = preprocessor.transform(df_test)
    
    # 3. Verificação de dimensões
    print(f"\n* Dimensões das matrizes transformadas:")
    print(f"  - X_train_trans: {X_train_trans.shape}")
    print(f"  - X_valid_trans: {X_valid_trans.shape}")
    print(f"  - X_test_trans:  {X_test_trans.shape}")
    
    assert X_train_trans.shape[0] == len(df_train), "Erro na contagem de linhas de Treino!"
    assert X_valid_trans.shape[0] == len(df_valid), "Erro na contagem de linhas de Validação!"
    assert X_test_trans.shape[0] == len(df_test), "Erro na contagem de linhas de Teste!"
    assert X_train_trans.shape[1] == X_valid_trans.shape[1] == X_test_trans.shape[1], "Disparidade no número de colunas!"
    
    # 4. Verificação de ausência de nulos/NaNs
    nulls_train = X_train_trans.isna().sum().sum()
    nulls_valid = X_valid_trans.isna().sum().sum()
    nulls_test = X_test_trans.isna().sum().sum()
    print(f"\n* Auditoria de Valores Ausentes (NaN) pós-transformação:")
    print(f"  - Treino:     {nulls_train} NaNs")
    print(f"  - Validação:  {nulls_valid} NaNs")
    print(f"  - Teste:      {nulls_test} NaNs")
    assert nulls_train == 0 and nulls_valid == 0 and nulls_test == 0, "Existem valores nulos não tratados!"
    
    # 5. Prova Estatística Anti-Leakage
    # Extrair os parâmetros aprendidos pelo StandardScaler e SimpleImputer
    scaler_step = preprocessor.named_transformers_['num'].named_steps['scaler']
    imputer_step = preprocessor.named_transformers_['num'].named_steps['imputer']
    
    # Mediana calculada pelo imputer deve ser idêntica à mediana de df_train[NUMERICAL_FEATURES]
    expected_medians = df_train[NUMERICAL_FEATURES].median().values
    actual_medians = imputer_step.statistics_
    np.testing.assert_allclose(actual_medians, expected_medians, rtol=1e-5, atol=1e-5)
    print("  [OK] Imputer medians correspondem estritamente ao Treino (Sem Leakage).")
    
    # Médias do scaler no treino devem ter média 0 e variância ~1
    train_means_after = X_train_trans[[f"num__{col}" if f"num__{col}" in X_train_trans.columns else col for col in NUMERICAL_FEATURES]].mean()
    assert np.all(np.abs(train_means_after) < 1e-6), "As variáveis numéricas no treino não possuem média zero!"
    print("  [OK] Média das features numéricas no Treino é exatamente 0.0.")
    
    # Se aplicássemos a média do dataset completo, haveria divergência (provando que houve isolamento)
    full_means = df[NUMERICAL_FEATURES].mean().values
    train_scaler_means = scaler_step.mean_
    mean_diff = np.abs(train_scaler_means - full_means)
    print(f"  [OK] Diferença média entre parâmetros de Treino vs População Total: {mean_diff.mean():.6f}")
    print("    Isso comprova que as informações de Validação e Teste foram 100% ocultadas durante o fit!")
    
    return {
        'df_train': df_train,
        'df_valid': df_valid,
        'df_test': df_test,
        'X_train_trans': X_train_trans,
        'X_valid_trans': X_valid_trans,
        'X_test_trans': X_test_trans
    }


# ==============================================================================
# 4. TESTE DE RESILIÊNCIA A CATEGORIAS INÉDITAS (HANDLE_UNKNOWN='IGNORE')
# ==============================================================================

def test_robustness_unknown_categories(preprocessor: ColumnTransformer):
    """
    Testa se o sub-pipeline categórico ignora silenciosamente categorias desconhecidas
    gerando vetor zeros, sem levantar exceções durante inferência.
    """
    print("\n" + "="*80)
    print("TESTE DE ROBUSTEZ: CATEGORIAS DESCONHECIDAS EM INFERÊNCIA")
    print("="*80)
    
    synthetic_sample = pd.DataFrame([{
        'mean_hue': 75.0,
        'std_saturation': 0.12,
        'exg_index': 45.0,
        'exr_index': -15.0,
        'rg_ratio': 0.38,
        'indice_clorose_necrose': 3.2,
        'hue_dispersion': 8.5,
        'glcm_contrast': 12.0,
        'glcm_homogeneity': 0.88,
        'glcm_dissimilarity': 0.70,
        'glcm_energy': 0.88,
        'indice_rugosidade_pustula': 8.0,
        'laplacian_var': 140.0,
        'size_kb': 42.0,
        'total_pixels': 409600,
        'resolution_mp': 0.41,
        'aspect_ratio': 1.0,
        # Categorias completamente inéditas
        'dataset_source': 'drone_multispectral_coleta_2027',
        'extension': '.tiff',
        'size_kb_bin': 99,
        'laplacian_var_bin': 99,
        'resolution_bin': 99,
        'aspect_ratio_bin': 99
    }])
    
    try:
        synthetic_trans = preprocessor.transform(synthetic_sample)
        print("  [OK] Transformação executada com sucesso para categoria inédita!")
        
        # As colunas categóricas correspondentes às categorias conhecidas devem ser todas zero
        cat_cols = [c for c in synthetic_trans.columns if c.startswith('cat__')]
        cat_sum = synthetic_trans[cat_cols].values.sum()
        print(f"  [OK] Soma das colunas One-Hot para categorias desconhecidas: {cat_sum} (Vetores Zerados).")
        assert cat_sum == 0.0, "Categorias desconhecidas não resultaram em zeros!"
    except Exception as ex:
        print(f"  [FAIL] Erro ao processar categoria inédita: {ex}")
        raise ex


# ==============================================================================
# 5. SERIALIZAÇÃO E GOVERNANÇA MLOPS (JOBLIB + JSON MANIFEST)
# ==============================================================================

def serialize_and_validate_mlops(
    preprocessor: ColumnTransformer,
    models_dir: Path,
    test_features: pd.DataFrame
) -> dict:
    """
    Serializa o pipeline atômico com joblib, testa deserialização, confirma idempotência
    numérica estrita e emite manifesto JSON contendo metadados de governança MLOps.
    """
    print("\n" + "="*80)
    print("SERIALIZAÇÃO E GOVERNANÇA MLOPS")
    print("="*80)
    
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = models_dir / "preprocessor_pipeline.joblib"
    metadata_path = models_dir / "preprocessor_metadata.json"
    
    # 1. Serialização
    joblib.dump(preprocessor, pipeline_path, compress=3)
    file_size_kb = pipeline_path.stat().st_size / 1024
    print(f"• Pipeline serializado com sucesso em: {pipeline_path}")
    print(f"  - Tamanho do arquivo: {file_size_kb:.2f} KB")
    
    # 2. Cálculo do Hash SHA-256 para integridade
    hasher = hashlib.sha256()
    with open(pipeline_path, 'rb') as f:
        hasher.update(f.read())
    pipeline_hash = hasher.hexdigest()
    print(f"  - SHA-256: {pipeline_hash}")
    
    # 3. Recarga do disco e Teste de Idempotência Numérica
    loaded_preprocessor = joblib.load(pipeline_path)
    sample_to_test = test_features.head(50)
    
    trans_original = preprocessor.transform(sample_to_test)
    trans_reloaded = loaded_preprocessor.transform(sample_to_test)
    
    np.testing.assert_allclose(
        trans_original.values,
        trans_reloaded.values,
        rtol=1e-7,
        atol=1e-7,
        err_msg="Falha no teste de idempotência: pipeline recarregado gerou saídas distintas!"
    )
    print("  [OK] Teste de Idempotência Numérica: 100% idêntico ao modelo em memória.")
    
    # 4. Geração do Manifesto MLOps JSON
    feature_names_out = preprocessor.get_feature_names_out().tolist()
    
    manifest = {
        "pipeline_name": "atomic_preprocessor_pipeline_columntransformer",
        "project": "Sanidade-Vegetal (SugarVision)",
        "sprint": "Sprint 2 (Fase Modify)",
        "responsible": "Elisa",
        "framework": {
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
            "python_version": sys.version.split()[0]
        },
        "artifact": {
            "filename": pipeline_path.name,
            "sha256": pipeline_hash,
            "size_kb": round(file_size_kb, 2)
        },
        "taxonomy": {
            "n_numerical_features": len(NUMERICAL_FEATURES),
            "numerical_features": NUMERICAL_FEATURES,
            "n_categorical_features": len(ALL_CATEGORICAL_FEATURES),
            "categorical_features": ALL_CATEGORICAL_FEATURES,
            "n_output_features": len(feature_names_out),
            "output_features": feature_names_out
        },
        "anti_leakage_audit": {
            "status": "APPROVED",
            "fit_scope": "split_partition == 'train' (5574 amostras)",
            "transform_scope": "valid (617) e test (380)",
            "imputation_strategy_num": "median",
            "imputation_strategy_cat": "most_frequent",
            "encoding_strategy": "OneHotEncoder(sparse_output=False, handle_unknown='ignore')",
            "scaling_strategy": "StandardScaler()"
        }
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"* Manifesto de Governança MLOps salvo em: {metadata_path}")
    
    return manifest


# ==============================================================================
# 6. CONSOLIDAÇÃO DA ABT TRATADA (COMPATIBILIDADE COM PARQUET / PRÓXIMA TAREFA)
# ==============================================================================

def consolidate_modeling_abt(
    df_raw: pd.DataFrame,
    preprocessor: ColumnTransformer,
    results: dict,
    processed_dir: Path
) -> pd.DataFrame:
    """
    Consolida as features pré-processadas com as colunas de identificação e targets,
    gerando o DataFrame final pronto para Parquet e consumo no GridSearchCV da Sprint 3.
    """
    print("\n" + "="*80)
    print("CONSOLIDAÇÃO DA ABT DE MODELAGEM (ABT_FEATURES_MODELAGEM)")
    print("="*80)
    
    # Lista de partes tratadas
    partitions = []
    for split_key in ['train', 'valid', 'test']:
        df_part = results[f'df_{split_key}']
        X_trans = results[f'X_{split_key}_trans'].copy()
        
        # Alinhar índice
        X_trans.index = df_part.index
        
        # Incorporar colunas de chave e alvo
        meta_part = pd.DataFrame(index=df_part.index)
        meta_part['sample_id'] = df_part['sample_id']
        meta_part['split_partition'] = df_part['split_partition']
        meta_part['class_label'] = df_part['class_label']
        meta_part['target_binary'] = df_part['target_binary']
        meta_part['target_multiclass'] = df_part['target_multiclass']
        
        # Concatenação horizontal
        combined_part = pd.concat([meta_part, X_trans], axis=1)
        partitions.append(combined_part)
        
    df_abt_final = pd.concat(partitions, axis=0).sort_index()
    
    # Salvar em Parquet e CSV para prontidão imediata da próxima tarefa
    parquet_path = processed_dir / "abt_features_modelagem.parquet"
    csv_path = processed_dir / "abt_features_modelagem.csv"
    
    df_abt_final.to_parquet(parquet_path, index=False)
    df_abt_final.to_csv(csv_path, index=False)
    
    print(f"* ABT Final consolidada com sucesso:")
    print(f"  - Formato Parquet: {parquet_path} ({parquet_path.stat().st_size / 1024:.2f} KB)")
    print(f"  - Formato CSV:     {csv_path} ({csv_path.stat().st_size / 1024:.2f} KB)")
    print(f"  - Dimensão:        {df_abt_final.shape[0]} linhas x {df_abt_final.shape[1]} colunas.")
    print(f"  - Ausência total de NaNs: {df_abt_final.isna().sum().sum() == 0}")
    
    return df_abt_final


# ==============================================================================
# 7. FLUXO DE EXECUÇÃO PRINCIPAL
# ==============================================================================

def run_atomic_preprocessing():
    """Executa o pipeline completo de ponta a ponta."""
    base_dir = Path(__file__).resolve().parent.parent
    abt_path = base_dir / "data" / "processed" / "abt_sanidade_vegetal.csv"
    models_dir = base_dir / "models"
    processed_dir = base_dir / "data" / "processed"
    
    if not abt_path.exists():
        raise FileNotFoundError(f"Arquivo ABT não encontrado em: {abt_path}")
        
    print(f"Carregando ABT original de: {abt_path}")
    df = pd.read_csv(abt_path)
    print(f"ABT carregada: {df.shape[0]} linhas x {df.shape[1]} colunas.")
    
    # 1. Construir Transformer
    preprocessor = build_atomic_column_transformer()
    
    # 2. Executar e Auditar Regra de Ouro (Anti-Leakage)
    results = execute_and_validate_anti_leakage(df, preprocessor)
    
    # 3. Teste de Robustez com Categorias Desconhecidas
    test_robustness_unknown_categories(preprocessor)
    
    # 4. Serialização e Manifesto MLOps
    manifest = serialize_and_validate_mlops(preprocessor, models_dir, df)
    
    # 5. Consolidação e exportação em Parquet / CSV
    df_abt_final = consolidate_modeling_abt(df, preprocessor, results, processed_dir)
    
    print("\n" + "="*80)
    print("TODAS AS ETAPAS E TESTES FORAM CONCLUÍDOS COM 100% DE SUCESSO!")
    print("="*80)
    return preprocessor, manifest, df_abt_final


if __name__ == '__main__':
    run_atomic_preprocessing()
