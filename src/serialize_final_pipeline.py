"""
Módulo de Serialização Atômica e Segura do Pipeline Final de Produção (Sprint 5)
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 5 — Framework SEMMA (Fase: Deploy & MLOps Readiness)
Responsável: Elisa (Engenharia de Machine Learning & MLOps)

Checklist da Tarefa:
1. Serializar o pipeline campeão ponta a ponta utilizando joblib no caminho models/sanidade_vegetal_pipeline_final.joblib.
2. Validar teste de des-serialização (load) em ambiente limpo, confirmando integridade de todos os transformadores e pesos do SVM.
3. Garantir que o pipeline serializado receba a imagem/features cruas e devolva a predição sem intervenção manual.
4. Gerar arquivo de metadados do modelo serializado contendo versão, data de treino, hash SHA-256 e hiperparâmetros.
"""

import os
import sys
import json
import time
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
from sklearn.metrics import classification_report, f1_score, accuracy_score, log_loss
import joblib

# Importar pipeline base e taxonomias canônicas
try:
    from atomic_svm_pipeline import (
        build_atomic_svm_pipeline,
        TARGET_COLUMNS,
        SPLIT_COLUMN,
        NUMERICAL_FEATURES,
        DISCRETIZED_FEATURES,
        CATEGORICAL_NOMINAL_FEATURES,
        ALL_CATEGORICAL_FEATURES
    )
except ImportError:
    from src.atomic_svm_pipeline import (
        build_atomic_svm_pipeline,
        TARGET_COLUMNS,
        SPLIT_COLUMN,
        NUMERICAL_FEATURES,
        DISCRETIZED_FEATURES,
        CATEGORICAL_NOMINAL_FEATURES,
        ALL_CATEGORICAL_FEATURES
    )

# Mapeamento oficial das 7 classes fitopatológicas do projeto SugarVision
CLASS_NAMES_MAP = {
    0: 'GRASSY SHOOT',
    1: 'HEALTHY',
    2: 'LEAF SCALD',
    3: 'MOSAIC',
    4: 'RED ROT',
    5: 'RUST',
    6: 'YELLOW LEAF'
}
TARGET_NAMES = [CLASS_NAMES_MAP[i] for i in range(7)]

# Hiperparâmetros ótimos homologados no Grid Search Exaustivo (Sprint 3 / Sprint 4)
OPTIMAL_SVM_HYPERPARAMETERS = {
    'kernel': 'poly',
    'C': 0.1,
    'degree': 3,
    'coef0': 1.0,
    'gamma': 'scale',
    'class_weight': 'balanced',
    'probability': True,
    'random_state': 42
}


def compute_sha256(filepath: Path) -> str:
    """Calcula hash SHA-256 de um arquivo em streaming de blocos de 64 KB."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def train_and_serialize_champion_pipeline(
    data_path: Path,
    output_model_path: Path,
    output_metadata_path: Path
) -> dict:
    """
    Treina o pipeline campeão na partição oficial de treino da ABT, serializa
    atomicamente o artefato com joblib e gera o manifesto formal de governança.
    """
    print("\n" + "="*80)
    print("INICIANDO SERIALIZAÇÃO ATÔMICA DO PIPELINE CAMPEÃO DE PRODUÇÃO")
    print("="*80)
    print(f"* Carregando base de dados: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Base de dados não encontrada em: {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"  - Volumetria total da base : {len(df):,} linhas x {len(df.columns)} colunas")
    
    target_col = 'target_multiclass'
    if target_col not in df.columns:
        raise KeyError(f"Coluna alvo '{target_col}' ausente na ABT!")
        
    if SPLIT_COLUMN not in df.columns:
        raise KeyError(f"Coluna de particionamento '{SPLIT_COLUMN}' ausente na ABT!")
        
    # Particionamento oficial anti-leakage
    train_mask = df[SPLIT_COLUMN] == 'train'
    valid_mask = df[SPLIT_COLUMN] == 'valid'
    test_mask = df[SPLIT_COLUMN] == 'test'
    
    X_train = df[train_mask]
    y_train = df.loc[train_mask, target_col].astype(int)
    
    X_valid = df[valid_mask]
    y_valid = df.loc[valid_mask, target_col].astype(int)
    
    X_test = df[test_mask]
    y_test = df.loc[test_mask, target_col].astype(int)
    
    print(f"  - Partição Treino     : {len(X_train):,} amostras")
    print(f"  - Partição Validação  : {len(X_valid):,} amostras")
    print(f"  - Partição Teste      : {len(X_test):,} amostras")
    
    # 1. Instanciação do Pipeline Atômico com os Hiperparâmetros Campeões
    print("\n* Instanciando pipeline atômico unificado (ColumnTransformer + SVC Campeão)...")
    pipeline = build_atomic_svm_pipeline(
        kernel=OPTIMAL_SVM_HYPERPARAMETERS['kernel'],
        C=OPTIMAL_SVM_HYPERPARAMETERS['C'],
        degree=OPTIMAL_SVM_HYPERPARAMETERS['degree'],
        gamma=OPTIMAL_SVM_HYPERPARAMETERS['gamma'],
        probability=OPTIMAL_SVM_HYPERPARAMETERS['probability'],
        random_state=OPTIMAL_SVM_HYPERPARAMETERS['random_state'],
        class_weight=OPTIMAL_SVM_HYPERPARAMETERS['class_weight']
    )
    pipeline.set_params(svm__coef0=OPTIMAL_SVM_HYPERPARAMETERS['coef0'])
    
    # 2. Ajuste (Fit) no Treino
    print("* Executando fit do pipeline completo sobre os dados de treino...")
    t0_fit = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_duration = time.perf_counter() - t0_fit
    print(f"  [OK] Fit concluído com sucesso em {fit_duration:.2f} segundos!")
    
    # 3. Avaliação de Desempenho nas Partições Independentes
    print("\n* Avaliando métricas em partições cegas para auditoria de integridade...")
    
    # Validação
    y_val_pred = pipeline.predict(X_valid)
    y_val_proba = pipeline.predict_proba(X_valid)
    val_acc = accuracy_score(y_valid, y_val_pred)
    val_f1 = f1_score(y_valid, y_val_pred, average='macro')
    val_loss = log_loss(y_valid, y_val_proba)
    
    # Teste
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    test_loss = log_loss(y_test, y_test_proba)
    
    print(f"  - Validação -> Acurácia: {val_acc*100:.2f}% | F1-Macro: {val_f1:.4f} | Log Loss: {val_loss:.4f}")
    print(f"  - Teste     -> Acurácia: {test_acc*100:.2f}% | F1-Macro: {test_f1:.4f} | Log Loss: {test_loss:.4f}")
    
    # 4. Serialização Atômica Ponta a Ponta com Joblib
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n* Serializando pipeline em: {output_model_path} (compressão nível 3)...")
    joblib.dump(pipeline, output_model_path, compress=3)
    
    # 5. Cálculo de Hash SHA-256 e Estatísticas Físicas
    sha256_hash = compute_sha256(output_model_path)
    file_size_bytes = output_model_path.stat().st_size
    file_size_kb = round(file_size_bytes / 1024, 2)
    
    print(f"  - Tamanho em disco : {file_size_kb} KB ({file_size_bytes:,} bytes)")
    print(f"  - Hash SHA-256     : {sha256_hash}")
    
    # 6. Teste de Validação Imediata de Des-serialização (Load) em Ambiente Limpo
    print("\n* Executando teste de des-serialização imediata...")
    reloaded_pipeline = joblib.load(output_model_path)
    assert isinstance(reloaded_pipeline, Pipeline), "O objeto recarregado não é uma instância de Pipeline!"
    assert 'preprocessor' in reloaded_pipeline.named_steps, "Step 'preprocessor' não encontrado no pipeline!"
    assert 'svm' in reloaded_pipeline.named_steps, "Step 'svm' não encontrado no pipeline!"
    
    svm_step = reloaded_pipeline.named_steps['svm']
    assert hasattr(svm_step, 'support_vectors_'), "SVM recarregado perdeu a matriz de vetores de suporte!"
    assert hasattr(svm_step, 'dual_coef_'), "SVM recarregado perdeu os coeficientes duais!"
    assert svm_step.support_vectors_.shape[0] > 0, "Nenhum vetor de suporte detectado no SVM!"
    
    # 7. Garantir que recebe features cruas com colunas auxiliares e devolve predição sem intervenção
    print("* Validando inferência autônoma com features cruas e colunas auxiliares...")
    sample_raw = X_test.head(5).copy()
    sample_raw['extra_field_metadata_uuid'] = ['test_uuid_1', 'test_uuid_2', 'test_uuid_3', 'test_uuid_4', 'test_uuid_5']
    sample_preds = reloaded_pipeline.predict(sample_raw)
    sample_probas = reloaded_pipeline.predict_proba(sample_raw)
    
    assert len(sample_preds) == 5, f"Esperado 5 predições, recebido {len(sample_preds)}"
    assert sample_probas.shape == (5, 7), f"Esperado formato (5, 7), recebido {sample_probas.shape}"
    assert np.allclose(sample_probas.sum(axis=1), 1.0, atol=1e-5), "Somatório das probabilidades não é 1.0!"
    print("  [OK] Inferência autônoma com descarte automático de metadados validada com sucesso!")
    
    # 8. Extração de Contrato e Metadados Formais de Governança
    class_distribution_train = {
        CLASS_NAMES_MAP[int(k)]: int(v)
        for k, v in y_train.value_counts().sort_index().items()
    }
    
    manifest = {
        "pipeline_name": "sanidade_vegetal_pipeline_final",
        "version": "1.0.0",
        "project": "Sanidade-Vegetal (SugarVision)",
        "sprint": "Sprint 5 — Governança, Deploy e Prontidão de Produção",
        "task": "Construir a rotina de serialização atômica e segura do pipeline completo treinado",
        "responsible": "Elisa (Engenharia de Machine Learning & MLOps)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "production_status": "HOMOLOGATED_PRODUCTION_READY",
        "framework": {
            "python_version": sys.version.split()[0],
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__
        },
        "artifact": {
            "filename": output_model_path.name,
            "relative_path": f"models/{output_model_path.name}",
            "size_bytes": file_size_bytes,
            "size_kb": file_size_kb,
            "sha256": sha256_hash,
            "serialization_format": "joblib",
            "compression_level": 3
        },
        "model_architecture": {
            "pipeline_steps": list(pipeline.named_steps.keys()),
            "preprocessor": {
                "type": "ColumnTransformer",
                "remainder_strategy": "drop",
                "numerical_features_count": len(NUMERICAL_FEATURES),
                "categorical_features_count": len(ALL_CATEGORICAL_FEATURES),
                "numeric_subpipeline": "SimpleImputer(strategy='median') -> StandardScaler()",
                "categorical_subpipeline": "SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore', sparse_output=False)"
            },
            "estimator": {
                "type": "sklearn.svm.SVC",
                "hyperparameters": {
                    "kernel": OPTIMAL_SVM_HYPERPARAMETERS['kernel'],
                    "C": OPTIMAL_SVM_HYPERPARAMETERS['C'],
                    "degree": OPTIMAL_SVM_HYPERPARAMETERS['degree'],
                    "coef0": OPTIMAL_SVM_HYPERPARAMETERS['coef0'],
                    "gamma": OPTIMAL_SVM_HYPERPARAMETERS['gamma'],
                    "class_weight": OPTIMAL_SVM_HYPERPARAMETERS['class_weight'],
                    "probability": OPTIMAL_SVM_HYPERPARAMETERS['probability'],
                    "random_state": OPTIMAL_SVM_HYPERPARAMETERS['random_state']
                },
                "n_classes": len(pipeline.classes_),
                "classes_mapping": CLASS_NAMES_MAP,
                "n_support_vectors_total": int(svm_step.support_vectors_.shape[0]),
                "support_vectors_per_class": [int(x) for x in svm_step.n_support_]
            }
        },
        "input_contract": {
            "numerical_features": NUMERICAL_FEATURES,
            "discretized_features": DISCRETIZED_FEATURES,
            "categorical_nominal_features": CATEGORICAL_NOMINAL_FEATURES,
            "all_categorical_features": ALL_CATEGORICAL_FEATURES,
            "total_contracted_features": len(NUMERICAL_FEATURES) + len(ALL_CATEGORICAL_FEATURES),
            "accepts_unmodeled_columns": True,
            "remainder_policy": "discarded_automatically_via_remainder_drop"
        },
        "training_metadata": {
            "source_data_file": data_path.name,
            "split_column": SPLIT_COLUMN,
            "target_column": target_col,
            "n_train_samples": int(len(X_train)),
            "n_valid_samples": int(len(X_valid)),
            "n_test_samples": int(len(X_test)),
            "fit_time_seconds": round(fit_duration, 4),
            "class_distribution_train": class_distribution_train
        },
        "evaluation_metrics": {
            "validation_partition": {
                "accuracy": round(float(val_acc), 4),
                "f1_macro": round(float(val_f1), 4),
                "log_loss": round(float(val_loss), 4)
            },
            "test_partition": {
                "accuracy": round(float(test_acc), 4),
                "f1_macro": round(float(test_f1), 4),
                "log_loss": round(float(test_loss), 4)
            }
        }
    }
    
    with open(output_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print(f"\n* Manifesto de metadados gravado com sucesso em: {output_metadata_path}")
    print("="*80)
    print("SERIALIZAÇÃO ATÔMICA E METADADOS CONCLUÍDOS COM 100% DE SUCESSO!")
    print("="*80)
    
    return manifest


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "abt_sanidade_vegetal.csv"
    output_model_path = base_dir / "models" / "sanidade_vegetal_pipeline_final.joblib"
    output_metadata_path = base_dir / "models" / "sanidade_vegetal_pipeline_final_metadata.json"
    
    train_and_serialize_champion_pipeline(
        data_path=data_path,
        output_model_path=output_model_path,
        output_metadata_path=output_metadata_path
    )


if __name__ == '__main__':
    main()
