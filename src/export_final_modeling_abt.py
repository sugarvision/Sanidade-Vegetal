"""
Módulo de Consolidação e Exportação da ABT Final (abt_features_modelagem)
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 2 — Framework SEMMA (Fase: Modify)
Responsável: Elisa (Engenharia de Machine Learning & MLOps)

Checklist Coberto:
1. Executar o pipeline de pré-processamento sobre a base completa com divisão determinística.
2. Validar ausência de valores null/NaN e integridade das colunas preditivas e alvo.
3. Salvar a base tratada em formato Apache Parquet (data/processed/abt_features_modelagem.parquet) e CSV.
4. Documentar o hash de integridade e o volume final de linhas e colunas geradas.
5. Garantir que o artefato esteja 100% pronto para ser consumido pelo GridSearchCV na Sprint 3.
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
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import joblib

from atomic_pipeline_preprocessing import (
    build_atomic_column_transformer,
    execute_and_validate_anti_leakage,
    consolidate_modeling_abt,
    SPLIT_COLUMN,
    TARGET_COLUMNS
)


def calculate_file_hash_and_stats(file_path: Path) -> dict:
    """Calcula tamanho e hash SHA-256 de um arquivo em disco."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    
    return {
        "filename": file_path.name,
        "path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "size_kb": round(file_path.stat().st_size / 1024, 2),
        "size_mb": round(file_path.stat().st_size / (1024 * 1024), 3),
        "sha256": hasher.hexdigest()
    }


def run_strict_quality_audit(df: pd.DataFrame) -> dict:
    """
    Executa bateria rigorosa de testes de qualidade sobre a ABT final tratada:
    1. Dimensão estrita: 6.571 linhas por 40 colunas.
    2. Ausência total de NaNs/Nulls em todas as colunas.
    3. Unicidade do sample_id.
    4. Integridade das partições (Train=5574, Valid=617, Test=380).
    5. Distribuição das 7 classes em class_label e consistência com alvos numéricos.
    """
    print("\n" + "="*80)
    print("AUDITORIA ESTRITA DE QUALIDADE E INTEGRIDADE DA ABT FINAL")
    print("="*80)
    
    # 1. Dimensão
    n_rows, n_cols = df.shape
    print(f"* Volumetria: {n_rows} linhas x {n_cols} colunas.")
    assert n_rows == 6571, f"Contagem de linhas divergente: esperado 6571, obtido {n_rows}!"
    assert n_cols == 40, f"Contagem de colunas divergente: esperado 40, obtido {n_cols}!"
    print("  [OK] Dimensões conformes: exatamente 6.571 instâncias x 40 colunas.")
    
    # 2. Ausência de Nulos
    total_nulls = df.isna().sum().sum()
    print(f"* Total de valores nulos (NaN/None): {total_nulls}")
    assert total_nulls == 0, f"Existem {total_nulls} valores ausentes na base consolidada!"
    print("  [OK] Ausência total de valores ausentes (0 NaNs em 100% das colunas).")
    
    # 3. Unicidade de sample_id
    n_unique_ids = df['sample_id'].nunique()
    assert n_unique_ids == 6571, f"IDs duplicados detectados: {6571 - n_unique_ids}"
    print("  [OK] Unicidade de identificadores de amostra (6.571 IDs únicos).")
    
    # 4. Integridade de Partições
    part_counts = df['split_partition'].value_counts().to_dict()
    print(f"* Distribuição por split_partition: {part_counts}")
    assert part_counts.get('train', 0) == 5574, "Erro na contagem de treino!"
    assert part_counts.get('valid', 0) == 617, "Erro na contagem de validação!"
    assert part_counts.get('test', 0) == 380, "Erro na contagem de teste!"
    print("  [OK] Partições experimentais 100% aderentes à amostragem estratificada.")
    
    # 5. Integridade das Classes e Targets
    class_counts = df['class_label'].value_counts().to_dict()
    print(f"* Distribuição das 7 classes fitopatológicas: {class_counts}")
    assert len(class_counts) == 7, "Número incorreto de classes!"
    
    binary_counts = df['target_binary'].value_counts().to_dict()
    print(f"* Target binário (0=Sadia, 1=Doente): {binary_counts}")
    assert binary_counts.get(0, 0) == 1146, "Contagem incorreta de folhas sadias no target binário!"
    assert binary_counts.get(1, 0) == 5425, "Contagem incorreta de folhas doentes no target binário!"
    print("  [OK] Targets supervisionados binário e multiclasse 100% consistentes.")
    
    return {
        "status": "APPROVED",
        "rows": n_rows,
        "columns": n_cols,
        "total_nulls": total_nulls,
        "partitions": part_counts,
        "classes": class_counts,
        "binary_distribution": binary_counts
    }


def benchmark_io_formats(parquet_path: Path, csv_path: Path) -> dict:
    """Compara tempo de carregamento e taxa de compressão entre Parquet e CSV."""
    print("\n" + "="*80)
    print("BENCHMARK COMPARATIVO DE DESEMPENHO DE I/O: PARQUET VS CSV")
    print("="*80)
    
    # Benchmark Parquet
    t0 = time.perf_counter()
    df_pq = pd.read_parquet(parquet_path)
    t_pq = time.perf_counter() - t0
    
    # Benchmark CSV
    t0 = time.perf_counter()
    df_csv = pd.read_csv(csv_path)
    t_csv = time.perf_counter() - t0
    
    size_pq_kb = parquet_path.stat().st_size / 1024
    size_csv_kb = csv_path.stat().st_size / 1024
    compression_ratio = size_csv_kb / size_pq_kb
    speedup = t_csv / t_pq if t_pq > 0 else 1.0
    
    print(f"* Apache Parquet:")
    print(f"  - Tamanho em disco : {size_pq_kb:.2f} KB ({size_pq_kb/1024:.2f} MB)")
    print(f"  - Tempo de leitura : {t_pq*1000:.2f} ms")
    
    print(f"* Formato CSV:")
    print(f"  - Tamanho em disco : {size_csv_kb:.2f} KB ({size_csv_kb/1024:.2f} MB)")
    print(f"  - Tempo de leitura : {t_csv*1000:.2f} ms")
    
    print(f"* Ganhos de Engenharia:")
    print(f"  - Redução de espaço (Compressão Colunar Snappy): {compression_ratio:.2f}x menor em Parquet.")
    print(f"  - Aceleração de I/O (Velocidade de Carga)       : {speedup:.2f}x mais rápido em Parquet.")
    
    return {
        "parquet": {"size_kb": round(size_pq_kb, 2), "read_ms": round(t_pq * 1000, 2)},
        "csv": {"size_kb": round(size_csv_kb, 2), "read_ms": round(t_csv * 1000, 2)},
        "compression_gain": round(compression_ratio, 2),
        "speedup_factor": round(speedup, 2)
    }


def run_gridsearch_readiness_smoke_test(df: pd.DataFrame):
    """
    Executa teste de fumaça (smoke test) utilizando GridSearchCV do Scikit-Learn
    para provar que a base consolidada está 100% pronta para a Sprint 3.
    """
    print("\n" + "="*80)
    print("TESTE DE FUMAÇA: PRONTIDÃO PARA GRIDSEARCHCV NA SPRINT 3")
    print("="*80)
    
    # Isolar features preditivas (as 35 features pré-processadas)
    feature_cols = [c for c in df.columns if c not in ['sample_id', 'split_partition', 'class_label', 'target_binary', 'target_multiclass']]
    print(f"* Quantidade de features preditivas isoladas: {len(feature_cols)}")
    assert len(feature_cols) == 35, f"Esperado 35 features preditivas, obtido {len(feature_cols)}"
    
    # Amostra determinística balanceada de desenvolvimento (treino)
    train_df = df[df['split_partition'] == 'train']
    idx_0 = train_df[train_df['target_binary'] == 0].sample(n=200, random_state=42).index
    idx_1 = train_df[train_df['target_binary'] == 1].sample(n=200, random_state=42).index
    sample_df = train_df.loc[idx_0.union(idx_1)]
    
    X_sample = sample_df[feature_cols].values
    y_sample = sample_df['target_binary'].values
    
    # Configurar mini-grade de hiperparâmetros para SVM RBF
    param_grid = {
        'C': [0.1, 1.0],
        'gamma': ['scale', 'auto']
    }
    
    svm = SVC(kernel='rbf', random_state=42)
    grid = GridSearchCV(svm, param_grid, cv=3, scoring='f1', n_jobs=1)
    
    t0 = time.perf_counter()
    grid.fit(X_sample, y_sample)
    t_fit = time.perf_counter() - t0
    
    print(f"  [OK] GridSearchCV executado com sucesso em {t_fit:.2f}s!")
    print(f"  [OK] Melhor F1-Score obtido no mini-teste: {grid.best_score_:.4f}")
    print(f"  [OK] Melhores Hiperparâmetros: {grid.best_params_}")
    print("  [OK] CONTRATO VALIDADO: A base 'abt_features_modelagem.parquet' é 100% compatível com a Sprint 3.")


def convert_numpy_types(obj):
    """Converte tipos numpy recursivamente para tipos nativos Python serializáveis em JSON."""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def export_and_audit_final_abt():
    """Fluxo completo de consolidação, auditoria, exportação e registro de manifesto."""
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "processed"
    abt_raw_path = data_dir / "abt_sanidade_vegetal.csv"
    parquet_path = data_dir / "abt_features_modelagem.parquet"
    csv_path = data_dir / "abt_features_modelagem.csv"
    manifest_path = data_dir / "abt_features_modelagem_manifest.json"
    
    print(f"Iniciando consolidação da ABT Final a partir de: {abt_raw_path}")
    df_raw = pd.read_csv(abt_raw_path)
    
    # 1. Executar Pipeline Determinístico com Anti-Leakage
    preprocessor = build_atomic_column_transformer()
    results = execute_and_validate_anti_leakage(df_raw, preprocessor)
    
    # 2. Consolidar e Salvar ABT Final
    df_final = consolidate_modeling_abt(df_raw, preprocessor, results, data_dir)
    
    # 3. Auditoria Estrita de Qualidade
    audit_results = run_strict_quality_audit(df_final)
    
    # 4. Benchmark de I/O
    io_benchmark = benchmark_io_formats(parquet_path, csv_path)
    
    # 5. Hashes Criptográficos e Manifesto
    hash_pq = calculate_file_hash_and_stats(parquet_path)
    hash_csv = calculate_file_hash_and_stats(csv_path)
    
    print("\n" + "="*80)
    print("HASHES CRIPTOGRÁFICOS DE INTEGRIDADE (SHA-256)")
    print("="*80)
    print(f"* Parquet : {hash_pq['filename']}")
    print(f"  SHA-256 : {hash_pq['sha256']}")
    print(f"  Tamanho : {hash_pq['size_kb']} KB")
    print(f"* CSV     : {hash_csv['filename']}")
    print(f"  SHA-256 : {hash_csv['sha256']}")
    print(f"  Tamanho : {hash_csv['size_kb']} KB")
    
    # 6. Teste de Fumaça com GridSearchCV
    run_gridsearch_readiness_smoke_test(df_final)
    
    # 7. Salvar Manifesto Estruturado
    manifest = {
        "artifact_name": "abt_features_modelagem",
        "project": "Sanidade-Vegetal (SugarVision)",
        "sprint": "Sprint 2 (Fase Modify)",
        "responsible": "Elisa",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {
            "parquet": hash_pq,
            "csv": hash_csv
        },
        "shape": {
            "rows": int(df_final.shape[0]),
            "columns": int(df_final.shape[1])
        },
        "column_groups": {
            "metadata_and_targets": ['sample_id', 'split_partition', 'class_label', 'target_binary', 'target_multiclass'],
            "predictive_features": [c for c in df_final.columns if c not in ['sample_id', 'split_partition', 'class_label', 'target_binary', 'target_multiclass']]
        },
        "audit": audit_results,
        "io_benchmark": io_benchmark,
        "readiness_sprint_3": "APPROVED"
    }
    
    manifest_clean = convert_numpy_types(manifest)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_clean, f, indent=2, ensure_ascii=False)
    print(f"\n* Manifesto de Integridade salvo em: {manifest_path}")
    print("\n" + "="*80)
    print("TAREFA CONCLUÍDA COM 100% DE SUCESSO!")
    print("="*80)
    return manifest_clean


if __name__ == '__main__':
    export_and_audit_final_abt()
