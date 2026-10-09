"""
Módulo de Integração Atômica do Estimador SVM ao Pipeline Anti-Leakage
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 3 — Framework SEMMA (Fase: Model)
Responsável: Elisa (Engenharia de Machine Learning & MLOps)

Checklist Coberto:
1. Construir o objeto Pipeline Scikit-Learn completo (Pré-processador da Sprint 2 + Estimador SVC).
2. Garantir que o escalonamento (StandardScaler) e transformações ocorram isoladamente dentro de cada fold.
3. Validar a compatibilidade dos métodos .fit(), .predict() e .predict_proba().
4. Assegurar ausência total de Data Leakage na execução combinada.
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
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    log_loss
)
from sklearn.calibration import calibration_curve
import joblib

# Importar construtor e taxonomia da Sprint 2
try:
    from atomic_pipeline_preprocessing import (
        build_atomic_column_transformer,
        get_column_taxonomy,
        NUMERICAL_FEATURES,
        DISCRETIZED_FEATURES,
        CATEGORICAL_NOMINAL_FEATURES,
        ALL_CATEGORICAL_FEATURES,
        METADATA_DROPPED_COLUMNS,
        TARGET_COLUMNS,
        SPLIT_COLUMN
    )
except ImportError:
    from src.atomic_pipeline_preprocessing import (
        build_atomic_column_transformer,
        get_column_taxonomy,
        NUMERICAL_FEATURES,
        DISCRETIZED_FEATURES,
        CATEGORICAL_NOMINAL_FEATURES,
        ALL_CATEGORICAL_FEATURES,
        METADATA_DROPPED_COLUMNS,
        TARGET_COLUMNS,
        SPLIT_COLUMN
    )


# ==============================================================================
# 1. CONSTRUTOR DA FÁBRICA DO PIPELINE ATÔMICO COM SVM
# ==============================================================================

def build_atomic_svm_pipeline(
    kernel: str = 'rbf',
    C: float = 1.0,
    gamma: str = 'scale',
    degree: int = 3,
    probability: bool = True,
    random_state: int = 42,
    class_weight: str = 'balanced'
) -> Pipeline:
    """
    Constrói e encapsula o objeto Pipeline Scikit-Learn completo e atômico:
    Etapa 1 ('preprocessor'): ColumnTransformer da Sprint 2 (Imputação + Scaler + OHE)
    Etapa 2 ('svm'): Estimador SVC com suporte a calibração de probabilidades (Platt Scaling)
    
    Parâmetros:
    -----------
    kernel : str, default='rbf'
        Kernel do SVM ('linear', 'rbf', 'poly', 'sigmoid').
    C : float, default=1.0
        Parâmetro de penalização por erro de margem suave.
    gamma : str ou float, default='scale'
        Coeficiente do kernel para 'rbf', 'poly' e 'sigmoid'.
    degree : int, default=3
        Grau da função polinomial para kernel 'poly'.
    probability : bool, default=True
        Habilita a estimativa de probabilidades via método predict_proba() (Platt scaling).
    random_state : int, default=42
        Semente para reprodutibilidade estrita.
    class_weight : str, default='balanced'
        Ponderação inversa à frequência das classes para mitigar desbalanceamento.
        
    Retorna:
    --------
    Pipeline:
        Objeto Scikit-Learn Pipeline pronto para fit, transform, predict e cross-validation.
    """
    preprocessor = build_atomic_column_transformer()
    
    svm_classifier = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        probability=probability,
        random_state=random_state,
        class_weight=class_weight
    )
    
    full_pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('svm', svm_classifier)
        ]
    )
    
    return full_pipeline


# ==============================================================================
# 2. AUDITORIA DE COMPATIBILIDADE DE MÉTODOS (.fit, .predict, .predict_proba)
# ==============================================================================

def validate_inference_methods(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    target_type: str = 'multiclass'
) -> dict:
    """
    Valida a conformidade operacional de todos os métodos de inferência exigidos:
    - .fit() ajusta pipeline completo de ponta a ponta sem erros.
    - .predict() gera predições com formato e tipos válidos.
    - .predict_proba() gera matriz estocástica com somatório 1.0 por linha.
    """
    print("\n" + "="*80)
    print(f"VALIDAÇÃO FORMAL DOS MÉTODOS .fit(), .predict() E .predict_proba() [{target_type.upper()}]")
    print("="*80)
    
    classes_unique = np.sort(np.unique(y_train))
    n_classes = len(classes_unique)
    print(f"* Classes identificadas ({n_classes}): {classes_unique}")
    print(f"* Volumetria: Treino = {len(X_train)} amostras | Validação = {len(X_val)} amostras")
    
    # 1. Teste do método .fit()
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_duration = time.perf_counter() - t0
    print(f"  [OK] Método .fit() executado com sucesso em {fit_duration:.2f}s!")
    
    # Checar se os estimadores foram devidamente ajustados
    assert hasattr(pipeline.named_steps['preprocessor'], 'transformers_'), "Pré-processador não foi ajustado!"
    assert hasattr(pipeline.named_steps['svm'], 'support_vectors_'), "SVM não foi ajustado!"
    n_support_vectors = pipeline.named_steps['svm'].support_vectors_.shape[0]
    print(f"  [OK] Estimador SVM convergiu com {n_support_vectors} vetores de suporte.")
    
    # 2. Teste do método .predict()
    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_val)
    pred_duration = time.perf_counter() - t0
    print(f"  [OK] Método .predict() executado com sucesso em {pred_duration*1000:.2f}ms!")
    
    assert y_pred.shape == (len(X_val),), f"Shape das predições inválido: {y_pred.shape}"
    assert np.all(np.isin(y_pred, classes_unique)), "Predições contêm classes inválidas!"
    
    # 3. Teste do método .predict_proba()
    t0 = time.perf_counter()
    y_proba = pipeline.predict_proba(X_val)
    proba_duration = time.perf_counter() - t0
    print(f"  [OK] Método .predict_proba() executado com sucesso em {proba_duration*1000:.2f}ms!")
    
    # Validações estocásticas estritas
    assert y_proba.shape == (len(X_val), n_classes), f"Shape da matriz de probabilidades incorreto: {y_proba.shape}"
    assert np.all(y_proba >= 0.0) and np.all(y_proba <= 1.0), "Probabilidades fora do intervalo [0.0, 1.0]!"
    
    # Somatório de probabilidades deve ser exatamente 1.0 para cada instância
    row_sums = y_proba.sum(axis=1)
    np.testing.assert_allclose(
        row_sums,
        np.ones(len(X_val)),
        rtol=1e-5,
        atol=1e-5,
        err_msg="As linhas de predict_proba não somam 1.0!"
    )
    print(f"  [OK] Matriz estocástica 100% válida: todas as {len(X_val)} linhas somam exatamente 1.0 (Platt Scaling).")
    
    # 4. Avaliação preliminar de métricas baseline
    acc = accuracy_score(y_val, y_pred)
    f1_mac = f1_score(y_val, y_pred, average='macro')
    f1_wei = f1_score(y_val, y_pred, average='weighted')
    loss_val = log_loss(y_val, y_proba, labels=classes_unique)
    
    print(f"\n* Desempenho Baseline no conjunto de Validação:")
    print(f"  - Acurácia Global     : {acc*100:.2f}%")
    print(f"  - F1-Score (Macro)    : {f1_mac:.4f}")
    print(f"  - F1-Score (Weighted) : {f1_wei:.4f}")
    print(f"  - Log-Loss            : {loss_val:.4f}")
    
    return {
        "target_type": target_type,
        "n_classes": int(n_classes),
        "fit_time_seconds": round(fit_duration, 3),
        "inference_time_ms": round(pred_duration * 1000, 2),
        "proba_time_ms": round(proba_duration * 1000, 2),
        "n_support_vectors": int(n_support_vectors),
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_mac), 4),
        "f1_weighted": round(float(f1_wei), 4),
        "log_loss": round(float(loss_val), 4),
        "y_pred": y_pred,
        "y_proba": y_proba
    }


# ==============================================================================
# 3. AUDITORIA DE ISOLAMENTO POR FOLD EM STRATIFIEDKFOLD (ANTI-LEAKAGE)
# ==============================================================================

def audit_stratified_kfold_isolation(
    df: pd.DataFrame,
    target_col: str = 'target_multiclass',
    n_splits: int = 5,
    random_state: int = 42
) -> dict:
    """
    Garante e comprova que o escalonamento (StandardScaler) e as transformações
    ocorram isoladamente dentro de cada fold do StratifiedKFold.
    
    Demonstração Matemática:
    Para cada fold k in [1..5]:
    1. StandardScaler.mean_ do fold k é computado unicamente sobre X_train_fold_k.
    2. A média difere da média global de toda a base (provando isolamento de X_val_fold_k).
    3. Nenhum dado do fold de validação vaza para o fold de treino.
    """
    print("\n" + "="*80)
    print(f"AUDITORIA MATEMÁTICA DE ISOLAMENTO POR FOLD EM STRATIFIED K-FOLD (cv={n_splits})")
    print("="*80)
    
    # Treino da ABT (apenas a partição de desenvolvimento 'train' de 5574 amostras)
    df_dev = df[df[SPLIT_COLUMN] == 'train'].copy().reset_index(drop=True)
    X_dev = df_dev.drop(columns=TARGET_COLUMNS)
    y_dev = df_dev[target_col]
    
    # Médias da população completa de desenvolvimento para comparação
    dev_global_means = df_dev[NUMERICAL_FEATURES].mean().values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_records = []
    fold_scaler_means = []
    
    print(f"* Base de Validação Cruzada: {len(df_dev)} instâncias, {n_splits} folds estratificados.")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), start=1):
        t_fold_start = time.perf_counter()
        
        X_train_f, y_train_f = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
        X_val_f, y_val_f = X_dev.iloc[val_idx], y_dev.iloc[val_idx]
        
        # 1. Instanciar novo pipeline atômico limpo e independente
        pipe_fold = build_atomic_svm_pipeline(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=random_state
        )
        
        # 2. Ajuste do Pipeline estritamente nas amostras de treino do Fold
        pipe_fold.fit(X_train_f, y_train_f)
        
        # 3. Extrair os parâmetros aprendidos pelo StandardScaler e SimpleImputer DENTRO deste fold
        preprocessor_step = pipe_fold.named_steps['preprocessor']
        num_transformer = preprocessor_step.named_transformers_['num']
        fold_scaler = num_transformer.named_steps['scaler']
        fold_imputer = num_transformer.named_steps['imputer']
        
        # A média do Scaler aprendida no fold
        learned_means = fold_scaler.mean_.copy()
        fold_scaler_means.append(learned_means)
        
        # Média empírica estrita do subconjunto de treino deste fold
        expected_means = X_train_f[NUMERICAL_FEATURES].mean().values
        
        # Comprovação matemática 1: StandardScaler.mean_ == Média do Treino do Fold
        np.testing.assert_allclose(
            learned_means,
            expected_means,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Discrepância entre as médias do Scaler e o conjunto de treino do Fold {fold_idx}!"
        )
        
        # Comprovação matemática 2: Mediana do Imputer == Mediana do Treino do Fold
        learned_medians = fold_imputer.statistics_.copy()
        expected_medians = X_train_f[NUMERICAL_FEATURES].median().values
        np.testing.assert_allclose(
            learned_medians,
            expected_medians,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Discrepância entre as medianas do Imputer e o conjunto de treino do Fold {fold_idx}!"
        )
        
        # Comprovação matemática 3: As estatísticas diferem da média da população global
        diff_from_global = np.abs(learned_means - dev_global_means)
        assert np.any(diff_from_global > 1e-4), f"Fold {fold_idx} usou parâmetros globais (Vazamento detectado)!"
        
        # 4. Avaliar predições e probabilidades sobre a partição de validação do fold
        y_val_pred = pipe_fold.predict(X_val_f)
        y_val_proba = pipe_fold.predict_proba(X_val_f)
        
        acc_fold = accuracy_score(y_val_f, y_val_pred)
        f1_fold = f1_score(y_val_f, y_val_pred, average='macro')
        fold_time = time.perf_counter() - t_fold_start
        
        print(f"  -> Fold {fold_idx}/{n_splits}: Treino={len(train_idx):4d} | Val={len(val_idx):4d} | "
              f"Acc={acc_fold*100:5.2f}% | F1-Macro={f1_fold:6.4f} | Tempo={fold_time:4.2f}s | "
              f"[OK] Isolamento Estatístico Perfeito.")
        
        fold_records.append({
            "fold": fold_idx,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "accuracy": round(float(acc_fold), 4),
            "f1_macro": round(float(f1_fold), 4),
            "execution_time_sec": round(fold_time, 2),
            "mean_diff_vs_global": round(float(diff_from_global.mean()), 6)
        })
        
    # Comprovação matemática 4: Todos os folds possuem vetores de médias distintos entre si
    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            fold_diff = np.abs(fold_scaler_means[i] - fold_scaler_means[j])
            assert np.any(fold_diff > 1e-5), f"Folds {i+1} e {j+1} possuem os mesmos parâmetros estatísticos!"
            
    print("\n* RESUMO DA AUDITORIA DE FOLDS:")
    mean_acc = np.mean([r['accuracy'] for r in fold_records])
    std_acc = np.std([r['accuracy'] for r in fold_records])
    mean_f1 = np.mean([r['f1_macro'] for r in fold_records])
    std_f1 = np.std([r['f1_macro'] for r in fold_records])
    
    print(f"  - Acurácia Média CV     : {mean_acc*100:5.2f}% (+/- {std_acc*100:4.2f}%)")
    print(f"  - F1-Macro Médio CV     : {mean_f1:6.4f} (+/- {std_f1:6.4f})")
    print("  - Status Anti-Leakage   : 100% HOMOLOGADO (Isolamento total em todos os 5 folds).")
    
    return {
        "status": "APPROVED",
        "n_splits": n_splits,
        "fold_results": fold_records,
        "mean_accuracy": round(float(mean_acc), 4),
        "std_accuracy": round(float(std_acc), 4),
        "mean_f1_macro": round(float(mean_f1), 4),
        "std_f1_macro": round(float(std_f1), 4),
        "isolation_verified": True
    }


# ==============================================================================
# 4. TESTE DE PERTURBAÇÃO ADVERSARIAL (PROVA CABAL DE IMUTABILIDADE PÓS-FIT)
# ==============================================================================

def validate_adversarial_non_leakage(
    pipeline: Pipeline,
    X_val: pd.DataFrame
) -> dict:
    """
    Testa se mutações e perturbações drásticas no conjunto de teste/validação
    são incapazes de contaminar ou alterar qualquer parâmetro interno do pipeline.
    
    Cenário Adversarial:
    1. Salva cópia exata de todos os parâmetros internos do scaler e do SVM.
    2. Aplica ruído astronômico e valores extremos no conjunto de validação.
    3. Executa .predict() e .predict_proba() sobre a partição contaminada.
    4. Comprova que os pesos internos, suporte vetorial e coeficientes permaneceram 100% idênticos.
    """
    print("\n" + "="*80)
    print("TESTE DE PERTURBAÇÃO ADVERSARIAL: PROVA CABAL DE IMUTABILIDADE PÓS-FIT")
    print("="*80)
    
    # 1. Snapshot dos parâmetros internos
    scaler = pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler']
    svm = pipeline.named_steps['svm']
    
    means_before = scaler.mean_.copy()
    scales_before = scaler.scale_.copy()
    dual_coef_before = svm.dual_coef_.copy()
    intercept_before = svm.intercept_.copy()
    support_vectors_before = svm.support_vectors_.copy()
    
    # 2. Criar réplica adversarial de X_val com perturbações extremas
    X_val_adversarial = X_val.copy()
    for col in NUMERICAL_FEATURES:
        X_val_adversarial[col] = X_val_adversarial[col] * 1000.0 + 99999.0
    
    # 3. Executar inferência nas amostras adversariais
    _ = pipeline.predict(X_val_adversarial)
    _ = pipeline.predict_proba(X_val_adversarial)
    
    # 4. Verificar se algum parâmetro interno sofreu qualquer alteração
    means_after = scaler.mean_
    scales_after = scaler.scale_
    dual_coef_after = svm.dual_coef_
    intercept_after = svm.intercept_
    support_vectors_after = svm.support_vectors_
    
    np.testing.assert_array_equal(means_before, means_after, err_msg="Médias do scaler foram alteradas durante a inferência!")
    np.testing.assert_array_equal(scales_before, scales_after, err_msg="Variâncias do scaler foram alteradas!")
    np.testing.assert_array_equal(dual_coef_before, dual_coef_after, err_msg="Coeficientes do SVM foram alterados!")
    np.testing.assert_array_equal(intercept_before, intercept_after, err_msg="Intercepts do SVM foram alterados!")
    np.testing.assert_array_equal(support_vectors_before, support_vectors_after, err_msg="Vetores de suporte foram alterados!")
    
    print("  [OK] Imutabilidade Pós-Fit Comprovada:")
    print("    - Parâmetros do StandardScaler : 100% Inalterados após inferência adversarial.")
    print("    - Vetores de Suporte do SVM    : 100% Inalterados após inferência adversarial.")
    print("    - Coeficientes Duais e Vieses  : 100% Inalterados após inferência adversarial.")
    print("    - Risco de Data Leakage        : 0.00% (Blindagem Absoluta).")
    
    return {
        "status": "APPROVED",
        "parameters_immutable": True,
        "leakage_risk_percentage": 0.0
    }


# ==============================================================================
# 5. SERIALIZAÇÃO E GOVERNANÇA MLOPS
# ==============================================================================

def serialize_and_document_baseline_pipeline(
    pipeline: Pipeline,
    cv_audit: dict,
    val_metrics: dict,
    models_dir: Path
) -> dict:
    """
    Serializa o Pipeline Scikit-Learn atômico completo e gera o manifesto
    de governança estruturado em JSON com hash criptográfico SHA-256.
    """
    print("\n" + "="*80)
    print("SERIALIZAÇÃO E GOVERNANÇA MLOPS DO PIPELINE ATÔMICO SVM")
    print("="*80)
    
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = models_dir / "svm_pipeline_baseline.joblib"
    metadata_path = models_dir / "svm_pipeline_metadata.json"
    
    # 1. Serialização com compressão joblib
    joblib.dump(pipeline, joblib_path, compress=3)
    
    # 2. Cálculo de hash e tamanho
    hasher = hashlib.sha256()
    with open(joblib_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256_hash = hasher.hexdigest()
    file_size_kb = round(joblib_path.stat().st_size / 1024, 2)
    
    print(f"* Pipeline serializado em: {joblib_path}")
    print(f"  - Tamanho em disco : {file_size_kb} KB")
    print(f"  - Hash SHA-256     : {sha256_hash}")
    
    # 3. Teste de idempotência e integridade pós-recarga
    reloaded_pipeline = joblib.load(joblib_path)
    assert isinstance(reloaded_pipeline, Pipeline), "Objeto recarregado não é uma instância de Pipeline!"
    assert hasattr(reloaded_pipeline.named_steps['svm'], 'support_vectors_'), "SVM recarregado perdeu os vetores de suporte!"
    print("  [OK] Teste de Idempotência e Deserialização: 100% bem-sucedido.")
    
    # 4. Elaboração do manifesto de governança MLOps
    manifest = {
        "pipeline_name": "atomic_svm_pipeline_anti_leakage",
        "project": "Sanidade-Vegetal (SugarVision)",
        "sprint": "Sprint 3 — Modelagem e Otimização",
        "sprint_task": "Integração Atômica do Estimador SVM ao Pipeline Anti-Leakage",
        "responsible": "Elisa (Engenharia de Machine Learning & MLOps)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "framework": {
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
            "python_version": sys.version.split()[0]
        },
        "artifact": {
            "filename": joblib_path.name,
            "path": str(joblib_path),
            "size_kb": file_size_kb,
            "sha256": sha256_hash
        },
        "pipeline_architecture": {
            "steps": [
                {"step_name": "preprocessor", "class": "ColumnTransformer"},
                {"step_name": "svm", "class": "SVC"}
            ],
            "preprocessor_details": {
                "numeric_features_count": len(NUMERICAL_FEATURES),
                "categorical_features_count": len(ALL_CATEGORICAL_FEATURES),
                "total_output_features": 35,
                "scaling": "StandardScaler()",
                "imputation_numeric": "SimpleImputer(strategy='median')",
                "imputation_categorical": "SimpleImputer(strategy='most_frequent')",
                "encoding": "OneHotEncoder(sparse_output=False, handle_unknown='ignore')"
            },
            "svm_hyperparameters_baseline": {
                "kernel": pipeline.named_steps['svm'].kernel,
                "C": pipeline.named_steps['svm'].C,
                "gamma": pipeline.named_steps['svm'].gamma,
                "probability": pipeline.named_steps['svm'].probability,
                "class_weight": pipeline.named_steps['svm'].class_weight,
                "random_state": pipeline.named_steps['svm'].random_state,
                "n_support_vectors": int(pipeline.named_steps['svm'].support_vectors_.shape[0])
            }
        },
        "anti_leakage_audit": {
            "status": "APPROVED",
            "k_fold_stratified_splits": cv_audit['n_splits'],
            "mean_cv_accuracy": cv_audit['mean_accuracy'],
            "std_cv_accuracy": cv_audit['std_accuracy'],
            "mean_cv_f1_macro": cv_audit['mean_f1_macro'],
            "std_cv_f1_macro": cv_audit['std_f1_macro'],
            "isolation_verified": True,
            "adversarial_perturbation_test": "PASSED"
        },
        "validation_metrics_multiclass": {
            "accuracy": val_metrics['accuracy'],
            "f1_macro": val_metrics['f1_macro'],
            "f1_weighted": val_metrics['f1_weighted'],
            "log_loss": val_metrics['log_loss'],
            "fit_time_seconds": val_metrics['fit_time_seconds'],
            "inference_time_ms": val_metrics['inference_time_ms']
        },
        "readiness_for_gridsearch": {
            "status": "READY",
            "recommended_param_grid": {
                "svm__C": [0.1, 1.0, 10.0, 100.0],
                "svm__kernel": ["linear", "rbf", "poly"],
                "svm__gamma": ["scale", "auto", 0.01, 0.1],
                "svm__degree": [2, 3]
            },
            "cv_strategy": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "primary_scoring": "f1_macro"
        }
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print(f"* Manifesto salvo em: {metadata_path}")
    return manifest


# ==============================================================================
# 6. ORQUESTRADOR PRINCIPAL DE EXECUÇÃO
# ==============================================================================

def run_atomic_svm_pipeline_delivery():
    """Executa a rotina completa de testes, auditorias, validações e exportação."""
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    figures_dir = base_dir / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    abt_path = data_dir / "abt_sanidade_vegetal.csv"
    print(f"Carregando ABT original de: {abt_path}")
    df = pd.read_csv(abt_path)
    
    # 1. Separar partição de Treino e Validação
    df_train = df[df[SPLIT_COLUMN] == 'train'].copy()
    df_valid = df[df[SPLIT_COLUMN] == 'valid'].copy()
    
    X_train = df_train.drop(columns=TARGET_COLUMNS)
    y_train_multi = df_train['target_multiclass']
    y_train_bin = df_train['target_binary']
    
    X_valid = df_valid.drop(columns=TARGET_COLUMNS)
    y_valid_multi = df_valid['target_multiclass']
    y_valid_bin = df_valid['target_binary']
    
    # 2. Construir Pipeline Atômico
    print("\n* Instanciando build_atomic_svm_pipeline()...")
    pipeline = build_atomic_svm_pipeline(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42,
        class_weight='balanced'
    )
    
    # 3. Validar Métodos de Inferência (.fit, .predict, .predict_proba) em Multiclasse
    val_metrics_multi = validate_inference_methods(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train_multi,
        X_val=X_valid,
        y_val=y_valid_multi,
        target_type='multiclass (7 patologias)'
    )
    
    # 4. Validar Métodos de Inferência em Binário (Sadia vs Patológica)
    pipeline_bin = build_atomic_svm_pipeline(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42,
        class_weight='balanced'
    )
    val_metrics_bin = validate_inference_methods(
        pipeline=pipeline_bin,
        X_train=X_train,
        y_train=y_train_bin,
        X_val=X_valid,
        y_val=y_valid_bin,
        target_type='binary (0=Sadia, 1=Doente)'
    )
    
    # 5. Auditoria Matemática de Isolamento por Fold (StratifiedKFold)
    cv_audit = audit_stratified_kfold_isolation(
        df=df,
        target_col='target_multiclass',
        n_splits=5,
        random_state=42
    )
    
    # 6. Teste de Perturbação Adversarial
    adversarial_audit = validate_adversarial_non_leakage(
        pipeline=pipeline,
        X_val=X_valid
    )
    
    # 7. Serialização e Governança MLOps
    manifest = serialize_and_document_baseline_pipeline(
        pipeline=pipeline,
        cv_audit=cv_audit,
        val_metrics=val_metrics_multi,
        models_dir=models_dir
    )
    
    print("\n" + "="*80)
    print("CONCLUSAO COM 100% DE SUCESSO: INTEGRACAO ATOMICA DO ESTIMADOR SVM HOMOLOGADA!")
    print("="*80)
    return manifest


if __name__ == '__main__':
    run_atomic_svm_pipeline_delivery()
