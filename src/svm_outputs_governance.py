"""
Governança das Saídas do SVM e Auditoria de Isolamento de Partições
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 3 — Framework SEMMA (Fase: Model)

Checklist coberto:
1. Catalogar colunas de saída: y_pred, y_proba (por classe) e decision_function.
2. Gerar a tabela de predições que materializa essas colunas (valid e test).
3. Auditar o isolamento estrito das partições (train / valid / test).

Saídas:
- data/processed/svm_predicoes_valid_test.csv
- models/svm_outputs_governance_audit.json
"""

import sys
import json
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import joblib

try:
    from atomic_svm_pipeline import TARGET_COLUMNS, SPLIT_COLUMN, NUMERICAL_FEATURES
except ImportError:
    from src.atomic_svm_pipeline import TARGET_COLUMNS, SPLIT_COLUMN, NUMERICAL_FEATURES

CLASS_NAMES_MAP = {
    0: 'GRASSY SHOOT', 1: 'HEALTHY', 2: 'LEAF SCALD', 3: 'MOSAIC',
    4: 'RED ROT', 5: 'RUST', 6: 'YELLOW LEAF'
}


def _slug(name: str) -> str:
    """Converte o nome da classe para snake_case minúsculo (ex.: RED ROT -> red_rot)."""
    return name.lower().replace(' ', '_')


# ==============================================================================
# 1. TABELA DE SAÍDAS DO MODELO (y_pred, y_proba_*, decision_function_*)
# ==============================================================================

def build_predictions_table(pipeline, df: pd.DataFrame, partitions=('valid', 'test')) -> pd.DataFrame:
    """Gera, por amostra, as saídas do SVM: classe prevista, probabilidades e distâncias à margem."""
    frames = []
    classes = list(pipeline.named_steps['svm'].classes_)
    for part in partitions:
        d = df[df[SPLIT_COLUMN] == part].copy()
        X = d.drop(columns=TARGET_COLUMNS)

        y_pred = pipeline.predict(X)
        y_proba = pipeline.predict_proba(X)
        y_dec = pipeline.decision_function(X)

        out = pd.DataFrame({
            'sample_id': d['sample_id'].values,
            'split_partition': part,
            'target_multiclass': d['target_multiclass'].values,
            'y_pred': y_pred.astype(int),
            'y_pred_label': [CLASS_NAMES_MAP[int(c)] for c in y_pred],
        })
        for j, c in enumerate(classes):
            out[f'y_proba_{_slug(CLASS_NAMES_MAP[int(c)])}'] = y_proba[:, j]
        for j, c in enumerate(classes):
            out[f'decision_function_{_slug(CLASS_NAMES_MAP[int(c)])}'] = y_dec[:, j]
        out['y_proba_max'] = y_proba.max(axis=1)
        out['pred_correct'] = (out['y_pred'] == out['target_multiclass']).astype(int)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


# ==============================================================================
# 2. AUDITORIA DE ISOLAMENTO DAS PARTIÇÕES
# ==============================================================================

def audit_partition_isolation(df: pd.DataFrame, pipeline=None) -> dict:
    """Verifica que nenhum dado de treino aparece em valid/test e que o pré-processador só viu treino."""
    parts = {p: df[df[SPLIT_COLUMN] == p] for p in ('train', 'valid', 'test')}
    report = {'partition_sizes': {p: int(len(d)) for p, d in parts.items()}, 'checks': {}}

    unknown = set(df[SPLIT_COLUMN].unique()) - set(parts)
    report['checks']['only_known_partitions'] = not unknown

    pairs = [('train', 'valid'), ('train', 'test'), ('valid', 'test')]
    for key in ('sample_id', 'filepath', 'filename'):
        for a, b in pairs:
            inter = set(parts[a][key]) & set(parts[b][key])
            report['checks'][f'{key}_disjoint_{a}_{b}'] = len(inter) == 0
            if inter:
                report['checks'][f'{key}_overlap_count_{a}_{b}'] = len(inter)

    # Duplicatas de conteúdo: linhas de features numéricas idênticas entre partições
    def feat_hash(d):
        return set(pd.util.hash_pandas_object(d[NUMERICAL_FEATURES].round(6), index=False))
    hashes = {p: feat_hash(d) for p, d in parts.items()}
    for a, b in pairs:
        n = len(hashes[a] & hashes[b])
        report['checks'][f'feature_duplicates_{a}_{b}'] = n
        report['checks'][f'feature_content_disjoint_{a}_{b}'] = n == 0

    # O pré-processador deve ter sido ajustado apenas com treino
    if pipeline is not None:
        scaler = pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler']
        train_mean = parts['train'][NUMERICAL_FEATURES].mean().values
        all_mean = df[NUMERICAL_FEATURES].mean().values
        report['checks']['scaler_fitted_on_train_only'] = bool(np.allclose(scaler.mean_, train_mean, rtol=1e-4, atol=1e-4))
        report['checks']['scaler_differs_from_global'] = bool(not np.allclose(scaler.mean_, all_mean, rtol=1e-6, atol=1e-6))

    bool_checks = [v for k, v in report['checks'].items() if isinstance(v, bool)]
    report['status'] = 'APPROVED' if all(bool_checks) else 'FAILED'
    return report


# ==============================================================================
# 3. EXECUÇÃO
# ==============================================================================

def load_champion_pipeline(base: Path, df: pd.DataFrame):
    """Carrega o SVM campeão; se o .joblib for incompatível com a versão do sklearn,
    reajusta o mesmo estimador (best_params do metadata) usando EXCLUSIVAMENTE a partição train."""
    try:
        return joblib.load(base / 'models' / 'svm_pipeline_optimal.joblib'), 'joblib'
    except Exception as exc:
        print(f"[AVISO] joblib incompatível ({type(exc).__name__}); reajustando campeão apenas no treino.")
        try:
            from atomic_svm_pipeline import build_atomic_svm_pipeline
        except ImportError:
            from src.atomic_svm_pipeline import build_atomic_svm_pipeline
        meta = json.loads((base / 'models' / 'svm_gridsearch_metadata.json').read_text(encoding='utf-8'))
        bp = meta['best_hyperparameters']
        pipe = build_atomic_svm_pipeline(kernel=bp['svm__kernel'], C=bp['svm__C'], gamma=bp['svm__gamma'],
                                         probability=True, random_state=42, class_weight='balanced')
        tr = df[df[SPLIT_COLUMN] == 'train']
        pipe.fit(tr.drop(columns=TARGET_COLUMNS), tr['target_multiclass'])
        return pipe, 'refit_train_only'


def main():
    base = Path(__file__).resolve().parent.parent
    df = pd.read_csv(base / 'data' / 'processed' / 'abt_sanidade_vegetal.csv')
    pipeline, model_origin = load_champion_pipeline(base, df)

    audit = audit_partition_isolation(df, pipeline)
    print(json.dumps(audit, indent=2, ensure_ascii=False))

    preds = build_predictions_table(pipeline, df)
    out_csv = base / 'data' / 'processed' / 'svm_predicoes_valid_test.csv'
    preds.to_csv(out_csv, index=False)
    print(f"\nPredições exportadas: {out_csv} {preds.shape}")
    print("Colunas:", list(preds.columns))

    # Sanidade das saídas
    proba_cols = [c for c in preds.columns if c.startswith('y_proba_') and c != 'y_proba_max']
    assert np.allclose(preds[proba_cols].sum(axis=1), 1.0, atol=1e-6), "y_proba não soma 1"
    audit['model_origin'] = model_origin
    audit['outputs'] = {
        'rows': int(len(preds)),
        'columns': list(preds.columns),
        'accuracy_valid': float(preds[preds.split_partition == 'valid'].pred_correct.mean()),
        'accuracy_test': float(preds[preds.split_partition == 'test'].pred_correct.mean()),
    }
    with open(base / 'models' / 'svm_outputs_governance_audit.json', 'w', encoding='utf-8') as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    return 0 if audit['status'] == 'APPROVED' else 1


if __name__ == '__main__':
    sys.exit(main())
