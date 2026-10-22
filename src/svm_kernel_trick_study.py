"""
Modelagem SVM com Kernel Trick (RBF e Polinomial)
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 3 — Framework SEMMA (Fase: Model)

Checklist coberto:
1. SVC(kernel='rbf'): influencia de gamma in [0.001, 0.01, 0.1, 1.0] e de C.
2. SVC(kernel='poly'): graus d in {2, 3}.
3. Comportamento da fronteira de decisao frente a ruido fotometrico e sombras.
4. Acuracia balanceada e tempo de treino: Linear x RBF x Polinomial.
5. Formulacao matematica do Kernel RBF (ver relatorio em Markdown).

Protocolo anti-leakage (herdado das tasks anteriores):
- Todo ajuste ocorre em 'train' (pre-processador + SVC dentro do Pipeline).
- Hiperparametros sao escolhidos pela particao 'valid'.
- 'test' e tocado UMA vez por kernel, apenas com o campeao ja escolhido.

Saidas:
- models/svm_kernel_study_results.csv   (todas as configuracoes avaliadas)
- models/svm_kernel_study.json          (campeoes, tempos, robustez)
- docs/figures/svm_kernel_*.png
"""

import sys
import json
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

try:
    from atomic_svm_pipeline import build_atomic_svm_pipeline, TARGET_COLUMNS, SPLIT_COLUMN
except ImportError:
    from src.atomic_svm_pipeline import build_atomic_svm_pipeline, TARGET_COLUMNS, SPLIT_COLUMN

SEED = 42
GAMMAS_RBF = [0.001, 0.01, 0.1, 1.0]
C_RBF = [0.1, 1.0, 10.0, 100.0]
DEGREES_POLY = [2, 3]
C_POLY = [0.1, 1.0, 10.0]
GAMMAS_POLY = [0.01, 0.1, 'scale']
C_LINEAR = [0.1, 1.0, 10.0]
N_TIME_REPEATS = 3


def split_abt(df):
    out = {}
    for p in ('train', 'valid', 'test'):
        d = df[df[SPLIT_COLUMN] == p]
        out[p] = (d.drop(columns=TARGET_COLUMNS), d['target_multiclass'].values)
    return out


def metrics(pipe, X, y):
    pred = pipe.predict(X)
    return {
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'accuracy': float(accuracy_score(y, pred)),
        'f1_macro': float(f1_score(y, pred, average='macro')),
    }


# ==============================================================================
# 1. VARREDURA POR KERNEL (treino em train, selecao em valid)
# ==============================================================================

def sweep_kernel(kernel, param_sets, data):
    Xtr, ytr = data['train']
    Xva, yva = data['valid']
    rows = []
    for params in param_sets:
        pipe = build_atomic_svm_pipeline(kernel=kernel, probability=False, random_state=SEED,
                                         class_weight='balanced', **params)
        t0 = time.perf_counter()
        pipe.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        m = metrics(pipe, Xva, yva)
        rows.append({
            'kernel': kernel,
            'C': params['C'],
            'gamma': params.get('gamma', np.nan if kernel == 'linear' else 'scale'),
            'degree': params.get('degree', np.nan),
            'fit_time_s': round(fit_s, 3),
            'n_support_vectors': int(pipe.named_steps['svm'].n_support_.sum()),
            **{f'valid_{k}': round(v, 4) for k, v in m.items()},
        })
        print(f"  {kernel:6s} {params} -> bal_acc={m['balanced_accuracy']:.4f} t={fit_s:.2f}s")
    return pd.DataFrame(rows)


def grids():
    rbf = [dict(C=c, gamma=g) for g in GAMMAS_RBF for c in C_RBF]
    poly = [dict(C=c, gamma=g, degree=d) for d in DEGREES_POLY for g in GAMMAS_POLY for c in C_POLY]
    lin = [dict(C=c) for c in C_LINEAR]
    return {'linear': lin, 'rbf': rbf, 'poly': poly}


def champion_params(df_k):
    best = df_k.sort_values(['valid_balanced_accuracy', 'valid_f1_macro'], ascending=False).iloc[0]
    p = {'C': float(best['C'])}
    if best['kernel'] in ('rbf', 'poly'):
        g = best['gamma']
        p['gamma'] = g if isinstance(g, str) else float(g)
    if best['kernel'] == 'poly':
        p['degree'] = int(best['degree'])
    return p


# ==============================================================================
# 2. COMPARACAO FINAL E TEMPO DE TREINO (mediana de N repeticoes)
# ==============================================================================

def evaluate_champion(kernel, params, data):
    Xtr, ytr = data['train']
    times, pipe = [], None
    for _ in range(N_TIME_REPEATS):
        pipe = build_atomic_svm_pipeline(kernel=kernel, probability=False, random_state=SEED,
                                         class_weight='balanced', **params)
        t0 = time.perf_counter()
        pipe.fit(Xtr, ytr)
        times.append(time.perf_counter() - t0)
    return pipe, {
        'params': params,
        'fit_time_median_s': round(float(np.median(times)), 3),
        'n_support_vectors': int(pipe.named_steps['svm'].n_support_.sum()),
        'valid': {k: round(v, 4) for k, v in metrics(pipe, *data['valid']).items()},
        'test': {k: round(v, 4) for k, v in metrics(pipe, *data['test']).items()},
    }


# ==============================================================================
# 3. ROBUSTEZ A RUIDO FOTOMETRICO E SOMBRAS (simulacao no espaco de features)
# ==============================================================================
# Nao ha imagens brutas no repositorio (data/raw vazio); logo as perturbacoes sao
# aplicadas as features extraidas, respeitando a fisica de cada descritor:
#  - Ruido fotometrico: N(0, (s * sigma_treino)^2) nas features cromaticas.
#  - Sombra (escurecimento multiplicativo f dos canais RGB):
#      ExG, ExR ~ f | laplacian_var, glcm_contrast ~ f^2 | glcm_dissimilarity ~ f
#      IRFP = contraste*dissimilaridade/homogeneidade ~ f^3
#      Hue, razao R/G e ICN sao (aprox.) invariantes a brilho.

CHROMA_FEATURES = ['mean_hue', 'std_saturation', 'exg_index', 'exr_index', 'rg_ratio',
                   'indice_clorose_necrose', 'hue_dispersion']
SHADOW_POWERS = {'exg_index': 1, 'exr_index': 1, 'laplacian_var': 2, 'glcm_contrast': 2,
                 'glcm_dissimilarity': 1, 'indice_rugosidade_pustula': 3}


def add_photometric_noise(X, train_std, level, rng):
    Xp = X.copy()
    for c in CHROMA_FEATURES:
        Xp[c] = Xp[c] + rng.normal(0.0, level * train_std[c], size=len(Xp))
    return Xp


def apply_shadow(X, factor):
    Xp = X.copy()
    for c, p in SHADOW_POWERS.items():
        Xp[c] = Xp[c] * (factor ** p)
    return Xp


def robustness(pipes, data):
    Xva, yva = data['valid']
    Xtr, _ = data['train']
    train_std = Xtr[CHROMA_FEATURES].std()
    scenarios = {'limpo': Xva}
    for lvl in (0.1, 0.25, 0.5, 1.0):
        scenarios[f'ruido_{lvl}sigma'] = add_photometric_noise(Xva, train_std, lvl, np.random.default_rng(SEED))
    for f in (0.8, 0.6, 0.4):
        scenarios[f'sombra_f{f}'] = apply_shadow(Xva, f)
    scenarios['sombra_f0.6+ruido_0.25sigma'] = add_photometric_noise(
        apply_shadow(Xva, 0.6), train_std, 0.25, np.random.default_rng(SEED))

    rows = []
    for k, pipe in pipes.items():
        base_pred = pipe.predict(Xva)
        base_ba = balanced_accuracy_score(yva, base_pred)
        base_dec = np.abs(pipe.decision_function(Xva)).mean()
        for name, Xs in scenarios.items():
            pred = pipe.predict(Xs)
            ba = balanced_accuracy_score(yva, pred)
            rows.append({
                'kernel': k, 'cenario': name,
                'balanced_accuracy': round(float(ba), 4),
                'queda_vs_limpo': round(float(base_ba - ba), 4),
                'taxa_troca_predicao': round(float((pred != base_pred).mean()), 4),
                'margem_media_abs_decision_function': round(float(np.abs(pipe.decision_function(Xs)).mean()), 4),
                'margem_relativa_ao_limpo': round(float(np.abs(pipe.decision_function(Xs)).mean() / base_dec), 4),
            })
    return pd.DataFrame(rows)


# ==============================================================================
# 4. FIGURAS
# ==============================================================================

def make_figures(res, comp, rob, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    rbf = res[res.kernel == 'rbf']
    piv = rbf.pivot(index='gamma', columns='C', values='valid_balanced_accuracy').sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(piv.values, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(g) for g in piv.index])
    ax.set_xlabel('C'); ax.set_ylabel('gamma'); ax.set_title('RBF: acuracia balanceada (valid)')
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha='center', va='center', color='w', fontsize=8)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(fig_dir / 'svm_kernel_rbf_gamma_C_heatmap.png', dpi=130); plt.close(fig)

    names = list(comp.keys())
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    axs[0].bar(names, [comp[n]['valid']['balanced_accuracy'] for n in names], label='valid')
    axs[0].bar(names, [comp[n]['test']['balanced_accuracy'] for n in names], width=0.4, label='test')
    axs[0].set_title('Acuracia balanceada'); axs[0].legend()
    axs[1].bar(names, [comp[n]['fit_time_median_s'] for n in names], color='tab:orange')
    axs[1].set_title('Tempo de treino (s, mediana)')
    fig.tight_layout(); fig.savefig(fig_dir / 'svm_kernel_comparacao.png', dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for k in rob.kernel.unique():
        d = rob[rob.kernel == k]
        ax.plot(d.cenario, d.balanced_accuracy, marker='o', label=k)
    ax.set_ylabel('Acuracia balanceada (valid)'); ax.set_title('Robustez a ruido fotometrico e sombras')
    ax.tick_params(axis='x', rotation=45); ax.legend(); fig.tight_layout()
    fig.savefig(fig_dir / 'svm_kernel_robustez.png', dpi=130); plt.close(fig)


# ==============================================================================
# 5. EXECUCAO
# ==============================================================================

def main():
    base = Path(__file__).resolve().parent.parent
    df = pd.read_csv(base / 'data' / 'processed' / 'abt_sanidade_vegetal.csv')
    data = split_abt(df)
    print({k: len(v[1]) for k, v in data.items()})

    all_res, champs = [], {}
    for kernel, ps in grids().items():
        print(f"\n== Kernel {kernel} ({len(ps)} configuracoes) ==")
        all_res.append(sweep_kernel(kernel, ps, data))
    res = pd.concat(all_res, ignore_index=True)
    res.to_csv(base / 'models' / 'svm_kernel_study_results.csv', index=False)

    pipes, comp = {}, {}
    for kernel in ('linear', 'rbf', 'poly'):
        p = champion_params(res[res.kernel == kernel])
        pipes[kernel], comp[kernel] = evaluate_champion(kernel, p, data)
        print(f"\nCampeao {kernel}: {comp[kernel]}")

    rob = robustness(pipes, data)
    print("\n", rob.to_string(index=False))
    make_figures(res, comp, rob, base / 'docs' / 'figures')

    # Sanidade anti-leakage: scaler dos campeoes ajustado so com train
    from atomic_svm_pipeline import NUMERICAL_FEATURES
    tr_mean = data['train'][0][NUMERICAL_FEATURES].mean().values
    for k, p in pipes.items():
        sc = p.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler']
        assert np.allclose(sc.mean_, tr_mean, rtol=1e-4, atol=1e-4), f"scaler de {k} nao e train-only"

    out = {
        'protocol': 'fit=train | selecao de hiperparametros=valid | test tocado 1x por campeao',
        'grids': {'gamma_rbf': GAMMAS_RBF, 'C_rbf': C_RBF, 'degree_poly': DEGREES_POLY,
                  'C_poly': C_POLY, 'gamma_poly': [str(g) for g in GAMMAS_POLY], 'C_linear': C_LINEAR},
        'champions': comp,
        'rbf_gamma_effect_best_C': {
            str(g): float(res[(res.kernel == 'rbf') & (res.gamma == g)].valid_balanced_accuracy.max())
            for g in GAMMAS_RBF},
        'poly_degree_effect_best': {
            str(d): float(res[(res.kernel == 'poly') & (res.degree == d)].valid_balanced_accuracy.max())
            for d in DEGREES_POLY},
        'robustness': rob.to_dict(orient='records'),
        'scaler_train_only': True,
    }
    (base / 'models' / 'svm_kernel_study.json').write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print("\nArtefatos gravados em models/ e docs/figures/")


if __name__ == '__main__':
    main()
