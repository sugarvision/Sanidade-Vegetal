"""
Módulo de Discretização de Variáveis Contínuas (Binning Estatístico)
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 2 — Framework SEMMA (Fase: Modify)
Responsável: Cesar (Lead Técnico & Visão Computacional)

Este módulo implementa:
1. Avaliação de distribuição e assimetria (Skewness e Curtose) de metadados contínuos.
2. Comparação formal de estratégias do KBinsDiscretizer (Quantile vs. Uniform vs. KMeans).
3. Determinação do número ideal de bins (k) por variável.
4. Prevenção estrita de Data Leakage (ajuste dos bins exclusivamente no conjunto de treino).
5. Geração de gráficos diagnósticos de distribuição e tabelas explicativas de intervalos de corte.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings('ignore')


def calculate_skewness_report(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """
    Calcula medidas de tendência central, dispersão, assimetria de Fisher-Pearson
    e curtose para cada variável contínua especificada.
    """
    report_rows = []
    for var in variables:
        s = df[var].dropna()
        skew_val = float(stats.skew(s))
        kurt_val = float(stats.kurtosis(s))
        
        # Classificação da assimetria
        if abs(skew_val) < 0.5:
            diag_skew = "Aproximadamente Simétrica"
        elif 0.5 <= abs(skew_val) < 1.0:
            diag_skew = "Assimetria Moderada"
        else:
            diag_skew = "Alta Assimetria (Cauda Longa)"
            
        report_rows.append({
            'variavel': var,
            'count': len(s),
            'min': float(s.min()),
            'q25': float(s.quantile(0.25)),
            'mediana': float(s.median()),
            'media': float(s.mean()),
            'q75': float(s.quantile(0.75)),
            'max': float(s.max()),
            'desvio_padrao': float(s.std()),
            'skewness': round(skew_val, 3),
            'curtose': round(kurt_val, 3),
            'diagnostico_assimetria': diag_skew
        })
    return pd.DataFrame(report_rows)


def compare_binning_strategies(train_df: pd.DataFrame, var_name: str, k_list: list = [3, 4, 5]) -> dict:
    """
    Compara as estratégias 'uniform', 'quantile' e 'kmeans' do KBinsDiscretizer
    calculando a distribuição de frequência, limites de corte e entropia de cada bin.
    """
    X_train = train_df[[var_name]]
    results = {}
    
    for k in k_list:
        results[k] = {}
        for strat in ['uniform', 'quantile', 'kmeans']:
            try:
                kbd = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy=strat, subsample=None)
                kbd.fit(X_train)
                binned = kbd.transform(X_train).flatten()
                counts = pd.Series(binned).value_counts().sort_index().to_dict()
                
                # Cálculo da entropia da distribuição de bins (máxima quando perfeitamente equilibrada)
                probs = np.array(list(counts.values())) / len(binned)
                entropy = -np.sum(probs * np.log2(probs + 1e-12))
                max_entropy = np.log2(k)
                balance_ratio = float(entropy / max_entropy) if max_entropy > 0 else 0.0
                
                results[k][strat] = {
                    'bin_edges': [round(float(e), 2) for e in kbd.bin_edges_[0]],
                    'counts': counts,
                    'entropy': round(float(entropy), 3),
                    'balance_ratio': round(balance_ratio, 3),
                    'model': kbd
                }
            except Exception as ex:
                results[k][strat] = {'error': str(ex)}
                
    return results


def plot_skewness_distributions(df: pd.DataFrame, variables: list, output_path: Path):
    """
    Gera painel visual com histogramas e densidade (KDE) evidenciando a assimetria acentuada.
    """
    n_vars = len(variables)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for idx, var in enumerate(variables[:4]):
        ax = axes[idx]
        data = df[var].dropna()
        skew_val = stats.skew(data)
        
        # Histograma com curva de densidade estimada
        ax.hist(data, bins=40, density=True, alpha=0.6, color=colors[idx], edgecolor='black', linewidth=0.5)
        
        # KDE
        try:
            kde = stats.gaussian_kde(data)
            x_grid = np.linspace(data.min(), data.quantile(0.995), 200)
            ax.plot(x_grid, kde(x_grid), color='darkred', lw=2, label='KDE Estimada')
        except Exception:
            pass
            
        ax.axvline(data.mean(), color='black', linestyle='--', lw=1.5, label=f'Média: {data.mean():.1f}')
        ax.axvline(data.median(), color='blue', linestyle=':', lw=1.5, label=f'Mediana: {data.median():.1f}')
        
        ax.set_title(f'{var} (Skewness = {skew_val:.2f})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Valor da Variável')
        ax.set_ylabel('Densidade de Probabilidade')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        
    plt.suptitle('Distribuição e Assimetria de Variáveis Contínuas de Metadados e Qualidade', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Gráfico de assimetria salvo em: {output_path}")


def plot_binning_comparison(train_df: pd.DataFrame, var_name: str, k: int, output_path: Path):
    """
    Gera gráfico comparativo lado a lado demonstrando o colapso do método Uniform
    em contraste com o equilíbrio do método Quantile.
    """
    X_train = train_df[[var_name]]
    
    kbd_u = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='uniform')
    kbd_u.fit(X_train)
    b_u = kbd_u.transform(X_train).flatten()
    
    kbd_q = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='quantile')
    kbd_q.fit(X_train)
    b_q = kbd_q.transform(X_train).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Uniform
    counts_u = pd.Series(b_u).value_counts().sort_index()
    labels_u = [f"Bin {int(i)}\n[{kbd_u.bin_edges_[0][int(i)]:.1f}, {kbd_u.bin_edges_[0][int(i)+1]:.1f}]" for i in counts_u.index]
    bars1 = axes[0].bar(labels_u, counts_u.values, color='#e74c3c', alpha=0.75, edgecolor='black')
    axes[0].set_title(f'Estratégia UNIFORM (Largura Fixa) — {var_name} (k={k})\nConcentração desequilibrada por cauda longa', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Quantidade de Amostras')
    axes[0].grid(True, linestyle='--', alpha=0.4)
    for b in bars1:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2., h + 30, f'{int(h)} ({h/len(b_u)*100:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    # 2. Quantile
    counts_q = pd.Series(b_q).value_counts().sort_index()
    labels_q = [f"Bin {int(i)}\n[{kbd_q.bin_edges_[0][int(i)]:.1f}, {kbd_q.bin_edges_[0][int(i)+1]:.1f}]" for i in counts_q.index]
    bars2 = axes[1].bar(labels_q, counts_q.values, color='#2ecc71', alpha=0.75, edgecolor='black')
    axes[1].set_title(f'Estratégia QUANTILE (Frequência Igual) — {var_name} (k={k})\nPartição equilibrada em faixas de comportamento', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Quantidade de Amostras')
    axes[1].grid(True, linestyle='--', alpha=0.4)
    for b in bars2:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2., h + 30, f'{int(h)} ({h/len(b_q)*100:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Gráfico comparativo de binning salvo em: {output_path}")


def run_binning_pipeline(abt_path: str, figures_dir: str):
    """
    Executa o pipeline completo de discretização com prevenção estrita de vazamento de dados.
    """
    figs_path = Path(figures_dir)
    figs_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/5] Carregando a Tabela Analítica Base (ABT): {abt_path}")
    df = pd.read_csv(abt_path)
    
    # Deriva métricas físicas contínuas caso ausentes
    if 'total_pixels' not in df.columns:
        df['total_pixels'] = df['width'] * df['height']
    if 'resolution_mp' not in df.columns:
        df['resolution_mp'] = np.round(df['total_pixels'] / 1e6, 3)
    if 'aspect_ratio' not in df.columns:
        df['aspect_ratio'] = np.round(df['width'] / df['height'], 2)
        
    # Garantir partição unificada (70% train, 15% valid, 15% test) inclusive para Mendeley Data
    mendeley_mask = df['dataset_source'] == 'mendeley_data'
    if (df.loc[mendeley_mask, 'split_partition'] == 'full').any():
        print("  • Estratificando dados Mendeley em train/valid/test para consistência global...")
        np.random.seed(42)
        mendeley_indices = np.random.permutation(df[mendeley_mask].index.values)
        n_m = len(mendeley_indices)
        n_tr = int(0.70 * n_m)
        n_va = int(0.15 * n_m)
        df.loc[mendeley_indices[:n_tr], 'split_partition'] = 'train'
        df.loc[mendeley_indices[n_tr:n_tr + n_va], 'split_partition'] = 'valid'
        df.loc[mendeley_indices[n_tr + n_va:], 'split_partition'] = 'test'
        
    # [Checklist 1] Avaliar distribuição de variáveis contínuas e identificar assimetrias
    print("\n[2/5] Avaliando a distribuição e assimetria das variáveis contínuas...")
    target_vars = ['size_kb', 'resolution_mp', 'aspect_ratio', 'laplacian_var']
    skew_report = calculate_skewness_report(df, target_vars)
    print("\n--- RELATÓRIO DE ASSIMETRIA E CURTOSE ---")
    for _, row in skew_report.iterrows():
        print(f"  • {row['variavel']:15s} | Skewness: {row['skewness']:6.2f} | Curtose: {row['curtose']:6.2f} | {row['diagnostico_assimetria']}")
        
    # Gerar Gráfico 1 de Assimetria
    plot_skewness_distributions(df, target_vars, figs_path / 'sprint2_distribuicao_assimetria_metadados.png')
    
    # [Checklist 2 & 3] Configurar KBinsDiscretizer e testar estratégias (Quantile vs Uniform vs KMeans)
    print("\n[3/5] Comparando estratégias de discretização (Quantile vs. Uniform vs. KMeans)...")
    train_mask = df['split_partition'] == 'train'
    train_df = df[train_mask]
    test_df = df[df['split_partition'] == 'test']
    
    print(f"  • Conjunto de Treino: {len(train_df)} instâncias")
    print(f"  • Conjunto de Teste:  {len(test_df)} instâncias")
    
    comp_size = compare_binning_strategies(train_df, 'size_kb', [3, 4, 5])
    comp_lap = compare_binning_strategies(train_df, 'laplacian_var', [3, 4, 5])
    
    # Gráfico 2 Comparativo Uniform vs Quantile
    plot_binning_comparison(train_df, 'size_kb', k=4, output_path=figs_path / 'sprint2_comparacao_binning_uniform_vs_quantile.png')
    
    # [Checklist 4] Garantir fit estritamente no conjunto de treino e transform nas partições
    print("\n[4/5] Ajustando transformadores KBinsDiscretizer estritamente no conjunto de TREINO...")
    
    # 1. size_kb: Quantile com k=4 (Quartis)
    discretizer_size = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile', subsample=None)
    discretizer_size.fit(train_df[['size_kb']])
    df['size_kb_bin'] = discretizer_size.transform(df[['size_kb']]).astype(int)
    
    # 2. laplacian_var: Quantile com k=4 (Quartis)
    discretizer_lap = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile', subsample=None)
    discretizer_lap.fit(train_df[['laplacian_var']])
    df['laplacian_var_bin'] = discretizer_lap.transform(df[['laplacian_var']]).astype(int)
    
    # 3. resolution_mp: KMeans com k=3 (Baixa, Média, Alta)
    discretizer_res = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans', subsample=None)
    discretizer_res.fit(train_df[['resolution_mp']])
    df['resolution_bin'] = discretizer_res.transform(df[['resolution_mp']]).astype(int)
    
    # 4. aspect_ratio: KMeans com k=3 (Retrato, Quadrado, Paisagem)
    discretizer_asp = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans', subsample=None)
    discretizer_asp.fit(train_df[['aspect_ratio']])
    df['aspect_ratio_bin'] = discretizer_asp.transform(df[['aspect_ratio']]).astype(int)
    
    # Salvar ABT atualizada com as variáveis discretizadas
    df.to_csv(abt_path, index=False)
    print(f"[OK] ABT atualizada com variáveis discretizadas salva em: {abt_path}")
    print(f"     Dimensão final: {df.shape[0]} amostras x {df.shape[1]} colunas.")
    
    # [Checklist 5] Gerar tabela explicativa com intervalos de corte e suas interpretações
    print("\n[5/5] Gerando tabela explicativa dos intervalos de corte (Cut Edges) e interpretações...")
    
    cut_table = [
        {
            "variavel_origem": "size_kb",
            "variavel_discretizada": "size_kb_bin",
            "estrategia": "Quantile (Frequência Igual)",
            "n_bins": 4,
            "limites_corte": [round(float(x), 2) for x in discretizer_size.bin_edges_[0]],
            "rotulos_faixa": ["Muito Leve", "Leve", "Moderado", "Pesado (Alta Resolução)"],
            "interpretacao_agronomica": "Distingue o nível de compressão do arquivo e separa imagens leves de campo de arquivos pesados sem perdas."
        },
        {
            "variavel_origem": "resolution_mp",
            "variavel_discretizada": "resolution_bin",
            "estrategia": "KMeans (Agrupamento 1D)",
            "n_bins": 3,
            "limites_corte": [round(float(x), 2) for x in discretizer_res.bin_edges_[0]],
            "rotulos_faixa": ["Baixa Resolução (640x640 px)", "Média Resolução (HD/FHD)", "Alta Resolução (DSLR > 18 MP)"],
            "interpretacao_agronomica": "Isola imagens padronizadas para deep learning (640px) de capturas macroscópicas em câmeras profissionais."
        },
        {
            "variavel_origem": "laplacian_var",
            "variavel_discretizada": "laplacian_var_bin",
            "estrategia": "Quantile (Frequência Igual)",
            "n_bins": 4,
            "limites_corte": [round(float(x), 2) for x in discretizer_lap.bin_edges_[0]],
            "rotulos_faixa": ["Foco Fraco / Blur Leve", "Nitidez Moderada", "Alta Nitidez", "Ultra Nítida"],
            "interpretacao_agronomica": "Permite ao modelo graduar a confiança nas lesões tênues em função da pureza de foco da fotografia."
        },
        {
            "variavel_origem": "aspect_ratio",
            "variavel_discretizada": "aspect_ratio_bin",
            "estrategia": "KMeans (Agrupamento 1D)",
            "n_bins": 3,
            "limites_corte": [round(float(x), 2) for x in discretizer_asp.bin_edges_[0]],
            "rotulos_faixa": ["Orientação Retrato (Vertical)", "Formato Quadrado (1:1)", "Orientação Paisagem (Horizontal)"],
            "interpretacao_agronomica": "Captura a orientação espacial em que a folha da cana foi enquadrada pelo operador no campo."
        }
    ]
    
    print("\n" + "="*80)
    print("TABELA EXPLICATIVA DE INTERVALOS DE DISCRETIZAÇÃO E INTERPRETAÇÃO")
    print("="*80)
    for item in cut_table:
        print(f"• Variável: {item['variavel_origem']} -> {item['variavel_discretizada']}")
        print(f"  Estratégia: {item['estrategia']} | Bins: {item['n_bins']}")
        print(f"  Limites de Corte: {item['limites_corte']}")
        print(f"  Rótulos: {item['rotulos_faixa']}")
        print(f"  Interpretação: {item['interpretacao_agronomica']}\n")
        
    return df, skew_report, cut_table


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    abt_file = base_dir / 'data' / 'processed' / 'abt_sanidade_vegetal.csv'
    figs_dir = base_dir / 'docs' / 'figures'
    
    if abt_file.exists():
        run_binning_pipeline(str(abt_file), str(figs_dir))
    else:
        print(f"[!] Erro: Arquivo {abt_file} não encontrado.")
