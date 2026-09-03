"""
Módulo de Engenharia de Features Visuais: Cor (HSV/LAB/ExG) e Textura Haralick (GLCM)
Projeto: Sanidade-Vegetal (SugarVision)
Responsável: Cesar (Lead Técnico & Visão Computacional)
Sprint: 2 (SEMMA - Modify)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

def compute_color_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera métricas complementares de cor e índices espectrais agronômicos
    baseados no espaço HSV e índices de vegetação RGB.
    """
    df_feat = df.copy()
    
    # Razão Vermelho / Verde (R/G) estimada a partir de ExG e ExR
    # ExG = 2G - R - B, ExR = 1.4R - G
    # Aproximação espectral consistente para o banco tabular
    if 'rg_ratio' not in df_feat.columns:
        # Se não existir explicitamente, derivamos da relação inversa de ExG
        df_feat['rg_ratio'] = np.clip(1.0 + (df_feat['exr_index'] - df_feat['exg_index']) / 100.0, 0.2, 4.0)
    
    # Índice de Clorose e Necrose Foliar (ICN)
    df_feat['indice_clorose_necrose'] = df_feat['rg_ratio'] / (df_feat['std_saturation'] + 1e-4)
    
    # Variação Relativa de Matiz (Hue Dispersion)
    df_feat['hue_dispersion'] = df_feat['mean_hue'] * df_feat['std_saturation']
    
    return df_feat

def compute_glcm_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida as propriedades de textura de Haralick (GLCM).
    """
    df_feat = df.copy()
    
    # Dissimilaridade estimada a partir de GLCM Contrast e Homogeneidade
    # Dissimilarity = sqrt(Contrast) * (1 - Homogeneity + 0.1)
    if 'glcm_dissimilarity' not in df_feat.columns:
        df_feat['glcm_dissimilarity'] = np.sqrt(np.maximum(df_feat['glcm_contrast'], 0)) * (1.1 - df_feat['glcm_homogeneity'])
    
    # Energia de Haralick (Uniformidade Angular / ASM)
    if 'glcm_energy' not in df_feat.columns:
        df_feat['glcm_energy'] = np.sqrt(np.clip(df_feat['glcm_homogeneity'] ** 2, 0, 1))
        
    # Índice de Rugosidade Foliar Pústula (IRFP) = Contraste * Dissimilaridade / Homogeneidade
    df_feat['indice_rugosidade_pustula'] = (df_feat['glcm_contrast'] * df_feat['glcm_dissimilarity']) / (df_feat['glcm_homogeneity'] + 1e-3)
    
    return df_feat

def run_feature_analysis_and_plots(abt_path: str, figures_dir: str):
    """
    Executa a análise estatística completa, gera visualizações diagnósticas
    e salva os gráficos de alta resolução para a documentação da Sprint 2.
    """
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/4] Carregando dados da ABT: {abt_path}")
    df = pd.read_csv(abt_path)
    print(f"Total de registros: {len(df)} amostras")
    
    print("[2/4] Aplicando engenharia de features cromáticas e texturais...")
    df = compute_color_descriptors(df)
    df = compute_glcm_descriptors(df)
    
    # Separação por classes principais
    healthy = df[df['class_label'] == 'HEALTHY']
    rust = df[df['class_label'] == 'RUST']
    others = df[(df['class_label'] != 'HEALTHY') & (df['class_label'] != 'RUST')]
    
    print(f"  • Saudável: {len(healthy)} | Ferrugem: {len(rust)} | Outras Patologias: {len(others)}")
    
    # -------------------------------------------------------------
    # Gráfico 1: Análise Cromática (HSV, ExG, ExR, R/G Ratio)
    # -------------------------------------------------------------
    print("[3/4] Gerando Gráfico 1: Análise Cromática e Índices Espectrais...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1.1 Boxplot de ExG (Excesso de Verde)
    data_exg = [healthy['exg_index'], rust['exg_index'], others['exg_index']]
    bp1 = axes[0, 0].boxplot(data_exg, patch_artist=True, labels=['Saudável', 'Ferrugem', 'Outras'])
    colors = ['#2ecc71', '#e67e22', '#3498db']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 0].set_title('Índice de Excesso de Verde (ExG = 2G - R - B)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Valor do Índice ExG')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    # 1.2 Scatter Matiz (Hue) vs Saturação (Std Saturation)
    axes[0, 1].scatter(healthy['mean_hue'], healthy['std_saturation'], c='#2ecc71', label='Saudável', alpha=0.4, s=20)
    axes[0, 1].scatter(others['mean_hue'], others['std_saturation'], c='#3498db', label='Outras Patologias', alpha=0.4, s=20)
    axes[0, 1].scatter(rust['mean_hue'], rust['std_saturation'], c='#e67e22', label='Ferrugem (Puccinia spp.)', alpha=0.6, s=25)
    axes[0, 1].set_title('Espaço HSV: Matiz Médio (Hue) x Dispersão de Saturação', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Matiz Médio (Hue em graus/escala normalizada)')
    axes[0, 1].set_ylabel('Desvio Padrão da Saturação (Contraste de Lesão)')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    # 1.3 Histograma Comparativo de ExR (Excesso de Vermelho)
    axes[1, 0].hist(healthy['exr_index'], bins=30, alpha=0.5, color='#2ecc71', label='Saudável', density=True)
    axes[1, 0].hist(rust['exr_index'], bins=30, alpha=0.5, color='#e67e22', label='Ferrugem', density=True)
    axes[1, 0].set_title('Distribuição de Densidade do Índice ExR (Excesso de Vermelho)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('ExR (1.4R - G)')
    axes[1, 0].set_ylabel('Densidade de Probabilidade')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    
    # 1.4 Razão Espectral R/G vs Índice de Clorose/Necrose
    axes[1, 1].scatter(healthy['rg_ratio'], healthy['indice_clorose_necrose'], c='#2ecc71', label='Saudável', alpha=0.4, s=20)
    axes[1, 1].scatter(rust['rg_ratio'], rust['indice_clorose_necrose'], c='#e67e22', label='Ferrugem', alpha=0.5, s=25)
    axes[1, 1].set_title('Razão Espectral R/G x Índice Composto de Clorose/Necrose', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Razão R/G (Vermelho / Verde)')
    axes[1, 1].set_ylabel('Índice de Clorose/Necrose (ICN)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig1_path = fig_dir / 'sprint2_analise_cromaticas_hsv_exg.png'
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[OK] Salvo: {fig1_path}")
    
    # -------------------------------------------------------------
    # Gráfico 2: Análise Textural Haralick (GLCM) e Rugosidade
    # -------------------------------------------------------------
    print("[4/4] Gerando Gráfico 2: Texturas Haralick GLCM e Rugosidade...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 2.1 Contraste GLCM por Classe
    data_contrast = [healthy['glcm_contrast'], rust['glcm_contrast'], others['glcm_contrast']]
    bp2 = axes[0, 0].boxplot(data_contrast, patch_artist=True, labels=['Saudável', 'Ferrugem', 'Outras'])
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 0].set_title('Contraste Haralick (GLCM Contrast)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Contraste Médio Multidirecional')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    # 2.2 Homogeneidade GLCM por Classe
    data_homo = [healthy['glcm_homogeneity'], rust['glcm_homogeneity'], others['glcm_homogeneity']]
    bp3 = axes[0, 1].boxplot(data_homo, patch_artist=True, labels=['Saudável', 'Ferrugem', 'Outras'])
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 1].set_title('Homogeneidade Haralick (GLCM Homogeneity)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Homogeneidade do Limbo Foliar')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    # 2.3 Scatter Contraste x Homogeneidade (Separabilidade de Textura)
    axes[1, 0].scatter(healthy['glcm_contrast'], healthy['glcm_homogeneity'], c='#2ecc71', label='Saudável (Cutícula Lisa)', alpha=0.4, s=20)
    axes[1, 0].scatter(rust['glcm_contrast'], rust['glcm_homogeneity'], c='#e67e22', label='Ferrugem (Pústulas Rugosas)', alpha=0.5, s=25)
    axes[1, 0].set_title('Espaço Textural: Contraste x Homogeneidade GLCM', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Contraste Haralick (Intensidade de Variação)')
    axes[1, 0].set_ylabel('Homogeneidade (Uniformidade)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    
    # 2.4 Índice de Rugosidade Foliar de Pústula (IRFP)
    axes[1, 1].hist(healthy['indice_rugosidade_pustula'], bins=35, alpha=0.6, color='#2ecc71', label='Saudável', density=True)
    axes[1, 1].hist(rust['indice_rugosidade_pustula'], bins=35, alpha=0.6, color='#e67e22', label='Ferrugem', density=True)
    axes[1, 1].set_title('Distribuição do Índice Composto de Rugosidade Foliar (IRFP)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('IRFP = (Contraste * Dissimilaridade) / Homogeneidade')
    axes[1, 1].set_ylabel('Densidade')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig2_path = fig_dir / 'sprint2_analise_texturas_glcm_haralick.png'
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[OK] Salvo: {fig2_path}")
    
    # -------------------------------------------------------------
    # 3. Testes Estatísticos de Validação de Hipóteses (Mann-Whitney & Cohen's d)
    # -------------------------------------------------------------
    print("\n--- TESTES ESTATÍSTICOS DE HIPÓTESES AGRONÔMICAS (CESAR) ---")
    
    def calculate_effect_size(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_se
    
    stat_exg, p_exg = stats.mannwhitneyu(healthy['exg_index'], rust['exg_index'], alternative='two-sided')
    d_exg = calculate_effect_size(healthy['exg_index'], rust['exg_index'])
    
    stat_contrast, p_contrast = stats.mannwhitneyu(rust['glcm_contrast'], healthy['glcm_contrast'], alternative='two-sided')
    d_contrast = calculate_effect_size(rust['glcm_contrast'], healthy['glcm_contrast'])
    
    stat_homo, p_homo = stats.mannwhitneyu(healthy['glcm_homogeneity'], rust['glcm_homogeneity'], alternative='two-sided')
    d_homo = calculate_effect_size(healthy['glcm_homogeneity'], rust['glcm_homogeneity'])
    
    stats_results = {
        "ExG_Index": {"p_value": float(p_exg), "cohens_d": float(d_exg), "healthy_mean": float(healthy['exg_index'].mean()), "rust_mean": float(rust['exg_index'].mean())},
        "GLCM_Contrast": {"p_value": float(p_contrast), "cohens_d": float(d_contrast), "rust_mean": float(rust['glcm_contrast'].mean()), "healthy_mean": float(healthy['glcm_contrast'].mean())},
        "GLCM_Homogeneity": {"p_value": float(p_homo), "cohens_d": float(d_homo), "healthy_mean": float(healthy['glcm_homogeneity'].mean()), "rust_mean": float(rust['glcm_homogeneity'].mean())}
    }
    
    for metric, res in stats_results.items():
        print(f"  • {metric}: p-valor = {res['p_value']:.4e} | Cohen's d = {res['cohens_d']:.3f}")
        
    return df, stats_results

if __name__ == '__main__':
    base_path = Path(__file__).resolve().parent.parent
    abt_csv = base_path / 'data' / 'processed' / 'abt_sanidade_vegetal.csv'
    figs_path = base_path / 'docs' / 'figures'
    
    if abt_csv.exists():
        run_feature_analysis_and_plots(str(abt_csv), str(figs_path))
    else:
        print(f"[!] Erro: Arquivo {abt_csv} não encontrado.")
