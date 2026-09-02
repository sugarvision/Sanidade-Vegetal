"""
Script de Análise de Padrões Visuais, Qualidade e Diagnóstico Estatístico
Projeto: Sanidade-Vegetal (SugarVision)
Responsável: Cesar (Exploração de Imagens e Padrões Visuais)
Sprint: 1 (SEMMA - Explore)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_visual_analysis(metadata_path: str, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/4] Carregando metadados de: {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    print(f"Total de registros carregados: {len(df)}")
    print(f"Colunas disponíveis: {list(df.columns)}")
    
    # 1. Gráfico de Distribuição de Classes e Fontes de Dados
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    class_counts = df['class_label'].value_counts()
    colors = ['#2ecc71' if c == 'HEALTHY' else '#e67e22' if c == 'RUST' else '#3498db' for c in class_counts.index]
    
    axes[0].barh(class_counts.index[::-1], class_counts.values[::-1], color=colors[::-1], edgecolor='black', alpha=0.85)
    axes[0].set_title('Distribuição de Amostras por Classe Fitopatológica', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Quantidade de Imagens')
    for i, v in enumerate(class_counts.values[::-1]):
        axes[0].text(v + 15, i, str(v), va='center', fontweight='bold')
        
    source_counts = df['dataset_source'].value_counts()
    axes[1].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', 
               colors=['#34495e', '#16a085'], startangle=140, explode=(0.05, 0), shadow=True)
    axes[1].set_title('Proporção por Fonte de Dados (Dataset Source)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig_path1 = output_path / 'distribuicao_classes_fontes.png'
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"[OK] Grafico salvo: {fig_path1}")
    
    # 2. Gráfico de Resolução e Tamanho de Arquivo (KB)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma de Tamanho em KB
    axes[0].hist(df['size_kb'], bins=40, color='#8e44ad', edgecolor='black', alpha=0.7)
    axes[0].set_title('Distribuição do Tamanho dos Arquivos (KB)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tamanho (KB)')
    axes[0].set_ylabel('Frequência')
    axes[0].axvline(df['size_kb'].median(), color='red', linestyle='--', label=f'Mediana: {df["size_kb"].median():.1f} KB')
    axes[0].legend()
    
    # Scatter Largura x Altura por Fonte
    for source, group in df.groupby('dataset_source'):
        axes[1].scatter(group['width'], group['height'], label=source, alpha=0.5, s=20)
    axes[1].set_title('Resolução Espacial das Imagens (Largura x Altura)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Largura (pixels)')
    axes[1].set_ylabel('Altura (pixels)')
    axes[1].legend()
    
    plt.tight_layout()
    fig_path2 = output_path / 'analise_resolucao_tamanho.png'
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"[OK] Grafico salvo: {fig_path2}")
    
    # 3. Métricas de Resumo Estatístico para a Documentação
    summary_report = {
        "total_images": len(df),
        "total_classes": df['class_label'].nunique(),
        "classes_list": df['class_label'].unique().tolist(),
        "healthy_count": int((df['class_label'] == 'HEALTHY').sum()),
        "rust_count": int((df['class_label'] == 'RUST').sum()),
        "other_diseases_count": int(((df['class_label'] != 'HEALTHY') & (df['class_label'] != 'RUST')).sum()),
        "mean_size_kb": float(df['size_kb'].mean()),
        "std_size_kb": float(df['size_kb'].std()),
        "min_size_kb": float(df['size_kb'].min()),
        "max_size_kb": float(df['size_kb'].max()),
        "resolutions": df.groupby(['width', 'height']).size().to_dict()
    }
    
    print("\n--- RESUMO QUANTITATIVO DE IMAGENS ---")
    for k, v in summary_report.items():
        print(f"  • {k}: {v}")
        
    return summary_report

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    meta_csv = base_dir / 'data' / 'processed' / 'metadata_raw_images.csv'
    figures_dir = base_dir / 'docs' / 'figures'
    
    if meta_csv.exists():
        run_visual_analysis(str(meta_csv), str(figures_dir))
    else:
        print(f"[!] Erro: Arquivo de metadados não encontrado em {meta_csv}")
