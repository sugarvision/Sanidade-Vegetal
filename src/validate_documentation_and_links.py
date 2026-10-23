"""
Módulo de Validação de Links, Integridade de Documentação e Catálogo SEMMA
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 5 — Consolidação Final do Dicionário de Dados e Catálogo de Produção
Responsável: Heitor Partezani (Engenharia de Machine Learning & IA SVM)

Checklist Coberto:
1. Fechar a versão final dos arquivos dicionario_de_dados.md e dicionario_de_dados.csv.
2. Registrar todos os artefatos gerados pelo projeto (tabelas ABT, pipelines serializados, scripts executáveis e relatórios).
3. Atrelar cada componente do repositório à fase correspondente do framework SEMMA.
4. Validar a integridade dos links, caminhos relativos e documentações técnicas antes do congelamento final do código.
"""

import os
import sys
import re
import json
import time
import urllib.parse
from pathlib import Path

# Assegurar saída UTF-8 no terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ==============================================================================
# 1. MAPEAMENTO OFICIAL DE ARTEFATOS POR FASE DO FRAMEWORK SEMMA
# ==============================================================================

SEMMA_COMPONENTS_MAP = [
    # 1. SAMPLE
    {
        "phase": "Sample",
        "component_name": "Estratégia de Amostragem Estratificada",
        "relative_path": "entregas/Definição da Estratégia de Amostragem/sampling_pipeline.py",
        "type": "Script Executável",
        "description": "Particionamento estratificado agronômico determinístico (70% treino, 15% validação, 15% teste) e cálculo de pesos de classe.",
        "status": "APPROVED"
    },
    {
        "phase": "Sample",
        "component_name": "Metadados de Amostragem e Balanceamento",
        "relative_path": "entregas/Definição da Estratégia de Amostragem/distribuicao_e_balanceamento.md",
        "type": "Documentação Técnica",
        "description": "Especificação da distribuição a priori e controle de vazamento entre talhões e plantas.",
        "status": "APPROVED"
    },
    # 2. EXPLORE
    {
        "phase": "Explore",
        "component_name": "Explorador de Padrões Visuais",
        "relative_path": "src/visual_patterns_analyzer.py",
        "type": "Script Executável",
        "description": "Diagnóstico de resolução, foco (variância laplaciana) e análise cromática exploratória univariada e multivariada.",
        "status": "APPROVED"
    },
    {
        "phase": "Explore",
        "component_name": "Metadados Físicos de Imagens Brutas",
        "relative_path": "data/processed/metadata_raw_images.csv",
        "type": "Tabela de Metadados",
        "description": "Inventário de 13.142 registros fotográficos com hashes SHA-256, dimensões, canais e extensões.",
        "status": "APPROVED"
    },
    # 3. MODIFY
    {
        "phase": "Modify",
        "component_name": "Pipeline de Engenharia de Features Visuais",
        "relative_path": "src/feature_engineering_visual.py",
        "type": "Script Executável",
        "description": "Extração de descritores cromáticos (HSV, ExG, ExR, R/G, ICN) e texturais Haralick GLCM.",
        "status": "APPROVED"
    },
    {
        "phase": "Modify",
        "component_name": "Discretização Estatística e Binning",
        "relative_path": "src/discretizacao_binning.py",
        "type": "Script Executável",
        "description": "Discretização ordinal de atributos físicos contínuos por quantis e clusters k-means.",
        "status": "APPROVED"
    },
    {
        "phase": "Modify",
        "component_name": "Tabela Analítica Base (ABT Sanidade Vegetal)",
        "relative_path": "data/processed/abt_sanidade_vegetal.csv",
        "type": "Tabela ABT Consolidada",
        "description": "Base analítica final tratada com 6.571 instâncias e 35 colunas de atributos preditivos e alvos.",
        "status": "APPROVED"
    },
    {
        "phase": "Modify",
        "component_name": "Pré-processador ColumnTransformer Atômico",
        "relative_path": "models/preprocessor_pipeline.joblib",
        "type": "Pipeline Serializado",
        "description": "Transformador atômico com imputação mediana, StandardScaler e OneHotEncoder integrado.",
        "status": "APPROVED"
    },
    # 4. MODEL
    {
        "phase": "Model",
        "component_name": "Pipeline Atômico SVM Anti-Leakage",
        "relative_path": "src/atomic_svm_pipeline.py",
        "type": "Script Executável / Pipeline",
        "description": "Integração atômica ColumnTransformer + SVC com Platt Scaling e blindagem total anti-leakage.",
        "status": "APPROVED"
    },
    {
        "phase": "Model",
        "component_name": "Estudo de Kernel Trick (RBF e Polinomial)",
        "relative_path": "src/svm_kernel_trick_study.py",
        "type": "Script Executável",
        "description": "Modelagem comparativa entre kernel Linear, RBF Gaussiano e Polinomial graus 2 e 3.",
        "status": "APPROVED"
    },
    {
        "phase": "Model",
        "component_name": "Grid Search Exaustivo de Hiperparâmetros",
        "relative_path": "src/grid_search_svm_optimization.py",
        "type": "Script Executável",
        "description": "Otimização hiperparamétrica exaustiva selecionando a configuração campeã (C=10.0, gamma=0.01).",
        "status": "APPROVED"
    },
    {
        "phase": "Model",
        "component_name": "Estimador SVM Campeão Serializado",
        "relative_path": "models/svm_pipeline_optimal.joblib",
        "type": "Modelo Serializado",
        "description": "Objeto Pipeline Scikit-Learn campeão ajustado e calibrado para inferência contínua em produção.",
        "status": "APPROVED"
    },
    # 5. ASSESS
    {
        "phase": "Assess",
        "component_name": "Auditoria de Partições e Repeated Stratified K-Fold",
        "relative_path": "src/repeated_stratified_kfold_audit.py",
        "type": "Script Executável",
        "description": "Validação cruzada de 15 folds provando ausência de viés amostral e estabilidade estocástica.",
        "status": "APPROVED"
    },
    {
        "phase": "Assess",
        "component_name": "Grid Search Paralelo Resiliente com Telemetria",
        "relative_path": "src/resilient_parallel_grid_search.py",
        "type": "Script Executável",
        "description": "Paralelização n_jobs=-1 com controle de RAM, monitoramento de CPU e checkpoints tolerantes a falhas.",
        "status": "APPROVED"
    },
    {
        "phase": "Assess",
        "component_name": "Auditoria de Linhagem e Métricas Formais",
        "relative_path": "src/audit_metadata_lineage.py",
        "type": "Script Executável",
        "description": "Cálculo formal de Precision, Recall, F1, AUC-ROC > 0.91 e Specificity > 0.93 com rastreabilidade 100%.",
        "status": "APPROVED"
    },
    {
        "phase": "Assess",
        "component_name": "Tabela de Predições Calibradas (Validação e Teste)",
        "relative_path": "data/processed/svm_predicoes_valid_test.csv",
        "type": "Tabela Analítica de Predições",
        "description": "997 instâncias avaliadas contendo y_pred, y_proba_* calibradas, scores de margem e diagnóstico de acerto.",
        "status": "APPROVED"
    },
    {
        "phase": "Assess",
        "component_name": "Auditoria Estrutural Canônica e Reprodutibilidade",
        "relative_path": "src/audit_repository_structure.py",
        "type": "Script Executável",
        "description": "Auditoria das pastas canônicas, congelamento requirements.txt e governança de conformidade.",
        "status": "APPROVED"
    },
    {
        "phase": "Assess",
        "component_name": "Dicionário de Dados Oficial Auditado (v3.0.0)",
        "relative_path": "entregas/Modelagem Inicial e Dicionário de Dados/dicionario_de_dados.md",
        "type": "Governança Documental",
        "description": "Catálogo unificado e auditado de todas as entidades, saídas do modelo, métricas e restrições.",
        "status": "APPROVED"
    }
]


# ==============================================================================
# 2. VALIDATOR DE LINKS, CAMINHOS RELATIVOS E INTEGRIDADE DE MARKDOWN
# ==============================================================================

def validate_all_markdown_links(base_dir: Path) -> dict:
    """
    Varre todos os arquivos Markdown do repositório, extrai links relativos e locais,
    e valida a existência física dos arquivos de destino.
    """
    print("\n" + "="*85)
    print("VALIDAÇÃO AUTOMATIZADA DE INTEGRIDADE DE LINKS E CAMINHOS RELATIVOS")
    print("="*85)
    
    md_files = list(base_dir.rglob('*.md'))
    # Ignorar .git e diretórios virtuais se houver
    md_files = [f for f in md_files if '.git' not in f.parts and '.venv' not in f.parts]
    
    link_regex = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    
    total_links_checked = 0
    valid_links_count = 0
    broken_links = []
    external_links_count = 0
    
    for md_path in md_files:
        content = md_path.read_text(encoding='utf-8', errors='ignore')
        matches = link_regex.findall(content)
        
        for text, link in matches:
            total_links_checked += 1
            link_clean = link.strip()
            
            # 1. Ignorar links web externos (http, https, mailto, ancoras locais #)
            if link_clean.startswith(('http://', 'https://', 'mailto:', '#')):
                external_links_count += 1
                continue
                
            # 2. Tratar scheme file:///
            if link_clean.startswith('file:///'):
                # Converter file:///C:/... para path local
                parsed_url = urllib.parse.urlparse(link_clean)
                target_path_str = urllib.parse.unquote(parsed_url.path)
                # No Windows, remover leading slash se for /C:/...
                if sys.platform == "win32" and target_path_str.startswith('/') and len(target_path_str) > 2 and target_path_str[2] == ':':
                    target_path_str = target_path_str[1:]
                target_path = Path(target_path_str)
            else:
                # 3. Tratar caminho relativo com base no diretório do arquivo .md atual
                # Separar âncoras internas caso existam (ex: file.md#section)
                link_no_anchor = link_clean.split('#')[0]
                if not link_no_anchor:
                    continue
                # Desfazer codificação URL (%20 etc) para checagem precisa no filesystem local
                link_decoded = urllib.parse.unquote(link_no_anchor)
                target_path = (md_path.parent / link_decoded).resolve()
                
            # Checar existência do arquivo ou diretório
            if target_path.exists():
                valid_links_count += 1
            else:
                broken_links.append({
                    "source_file": str(md_path.relative_to(base_dir)),
                    "link_text": text,
                    "target_link": link_clean,
                    "resolved_path": str(target_path)
                })
                
    status = "APPROVED" if len(broken_links) == 0 else "WARNING"
    print(f"* Total de Arquivos Markdown Analisados : {len(md_files)}")
    print(f"* Total de Links Inspecionados          : {total_links_checked}")
    print(f"* Links Locais/Relativos Válidos        : {valid_links_count}")
    print(f"* Links Externos / Âncoras Web          : {external_links_count}")
    print(f"* Links Quebrados Encontrados           : {len(broken_links)}")
    print(f"* Status da Integridade Documental      : {status}")
    
    return {
        "status": status,
        "total_markdown_files": len(md_files),
        "total_links_checked": total_links_checked,
        "valid_internal_links": valid_links_count,
        "external_links_or_anchors": external_links_count,
        "broken_links_count": len(broken_links),
        "broken_links": broken_links
    }


# ==============================================================================
# 3. AUDITORIA DE EXISTÊNCIA E RASTREABILIDADE DOS COMPONENTES SEMMA
# ==============================================================================

def audit_semma_components_integrity(base_dir: Path) -> dict:
    """
    Verifica a existência física e atesta cada componente do mapeamento SEMMA.
    """
    print("\n" + "="*85)
    print("AUDITORIA DA MATRIZ DE COMPONENTES ATRELADOS AO FRAMEWORK SEMMA")
    print("="*85)
    
    audited_components = []
    all_exist = True
    
    for comp in SEMMA_COMPONENTS_MAP:
        comp_path = base_dir / comp["relative_path"]
        exists = comp_path.exists()
        if not exists:
            all_exist = False
        size_kb = round(comp_path.stat().st_size / 1024, 2) if exists and comp_path.is_file() else (len(list(comp_path.glob('*'))) if exists else 0)
        
        audited_components.append({
            "phase": comp["phase"],
            "component_name": comp["component_name"],
            "relative_path": comp["relative_path"],
            "type": comp["type"],
            "exists": exists,
            "size_kb_or_items": size_kb,
            "description": comp["description"],
            "status": "APPROVED" if exists else "MISSING"
        })
        print(f"  [{comp['phase']:<7}] {comp['component_name']:<42} -> {'[OK]' if exists else '[FALHA]'} ({comp['relative_path']})")
        
    return {
        "status": "APPROVED" if all_exist else "FAILED",
        "total_semma_components": len(audited_components),
        "components": audited_components
    }


# ==============================================================================
# 4. EXPORTAÇÃO DO MANIFESTO MESTRE DE PRODUÇÃO
# ==============================================================================

def export_production_artifacts_catalog(
    semma_audit: dict,
    links_audit: dict,
    base_dir: Path
):
    models_dir = base_dir / "models"
    sprint_dir = base_dir / "entregas" / "Consolidacao Final do Dicionario de Dados e Catalogo de Producao"
    models_dir.mkdir(parents=True, exist_ok=True)
    sprint_dir.mkdir(parents=True, exist_ok=True)
    
    catalog_payload = {
        "catalog_title": "Catálogo Mestre de Produção e Matriz de Rastreabilidade SEMMA (Sprint 5)",
        "project": "Sanidade-Vegetal (SugarVision)",
        "responsible": "Heitor Partezani (Engenharia de Machine Learning & IA SVM)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "3.0.0",
        "semma_lifecycle_audit": semma_audit,
        "documentation_links_integrity": links_audit,
        "overall_catalog_status": "APPROVED" if (semma_audit["status"] == "APPROVED" and links_audit["status"] == "APPROVED") else "FAILED"
    }
    
    p_models = models_dir / "production_artifacts_catalog.json"
    p_sprint = sprint_dir / "production_artifacts_catalog.json"
    
    with open(p_models, 'w', encoding='utf-8') as f:
        json.dump(catalog_payload, f, indent=2, ensure_ascii=False)
        
    with open(p_sprint, 'w', encoding='utf-8') as f:
        json.dump(catalog_payload, f, indent=2, ensure_ascii=False)
        
    print(f"\n* [Manifesto MLOps Salvo]: {p_models}")
    return catalog_payload


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    semma_audit = audit_semma_components_integrity(base_dir)
    links_audit = validate_all_markdown_links(base_dir)
    export_production_artifacts_catalog(semma_audit, links_audit, base_dir)
