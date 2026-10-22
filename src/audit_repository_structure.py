"""
Módulo de Auditoria da Estrutura de Pastas e Conformidade do Repositório
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 5 — Auditoria Final, Padronização de Deploy e Reprodutibilidade
Responsável: Heitor Partezani (Engenharia de Machine Learning & IA SVM)

Checklist Coberto:
1. Auditar a estrutura de pastas do repositório (data/, docs/, models/, notebooks/, src/) garantindo conformidade.
2. Congelar e documentar o arquivo de dependências exatas requirements.txt com versões estritas.
3. Criar o guia didático e passo a passo de reprodução e instalação no README.md raiz do repositório.
4. Auditar a conformidade de licenças de código aberto e permissões de dados das bases utilizadas.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

# Assegurar saída UTF-8 no terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def audit_repository_canonical_structure(base_dir: Path) -> dict:
    """
    Verifica e audita formalmente a presença e integridade das 5 pastas obrigatórias:
    - data/ (raw/ e processed/ com ABT e metadados)
    - docs/ (figures/ e relatórios técnicos)
    - models/ (modelos joblib, metadados MLOps e checkpoints)
    - notebooks/ (notebooks interativos das fases SEMMA)
    - src/ (código modular de produção e suites de teste)
    """
    print("\n" + "="*85)
    print("AUDITORIA FORMAL DA ESTRUTURA CANÔNICA DE PASTAS DO REPOSITÓRIO")
    print("="*85)
    
    canonical_dirs = ['data', 'docs', 'models', 'notebooks', 'src']
    audit_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_directory": str(base_dir),
        "directories_audit": {},
        "critical_artifacts_audit": {},
        "cleanliness_audit": {},
        "test_suite_audit": {}
    }
    
    # 1. Auditoria das 5 Pastas Canônicas
    all_dirs_exist = True
    for d_name in canonical_dirs:
        d_path = base_dir / d_name
        exists = d_path.exists() and d_path.is_dir()
        if not exists:
            all_dirs_exist = False
        files_count = len(list(d_path.rglob('*'))) if exists else 0
        audit_report["directories_audit"][d_name] = {
            "status": "APPROVED" if exists else "MISSING",
            "path": str(d_path),
            "files_and_subdirs_count": files_count
        }
        print(f"  * Diretório [{d_name:<9}]: {'[OK] Conforme' if exists else '[FALHA] Ausente'} ({files_count} itens)")

    # 2. Auditoria de Artefatos Críticos em cada pasta
    critical_files = {
        "data_abt_sanidade": base_dir / "data" / "processed" / "abt_sanidade_vegetal.csv",
        "data_metadata_raw": base_dir / "data" / "processed" / "metadata_raw_images.csv",
        "data_predictions": base_dir / "data" / "processed" / "svm_predicoes_valid_test.csv",
        "models_optimal": base_dir / "models" / "svm_pipeline_optimal.joblib",
        "models_lineage_catalog": base_dir / "models" / "dataset_lineage_and_metrics_catalog.json",
        "models_computational_efficiency": base_dir / "models" / "computational_efficiency_metrics.json",
        "models_repeated_kfold": base_dir / "models" / "repeated_kfold_audit_metadata.json",
        "docs_figures_dir": base_dir / "docs" / "figures",
        "src_atomic_svm_pipeline": base_dir / "src" / "atomic_svm_pipeline.py",
        "src_repeated_stratified_kfold": base_dir / "src" / "repeated_stratified_kfold_audit.py",
        "src_resilient_parallel_grid_search": base_dir / "src" / "resilient_parallel_grid_search.py",
        "src_audit_metadata_lineage": base_dir / "src" / "audit_metadata_lineage.py"
    }
    
    all_critical_exist = True
    print("\n* Auditoria de Integridade de Artefatos Críticos:")
    for key, p in critical_files.items():
        exists = p.exists()
        if not exists:
            all_critical_exist = False
        size_kb = round(p.stat().st_size / 1024, 2) if exists and p.is_file() else (len(list(p.glob('*'))) if exists else 0)
        audit_report["critical_artifacts_audit"][key] = {
            "status": "APPROVED" if exists else "MISSING",
            "path": str(p),
            "size_kb_or_items": size_kb
        }
        print(f"  - {key:<35}: {'[OK] Presente' if exists else '[FALHA] Ausente'} ({size_kb} KB/itens)")

    # 3. Auditoria de Limpeza (Sem resíduos .tmp ou arquivos corrompidos)
    tmp_files = list(base_dir.rglob('*.tmp'))
    audit_report["cleanliness_audit"] = {
        "status": "APPROVED" if len(tmp_files) == 0 else "WARNING",
        "orphan_tmp_files_count": len(tmp_files),
        "orphan_tmp_files": [str(f) for f in tmp_files]
    }
    print(f"\n* Auditoria de Limpeza de Workspace: {len(tmp_files)} arquivos .tmp órfãos encontrados.")

    # 4. Execução da Suíte Completa de Testes Automatizados
    print("\n* Disparando Execução da Suíte Completa de Testes Unitários:")
    t0 = time.perf_counter()
    res = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "src", "-p", "test_*.py"],
        cwd=str(base_dir),
        capture_output=True,
        text=True
    )
    test_duration = time.perf_counter() - t0
    tests_passed = (res.returncode == 0)
    
    audit_report["test_suite_audit"] = {
        "status": "APPROVED" if tests_passed else "FAILED",
        "return_code": res.returncode,
        "execution_time_seconds": round(test_duration, 3),
        "output": res.stderr.strip() or res.stdout.strip()
    }
    print(f"  - Resultado dos Testes: {'[OK] 100% APROVADOS' if tests_passed else '[FALHA] Erro na execução'}")
    print(f"  - Tempo de Execução   : {test_duration:.3f} segundos")
    
    # Status Geral
    overall_status = "APPROVED" if (all_dirs_exist and all_critical_exist and tests_passed) else "FAILED"
    audit_report["overall_compliance_status"] = overall_status
    
    print("\n" + "="*85)
    print(f"STATUS FINAL DA AUDITORIA DE CONFORMIDADE: {overall_status}")
    print("="*85)
    
    return audit_report


def export_repository_audit_manifest(report: dict, base_dir: Path):
    """Salva o manifesto formal de conformidade em models/ e na pasta da sprint."""
    models_dir = base_dir / "models"
    sprint_dir = base_dir / "Auditoria Final de Codigo Deploy e Congelamento de Dependencias"
    models_dir.mkdir(parents=True, exist_ok=True)
    sprint_dir.mkdir(parents=True, exist_ok=True)
    
    p_models = models_dir / "repository_structure_audit.json"
    p_sprint = sprint_dir / "repository_structure_audit.json"
    
    with open(p_models, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    with open(p_sprint, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"* [Manifesto Salvo]: {p_models}")


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    report = audit_repository_canonical_structure(base_dir)
    export_repository_audit_manifest(report, base_dir)
