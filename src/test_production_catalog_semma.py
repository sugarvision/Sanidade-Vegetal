"""
Testes Unitários da Consolidação do Dicionário de Dados e Catálogo de Produção SEMMA
Sprint: 5 — Consolidação Final do Dicionário de Dados e Catálogo de Produção
Responsável: Heitor Partezani (Engenheiro de Machine Learning & IA SVM)

Checklist Validado:
1. Fechar a versão final dos arquivos dicionario_de_dados.md e dicionario_de_dados.csv.
2. Registrar todos os artefatos gerados pelo projeto (tabelas ABT, pipelines serializados, scripts executáveis e relatórios).
3. Atrelar cada componente do repositório à fase correspondente do framework SEMMA.
4. Validar a integridade dos links, caminhos relativos e documentações técnicas antes do congelamento final do código.
"""

import unittest
import json
import csv
from pathlib import Path

from src.validate_documentation_and_links import (
    SEMMA_COMPONENTS_MAP,
    audit_semma_components_integrity,
    validate_all_markdown_links
)


class TestProductionCatalogSEMMA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent
        dict_dir = cls.base_dir / "entregas" / "Modelagem Inicial e Dicionário de Dados"
        if not dict_dir.exists(): dict_dir = cls.base_dir / "Modelagem Inicial e Dicionário de Dados"
        cls.dict_md = dict_dir / "dicionario_de_dados.md"
        cls.dict_csv = dict_dir / "dicionario_de_dados.csv"
        cls.catalog_json = cls.base_dir / "models" / "production_artifacts_catalog.json"

    def test_checklist_item1_dicionario_version_and_structure(self):
        """Verifica se o dicionário de dados foi promovido a v3.0.0 com Seção 12 e entidade ARTEFATO_PRODUCAO."""
        self.assertTrue(self.dict_md.exists(), "dicionario_de_dados.md não encontrado")
        self.assertTrue(self.dict_csv.exists(), "dicionario_de_dados.csv não encontrado")

        content_md = self.dict_md.read_text(encoding='utf-8')
        self.assertIn("Versão:** 3.0.0", content_md, "Versão 3.0.0 ausente no dicionario_de_dados.md")
        self.assertIn("## 12. Catálogo Oficial de Artefatos de Produção e Matriz de Rastreabilidade SEMMA", content_md, "Seção 12 ausente no markdown")

        # Verificar se o CSV contém a entidade ARTEFATO_PRODUCAO
        with open(self.dict_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            entities = {row.get('entidade') for row in reader}
            self.assertIn("ARTEFATO_PRODUCAO", entities, "Entidade ARTEFATO_PRODUCAO ausente no dicionario_de_dados.csv")

    def test_checklist_item2_artifacts_registration_and_presence(self):
        """Verifica se os artefatos canônicos gerados pelo projeto existem fisicamente em disco."""
        critical_artifacts = [
            "data/processed/abt_sanidade_vegetal.csv",
            "data/processed/abt_features_modelagem.parquet",
            "data/processed/metadata_raw_images.csv",
            "data/processed/svm_predicoes_valid_test.csv",
            "models/svm_pipeline_optimal.joblib",
            "models/preprocessor_pipeline.joblib",
            "models/gridsearch_checkpoint.json",
            "models/production_artifacts_catalog.json",
            "entregas/Consolidacao Final do Dicionario de Dados e Catalogo de Producao/relatorio_consolidacao_dicionario_e_catalogo_producao.md"
        ]
        for rel_path in critical_artifacts:
            full_path = self.base_dir / rel_path
            self.assertTrue(full_path.exists(), f"Artefato crítico não encontrado: {rel_path}")

    def test_checklist_item3_semma_matrix_integrity(self):
        """Verifica se cada componente está atrelado às 5 fases do SEMMA e se todos estão homologados."""
        semma_audit = audit_semma_components_integrity(self.base_dir)
        self.assertEqual(semma_audit["status"], "APPROVED", "Auditoria SEMMA não foi aprovada")
        
        phases = {c["phase"] for c in semma_audit["components"]}
        expected_phases = {"Sample", "Explore", "Modify", "Model", "Assess"}
        self.assertEqual(phases, expected_phases, "Fases SEMMA incompletas no catálogo")
        
        # Verificar se não há componentes com status MISSING
        missing = [c for c in semma_audit["components"] if c["status"] != "APPROVED"]
        self.assertEqual(len(missing), 0, f"Componentes SEMMA ausentes: {missing}")

    def test_checklist_item4_markdown_links_integrity_zero_broken(self):
        """Valida a integridade dos links em todos os markdowns antes do congelamento final do código."""
        links_audit = validate_all_markdown_links(self.base_dir)
        self.assertEqual(links_audit["status"], "APPROVED", "Status da integridade documental não é APPROVED")
        self.assertEqual(len(links_audit["broken_links"]), 0, f"Existem links quebrados: {links_audit['broken_links']}")
        self.assertGreater(links_audit["valid_internal_links"], 200, "Contagem anormalmente baixa de links válidos")

    def test_production_catalog_json_validity(self):
        """Valida a integridade do manifesto JSON exportado em models/."""
        self.assertTrue(self.catalog_json.exists(), "production_artifacts_catalog.json não existe")
        with open(self.catalog_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(data.get("overall_catalog_status"), "APPROVED")
        self.assertEqual(data.get("version"), "3.0.0")


if __name__ == '__main__':
    unittest.main()
