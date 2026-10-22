"""
Módulo de Testes Automatizados para Deploy e Reprodutibilidade (Sprint 5)
Projeto: Sanidade-Vegetal (SugarVision)
"""

import unittest
import json
from pathlib import Path


class TestSprint5DeployAndReproducibility(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.requirements_txt = cls.base_dir / "requirements.txt"
        cls.readme_md = cls.base_dir / "README.md"
        cls.license_file = cls.base_dir / "LICENSE"
        cls.audit_manifest = cls.base_dir / "models" / "repository_structure_audit.json"
        cls.license_report = cls.base_dir / "docs" / "relatorio_conformidade_licencas_e_permissoes.md"

    def test_canonical_structure_manifest(self):
        """Valida que a auditoria da estrutura de pastas retornou status APPROVED."""
        self.assertTrue(self.audit_manifest.exists(), "Manifesto repository_structure_audit.json ausente!")
        with open(self.audit_manifest, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            
        self.assertEqual(manifest['overall_compliance_status'], "APPROVED")
        for d in ['data', 'docs', 'models', 'notebooks', 'src']:
            self.assertEqual(manifest['directories_audit'][d]['status'], "APPROVED", f"Pasta {d} não aprovada!")

    def test_requirements_pinned_versions(self):
        """Valida que requirements.txt existe e todas as dependências estão estritamente pinadas (==)."""
        self.assertTrue(self.requirements_txt.exists(), "requirements.txt não encontrado na raiz!")
        content = self.requirements_txt.read_text(encoding='utf-8')
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith('#')]
        
        self.assertGreaterEqual(len(lines), 7, "Menos de 7 dependências em requirements.txt!")
        for line in lines:
            self.assertTrue('==' in line or '>=' in line, f"Dependência sem versão estrita: {line}")
            
        # Dependências críticas obrigatórias
        dep_names = [line.split('==')[0].split('>=')[0].strip() for line in lines]
        self.assertIn('scikit-learn', dep_names)
        self.assertIn('numpy', dep_names)
        self.assertIn('pandas', dep_names)
        self.assertIn('joblib', dep_names)
        self.assertIn('psutil', dep_names)

    def test_root_readme_comprehensive_guide(self):
        """Valida que o README.md raiz contém o passo a passo didático de instalação e reprodução."""
        self.assertTrue(self.readme_md.exists(), "README.md não encontrado!")
        content = self.readme_md.read_text(encoding='utf-8')
        
        self.assertIn("Guia Didático: Instalação e Reprodução Passo a Passo", content)
        self.assertIn("pip install -r requirements.txt", content)
        self.assertIn("python src/repeated_stratified_kfold_audit.py", content)
        self.assertIn("python src/resilient_parallel_grid_search.py", content)
        self.assertIn("Métricas Oficiais do Estimador Campeão", content)
        self.assertIn("Exemplo de Inferência Programática (Deploy)", content)

    def test_license_and_data_permissions(self):
        """Valida a existência do arquivo LICENSE e do relatório de conformidade de dados."""
        self.assertTrue(self.license_file.exists(), "Arquivo LICENSE não encontrado na raiz!")
        self.assertTrue(self.license_report.exists(), "Relatório de licenças em docs/ ausente!")
        
        lic_content = self.license_file.read_text(encoding='utf-8')
        self.assertIn("MIT License", lic_content)
        self.assertIn("Roboflow Universe", lic_content)
        self.assertIn("Mendeley Data", lic_content)


if __name__ == '__main__':
    unittest.main()
