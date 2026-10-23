"""
Módulo de Validação de Schema e Governança de Inferência em Produção (Sprint 5)
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 5 — Framework SEMMA (Fase: Deploy & MLOps Governance)
Responsável: Elisa (Engenharia de Machine Learning & MLOps)

Checklist Coberto:
1. Definir o schema formal de entrada de dados (tipos, domínios e faixas aceitáveis das features de imagem).
2. Implementar validação automática de dados de entrada na predição para rejeitar amostras inconsistentes ou corrompidas.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
import joblib

# Configuração de Logger MLOps
logger = logging.getLogger("SugarVisionSchemaValidator")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class SchemaValidationError(Exception):
    """Exceção levantada quando amostras de dados violam o contrato formal de entrada."""
    def __init__(self, message: str, violations: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.violations = violations or []


class SchemaValidator:
    """
    Validador de contrato formal de dados para produção.
    Garante integridade de tipos, limites físicos de domínio e categorias autorizadas.
    """
    
    def __init__(self, schema_path: Optional[Union[str, Path]] = None):
        base_dir = Path(__file__).resolve().parent.parent
        self.schema_path = Path(schema_path) if schema_path else (base_dir / "models" / "model_schema_contract.json")
        
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Arquivo de contrato de schema não encontrado em: {self.schema_path}")
            
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            self.contract = json.load(f)
            
        input_contract = self.contract.get("input_features_contract", {})
        self.numerical_rules = input_contract.get("numerical_features", {})
        self.categorical_rules = input_contract.get("categorical_features", {})
        
        self.required_numerical_features = list(self.numerical_rules.keys())
        self.required_categorical_features = list(self.categorical_rules.keys())
        self.all_required_features = self.required_numerical_features + self.required_categorical_features
        
        target_contract = self.contract.get("target_contract", {})
        self.classes_mapping = target_contract.get("classes_mapping", {})

    def validate_batch(
        self,
        df: pd.DataFrame,
        raise_on_error: bool = False,
        quarantine: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
        """
        Valida um lote de dados para predição.
        
        Retorna:
        --------
        valid_df : pd.DataFrame
            Subconjunto de registros em conformidade total com o contrato.
        quarantine_df : pd.DataFrame
            Subconjunto de registros rejeitados por violação de contrato.
        violations_report : List[Dict[str, Any]]
            Relatório auditável detalhando índice da linha, coluna e motivo da rejeição.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Entrada deve ser um pandas.DataFrame, recebido {type(df)}")
            
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), []

        # 1. Checagem de colunas obrigatórias
        missing_columns = [col for col in self.all_required_features if col not in df.columns]
        if missing_columns:
            msg = f"Violação estrutural crítica: colunas obrigatórias ausentes no lote: {missing_columns}"
            logger.error(msg)
            if raise_on_error:
                raise SchemaValidationError(msg, [{"type": "missing_columns", "columns": missing_columns}])
            # Se colunas essenciais faltam para todo o lote, o lote inteiro vai para quarentena
            violations_report = [{
                "row_index": idx,
                "reasons": [f"Coluna contratual ausente: {col}" for col in missing_columns]
            } for idx in df.index]
            return pd.DataFrame(columns=df.columns), df.copy(), violations_report

        violations_report = []
        quarantine_indices = set()

        for idx, row in df.iterrows():
            row_violations = []

            # 2. Validação das features numéricas
            for col, rules in self.numerical_rules.items():
                val = row[col]
                
                # Nulos não permitidos
                if pd.isna(val):
                    row_violations.append(f"Feature numérico '{col}' contém valor nulo (NaN)")
                    continue
                    
                # Conversão de tipo segura
                try:
                    num_val = float(val)
                except (ValueError, TypeError):
                    row_violations.append(f"Feature '{col}' com tipo incompatível não numérico: '{val}'")
                    continue
                    
                # Checagem de domínio (Out-of-Bounds)
                d_min = rules.get("domain_min")
                d_max = rules.get("domain_max")
                if d_min is not None and num_val < d_min:
                    row_violations.append(f"Valor {num_val} na feature '{col}' abaixo do limite físico permitido ({d_min})")
                if d_max is not None and num_val > d_max:
                    row_violations.append(f"Valor {num_val} na feature '{col}' acima do limite físico permitido ({d_max})")

            # 3. Validação das features categóricas
            for col, rules in self.categorical_rules.items():
                val = row[col]
                if pd.isna(val):
                    row_violations.append(f"Feature categórica '{col}' contém valor nulo (NaN)")
                    continue
                    
                allowed = rules.get("allowed_values", [])
                # Permitir compatibilidade flexível entre int e str para bins
                str_allowed = [str(x) for x in allowed]
                if str(val) not in str_allowed and val not in allowed:
                    row_violations.append(f"Valor '{val}' na feature '{col}' fora do vocabulário homologado: {allowed}")

            if row_violations:
                quarantine_indices.add(idx)
                violations_report.append({
                    "row_index": idx,
                    "reasons": row_violations
                })

        # Divisão em válidos e quarentena
        valid_indices = [idx for idx in df.index if idx not in quarantine_indices]
        valid_df = df.loc[valid_indices].copy()
        quarantine_df = df.loc[list(quarantine_indices)].copy()

        if violations_report:
            logger.warning(
                f"Validação concluída com alertas: {len(valid_df)}/{len(df)} amostras válidas. "
                f"{len(quarantine_df)} amostras isoladas em quarentena."
            )
            if raise_on_error and not quarantine:
                raise SchemaValidationError(
                    f"{len(quarantine_df)} amostras violaram o contrato de schema formal.",
                    violations_report
                )

        return valid_df, quarantine_df, violations_report


class ProductionInferenceService:
    """
    Serviço unificado de inferência em produção protegido por Schema Validation.
    Executa a sanitização, rejeição de amostras e predição com calibração Platt.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        schema_path: Optional[Union[str, Path]] = None
    ):
        base_dir = Path(__file__).resolve().parent.parent
        self.model_path = Path(model_path) if model_path else (base_dir / "models" / "sanidade_vegetal_pipeline_final.joblib")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Pipeline serializado não encontrado em: {self.model_path}")
            
        logger.info(f"Carregando pipeline de produção de: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        self.validator = SchemaValidator(schema_path=schema_path)
        self.classes_mapping = self.validator.classes_mapping

    def predict(self, df: pd.DataFrame, raise_on_error: bool = False) -> Dict[str, Any]:
        """
        Executa inferência protegida:
        1. Valida conformidade do lote com o contrato.
        2. Isola registros corrompidos em quarentena.
        3. Executa predição autônoma nas instâncias válidas.
        """
        n_total = len(df)
        valid_df, quarantine_df, violations = self.validator.validate_batch(
            df, raise_on_error=raise_on_error, quarantine=True
        )
        
        result_payload = {
            "status": "SUCCESS" if len(quarantine_df) == 0 else ("PARTIAL_SUCCESS" if len(valid_df) > 0 else "ALL_REJECTED"),
            "n_total_received": n_total,
            "n_valid_processed": len(valid_df),
            "n_quarantined": len(quarantine_df),
            "valid_indices": list(valid_df.index),
            "quarantined_indices": list(quarantine_df.index),
            "quarantine_violations": violations,
            "predictions_discrete": [],
            "predictions_labels": [],
            "prediction_probabilities": []
        }
        
        if not valid_df.empty:
            preds = self.pipeline.predict(valid_df)
            probas = self.pipeline.predict_proba(valid_df)
            
            result_payload["predictions_discrete"] = [int(p) for p in preds]
            result_payload["predictions_labels"] = [self.classes_mapping.get(str(p), f"CLASS_{p}") for p in preds]
            result_payload["prediction_probabilities"] = probas.tolist()
            
        return result_payload
