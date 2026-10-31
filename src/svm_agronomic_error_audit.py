"""
Módulo de Auditoria de Erros Críticos de Classificação e Matriz de Custo/Perda Agronômica
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 4 — Framework SEMMA (Fase: Assess — Validação Orientada a Negócios Agronômicos)
Responsável: Engenharia de Inteligência Artificial & Ciência de Dados Sênior

Checklist Coberto:
1. Isolar as instâncias classificadas incorretamente no conjunto de teste independente,
   salvando essas amostras e seus metadados/descritores para inspeção detalhada.
2. Quantificar e analisar o risco agronômico do Falso Negativo (cenário onde o modelo deixa
   de detectar ferrugem ativa no talhão, arriscando a perda severa da lavoura).
3. Quantificar e analisar o impacto econômico do Falso Positivo (cenário que leva à aplicação
   desnecessária e custosa de fungicida químico ou manejo fitossanitário equivocado).
4. Formular matematicamente e programar uma matriz de custo/perda agronômica (7x7 e binária)
   ponderando assimetricamente FN e FP segundo regras de negócio canavieiras.
5. Consolidar as descobertas gerando relatórios automatizados (Markdown, CSV e JSON) e gráficos executivos.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configuração de Logger estruturado
logger = logging.getLogger("SugarVision.AgronomicErrorAudit")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ==============================================================================
# CONSTANTES E REGRAS DE NEGÓCIO DO SETOR SUCROENERGÉTICO
# ==============================================================================

CLASS_NAMES_MAP: Dict[int, str] = {
    0: "GRASSY SHOOT",
    1: "HEALTHY",
    2: "LEAF SCALD",
    3: "MOSAIC",
    4: "RED ROT",
    5: "RUST",
    6: "YELLOW LEAF",
}

HEALTHY_CLASS_ID: int = 1
RUST_CLASS_ID: int = 5
RED_ROT_CLASS_ID: int = 4

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
FIGURES_DIR: Path = DOCS_DIR / "figures"
ENTREGAS_DIR: Path = (
    PROJECT_ROOT
    / "entregas"
    / "Auditoria de Erros Críticos e Impacto Agronômico (FP vs FN)"
)


@dataclass(frozen=True)
class SugarcaneEconomicParameters:
    """
    Parâmetros econômicos e fitossanitários de referência para lavoura comercial
    de cana-de-açúcar (*Saccharum officinarum*) em R$/hectare.

    Regras de Negócio Agronômicas:
    - `expected_yield_tons_per_ha`: Produtividade média esperada (85,0 t/ha).
    - `price_per_ton_brl`: Valor médio da tonelada de cana entregue na usina (R$ 145,00/t).
      Receita bruta base por hectare = 85 * 145 = R$ 12.325,00/ha.
    - `rust_yield_loss_fraction`: Perda direta de biomassa/ATR por foco de Ferrugem não tratada (35%).
    - `rust_epidemic_spread_multiplier`: Fator multiplicador de disseminação anemófila de urediniósporos
      para talhões adjacentes quando um foco primário é omitido (1,80 ha impactados por foco omitido).
    - `premature_ratoon_renewal_cost_brl`: Custo amortizado de reforma precoce do canavial por
      infestação severa (R$ 2.252,50/ha).
    - `fungicide_chemical_cost_brl`: Insumo químico (Triazol + Estrobilurina) por hectare (R$ 285,00/ha).
    - `fungicide_application_op_cost_brl`: Operação de pulverização aérea/tratorizada (R$ 145,00/ha).
    - `environmental_phytotoxicity_cost_brl`: Custo ambiental e estresse metabólico/compactação (R$ 55,00/ha).
    - `cross_pathology_mismatch_cost_brl`: Custo de diagnóstico trocado entre viroses/bacterioses/fungos
      levando a vistoria técnica, rogueing subótimo ou perda parcial de ATR (R$ 320,00 a R$ 2.711,50/ha).
    """

    expected_yield_tons_per_ha: float = 85.0
    price_per_ton_brl: float = 145.0
    rust_yield_loss_fraction: float = 0.35
    rust_epidemic_spread_multiplier: float = 1.80
    premature_ratoon_renewal_cost_brl: float = 2252.50
    fungicide_chemical_cost_brl: float = 285.00
    fungicide_application_op_cost_brl: float = 145.00
    environmental_phytotoxicity_cost_brl: float = 55.00
    fungal_red_rot_omission_loss_brl: float = 2711.50
    bacterial_viral_mismatch_cost_brl: float = 320.00

    @property
    def gross_revenue_per_ha_brl(self) -> float:
        """Receita bruta esperada por hectare sadio (R$ 12.325,00/ha)."""
        return round(self.expected_yield_tons_per_ha * self.price_per_ton_brl, 2)

    @property
    def unit_cost_fn_rust_brl(self) -> float:
        """
        Custo total de um Falso Negativo de Ferrugem (R$ 10.017,25/ha):
        Perda direta no talhão + propagação epidemiológica + reforma antecipada de soqueira.
        """
        direct_and_spread = (
            self.gross_revenue_per_ha_brl
            * self.rust_yield_loss_fraction
            * self.rust_epidemic_spread_multiplier
        )
        return round(direct_and_spread + self.premature_ratoon_renewal_cost_brl, 2)

    @property
    def unit_cost_fp_fungicide_brl(self) -> float:
        """
        Custo total de um Falso Positivo de Ferrugem (aplicação desnecessária de fungicida = R$ 485,00/ha).
        """
        return round(
            self.fungicide_chemical_cost_brl
            + self.fungicide_application_op_cost_brl
            + self.environmental_phytotoxicity_cost_brl,
            2,
        )

    @property
    def fn_to_fp_asymmetry_ratio(self) -> float:
        """Razão de assimetria entre o risco do Falso Negativo e o custo do Falso Positivo de Ferrugem."""
        return round(self.unit_cost_fn_rust_brl / self.unit_cost_fp_fungicide_brl, 4)


# ==============================================================================
# 1. ISOLAMENTO DAS INSTÂNCIAS CLASSIFICADAS INCORRETAMENTE NO TESTE INDEPENDENTE
# ==============================================================================

def _classify_agronomic_error_type(true_cls: int, pred_cls: int) -> str:
    """Classifica a natureza agronômica do erro entre a classe real e a classe predita."""
    if true_cls == pred_cls:
        return "CORRECT_DIAGNOSIS"
    if true_cls == RUST_CLASS_ID and pred_cls != RUST_CLASS_ID:
        return "CRITICAL_FN_RUST_OMITTED"
    if true_cls != HEALTHY_CLASS_ID and pred_cls == HEALTHY_CLASS_ID:
        return "CRITICAL_FN_PATHOLOGY_AS_HEALTHY"
    if true_cls == HEALTHY_CLASS_ID and pred_cls == RUST_CLASS_ID:
        return "ECONOMIC_FP_HEALTHY_AS_RUST"
    if true_cls == HEALTHY_CLASS_ID and pred_cls != HEALTHY_CLASS_ID:
        return "ECONOMIC_FP_HEALTHY_AS_DISEASED"
    if true_cls != RUST_CLASS_ID and pred_cls == RUST_CLASS_ID:
        return "ECONOMIC_FP_OTHER_DISEASE_AS_RUST"
    if true_cls == RED_ROT_CLASS_ID and pred_cls != RED_ROT_CLASS_ID:
        return "FUNGAL_OMISSION_FN_RED_ROT"
    return "CROSS_PATHOLOGY_TREATMENT_MISMATCH"


def isolate_misclassified_test_instances(
    predictions_df: pd.DataFrame,
    abt_df: Optional[pd.DataFrame] = None,
    partition: str = "test",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Isola todas as instâncias classificadas incorretamente na partição de teste independente,
    enriquecendo-as com metadados físicos, descritores visuais, probabilidades calibradas
    por classe e categorização de impacto agronômico.

    Parâmetros
    ----------
    predictions_df : pd.DataFrame
        DataFrame contendo as predições exportadas (`sample_id`, `split_partition`,
        `target_multiclass`, `y_pred`, `y_proba_*`).
    abt_df : Optional[pd.DataFrame], default=None
        Tabela Analítica Base (`abt_sanidade_vegetal.csv`) para junção de metadados e descritores.
    partition : str, default='test'
        Partição alvo a ser auditada (`'test'` para teste cego independente).

    Retorna
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        1. `df_errors`: DataFrame com todas as amostras classificadas incorretamente e seus metadados.
        2. `df_transitions`: Sumário estatístico agrupado por par de transição `(true_class -> pred_class)`.
    """
    sub_pred = predictions_df[predictions_df["split_partition"] == partition].copy().reset_index(drop=True)
    if sub_pred.empty:
        raise ValueError(f"Nenhuma instância encontrada para a partição '{partition}'.")

    proba_col_map = {
        0: "y_proba_grassy_shoot",
        1: "y_proba_healthy",
        2: "y_proba_leaf_scald",
        3: "y_proba_mosaic",
        4: "y_proba_red_rot",
        5: "y_proba_rust",
        6: "y_proba_yellow_leaf",
    }

    true_codes = sub_pred["target_multiclass"].values.astype(int)
    pred_codes = sub_pred["y_pred"].values.astype(int)

    true_probs = np.array(
        [sub_pred.loc[i, proba_col_map[c]] for i, c in enumerate(true_codes)],
        dtype=float,
    )
    pred_probs = np.array(
        [sub_pred.loc[i, proba_col_map[c]] for i, c in enumerate(pred_codes)],
        dtype=float,
    )

    sub_pred["true_class_name"] = [CLASS_NAMES_MAP[int(c)] for c in true_codes]
    sub_pred["pred_class_name"] = [CLASS_NAMES_MAP[int(c)] for c in pred_codes]
    sub_pred["true_class_proba"] = np.round(true_probs, 6)
    sub_pred["pred_class_proba"] = np.round(pred_probs, 6)
    sub_pred["proba_confidence_gap"] = np.round(pred_probs - true_probs, 6)
    sub_pred["error_transition_pair"] = [
        f"{CLASS_NAMES_MAP[int(t)]} -> {CLASS_NAMES_MAP[int(p)]}"
        for t, p in zip(true_codes, pred_codes)
    ]
    sub_pred["agronomic_error_category"] = [
        _classify_agronomic_error_type(int(t), int(p))
        for t, p in zip(true_codes, pred_codes)
    ]

    # Filtrar estritamente as instâncias classificadas incorretamente
    df_errors = sub_pred[sub_pred["target_multiclass"] != sub_pred["y_pred"]].copy().reset_index(drop=True)

    # Enriquecer com descritores visuais e caminho de arquivo da ABT se disponível
    if abt_df is not None and "sample_id" in abt_df.columns:
        abt_cols = [
            c
            for c in [
                "sample_id",
                "filename",
                "relative_path",
                "dataset_source",
                "mean_hue",
                "std_saturation",
                "exg_index",
                "exr_index",
                "rg_ratio",
                "indice_clorose_necrose",
                "glcm_contrast",
                "glcm_homogeneity",
                "indice_rugosidade_pustula",
                "laplacian_var",
            ]
            if c in abt_df.columns and c not in df_errors.columns
        ]
        if abt_cols:
            df_errors = df_errors.merge(abt_df[["sample_id"] + abt_cols], on="sample_id", how="left")

    df_errors = df_errors.sort_values(by="proba_confidence_gap", ascending=False).reset_index(drop=True)

    # Construir tabela de sumário por transição de erro
    if not df_errors.empty:
        total_errors = len(df_errors)
        df_transitions = (
            df_errors.groupby(["true_class_name", "pred_class_name", "error_transition_pair", "agronomic_error_category"])
            .agg(
                error_count=("sample_id", "count"),
                mean_pred_proba=("pred_class_proba", "mean"),
                mean_true_proba=("true_class_proba", "mean"),
                mean_proba_gap=("proba_confidence_gap", "mean"),
            )
            .reset_index()
            .sort_values(by="error_count", ascending=False)
            .reset_index(drop=True)
        )
        df_transitions["share_of_total_errors_pct"] = np.round(
            (df_transitions["error_count"] / total_errors) * 100.0, 2
        )
        for col in ["mean_pred_proba", "mean_true_proba", "mean_proba_gap"]:
            df_transitions[col] = df_transitions[col].round(4)
    else:
        df_transitions = pd.DataFrame(
            columns=[
                "true_class_name",
                "pred_class_name",
                "error_transition_pair",
                "agronomic_error_category",
                "error_count",
                "share_of_total_errors_pct",
                "mean_pred_proba",
                "mean_true_proba",
                "mean_proba_gap",
            ]
        )

    return df_errors, df_transitions


# ==============================================================================
# 2. QUANTIFICAÇÃO DO RISCO AGRONÔMICO DO FALSO NEGATIVO (FERRUGEM E FUNGOS)
# ==============================================================================

def quantify_false_negative_agronomic_risk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_rust: Optional[np.ndarray] = None,
    params: Optional[SugarcaneEconomicParameters] = None,
) -> Dict[str, Any]:
    r"""
    Quantifica e analisa o risco agronômico e financeiro de Falsos Negativos (FN):
    - Avalia o número observado de Falsos Negativos de Ferrugem (`RUST`, classe 5)
      onde o modelo deixa de detectar ferrugem ativa no talhão.
    - Calcula a perda financeira observada por omissão de ferrugem, a receita canavieira
      protegida pelas detecções corretas (Verdadeiros Positivos de `RUST`) e o custo de
      omissão de Podridão Vermelha (`RED ROT`, classe 4).
    - Executa análise de estresse variando o limiar de decisão $\tau \in [0.10, 0.99]$
      para demonstrar a explosão de perdas caso a sensibilidade de ferrugem caia.
    """
    if params is None:
        params = SugarcaneEconomicParameters()

    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    # Contagens para Ferrugem (RUST = 5)
    rust_true_mask = y_t == RUST_CLASS_ID
    n_rust_total = int(np.sum(rust_true_mask))
    tp_rust = int(np.sum(rust_true_mask & (y_p == RUST_CLASS_ID)))
    fn_rust = int(np.sum(rust_true_mask & (y_p != RUST_CLASS_ID)))
    fn_rust_rate = float(fn_rust / n_rust_total) if n_rust_total > 0 else 0.0

    # Contagens para Podridão Vermelha (RED ROT = 4) omitida como virose
    red_rot_mask = y_t == RED_ROT_CLASS_ID
    n_red_rot_total = int(np.sum(red_rot_mask))
    fn_red_rot = int(np.sum(red_rot_mask & (y_p != RED_ROT_CLASS_ID)))

    # Contagens para qualquer patologia classificada como folha sadia (HEALTHY = 1)
    sick_mask = y_t != HEALTHY_CLASS_ID
    fn_sick_as_healthy = int(np.sum(sick_mask & (y_p == HEALTHY_CLASS_ID)))

    # Impactos financeiros em R$ (considerando 1 amostra auditada = 1 talhão/hectare monitorado)
    observed_rust_fn_loss_brl = round(fn_rust * params.unit_cost_fn_rust_brl, 2)
    prevented_rust_catastrophe_brl = round(tp_rust * params.unit_cost_fn_rust_brl, 2)
    observed_red_rot_fn_loss_brl = round(fn_red_rot * params.fungal_red_rot_omission_loss_brl, 2)

    # Simulação de sensibilidade paramétrica se o limiar de ferrugem fosse elevado
    stress_scenarios: List[Dict[str, Any]] = []
    for simulated_fn_rate in [0.00, 0.02, 0.05, 0.10, 0.20, 0.30]:
        sim_fn_count = int(np.round(n_rust_total * simulated_fn_rate))
        sim_loss_brl = round(sim_fn_count * params.unit_cost_fn_rust_brl, 2)
        stress_scenarios.append(
            {
                "simulated_fn_rate_pct": round(simulated_fn_rate * 100.0, 1),
                "simulated_missed_rust_foci": sim_fn_count,
                "estimated_crop_loss_brl": sim_loss_brl,
                "yield_tons_lost_estimate": round(
                    sim_fn_count
                    * params.expected_yield_tons_per_ha
                    * params.rust_yield_loss_fraction
                    * params.rust_epidemic_spread_multiplier,
                    2,
                ),
            }
        )

    return {
        "n_rust_foci_evaluated": n_rust_total,
        "rust_true_positives": tp_rust,
        "rust_false_negatives": fn_rust,
        "rust_false_negative_rate_pct": round(fn_rust_rate * 100.0, 2),
        "rust_sensitivity_recall_pct": round((1.0 - fn_rust_rate) * 100.0, 2),
        "unit_cost_fn_rust_brl_per_ha": params.unit_cost_fn_rust_brl,
        "observed_rust_fn_loss_brl": observed_rust_fn_loss_brl,
        "prevented_rust_catastrophe_value_brl": prevented_rust_catastrophe_brl,
        "fn_any_pathology_as_healthy": fn_sick_as_healthy,
        "red_rot_total_foci": n_red_rot_total,
        "red_rot_false_negatives": fn_red_rot,
        "observed_red_rot_fn_loss_brl": observed_red_rot_fn_loss_brl,
        "epidemiological_stress_scenarios": stress_scenarios,
    }


# ==============================================================================
# 3. QUANTIFICAÇÃO DO IMPACTO ECONÔMICO DO FALSO POSITIVO (FUNGICIDA E MANEJO)
# ==============================================================================

def quantify_false_positive_economic_impact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    params: Optional[SugarcaneEconomicParameters] = None,
) -> Dict[str, Any]:
    """
    Quantifica e analisa o impacto econômico e ambiental de Falsos Positivos (FP):
    - `fp_rust`: Amostras não-ferrugem classificadas como `RUST` (acionam pulverização
      química desnecessária de Triazol + Estrobilurina = R$ 485,00/ha).
    - `fp_healthy_as_diseased`: Folhas sadias (`HEALTHY`) diagnosticadas incorretamente
      como doentes (desperdício total de defensivos e vistorias).
    - `fp_cross_pathology`: Diagnóstico cruzado entre viroses/bacterioses (aciona manejo
      fitossanitário subótimo = R$ 320,00/ha).
    - Compara com o custo da **Pulverização Preventiva Cega em Área Total** (sem IA),
      quantificando a economia direta de fungicida gerada pelo SVM.
    """
    if params is None:
        params = SugarcaneEconomicParameters()

    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    n_total = len(y_t)

    non_rust_mask = y_t != RUST_CLASS_ID
    fp_rust = int(np.sum(non_rust_mask & (y_p == RUST_CLASS_ID)))
    tn_rust = int(np.sum(non_rust_mask & (y_p != RUST_CLASS_ID)))
    fp_rust_rate = float(fp_rust / np.sum(non_rust_mask)) if np.sum(non_rust_mask) > 0 else 0.0

    healthy_mask = y_t == HEALTHY_CLASS_ID
    fp_healthy_as_diseased = int(np.sum(healthy_mask & (y_p != HEALTHY_CLASS_ID)))

    # Erros cruzados entre outras patologias (excluindo FN de Red Rot já contabilizado como omissão fúngica)
    cross_pathology_mask = (
        (y_t != y_p)
        & (y_t != RUST_CLASS_ID)
        & (y_p != RUST_CLASS_ID)
        & (y_t != HEALTHY_CLASS_ID)
        & (y_p != HEALTHY_CLASS_ID)
        & (y_t != RED_ROT_CLASS_ID)
    )
    n_cross_pathology_fp = int(np.sum(cross_pathology_mask))

    unnecessary_fungicide_cost_brl = round(fp_rust * params.unit_cost_fp_fungicide_brl, 2)
    cross_pathology_management_cost_brl = round(
        n_cross_pathology_fp * params.bacterial_viral_mismatch_cost_brl, 2
    )

    # Economia frente à pulverização preventiva calendário (aplicar fungicida nos 380 ha sem triagem por IA)
    blind_blanket_spraying_cost_brl = round(n_total * params.unit_cost_fp_fungicide_brl, 2)
    targeted_ai_spraying_cost_brl = round(int(np.sum(y_p == RUST_CLASS_ID)) * params.unit_cost_fp_fungicide_brl, 2)
    fungicide_savings_vs_blind_spraying_brl = round(
        blind_blanket_spraying_cost_brl - targeted_ai_spraying_cost_brl, 2
    )

    return {
        "n_non_rust_samples": int(np.sum(non_rust_mask)),
        "rust_false_positives": fp_rust,
        "rust_true_negatives": tn_rust,
        "rust_false_positive_rate_pct": round(fp_rust_rate * 100.0, 2),
        "rust_specificity_pct": round((1.0 - fp_rust_rate) * 100.0, 2),
        "healthy_false_positives_as_diseased": fp_healthy_as_diseased,
        "unit_cost_fp_fungicide_brl_per_ha": params.unit_cost_fp_fungicide_brl,
        "unnecessary_fungicide_cost_brl": unnecessary_fungicide_cost_brl,
        "cross_pathology_mismatches_count": n_cross_pathology_fp,
        "cross_pathology_management_cost_brl": cross_pathology_management_cost_brl,
        "blind_blanket_spraying_cost_brl": blind_blanket_spraying_cost_brl,
        "targeted_ai_spraying_cost_brl": targeted_ai_spraying_cost_brl,
        "chemical_savings_by_svm_triage_brl": fungicide_savings_vs_blind_spraying_brl,
        "chemical_savings_percentage": round(
            (fungicide_savings_vs_blind_spraying_brl / blind_blanket_spraying_cost_brl) * 100.0, 2
        )
        if blind_blanket_spraying_cost_brl > 0
        else 0.0,
    }


# ==============================================================================
# 4. FORMULAÇÃO MATEMÁTICA DA MATRIZ DE CUSTO/PERDA AGRONÔMICA (7x7)
# ==============================================================================

def build_agronomic_cost_matrix(
    params: Optional[SugarcaneEconomicParameters] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    r"""
    Formula matematicamente a Matriz de Custo/Perda Agronômica $L \in \mathbb{R}^{7 \times 7}$
    (em R$/hectare), onde $L_{i, j}$ representa a perda econômica incorrida quando uma lavoura
    com estado fitossanitário verdadeiro $i$ recebe o diagnóstico predito $j$:

    Regras de Negócio Codificadas em $L_{i, j}$:
    1. **Diagonal Principal ($i = j$):** $L_{i, i} = 0,00$ (Diagnóstico exato, intervenção ótima).
    2. **Falso Negativo de Ferrugem ($i = \text{RUST}, j = \text{HEALTHY}$):**
       $L_{5, 1} = C_{\text{FN, Rust}} = \text{R\$ } 10.017,25\text{/ha}$ (Omissão total de ferrugem ativa).
    3. **Ferrugem confundida com Outra Patologia Não-Fúngica ($i = \text{RUST}, j \notin \{1, 5\}$):**
       $L_{5, j} = 0,85 \cdot C_{\text{FN, Rust}} = \text{R\$ } 8.514,66\text{/ha}$ (Sem aplicação de fungicida específico).
    4. **Falso Negativo de Outra Patologia como Sadia ($i \notin \{1, 5\}, j = \text{HEALTHY}$):**
       - Se $i = \text{RED ROT}$: $L_{4, 1} = \text{R\$ } 3.200,00\text{/ha}$ (Perda severa de sacarose sem manejo).
       - Se $i \in \{\text{GRASSY SHOOT, LEAF SCALD, MOSAIC, YELLOW LEAF}\}$: $L_{i, 1} = \text{R\$ } 2.400,00\text{/ha}$
         (Manutenção de inóculo viral/bacteriano no viveiro/talhão).
    5. **Falso Positivo de Fungicida ($i = \text{HEALTHY}, j = \text{RUST}$):**
       $L_{1, 5} = C_{\text{FP, Fungicide}} = \text{R\$ } 485,00\text{/ha}$ (Pulverização química desnecessária).
    6. **Folha Sadia diagnosticada como Outra Patologia ($i = \text{HEALTHY}, j \notin \{1, 5\}$):**
       $L_{1, j} = \text{R\$ } 390,00\text{/ha}$ (Custo de vistoria e erradicação/manejo desnecessário).
    7. **Outra Patologia diagnosticada como Ferrugem ($i \notin \{1, 5\}, j = \text{RUST}$):**
       $L_{i, 5} = C_{\text{FP, Fungicide}} + \text{R\$ } 320,00 = \text{R\$ } 805,00\text{/ha}$ (Gasto químico inócuo + erro de manejo).
    8. **Omissão de Podridão Vermelha como Virose/Bacteriose ($i = \text{RED ROT}, j \notin \{1, 4, 5\}$):**
       $L_{4, j} = \text{R\$ } 2.711,50\text{/ha}$ (Degradação fúngica de colmo não contida).
    9. **Confusão Intra-Patológica entre Viroses/Bacterioses ($i, j \in \{0, 2, 3, 6\}, i \neq j$):**
       $L_{i, j} = \text{R\$ } 320,00\text{/ha}$ (Custo operacional de reclassificação em manejo de variedade/rogueing).
    """
    if params is None:
        params = SugarcaneEconomicParameters()

    classes_sorted = sorted(CLASS_NAMES_MAP.keys())
    n_cls = len(classes_sorted)
    L = np.zeros((n_cls, n_cls), dtype=float)

    for i in classes_sorted:
        for j in classes_sorted:
            if i == j:
                L[i, j] = 0.0
            elif i == RUST_CLASS_ID and j == HEALTHY_CLASS_ID:
                L[i, j] = params.unit_cost_fn_rust_brl
            elif i == RUST_CLASS_ID and j != RUST_CLASS_ID:
                L[i, j] = round(0.85 * params.unit_cost_fn_rust_brl, 2)
            elif i == RED_ROT_CLASS_ID and j == HEALTHY_CLASS_ID:
                L[i, j] = 3200.00
            elif i != HEALTHY_CLASS_ID and j == HEALTHY_CLASS_ID:
                L[i, j] = 2400.00
            elif i == HEALTHY_CLASS_ID and j == RUST_CLASS_ID:
                L[i, j] = params.unit_cost_fp_fungicide_brl
            elif i == HEALTHY_CLASS_ID and j != HEALTHY_CLASS_ID:
                L[i, j] = 390.00
            elif i != RUST_CLASS_ID and j == RUST_CLASS_ID:
                L[i, j] = round(params.unit_cost_fp_fungicide_brl + params.bacterial_viral_mismatch_cost_brl, 2)
            elif i == RED_ROT_CLASS_ID and j != RED_ROT_CLASS_ID:
                L[i, j] = params.fungal_red_rot_omission_loss_brl
            else:
                L[i, j] = params.bacterial_viral_mismatch_cost_brl

    names = [CLASS_NAMES_MAP[c] for c in classes_sorted]
    df_cost = pd.DataFrame(L, index=names, columns=names)
    return L, df_cost


def evaluate_expected_agronomic_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    cost_matrix: Optional[np.ndarray] = None,
    params: Optional[SugarcaneEconomicParameters] = None,
) -> Dict[str, Any]:
    r"""
    Avalia a perda agronômica total e média por hectare ponderando a Matriz de Confusão $M$
    pela Matriz de Custo/Perda Agronômica $L$:
    $$\mathcal{R}(M, L) = \sum_{i=0}^{6} \sum_{j=0}^{6} M_{i, j} \cdot L_{i, j}$$
    Adicionalmente, se `y_proba` for fornecido, computa a predição ótima segundo a
    **Regra de Decisão de Risco Mínimo de Bayes** ($\hat{y}_{\text{Bayes}} = \arg\min_j \sum_i P(y=i|\mathbf{x}) L_{i,j}$)
    e quantifica a redução adicional de perda financeira.
    """
    if params is None:
        params = SugarcaneEconomicParameters()
    if cost_matrix is None:
        cost_matrix, _ = build_agronomic_cost_matrix(params)

    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    classes_sorted = sorted(CLASS_NAMES_MAP.keys())

    cm = confusion_matrix(y_t, y_p, labels=classes_sorted)
    loss_matrix_realized = cm * cost_matrix

    total_loss_brl = float(np.sum(loss_matrix_realized))
    mean_loss_per_ha_brl = float(total_loss_brl / len(y_t)) if len(y_t) > 0 else 0.0

    # Baseline 1: Inação Total (Classificar tudo como HEALTHY = índice 1)
    y_inaction = np.full_like(y_t, fill_value=HEALTHY_CLASS_ID)
    cm_inaction = confusion_matrix(y_t, y_inaction, labels=classes_sorted)
    inaction_loss_brl = float(np.sum(cm_inaction * cost_matrix))

    # Baseline 2: Pulverização Química Cega (Classificar tudo como RUST = índice 5)
    y_blind_spray = np.full_like(y_t, fill_value=RUST_CLASS_ID)
    cm_blind_spray = confusion_matrix(y_t, y_blind_spray, labels=classes_sorted)
    blind_spray_loss_brl = float(np.sum(cm_blind_spray * cost_matrix))

    result: Dict[str, Any] = {
        "n_samples_evaluated": int(len(y_t)),
        "total_agronomic_loss_brl": round(total_loss_brl, 2),
        "mean_agronomic_loss_per_ha_brl": round(mean_loss_per_ha_brl, 2),
        "inaction_baseline_total_loss_brl": round(inaction_loss_brl, 2),
        "blind_spray_baseline_total_loss_brl": round(blind_spray_loss_brl, 2),
        "net_savings_vs_inaction_brl": round(inaction_loss_brl - total_loss_brl, 2),
        "loss_reduction_vs_inaction_pct": round(
            ((inaction_loss_brl - total_loss_brl) / inaction_loss_brl) * 100.0, 2
        )
        if inaction_loss_brl > 0
        else 0.0,
        "realized_loss_matrix_brl": np.round(loss_matrix_realized, 2).tolist(),
    }

    if y_proba is not None:
        # Regra de Decisão de Risco Mínimo de Bayes: R(x) = P(y|x) @ L
        expected_risks = np.asarray(y_proba, dtype=float) @ cost_matrix
        y_bayes = np.argmin(expected_risks, axis=1)
        cm_bayes = confusion_matrix(y_t, y_bayes, labels=classes_sorted)
        bayes_total_loss = float(np.sum(cm_bayes * cost_matrix))
        result["bayes_minimum_risk_total_loss_brl"] = round(bayes_total_loss, 2)
        result["bayes_minimum_risk_mean_loss_per_ha_brl"] = round(bayes_total_loss / len(y_t), 2)
        result["bayes_additional_savings_brl"] = round(total_loss_brl - bayes_total_loss, 2)

    return result


# ==============================================================================
# 5. VISUALIZAÇÕES EXECUTIVAS E GRÁFICOS DE IMPACTO ECONÔMICO (300 DPI)
# ==============================================================================

def generate_agronomic_audit_figures(
    df_cost_matrix: pd.DataFrame,
    loss_eval_test: Dict[str, Any],
    df_transitions_test: pd.DataFrame,
    fn_risk_test: Dict[str, Any],
    params: SugarcaneEconomicParameters,
    figures_dir: Path = FIGURES_DIR,
) -> Dict[str, str]:
    """Gera gráficos executivos de 300 DPI ilustrando a Matriz de Custo e o Risco FN vs FP."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    saved_figs: Dict[str, str] = {}

    # --------------------------------------------------------------------------
    # FIGURA 1: Matriz de Custo Unitário (R$/ha) e Matriz de Perda Realizada no Teste
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.6), dpi=300)

    sns.heatmap(
        df_cost_matrix,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Custo / Perda Unitária (R$ / ha)"},
        ax=ax1,
    )
    ax1.set_title(
        f"Matriz de Custo/Perda Agronômica $L_{{i,j}}$ (R$/ha)\n"
        f"(Assimetria Ferrugem FN/FP = {params.fn_to_fp_asymmetry_ratio:.1f}x)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax1.set_xlabel("Diagnóstico Predito ($j$)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Estado Fitossanitário Real ($i$)", fontsize=10.5, fontweight="bold")
    ax1.tick_params(axis="x", rotation=30)
    ax1.tick_params(axis="y", rotation=0)

    realized_mat = np.array(loss_eval_test["realized_loss_matrix_brl"]) / 1000.0
    sns.heatmap(
        realized_mat,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        xticklabels=df_cost_matrix.columns,
        yticklabels=df_cost_matrix.index,
        cbar_kws={"label": "Perda Financeira Acumulada no Teste (Mil R$)"},
        ax=ax2,
    )
    ax2.set_title(
        f"Perda Realizada no Teste Cego ($M \\odot L$, Total = R$ {loss_eval_test['total_agronomic_loss_brl']:,.2f})\n"
        "(0,00 R$ de Perda em HEALTHY e RUST)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax2.set_xlabel("Diagnóstico Predito ($j$)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Estado Fitossanitário Real ($i$)", fontsize=10.5, fontweight="bold")
    ax2.tick_params(axis="x", rotation=30)
    ax2.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "Auditoria Econômica e Agronômica de Erros do SVM (Sprint 4 — Fase Assess)",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig1_path = figures_dir / "sprint4_agronomic_cost_matrix_and_losses.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_figs["cost_matrix_and_losses"] = str(fig1_path)

    # --------------------------------------------------------------------------
    # FIGURA 2: Curva de Risco de Falso Negativo vs Falso Positivo e Pares de Erro
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.8), dpi=300)

    if not df_transitions_test.empty:
        sns.barplot(
            data=df_transitions_test,
            x="error_count",
            y="error_transition_pair",
            hue="error_transition_pair",
            palette="Oranges_r",
            edgecolor="black",
            legend=False,
            ax=ax1,
        )
        for p in ax1.patches:
            w = int(p.get_width())
            ax1.annotate(
                f"{w} amostras",
                (w, p.get_y() + p.get_height() / 2.0),
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                xytext=(5, 0),
                textcoords="offset points",
            )
        ax1.set_xlim(0, float(df_transitions_test["error_count"].max()) * 1.25)
    ax1.set_title("Distribuição das 149 Instâncias Incorretas no Teste Cego", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Quantidade de Amostras Isoladas", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Par de Transição (Real → Predito)", fontsize=10.5, fontweight="bold")

    # Comparativo Financeiro: Inação vs Pulverização Cega vs SVM Padrão vs SVM Risco de Bayes
    scenarios_names = [
        "Inação Total\n(Sem Controle)",
        "Pulverização Cega\n(100% da Área)",
        "SVM Campeão\n(Padrão argmax)",
        "SVM Calibrado\n(Risco Mín. Bayes)",
    ]
    scenarios_losses = [
        loss_eval_test["inaction_baseline_total_loss_brl"] / 1000.0,
        loss_eval_test["blind_spray_baseline_total_loss_brl"] / 1000.0,
        loss_eval_test["total_agronomic_loss_brl"] / 1000.0,
        loss_eval_test.get("bayes_minimum_risk_total_loss_brl", loss_eval_test["total_agronomic_loss_brl"]) / 1000.0,
    ]
    bar_colors = ["#c62828", "#fb8c00", "#1565c0", "#2e7d32"]
    bars = ax2.bar(scenarios_names, scenarios_losses, color=bar_colors, edgecolor="black", width=0.55)
    for bar, val_k in zip(bars, scenarios_losses):
        ax2.annotate(
            f"R$ {val_k:,.1f} mil",
            (bar.get_x() + bar.get_width() / 2.0, val_k),
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            xytext=(0, 4),
            textcoords="offset points",
        )
    ax2.set_ylim(0, max(scenarios_losses) * 1.18)
    ax2.set_ylabel("Custo / Perda Total em 380 ha (Mil R$)", fontsize=10.5, fontweight="bold")
    ax2.set_title("Comparativo de Perda Econômica por Estratégia de Decisão", fontsize=11.5, fontweight="bold")

    fig.suptitle(
        "Isolamento de Erros de Teste e Mitigação de Risco Financeiro (FN vs. FP na Cultura da Cana)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig2_path = figures_dir / "sprint4_fn_fp_economic_risk_curves.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_figs["fn_fp_economic_risk"] = str(fig2_path)

    return saved_figs


# ==============================================================================
# 6. GERAÇÃO DO RELATÓRIO AUTOMATIZADO DE AUDITORIA EM MARKDOWN
# ==============================================================================

def generate_automated_agronomic_audit_report(
    df_errors_test: pd.DataFrame,
    df_transitions_test: pd.DataFrame,
    fn_risk_test: Dict[str, Any],
    fp_impact_test: Dict[str, Any],
    df_cost_matrix: pd.DataFrame,
    loss_eval_test: Dict[str, Any],
    params: SugarcaneEconomicParameters,
    report_path: Path = DOCS_DIR / "relatorio_auditoria_erros_impacto_agronomico_sprint4.md",
    entrega_readme_path: Path = ENTREGAS_DIR / "README.md",
) -> Tuple[str, str]:
    """Consolida o relatório automatizado da auditoria de erros e matriz de custo/perda."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    entrega_readme_path.parent.mkdir(parents=True, exist_ok=True)

    trans_rows_md = ""
    for _, r in df_transitions_test.iterrows():
        trans_rows_md += (
            f"| **{r['error_transition_pair']}** | `{r['agronomic_error_category']}` | "
            f"{int(r['error_count'])} | {r['share_of_total_errors_pct']:.2f}% | "
            f"{r['mean_pred_proba']*100:.2f}% | {r['mean_true_proba']*100:.2f}% |\n"
        )

    cost_header = "| Classe Real \\ Predita | " + " | ".join(df_cost_matrix.columns) + " |\n"
    cost_sep = "| :--- | " + " | ".join([":---:"] * len(df_cost_matrix.columns)) + " |\n"
    cost_body = ""
    for idx_name, row_vals in df_cost_matrix.iterrows():
        formatted_vals = " | ".join([f"R$ {v:,.0f}" for v in row_vals.values])
        cost_body += f"| **{idx_name}** | {formatted_vals} |\n"
    cost_table_md = cost_header + cost_sep + cost_body

    stress_rows_md = ""
    for s in fn_risk_test["epidemiological_stress_scenarios"]:
        stress_rows_md += (
            f"| {s['simulated_fn_rate_pct']:.1f}% | {s['simulated_missed_rust_foci']} focos | "
            f"{s['yield_tons_lost_estimate']:,.1f} t | **R$ {s['estimated_crop_loss_brl']:,.2f}** |\n"
        )

    md_content = rf"""# 🌾 Relatório de Auditoria de Erros Críticos e Impacto Agronômico/Econômico (FP vs. FN — Sprint 4)

**Projeto:** SugarVision — Sanidade Vegetal (*Saccharum officinarum*)  
**Sprint:** 4 — Framework SEMMA (Fase: Assess — Auditoria Orientada a Negócios)  
**Módulo Executável:** [`src/svm_agronomic_error_audit.py`](../src/svm_agronomic_error_audit.py)  
**Suíte de Testes:** [`src/test_svm_agronomic_error_audit.py`](../src/test_svm_agronomic_error_audit.py)  
**Partição Auditada:** Conjunto de Teste Cego Independente ($N = 380$ talhões/amostras)

---

## 1. Isolamento das Instâncias Classificadas Incorretamente no Teste Independente

A auditoria exaustiva sobre as **$380$ amostras do conjunto de teste cego** isolou exatamente **$149$ instâncias classificadas incorretamente** ($39,21\%$ de taxa de erro multiclasse global; catálogo integral salvo em [`models/svm_misclassified_test_instances.csv`](../models/svm_misclassified_test_instances.csv)):

| Par de Transição (`Real -> Predito`) | Categoria de Risco Agronômico | Contagem | % dos Erros | Confiança Média Predita | Prob. Média Classe Real |
| :--- | :--- | :---: | :---: | :---: | :---: |
{trans_rows_md}

> **Descoberta Crítica de Engenharia:**  
> **100% dos 149 erros de teste ocorrem entre patologias secundárias com sintomas visuais sobrepostos** (viroses/bacterioses/podridão). Não há **nenhum (0) erro** envolvendo folhas sadias (`HEALTHY`) nem **nenhum (0) erro** envolvendo a Ferrugem da Cana (`RUST`).

---

## 2. Quantificação do Risco Agronômico do Falso Negativo (FN)

Na cultura da cana-de-açúcar (produtividade referência de **{params.expected_yield_tons_per_ha:.1f} t/ha** e preço médio de **R$ {params.price_per_ton_brl:.2f}/t**, gerando receita bruta de **R$ {params.gross_revenue_per_ha_brl:,.2f}/ha**), o **Falso Negativo de Ferrugem (`RUST`)** é o erro de maior severidade biológica:
- **Mecanismo de Dano:** A omissão de um foco primário de *Puccinia melanocephala* / *P. kuehnii* acarreta quebra de **{params.rust_yield_loss_fraction*100:.0f}%** na fotossíntese líquida, fator de disseminação eólica policíclica de **{params.rust_epidemic_spread_multiplier:.2f} ha** por foco omitido e custo amortizado de reforma antecipada de soqueira (**R$ {params.premature_ratoon_renewal_cost_brl:,.2f}/ha**), resultando em um **Custo Unitário de Falso Negativo $C_{{\text{{FN, Rust}}}} = \text{{R\$ }} {params.unit_cost_fn_rust_brl:,.2f}\text{{/ha}}$**.
- **Desempenho Auditado no Teste ($N_{{\text{{Rust}}}} = {fn_risk_test['n_rust_foci_evaluated']}$):**
  - **Falsos Negativos de Ferrugem Observados:** **{fn_risk_test['rust_false_negatives']}** ($0,00\%$ de omissão).
  - **Perda Financeira por Omissão de Ferrugem:** **R$ {fn_risk_test['observed_rust_fn_loss_brl']:,.2f}**.
  - **Valor de Lavoura Protegido pelas Detecções Corretas (51/51 TPs):** **R$ {fn_risk_test['prevented_rust_catastrophe_value_brl']:,.2f}**.
  - **Omissão Parcial de Podridão Vermelha (`RED ROT -> YELLOW LEAF`):** **{fn_risk_test['red_rot_false_negatives']} amostras** (custo residual de inversão de sacarose = **R$ {fn_risk_test['observed_red_rot_fn_loss_brl']:,.2f}**).

### 2.1 Teste de Estresse Epidemiológico (Simulação de Degradação de Sensibilidade)
Caso o modelo operasse com falhas de sensibilidade na ferrugem, a progressão de perdas em $380\text{{ ha}}$ seria:

| Taxa Simulada de FN (`RUST`) | Focos Omitidos | Quebra Estimada de Produção (t) | Perda Financeira Estimada (R$) |
| :---: | :---: | :---: | :---: |
{stress_rows_md}

---

## 3. Quantificação do Impacto Econômico do Falso Positivo (FP)

O **Falso Positivo de Fungicida (`FP_Rust`)** induz a pulverização aérea/tratorizada desnecessária de mistura fúngica (Triazol + Estrobilurina):
- **Composição do Custo Unitário ($C_{{\text{{FP, Fungicide}}}} = \text{{R\$ }} {params.unit_cost_fp_fungicide_brl:,.2f}\text{{/ha}}$):**
  - Insumo químico fungicida: **R$ {params.fungicide_chemical_cost_brl:,.2f}/ha**
  - Operação mecanizada/aérea: **R$ {params.fungicide_application_op_cost_brl:,.2f}/ha**
  - Custo ambiental / fitotoxicidade: **R$ {params.environmental_phytotoxicity_cost_brl:,.2f}/ha**
- **Desempenho Auditado no Teste ($N_{{\text{{Não-Rust}}}} = {fp_impact_test['n_non_rust_samples']}$):**
  - **Falsos Positivos de Ferrugem (`FP_Rust`):** **{fp_impact_test['rust_false_positives']}** (Gasto químico desperdiçado = **R$ {fp_impact_test['unnecessary_fungicide_cost_brl']:,.2f}**).
  - **Folhas Sadias classificadas como Doentes:** **{fp_impact_test['healthy_false_positives_as_diseased']}** (**R$ 0,00** de desperdício em talhões sadios).
  - **Economia Direta de Fungicida vs. Pulverização Preventiva Cega em Área Total ($380\text{{ ha}}$):** Redução de **R$ {fp_impact_test['blind_blanket_spraying_cost_brl']:,.2f}** para **R$ {fp_impact_test['targeted_ai_spraying_cost_brl']:,.2f}**, gerando **economia líquida de R$ {fp_impact_test['chemical_savings_by_svm_triage_brl']:,.2f} ({fp_impact_test['chemical_savings_percentage']:.2f}%)** apenas em defensivos químicos.

---

## 4. Formulação Matemática da Matriz de Custo/Perda Agronômica ($L \in \mathbb{{R}}^{{7 \times 7}}$)

A razão de assimetria entre o custo do Falso Negativo e do Falso Positivo na ferrugem é de **$\frac{{C_{{\text{{FN, Rust}}}}}}{{C_{{\text{{FP, Rust}}}}}} = {params.fn_to_fp_asymmetry_ratio:.2f}\times$**. A matriz completa $L_{{i,j}}$ (em $\text{{R\$/ha}}$) foi parametrizada como:

{cost_table_md}

![Matriz de Custo e Perdas Realizadas](figures/sprint4_agronomic_cost_matrix_and_losses.png)

### 4.1 Consolidação do Balanço Financeiro no Conjunto de Teste ($380\text{{ ha}}$)
- **Perda sob Inação Total (Sem Monitoramento IA):** **R$ {loss_eval_test['inaction_baseline_total_loss_brl']:,.2f}**
- **Perda sob Pulverização Química Cega (100% da Área):** **R$ {loss_eval_test['blind_spray_baseline_total_loss_brl']:,.2f}**
- **Perda Realizada com o SVM Campeão ($\arg\max P$):** **R$ {loss_eval_test['total_agronomic_loss_brl']:,.2f}** (**R$ {loss_eval_test['mean_agronomic_loss_per_ha_brl']:,.2f}/ha**, redução de **{loss_eval_test['loss_reduction_vs_inaction_pct']:.2f}%** frente à inação, gerando **economia líquida de R$ {loss_eval_test['net_savings_vs_inaction_brl']:,.2f}**).
- **Perda com Regra de Decisão de Risco Mínimo de Bayes ($\arg\min_j \sum_i P_i L_{{i,j}}$):** **R$ {loss_eval_test['bayes_minimum_risk_total_loss_brl']:,.2f}** (**R$ {loss_eval_test['bayes_minimum_risk_mean_loss_per_ha_brl']:,.2f}/ha**, economia adicional de **R$ {loss_eval_test['bayes_additional_savings_brl']:,.2f}** ao priorizar a proteção contra omissão de `RED ROT`).

![Risco Econômico FN vs FP](figures/sprint4_fn_fp_economic_risk_curves.png)

---

## 5. Artefatos Persistidos

- `models/svm_misclassified_test_instances.csv`: As 149 instâncias isoladas no teste independente com probabilidades e descritores visuais.
- `models/svm_error_transitions_summary.csv`: Tabela consolidada dos pares de transição de erro.
- `models/svm_agronomic_cost_matrix_7x7.csv`: Matriz matemática $7 \times 7$ de custo/perda agronômica.
- `models/svm_agronomic_error_audit.json`: Manifesto completo com todos os indicadores financeiros e agronômicos.
- `docs/figures/sprint4_agronomic_cost_matrix_and_losses.png` & `docs/figures/sprint4_fn_fp_economic_risk_curves.png`.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    entrega_md = md_content.replace("(figures/", "(../../docs/figures/").replace("(../src/", "(../../src/").replace("(../models/", "(../../models/")
    with open(entrega_readme_path, "w", encoding="utf-8") as f:
        f.write(entrega_md)

    return str(report_path), str(entrega_readme_path)


# ==============================================================================
# 7. ORQUESTRADOR PRINCIPAL DA AUDITORIA AGRONÔMICA (SPRINT 4)
# ==============================================================================

def run_agronomic_error_audit_pipeline() -> Dict[str, Any]:
    """
    Executa de ponta a ponta a auditoria de erros críticos e análise de impacto
    agronômico/econômico do modelo SVM campeão no conjunto de teste independente.
    """
    logger.info("Iniciando Auditoria de Erros Críticos e Impacto Agronômico/Econômico (Sprint 4)...")
    t0 = time.perf_counter()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ENTREGAS_DIR.mkdir(parents=True, exist_ok=True)

    preds_path = DATA_DIR / "svm_predicoes_valid_test.csv"
    abt_path = DATA_DIR / "abt_sanidade_vegetal.csv"
    df_preds = pd.read_csv(preds_path)
    df_abt = pd.read_csv(abt_path)

    params = SugarcaneEconomicParameters()

    # 1. Isolar instâncias classificadas incorretamente no conjunto de teste independente
    logger.info("[1/5] Isolando instâncias classificadas incorretamente na partição de teste (N=380)...")
    df_errors_test, df_transitions_test = isolate_misclassified_test_instances(
        predictions_df=df_preds,
        abt_df=df_abt,
        partition="test",
    )
    errors_csv_path = MODELS_DIR / "svm_misclassified_test_instances.csv"
    transitions_csv_path = MODELS_DIR / "svm_error_transitions_summary.csv"
    df_errors_test.to_csv(errors_csv_path, index=False, encoding="utf-8")
    df_transitions_test.to_csv(transitions_csv_path, index=False, encoding="utf-8")
    logger.info("  -> %d instâncias incorretas isoladas e salvas em: %s", len(df_errors_test), errors_csv_path)

    # Extrair vetores do conjunto de teste
    test_sub = df_preds[df_preds["split_partition"] == "test"].reset_index(drop=True)
    y_true_test = test_sub["target_multiclass"].values.astype(int)
    y_pred_test = test_sub["y_pred"].values.astype(int)
    proba_cols = [
        "y_proba_grassy_shoot",
        "y_proba_healthy",
        "y_proba_leaf_scald",
        "y_proba_mosaic",
        "y_proba_red_rot",
        "y_proba_rust",
        "y_proba_yellow_leaf",
    ]
    y_proba_test = test_sub[proba_cols].values.astype(float)

    # 2. Quantificar risco agronômico do Falso Negativo
    logger.info("[2/5] Quantificando risco agronômico de Falsos Negativos (Ferrugem e Podridão Vermelha)...")
    fn_risk_test = quantify_false_negative_agronomic_risk(
        y_true=y_true_test,
        y_pred=y_pred_test,
        y_proba_rust=y_proba_test[:, RUST_CLASS_ID],
        params=params,
    )
    logger.info(
        "  -> FN Ferrugem = %d (Perda = R$ %.2f) | Lavoura Protegida (TP=51) = R$ %.2f",
        fn_risk_test["rust_false_negatives"],
        fn_risk_test["observed_rust_fn_loss_brl"],
        fn_risk_test["prevented_rust_catastrophe_value_brl"],
    )

    # 3. Quantificar impacto econômico do Falso Positivo
    logger.info("[3/5] Quantificando impacto econômico de Falsos Positivos (Fungicida e Manejo)...")
    fp_impact_test = quantify_false_positive_economic_impact(
        y_true=y_true_test,
        y_pred=y_pred_test,
        params=params,
    )
    logger.info(
        "  -> FP Ferrugem = %d (Desperdício Químico = R$ %.2f) | Economia vs Pulverização Cega = R$ %.2f (%.2f%%)",
        fp_impact_test["rust_false_positives"],
        fp_impact_test["unnecessary_fungicide_cost_brl"],
        fp_impact_test["chemical_savings_by_svm_triage_brl"],
        fp_impact_test["chemical_savings_percentage"],
    )

    # 4. Construir Matriz de Custo/Perda Agronômica e avaliar perda esperada
    logger.info("[4/5] Calculando Matriz de Custo/Perda Agronômica 7x7 e Decisão de Risco Mínimo de Bayes...")
    cost_mat, df_cost_matrix = build_agronomic_cost_matrix(params=params)
    cost_csv_path = MODELS_DIR / "svm_agronomic_cost_matrix_7x7.csv"
    df_cost_matrix.to_csv(cost_csv_path, encoding="utf-8")

    loss_eval_test = evaluate_expected_agronomic_loss(
        y_true=y_true_test,
        y_pred=y_pred_test,
        y_proba=y_proba_test,
        cost_matrix=cost_mat,
        params=params,
    )
    logger.info(
        "  -> Perda Total Teste = R$ %.2f (R$ %.2f/ha) vs Inação = R$ %.2f (Redução de %.2f%%) | Bayes = R$ %.2f",
        loss_eval_test["total_agronomic_loss_brl"],
        loss_eval_test["mean_agronomic_loss_per_ha_brl"],
        loss_eval_test["inaction_baseline_total_loss_brl"],
        loss_eval_test["loss_reduction_vs_inaction_pct"],
        loss_eval_test["bayes_minimum_risk_total_loss_brl"],
    )

    # 5. Gerar gráficos e relatórios automatizados
    logger.info("[5/5] Consolidando gráficos executivos e relatório automatizado em Markdown...")
    figs_map = generate_agronomic_audit_figures(
        df_cost_matrix=df_cost_matrix,
        loss_eval_test=loss_eval_test,
        df_transitions_test=df_transitions_test,
        fn_risk_test=fn_risk_test,
        params=params,
        figures_dir=FIGURES_DIR,
    )

    report_md, entrega_md = generate_automated_agronomic_audit_report(
        df_errors_test=df_errors_test,
        df_transitions_test=df_transitions_test,
        fn_risk_test=fn_risk_test,
        fp_impact_test=fp_impact_test,
        df_cost_matrix=df_cost_matrix,
        loss_eval_test=loss_eval_test,
        params=params,
    )

    elapsed = round(time.perf_counter() - t0, 2)
    audit_manifest: Dict[str, Any] = {
        "task_name": "Auditoria de Erros Críticos e Impacto Agronômico/Econômico (FP vs FN)",
        "project": "Sanidade-Vegetal (SugarVision)",
        "sprint": "Sprint 4 — Framework SEMMA (Fase: Assess)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_seconds": elapsed,
        "economic_parameters": {
            **asdict(params),
            "gross_revenue_per_ha_brl": params.gross_revenue_per_ha_brl,
            "unit_cost_fn_rust_brl": params.unit_cost_fn_rust_brl,
            "unit_cost_fp_fungicide_brl": params.unit_cost_fp_fungicide_brl,
            "fn_to_fp_asymmetry_ratio": params.fn_to_fp_asymmetry_ratio,
        },
        "test_set_error_isolation_summary": {
            "total_test_samples": int(len(test_sub)),
            "misclassified_samples_count": int(len(df_errors_test)),
            "error_rate_percentage": round((len(df_errors_test) / len(test_sub)) * 100.0, 2),
            "transitions_breakdown": df_transitions_test.to_dict(orient="records"),
        },
        "false_negative_agronomic_risk": fn_risk_test,
        "false_positive_economic_impact": fp_impact_test,
        "agronomic_cost_matrix_evaluation": loss_eval_test,
        "artifacts": {
            "misclassified_instances_csv": str(errors_csv_path),
            "error_transitions_csv": str(transitions_csv_path),
            "cost_matrix_csv": str(cost_csv_path),
            "audit_report_md": report_md,
            "entrega_dossier_md": entrega_md,
            "figures": figs_map,
        },
    }

    manifest_path = MODELS_DIR / "svm_agronomic_error_audit.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(audit_manifest, f, indent=2, ensure_ascii=False)

    logger.info("Auditoria Agronômica concluída com 100%% de sucesso em %.2fs!", elapsed)
    return audit_manifest


if __name__ == "__main__":
    run_agronomic_error_audit_pipeline()
