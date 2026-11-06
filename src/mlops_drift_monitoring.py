"""
Módulo de Monitoramento Contínuo de Data/Concept Drift e Gatilhos de Retreinamento (MLOps)
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 5 — Framework SEMMA (Fase: Assess & Sustain / MLOps Governance)

Checklist Coberto:
1. Mapear as fontes potenciais de drift no campo (mudança de iluminação sazonal,
   novas câmeras/sensores, estádios fenológicos e novas cultivares).
2. Desenvolver funções para definir e calcular métricas de monitoramento da distribuição
   das features visuais de entrada, implementando especificamente o teste de
   Kolmogorov-Smirnov (KS) e o cálculo de PSI (Population Stability Index).
3. Programar as lógicas em código para estabelecer gatilhos e regras de alerta (thresholds)
   para indicar automaticamente quando o modelo precisará ser re-treinado com novos dados.
4. Exportar artefatos auditáveis de configuração e relatórios de telemetria MLOps.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Configuração de Logger Corporativo MLOps
logger = logging.getLogger("SugarVisionDriftMonitor")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

ArrayLike = Union[np.ndarray, pd.Series, Sequence[float]]

# Features visuais numéricas monitoradas continuamente em produção
MONITORED_VISUAL_FEATURES: List[str] = [
    "mean_hue",
    "std_saturation",
    "exg_index",
    "exr_index",
    "rg_ratio",
    "indice_clorose_necrose",
    "hue_dispersion",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_dissimilarity",
    "glcm_energy",
    "indice_rugosidade_pustula",
    "laplacian_var",
    "size_kb",
    "total_pixels",
    "resolution_mp",
    "aspect_ratio",
]

# Subconjunto de features críticas de alto impacto preditivo (Permutation Importance Sprint 4)
CRITICAL_AGRONOMIC_FEATURES: List[str] = [
    "exg_index",
    "exr_index",
    "rg_ratio",
    "indice_clorose_necrose",
    "mean_hue",
    "glcm_contrast",
    "glcm_homogeneity",
    "indice_rugosidade_pustula",
    "laplacian_var",
]


# ==============================================================================
# 1. MAPEAMENTO DE FONTES POTENCIAIS DE DRIFT NO CAMPO AGRÍCOLA
# ==============================================================================


@dataclass(frozen=True)
class FieldDriftSource:
    """
    Representa uma fonte física, óptica ou biológica de desvio estatístico (Drift)
    na operação de visão computacional em lavouras de cana-de-açúcar.
    """

    source_id: str
    name: str
    category: str
    drift_taxonomy: str  # "Data Drift (Covariate Shift)", "Concept Drift", "Prior Probability Shift"
    environmental_causes: List[str]
    affected_features: List[str]
    physical_mechanism: str
    mitigation_and_sustainment_action: str


FIELD_DRIFT_SOURCES_CATALOG: Dict[str, FieldDriftSource] = {
    "seasonal_illumination": FieldDriftSource(
        source_id="DRIFT-SRC-01",
        name="Mudança de Iluminação Sazonal e Geometria Solar",
        category="Mudança de Iluminação Sazonal",
        drift_taxonomy="Data Drift (Covariate Shift)",
        environmental_causes=[
            "Transição entre safra de verão (alta irradiação zenital, especularidade intensa) e inverno seco (luz difusa, ângulo solar oblíquo).",
            "Cobertura de nuvens intermitente, neblina matinal e poeira em suspensão durante colheita mecanizada.",
            "Horários extremos de captura fotográfica no talhão (alvorada/crepúsculo) alterando a temperatura de cor (Kelvin).",
        ],
        affected_features=[
            "mean_hue",
            "std_saturation",
            "exg_index",
            "exr_index",
            "rg_ratio",
            "hue_dispersion",
        ],
        physical_mechanism=(
            "Deslocamento sistemático nos canais cromáticos RGB/HSV: o excesso de reflexo especular "
            "satura o canal V e comprime a saturação (std_saturation), enquanto a iluminação oblíqua "
            "avermelhada inflaciona artificialmente a razão R/G (rg_ratio) e o índice ExR (exr_index)."
        ),
        mitigation_and_sustainment_action=(
            "Normalização radiométrica via StandardScaler re-ajustado por estação climática, "
            "calibração de balanço de branco na borda e retreinamento sazonal com amostras estratificadas por iluminância."
        ),
    ),
    "new_cameras_sensors": FieldDriftSource(
        source_id="DRIFT-SRC-02",
        name="Introdução de Novas Câmeras, Lentes e Sensores Ópticos",
        category="Novas Câmeras e Sensores",
        drift_taxonomy="Data Drift (Covariate Shift)",
        environmental_causes=[
            "Substituição ou heterogeneidade de dispositivos móveis de campo (smartphones distintos, câmeras embarcadas em tratores/drones).",
            "Mudança na resolução nativa do sensor CMOS, distância focal da lente e algoritmos internos de pós-processamento ISP (HDR, sharpening, compressão JPEG).",
            "Degradação física de lentes por microabrasão de sílica, umidade ou resíduos de calda fitossanitária.",
        ],
        affected_features=[
            "laplacian_var",
            "glcm_contrast",
            "glcm_homogeneity",
            "glcm_dissimilarity",
            "glcm_energy",
            "resolution_mp",
            "total_pixels",
            "size_kb",
            "aspect_ratio",
        ],
        physical_mechanism=(
            "Sensores de maior resolução ou com filtros de nitidez agressivos elevam abruptamente a "
            "variância laplaciana (laplacian_var) e o contraste GLCM (glcm_contrast), enquanto lentes "
            "embaçadas suavizam microtexturas, reduzindo o Índice de Rugosidade Foliar de Pústula "
            "(indice_rugosidade_pustula) e mascarando urédias incipientes de Puccinia melanocephala."
        ),
        mitigation_and_sustainment_action=(
            "Padronização de reamostragem espacial pré-GLCM, bloqueio de imagens com laplacian_var < 20 "
            "no SchemaValidator e inclusão de lotes de calibração óptica sempre que novos modelos de câmeras forem homologados."
        ),
    ),
    "phenological_stages_and_cultivars": FieldDriftSource(
        source_id="DRIFT-SRC-03",
        name="Estádios Fenológicos da Cana-de-Açúcar e Novas Cultivares",
        category="Estádios Fenológicos e Cultivares",
        drift_taxonomy="Concept Drift & Covariate Shift",
        environmental_causes=[
            "Evolução fisiológica da planta: brotação, perfilhamento, crescimento dos colmos e maturação/senescência natural.",
            "Introdução de novas variedades genéticas (ex.: RB, CTC, IAC) com arquitetura foliar, cerosidade epicuticular e teor basal de clorofila distintos.",
            "Estresse hídrico sazonal e deficiências nutricionais (N, K, Mg) que mimetizam sintomas visuais de clorose e escaldadura.",
        ],
        affected_features=[
            "indice_clorose_necrose",
            "indice_rugosidade_pustula",
            "exg_index",
            "exr_index",
            "rg_ratio",
            "mean_hue",
            "glcm_dissimilarity",
        ],
        physical_mechanism=(
            "Na maturação e senescência fisiológica, a degradação natural da clorofila reduz o ExG (exg_index) "
            "e eleva o Índice de Clorose/Necrose (indice_clorose_necrose) em folhas sadias, alterando a "
            "fronteira de decisão P(Y|X) (Concept Drift) entre folhas senescentes sadias e patologias "
            "como Yellow Leaf (SCYLV) ou Red Rot."
        ),
        mitigation_and_sustainment_action=(
            "Retreinamento programado a cada ciclo fenológico trimestral (90 dias) com auditoria "
            "fitopatológica humana (Active Learning) e expansão da ABT para contemplar novas cultivares comerciais."
        ),
    ),
}


def get_field_drift_sources_mapping() -> Dict[str, Dict[str, Any]]:
    """
    Retorna o mapeamento estruturado de todas as fontes de drift de campo,
    pronto para serialização JSON e auditoria de governança.

    Returns:
        Dict[str, Dict[str, Any]]: Dicionário mapeando cada chave de fonte de drift
        aos seus metadados agronômicos, físicos e lista de features afetadas.
    """
    return {key: asdict(source) for key, source in FIELD_DRIFT_SOURCES_CATALOG.items()}


# ==============================================================================
# 2. PARÂMETROS DE GOVERNANÇA E LIMIARES DE ALERTA (THRESHOLDS)
# ==============================================================================


@dataclass(frozen=True)
class DriftThresholds:
    """
    Regras quantitativas e limiares estatísticos (thresholds) para detecção de
    Data Drift, Concept Drift e disparo automático de retreinamento/rollback.

    Atributos:
        psi_warning (float): Limiar de alerta moderado para PSI (padrão: 0.10).
        psi_critical (float): Limiar crítico de PSI indicando drift severo (padrão: 0.25).
        ks_alpha_warning (float): Nível de significância de alerta para o teste KS (padrão: 0.05).
        ks_alpha_critical (float): Nível de significância crítico para o teste KS (padrão: 0.01).
        ks_statistic_min_effect (float): Magnitude mínima da estatística D do teste KS para
            evitar falsos alarmes por hipersensibilidade em amostras gigantes (padrão: 0.10).
        min_drifted_features_for_retrain (int): Número mínimo de features visuais em estado
            CRITICAL (PSI >= 0.25 ou KS p < 0.01 com D >= 0.10) para acionar retreinamento.
        critical_feature_fraction_threshold (float): Fração de features monitoradas com drift
            crítico que força retreinamento imediato (padrão: 0.20 = 20%).
        min_new_labeled_samples_trigger (int): Acúmulo de novas imagens validadas por
            fitopatologistas que dispara retreinamento por volume (padrão: 500).
        max_days_without_retrain (int): Janela máxima de operação contínua sem retreinamento,
            alinhada ao ciclo fenológico trimestral (padrão: 90 dias).
        min_acceptable_f1_macro (float): Limiar mínimo de F1-Macro em validação de campo antes
            de disparar retreinamento e rollback (padrão: 0.50).
        max_quarantine_rejection_rate (float): Taxa máxima tolerada de rejeição no
            SchemaValidator antes de acionar alerta crítico (padrão: 0.05 = 5%).
    """

    psi_warning: float = 0.10
    psi_critical: float = 0.25
    ks_alpha_warning: float = 0.05
    ks_alpha_critical: float = 0.01
    ks_statistic_min_effect: float = 0.10
    min_drifted_features_for_retrain: int = 2
    critical_feature_fraction_threshold: float = 0.20
    min_new_labeled_samples_trigger: int = 500
    max_days_without_retrain: int = 90
    min_acceptable_f1_macro: float = 0.50
    max_quarantine_rejection_rate: float = 0.05


# ==============================================================================
# 3. FUNÇÕES MATEMÁTICAS DE MONITORAMENTO (PSI & KOLMOGOROV-SMIRNOV)
# ==============================================================================


def _sanitize_numeric_array(data: ArrayLike, name: str = "array") -> np.ndarray:
    """
    Converte uma sequência numérica em um np.ndarray 1D float64 limpo de NaNs/Infs.

    Raises:
        ValueError: Se o array resultante for vazio após remoção de valores inválidos.
        TypeError: Se os elementos não puderem ser convertidos para float.
    """
    if data is None:
        raise ValueError(f"O parâmetro '{name}' não pode ser None.")

    try:
        arr = np.asarray(data, dtype=np.float64).ravel()
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"O parâmetro '{name}' deve conter valores numéricos conversíveis para float."
        ) from exc

    valid_mask = np.isfinite(arr)
    clean_arr = arr[valid_mask]

    if clean_arr.size < 2:
        raise ValueError(
            f"O parâmetro '{name}' deve conter pelo menos 2 valores numéricos finitos "
            f"(recebido: {clean_arr.size})."
        )
    return clean_arr


def calculate_psi(
    reference: ArrayLike,
    current: ArrayLike,
    bins: int = 10,
    strategy: str = "quantile",
    epsilon: float = 1e-6,
) -> float:
    """
    Calcula o Population Stability Index (PSI) entre uma distribuição de referência
    (treino/baseline) e uma distribuição atual (produção/inferência).

    Formulação Matemática:
        PSI = \\sum_{b=1}^{B} (P_b - Q_b) \\times \\ln(P_b / Q_b)
    onde P_b é a proporção empírica da amostra atual no bin b e Q_b é a proporção
    da amostra de referência no mesmo bin b, regularizadas por Laplace smoothing (epsilon).

    Interpretação Padrão MLOps:
        - PSI < 0.10       : Distribuição estável (Sem Drift relevante).
        - 0.10 <= PSI < 0.25: Desvio moderado (Alerta Amarelo / Monitoramento intensivo).
        - PSI >= 0.25      : Desvio severo (Alerta Crítico / Gatilho de Retreinamento/Rollback).

    Args:
        reference (ArrayLike): Vetor 1D com os valores da feature na base de referência (treino).
        current (ArrayLike): Vetor 1D com os valores da feature no lote de produção.
        bins (int): Número de faixas (bins) para discretização do histograma (padrão: 10).
        strategy (str): Estratégia de definição dos cortes dos bins a partir da referência:
            'quantile' (decis empíricos) ou 'uniform' (largura igual). Padrão: 'quantile'.
        epsilon (float): Constante de suavização para evitar divisão por zero ou ln(0).

    Returns:
        float: Valor não-negativo do PSI arredondado em precisão de ponto flutuante.

    Raises:
        ValueError: Se `bins < 2`, `epsilon <= 0`, `strategy` inválida ou arrays vazios.
    """
    if bins < 2:
        raise ValueError(f"O número de bins deve ser >= 2 (recebido: {bins}).")
    if epsilon <= 0:
        raise ValueError(f"O parâmetro epsilon deve ser > 0 (recebido: {epsilon}).")
    if strategy not in ("quantile", "uniform"):
        raise ValueError(
            f"Estratégia '{strategy}' inválida. Use 'quantile' ou 'uniform'."
        )

    ref_arr = _sanitize_numeric_array(reference, name="reference")
    cur_arr = _sanitize_numeric_array(current, name="current")

    ref_min, ref_max = float(np.min(ref_arr)), float(np.max(ref_arr))
    cur_min, cur_max = float(np.min(cur_arr)), float(np.max(cur_arr))

    # Caso especial: distribuição de referência constante (variância zero)
    if np.isclose(ref_min, ref_max, atol=1e-12):
        if np.isclose(cur_min, cur_max, atol=1e-12) and np.isclose(
            ref_min, cur_min, atol=1e-12
        ):
            return 0.0
        # Se a referência é constante mas a produção variou, criar borda simétrica ao redor da referência
        delta = max(abs(ref_min) * 0.01, 1e-3)
        bin_edges = np.array([-np.inf, ref_min - delta, ref_min + delta, np.inf])
    else:
        if strategy == "quantile":
            quantiles = np.linspace(0.0, 100.0, bins + 1)
            internal_edges = np.percentile(ref_arr, quantiles)
            # Remover bordas duplicadas caso existam empates (ties) nos quantis
            internal_edges = np.unique(internal_edges)
            if internal_edges.size < 2:
                internal_edges = np.linspace(ref_min, ref_max, bins + 1)
        else:
            internal_edges = np.linspace(ref_min, ref_max, bins + 1)

        # Estender os extremos para [-inf, +inf] garantindo que outliers de produção sejam contabilizados
        bin_edges = internal_edges.copy()
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_arr, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_arr, bins=bin_edges)

    ref_props = ref_counts.astype(np.float64) / float(ref_arr.size)
    cur_props = cur_counts.astype(np.float64) / float(cur_arr.size)

    # Suavização numérica estrita para evitar log(0) e divisão por zero
    ref_props = np.clip(ref_props, epsilon, None)
    cur_props = np.clip(cur_props, epsilon, None)

    # Normalizar após clip para manter soma unitária exata se houve substituição por epsilon
    ref_props /= np.sum(ref_props)
    cur_props /= np.sum(cur_props)

    psi_value = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
    return max(0.0, psi_value)


def calculate_ks_drift(
    reference: ArrayLike,
    current: ArrayLike,
    alpha: float = 0.01,
    alpha_warning: float = 0.05,
    min_effect_size: float = 0.10,
) -> Dict[str, Any]:
    """
    Executa o teste não-paramétrico bicaudal de Kolmogorov-Smirnov (KS de duas amostras)
    para verificar se a amostra de produção provém da mesma distribuição contínua
    que a amostra de referência do treinamento.

    Formulação Matemática:
        D_{n,m} = \\sup_{x} |F_{1,n}(x) - F_{2,m}(x)|
    Hipótese Nula (H_0): As duas amostras seguem a mesma distribuição F_1(x) = F_2(x).
    Regra de Rejeição: p_value < alpha (combinado ao tamanho de efeito D >= min_effect_size).

    Args:
        reference (ArrayLike): Vetor 1D com os valores da feature na referência (treino).
        current (ArrayLike): Vetor 1D com os valores da feature no lote de produção.
        alpha (float): Nível de significância crítico para rejeição de H_0 (padrão: 0.01).
        alpha_warning (float): Nível de significância para alerta preventivo (padrão: 0.05).
        min_effect_size (float): Estatística D mínima para considerar o desvio relevante na prática.

    Returns:
        Dict[str, Any]: Dicionário estruturado contendo:
            - 'ks_statistic' (float): Distância máxima D entre as CDFs empíricas [0, 1].
            - 'p_value' (float): Valor-p bicaudal assintótico/exato do teste KS [0, 1].
            - 'alpha' (float): Nível de significância crítico adotado.
            - 'n_reference' (int): Tamanho amostral válido da referência.
            - 'n_current' (int): Tamanho amostral válido do lote atual.
            - 'null_hypothesis_rejected' (bool): True se p_value < alpha.
            - 'drift_detected' (bool): True se p_value < alpha e ks_statistic >= min_effect_size.
            - 'severity' (str): 'STABLE', 'WARNING' ou 'CRITICAL'.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"O nível de significância alpha deve estar em (0, 1): {alpha}")
    if not (0.0 < alpha_warning < 1.0):
        raise ValueError(
            f"O nível alpha_warning deve estar em (0, 1): {alpha_warning}"
        )

    ref_arr = _sanitize_numeric_array(reference, name="reference")
    cur_arr = _sanitize_numeric_array(current, name="current")

    ks_res = stats.ks_2samp(ref_arr, cur_arr, alternative="two-sided", mode="auto")
    ks_stat = float(ks_res.statistic)
    p_val = float(ks_res.pvalue)

    null_rejected = bool(p_val < alpha)
    drift_detected = bool(null_rejected and (ks_stat >= min_effect_size))

    if drift_detected:
        severity = "CRITICAL"
    elif (p_val < alpha_warning) and (ks_stat >= min_effect_size * 0.5):
        severity = "WARNING"
    else:
        severity = "STABLE"

    return {
        "ks_statistic": round(ks_stat, 6),
        "p_value": round(p_val, 8),
        "alpha": alpha,
        "n_reference": int(ref_arr.size),
        "n_current": int(cur_arr.size),
        "null_hypothesis_rejected": null_rejected,
        "drift_detected": drift_detected,
        "severity": severity,
    }


def classify_psi_severity(
    psi_value: float, thresholds: Optional[DriftThresholds] = None
) -> str:
    """
    Classifica a severidade do Population Stability Index (PSI) segundo os thresholds MLOps.

    Args:
        psi_value (float): Valor calculado do PSI.
        thresholds (Optional[DriftThresholds]): Configuração de limiares.

    Returns:
        str: 'STABLE', 'WARNING' ou 'CRITICAL'.
    """
    cfg = thresholds or DriftThresholds()
    if psi_value >= cfg.psi_critical:
        return "CRITICAL"
    if psi_value >= cfg.psi_warning:
        return "WARNING"
    return "STABLE"


# ==============================================================================
# 4. AVALIAÇÃO MULTIVARIADA POR FEATURE E MOTOR DE GATILHOS DE RETREINAMENTO
# ==============================================================================


def evaluate_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    thresholds: Optional[DriftThresholds] = None,
    psi_bins: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    Calcula o diagnóstico completo de Data Drift (PSI + Teste KS + Deslocamento de Média/Desvio)
    para todas as features visuais monitoradas entre a base de referência e o lote atual.

    Args:
        reference_df (pd.DataFrame): DataFrame da partição de referência (treinamento).
        current_df (pd.DataFrame): DataFrame do lote de produção sob monitoramento.
        features (Optional[List[str]]): Lista de colunas numéricas a monitorar.
            Se None, utiliza a interseção com `MONITORED_VISUAL_FEATURES`.
        thresholds (Optional[DriftThresholds]): Configuração de limiares de alerta.
        psi_bins (int): Quantidade de decis/bins no cálculo do PSI (padrão: 10).

    Returns:
        Dict[str, Dict[str, Any]]: Relatório detalhado indexado pelo nome da feature.
    """
    if not isinstance(reference_df, pd.DataFrame) or not isinstance(
        current_df, pd.DataFrame
    ):
        raise TypeError("reference_df e current_df devem ser instâncias de pandas.DataFrame.")

    cfg = thresholds or DriftThresholds()
    target_features = (
        features
        if features is not None
        else [
            col
            for col in MONITORED_VISUAL_FEATURES
            if col in reference_df.columns and col in current_df.columns
        ]
    )

    if not target_features:
        raise ValueError(
            "Nenhuma feature visual monitorada foi encontrada em comum entre reference_df e current_df."
        )

    report: Dict[str, Dict[str, Any]] = {}
    severity_rank = {"STABLE": 0, "WARNING": 1, "CRITICAL": 2}

    for feat in target_features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            raise KeyError(
                f"A coluna obrigatória '{feat}' está ausente em um dos DataFrames."
            )

        ref_series = reference_df[feat].dropna()
        cur_series = current_df[feat].dropna()

        psi_val = calculate_psi(ref_series, cur_series, bins=psi_bins, strategy="quantile")
        psi_sev = classify_psi_severity(psi_val, cfg)

        ks_res = calculate_ks_drift(
            ref_series,
            cur_series,
            alpha=cfg.ks_alpha_critical,
            alpha_warning=cfg.ks_alpha_warning,
            min_effect_size=cfg.ks_statistic_min_effect,
        )
        ks_sev = ks_res["severity"]

        # Severidade combinada (conservadora: assume o pior caso entre PSI e KS)
        combined_severity = (
            psi_sev
            if severity_rank[psi_sev] >= severity_rank[ks_sev]
            else ks_sev
        )

        ref_mean = float(np.mean(ref_series))
        cur_mean = float(np.mean(cur_series))
        ref_std = float(np.std(ref_series, ddof=1)) if len(ref_series) > 1 else 0.0
        cur_std = float(np.std(cur_series, ddof=1)) if len(cur_series) > 1 else 0.0
        mean_shift_pct = (
            ((cur_mean - ref_mean) / abs(ref_mean)) * 100.0
            if abs(ref_mean) > 1e-9
            else 0.0
        )

        # Mapear possíveis causas de campo associadas a esta feature
        associated_field_sources = [
            src.name
            for src in FIELD_DRIFT_SOURCES_CATALOG.values()
            if feat in src.affected_features
        ]

        report[feat] = {
            "feature_name": feat,
            "is_critical_agronomic_feature": feat in CRITICAL_AGRONOMIC_FEATURES,
            "psi": round(psi_val, 6),
            "psi_severity": psi_sev,
            "ks_statistic": ks_res["ks_statistic"],
            "ks_p_value": ks_res["p_value"],
            "ks_drift_detected": ks_res["drift_detected"],
            "ks_severity": ks_sev,
            "combined_severity": combined_severity,
            "reference_mean": round(ref_mean, 4),
            "current_mean": round(cur_mean, 4),
            "reference_std": round(ref_std, 4),
            "current_std": round(cur_std, 4),
            "relative_mean_shift_pct": round(mean_shift_pct, 2),
            "associated_field_drift_sources": associated_field_sources,
        }

    return report


def evaluate_retraining_triggers(
    feature_drift_report: Dict[str, Dict[str, Any]],
    thresholds: Optional[DriftThresholds] = None,
    new_labeled_samples: int = 0,
    days_since_last_train: int = 0,
    current_f1_macro: Optional[float] = None,
    quarantine_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Avalia as regras de negócio e gatilhos estatísticos de MLOps para determinar
    automaticamente se o modelo em produção requer:
    1. Operação Normal ('NORMAL')
    2. Monitoramento Intensivo / Alerta Preventivo ('WARNING_MONITORING')
    3. Retreinamento Automático Programado ('CRITICAL_RETRAIN_REQUIRED')
    4. Rollback Imediato para v0.9.0 + Retreinamento de Emergência ('EMERGENCY_ROLLBACK_AND_RETRAIN')

    Args:
        feature_drift_report (Dict[str, Dict[str, Any]]): Saída de `evaluate_feature_drift`.
        thresholds (Optional[DriftThresholds]): Limiares de governança MLOps.
        new_labeled_samples (int): Quantidade de novas imagens rotuladas por agrônomos.
        days_since_last_train (int): Dias corridos desde o último treino homologado.
        current_f1_macro (Optional[float]): F1-Macro aferido em amostras auditadas em campo.
        quarantine_rate (float): Proporção [0.0, 1.0] de amostras retidas pelo SchemaValidator.

    Returns:
        Dict[str, Any]: Decisão automatizada contendo `retrain_required`, `rollback_required`,
        `alert_level`, `triggered_rules`, `probable_field_causes` e métricas agregadas.
    """
    cfg = thresholds or DriftThresholds()
    total_features = len(feature_drift_report)
    if total_features == 0:
        raise ValueError("O relatório feature_drift_report não pode ser vazio.")

    critical_psi_features = [
        f
        for f, m in feature_drift_report.items()
        if m["psi"] >= cfg.psi_critical
    ]
    warning_psi_features = [
        f
        for f, m in feature_drift_report.items()
        if cfg.psi_warning <= m["psi"] < cfg.psi_critical
    ]
    critical_ks_features = [
        f for f, m in feature_drift_report.items() if m["ks_drift_detected"]
    ]
    combined_critical_features = [
        f
        for f, m in feature_drift_report.items()
        if m["combined_severity"] == "CRITICAL"
    ]
    combined_warning_features = [
        f
        for f, m in feature_drift_report.items()
        if m["combined_severity"] == "WARNING"
    ]
    critical_agronomic_drifted = [
        f for f in combined_critical_features if f in CRITICAL_AGRONOMIC_FEATURES
    ]

    max_psi = max((m["psi"] for m in feature_drift_report.values()), default=0.0)
    mean_psi = float(np.mean([m["psi"] for m in feature_drift_report.values()]))
    max_ks_stat = max(
        (m["ks_statistic"] for m in feature_drift_report.values()), default=0.0
    )
    critical_fraction = len(combined_critical_features) / float(total_features)

    triggered_rules: List[Dict[str, Any]] = []
    retrain_required = False
    rollback_required = False

    # Regra 1: Divergência Populacional Severa (PSI >= 0.25)
    if len(critical_psi_features) >= cfg.min_drifted_features_for_retrain or (
        len(critical_psi_features) >= 1
        and any(f in CRITICAL_AGRONOMIC_FEATURES for f in critical_psi_features)
    ):
        retrain_required = True
        if max_psi >= cfg.psi_critical and len(critical_psi_features) >= cfg.min_drifted_features_for_retrain:
            rollback_required = True
        triggered_rules.append(
            {
                "rule_id": "RULE-PSI-CRITICAL",
                "metric": "Population Stability Index (PSI)",
                "observed_value": round(max_psi, 4),
                "threshold": cfg.psi_critical,
                "affected_features": critical_psi_features,
                "action": "TRIGGER_RETRAINING_AND_EVALUATE_ROLLBACK",
                "description": (
                    f"PSI crítico (>= {cfg.psi_critical}) detectado em {len(critical_psi_features)} "
                    f"feature(s): {critical_psi_features}."
                ),
            }
        )

    # Regra 2: Drift Estatístico de Kolmogorov-Smirnov (p < 0.01 e D >= 0.10)
    if len(critical_ks_features) >= cfg.min_drifted_features_for_retrain:
        retrain_required = True
        triggered_rules.append(
            {
                "rule_id": "RULE-KS-CONCEPT-DATA-DRIFT",
                "metric": "Kolmogorov-Smirnov Two-Sample Test",
                "observed_value": round(max_ks_stat, 4),
                "threshold": f"p < {cfg.ks_alpha_critical} & D >= {cfg.ks_statistic_min_effect}",
                "affected_features": critical_ks_features,
                "action": "TRIGGER_RETRAINING_PIPELINE",
                "description": (
                    f"Rejeição da hipótese nula de estabilidade no teste KS (p < {cfg.ks_alpha_critical}) "
                    f"em {len(critical_ks_features)} features: {critical_ks_features}."
                ),
            }
        )

    # Regra 3: Fração Sistêmica de Features com Drift Crítico (>= 20%)
    if critical_fraction >= cfg.critical_feature_fraction_threshold:
        retrain_required = True
        triggered_rules.append(
            {
                "rule_id": "RULE-MULTIVARIATE-DRIFT-FRACTION",
                "metric": "Drifted Features Ratio",
                "observed_value": round(critical_fraction, 4),
                "threshold": cfg.critical_feature_fraction_threshold,
                "affected_features": combined_critical_features,
                "action": "TRIGGER_FULL_PIPELINE_RETRAINING",
                "description": (
                    f"{critical_fraction * 100:.1f}% das features visuais apresentam drift crítico "
                    f"(limite: {cfg.critical_feature_fraction_threshold * 100:.1f}%)."
                ),
            }
        )

    # Regra 4: Acúmulo de Novas Imagens Rotuladas em Campo (>= 500 amostras)
    if new_labeled_samples >= cfg.min_new_labeled_samples_trigger:
        retrain_required = True
        triggered_rules.append(
            {
                "rule_id": "RULE-DATA-VOLUME-CADENCE",
                "metric": "New Audited Labeled Images",
                "observed_value": new_labeled_samples,
                "threshold": cfg.min_new_labeled_samples_trigger,
                "affected_features": [],
                "action": "TRIGGER_INCREMENTAL_RETRAINING",
                "description": (
                    f"Volume de novas imagens laudadas por fitopatologistas ({new_labeled_samples}) "
                    f"atingiu a meta de requalificação (>= {cfg.min_new_labeled_samples_trigger})."
                ),
            }
        )

    # Regra 5: Janela Fenológica Trimestral Expirada (>= 90 dias)
    if days_since_last_train >= cfg.max_days_without_retrain:
        retrain_required = True
        triggered_rules.append(
            {
                "rule_id": "RULE-PHENOLOGICAL-WINDOW-90D",
                "metric": "Days Since Last Homologated Training",
                "observed_value": days_since_last_train,
                "threshold": cfg.max_days_without_retrain,
                "affected_features": [],
                "action": "TRIGGER_SEASONAL_RETRAINING",
                "description": (
                    f"Decurso de {days_since_last_train} dias desde o último treinamento "
                    f"(limite do ciclo fenológico: {cfg.max_days_without_retrain} dias)."
                ),
            }
        )

    # Regra 6: Degradação Direta de Performance Agronômica (F1-Macro < 0.50)
    if current_f1_macro is not None and current_f1_macro < cfg.min_acceptable_f1_macro:
        retrain_required = True
        rollback_required = True
        triggered_rules.append(
            {
                "rule_id": "RULE-PERFORMANCE-DEGRADATION-F1",
                "metric": "Field Audited F1-Macro",
                "observed_value": round(current_f1_macro, 4),
                "threshold": cfg.min_acceptable_f1_macro,
                "affected_features": critical_agronomic_drifted,
                "action": "IMMEDIATE_ROLLBACK_TO_V0.9.0_AND_RETRAIN",
                "description": (
                    f"Queda severa de performance preditiva em campo (F1-Macro = {current_f1_macro:.4f} "
                    f"< {cfg.min_acceptable_f1_macro})."
                ),
            }
        )

    # Regra 7: Alta Taxa de Quarentena no SchemaValidator (> 5.0%)
    if quarantine_rate > cfg.max_quarantine_rejection_rate:
        rollback_required = True
        triggered_rules.append(
            {
                "rule_id": "RULE-QUARANTINE-SPIKE",
                "metric": "Schema Quarantine Rejection Rate",
                "observed_value": round(quarantine_rate, 4),
                "threshold": cfg.max_quarantine_rejection_rate,
                "affected_features": [],
                "action": "IMMEDIATE_SENSOR_AUDIT_AND_ROLLBACK",
                "description": (
                    f"Taxa de rejeição de schema ({quarantine_rate * 100:.2f}%) excedeu o limite "
                    f"operacional de {cfg.max_quarantine_rejection_rate * 100:.1f}%."
                ),
            }
        )

    # Inferir as fontes físicas/agronômicas mais prováveis com base nas features afetadas
    probable_causes: List[Dict[str, Any]] = []
    affected_set = set(combined_critical_features + combined_warning_features)
    if affected_set:
        for src_key, src_obj in FIELD_DRIFT_SOURCES_CATALOG.items():
            overlap = sorted(affected_set.intersection(src_obj.affected_features))
            if overlap:
                probable_causes.append(
                    {
                        "source_id": src_obj.source_id,
                        "source_key": src_key,
                        "name": src_obj.name,
                        "drift_taxonomy": src_obj.drift_taxonomy,
                        "matching_drifted_features": overlap,
                        "recommended_mitigation": src_obj.mitigation_and_sustainment_action,
                    }
                )

    # Determinação do nível global de alerta
    if rollback_required and retrain_required:
        alert_level = "EMERGENCY_ROLLBACK_AND_RETRAIN"
        recommended_action = (
            "Comutar imediatamente o tráfego de inferência para o modelo de backup v0.9.0 "
            "(svm_pipeline_optimal.joblib) e iniciar o pipeline de retreinamento com os lotes recentes."
        )
    elif retrain_required:
        alert_level = "CRITICAL_RETRAIN_REQUIRED"
        recommended_action = (
            "Disparar pipeline automatizado de retreinamento (GridSearchCV + Platt Scaling) "
            "para gerar versão candidata Minor (v1.1.0) no estágio Staging."
        )
    elif rollback_required:
        alert_level = "CRITICAL_QUARANTINE_ALERT"
        recommended_action = (
            "Acionar auditoria de hardware/câmeras em campo e avaliar rollback preventivo."
        )
    elif len(combined_warning_features) > 0 or len(warning_psi_features) > 0:
        alert_level = "WARNING_MONITORING"
        recommended_action = (
            "Manter modelo v1.0.0 ativo, reduzir janela de amostragem de telemetria para 24h "
            "e priorizar rotulagem de amostras limítrofes (Active Learning)."
        )
    else:
        alert_level = "NORMAL"
        recommended_action = (
            "Distribuições ópticas e texturais estáveis. Manter operação padrão do modelo v1.0.0."
        )

    return {
        "alert_level": alert_level,
        "retrain_required": retrain_required,
        "rollback_required": rollback_required,
        "recommended_action": recommended_action,
        "summary_metrics": {
            "total_monitored_features": total_features,
            "n_critical_features": len(combined_critical_features),
            "n_warning_features": len(combined_warning_features),
            "n_critical_psi_features": len(critical_psi_features),
            "n_critical_ks_features": len(critical_ks_features),
            "critical_feature_fraction": round(critical_fraction, 4),
            "max_psi": round(max_psi, 6),
            "mean_psi": round(mean_psi, 6),
            "max_ks_statistic": round(max_ks_stat, 6),
            "new_labeled_samples": new_labeled_samples,
            "days_since_last_train": days_since_last_train,
            "current_f1_macro": current_f1_macro,
            "quarantine_rate": round(quarantine_rate, 4),
        },
        "critical_features_list": combined_critical_features,
        "warning_features_list": combined_warning_features,
        "triggered_rules": triggered_rules,
        "probable_field_causes": probable_causes,
    }


# ==============================================================================
# 5. ORQUESTRADOR DE MONITORAMENTO CONTÍNUO E SIMULAÇÃO DE CENÁRIOS DE CAMPO
# ==============================================================================


class DriftMonitoringEngine:
    """
    Motor de monitoramento contínuo de MLOps do projeto SugarVision.
    Gerencia a distribuição de referência oficial (partição de treino da ABT),
    avalia lotes de produção e exporta configurações e relatórios de governança.
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        thresholds: Optional[DriftThresholds] = None,
        features: Optional[List[str]] = None,
    ) -> None:
        self.thresholds = thresholds or DriftThresholds()
        self.features = (
            features
            if features is not None
            else [
                c for c in MONITORED_VISUAL_FEATURES if c in reference_df.columns
            ]
        )
        if not self.features:
            raise ValueError("Nenhuma feature visual válida encontrada no reference_df.")
        self.reference_df = reference_df[self.features].copy()

    @classmethod
    def from_abt_file(
        cls,
        abt_path: Optional[Union[str, Path]] = None,
        thresholds: Optional[DriftThresholds] = None,
    ) -> "DriftMonitoringEngine":
        """
        Inicializa o motor carregando a partição de treinamento ('train') da
        Tabela Analítica Base oficial (`data/processed/abt_sanidade_vegetal.csv`).
        """
        base_dir = Path(__file__).resolve().parent.parent
        resolved_path = (
            Path(abt_path)
            if abt_path
            else (base_dir / "data" / "processed" / "abt_sanidade_vegetal.csv")
        )
        if not resolved_path.exists():
            raise FileNotFoundError(f"Arquivo ABT não encontrado em: {resolved_path}")

        df = pd.read_csv(resolved_path)
        if "split_partition" in df.columns:
            ref_df = df[df["split_partition"] == "train"].copy()
        else:
            ref_df = df.copy()

        return cls(reference_df=ref_df, thresholds=thresholds)

    def inspect_batch(
        self,
        current_df: pd.DataFrame,
        batch_name: str = "production_batch",
        new_labeled_samples: int = 0,
        days_since_last_train: int = 15,
        current_f1_macro: Optional[float] = None,
        quarantine_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Executa a bateria completa de testes estatísticos (PSI + KS) em um lote de
        dados e avalia os gatilhos de retreinamento e rollback.
        """
        feature_report = evaluate_feature_drift(
            reference_df=self.reference_df,
            current_df=current_df,
            features=self.features,
            thresholds=self.thresholds,
        )
        decision = evaluate_retraining_triggers(
            feature_drift_report=feature_report,
            thresholds=self.thresholds,
            new_labeled_samples=new_labeled_samples,
            days_since_last_train=days_since_last_train,
            current_f1_macro=current_f1_macro,
            quarantine_rate=quarantine_rate,
        )
        return {
            "batch_name": batch_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_reference_samples": int(len(self.reference_df)),
            "n_current_samples": int(len(current_df)),
            "decision": decision,
            "feature_metrics": feature_report,
        }


def export_mlops_drift_artifacts(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Executa a auditoria de monitoramento contínuo sobre a base real do projeto
    (partição de teste cego vs. treino) e simula 3 cenários de campo (Iluminação Sazonal,
    Novos Sensores e Estádios Fenológicos), salvando os artefatos JSON em `models/`.
    """
    root_dir = base_dir or Path(__file__).resolve().parent.parent
    abt_path = root_dir / "data" / "processed" / "abt_sanidade_vegetal.csv"
    models_dir = root_dir / "models"
    sprint_dir = (
        root_dir
        / "entregas"
        / "Monitoramento Continuo de Drift e Politica de Retreinamento MLOps"
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    sprint_dir.mkdir(parents=True, exist_ok=True)

    thresholds = DriftThresholds()

    # 1. Exportar arquivo de configuração de fontes de drift e thresholds
    config_payload = {
        "config_name": "sugarvision_mlops_drift_monitoring_config",
        "version": "1.0.0",
        "project": "Sanidade-Vegetal (SugarVision)",
        "sprint": "Sprint 5 — MLOps Continuous Monitoring & Drift Governance",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "active_production_model": {
            "version": "1.0.0",
            "artifact_path": "models/sanidade_vegetal_pipeline_final.joblib",
            "rollback_version": "0.9.0",
            "rollback_artifact_path": "models/svm_pipeline_optimal.joblib",
        },
        "monitored_visual_features": MONITORED_VISUAL_FEATURES,
        "critical_agronomic_features": CRITICAL_AGRONOMIC_FEATURES,
        "thresholds": asdict(thresholds),
        "field_drift_sources_catalog": get_field_drift_sources_mapping(),
    }

    config_path = models_dir / "mlops_drift_monitoring_config.json"
    sprint_config_path = sprint_dir / "mlops_drift_monitoring_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)
    with open(sprint_config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)

    # 2. Avaliar cenários reais e simulados usando a ABT oficial
    df_all = pd.read_csv(abt_path)
    df_train = df_all[df_all["split_partition"] == "train"].copy()
    df_test = df_all[df_all["split_partition"] == "test"].copy()

    engine = DriftMonitoringEngine(reference_df=df_train, thresholds=thresholds)

    # Amostra estratificada de produção em regime nominal estável
    df_nominal = df_train.sample(n=500, random_state=42).copy()

    # Cenário A: Operação Nominal Estável em Campo (Sem Drift)
    scenario_nominal = engine.inspect_batch(
        current_df=df_nominal,
        batch_name="Cenario_01_Operacao_Nominal_Estavel",
        new_labeled_samples=120,
        days_since_last_train=18,
        current_f1_macro=0.6048,
        quarantine_rate=0.005,
    )

    # Cenário B: Mudança de Iluminação Sazonal Moderada (Alerta Preventivo WARNING)
    rng = np.random.default_rng(42)
    df_seasonal = df_nominal.copy()
    df_seasonal["mean_hue"] = np.clip(
        df_seasonal["mean_hue"] + rng.normal(4.8, 2.0, size=len(df_seasonal)),
        0.0,
        180.0,
    )
    scenario_seasonal_warning = engine.inspect_batch(
        current_df=df_seasonal,
        batch_name="Cenario_02_Transicao_Sazonal_Iluminacao_Moderada",
        new_labeled_samples=240,
        days_since_last_train=45,
        current_f1_macro=0.5710,
        quarantine_rate=0.018,
    )

    # Cenário C: Drift Severo Combinado (Novas Câmeras + Nova Cultivar em Senescência -> Retreino)
    df_severe = df_nominal.copy()
    df_severe["exg_index"] = df_severe["exg_index"] - 18.5
    df_severe["exr_index"] = df_severe["exr_index"] + 16.0
    df_severe["indice_clorose_necrose"] = (
        df_severe["indice_clorose_necrose"] * 1.85 + 6.0
    )
    df_severe["laplacian_var"] = df_severe["laplacian_var"] * 2.40 + 80.0
    df_severe["glcm_contrast"] = df_severe["glcm_contrast"] * 1.75 + 12.0
    scenario_severe_drift = engine.inspect_batch(
        current_df=df_severe,
        batch_name="Cenario_03_Drift_Severo_Novas_Cultivares_e_Sensores",
        new_labeled_samples=620,
        days_since_last_train=95,
        current_f1_macro=0.4620,
        quarantine_rate=0.068,
    )

    # Cenário D: Auditoria Real entre Partições (Teste Cego vs. Treino)
    scenario_blind_test_audit = engine.inspect_batch(
        current_df=df_test,
        batch_name="Cenario_04_Auditoria_Real_Particao_Teste_vs_Treino",
        new_labeled_samples=0,
        days_since_last_train=10,
        current_f1_macro=0.6048,
        quarantine_rate=0.0,
    )

    report_payload = {
        "report_title": "Relatório de Telemetria de Data/Concept Drift e Gatilhos de Retreinamento",
        "project": "Sanidade-Vegetal (SugarVision)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenarios_evaluated": {
            "nominal_operation_stable": scenario_nominal,
            "seasonal_illumination_shift": scenario_seasonal_warning,
            "severe_multivariate_field_drift": scenario_severe_drift,
            "real_blind_test_partition_audit": scenario_blind_test_audit,
        },
    }

    report_path = models_dir / "mlops_drift_monitoring_report.json"
    sprint_report_path = sprint_dir / "mlops_drift_monitoring_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)
    with open(sprint_report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Configuração de Drift salva em: {config_path}")
    logger.info(f"Relatório de Monitoramento salvo em: {report_path}")
    return report_payload


if __name__ == "__main__":
    export_mlops_drift_artifacts()
