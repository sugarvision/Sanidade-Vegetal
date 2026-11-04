"""
Interface Interativa para Demonstração em Tempo Real — Demoday 5
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 5 — Framework SEMMA (Fase: Assess & Deploy Readiness)
Responsável: Cesar (Lead Técnico & Visão Computacional)

Como Executar:
    streamlit run src/app_demoday5_sugarvision.py
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

# Ajuste de path para importação dos módulos locais
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.vision_inference_engine import VisionInferenceEngine

# Configuração da Página do Streamlit
st.set_page_config(
    page_title="SugarVision AI — Demoday 5",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização CSS Customizada para Apresentação Profissional de Alto Nível
st.markdown(
    """
    <style>
    /* 1. Ocultação de menus padrão de desenvolvimento em inglês (3 pontinhos e Deploy) */
    #MainMenu {
        visibility: hidden !important;
        display: none !important;
    }
    .stDeployButton {
        visibility: hidden !important;
        display: none !important;
    }
    header[data-testid="stHeader"] {
        visibility: hidden !important;
        height: 0px !important;
    }
    footer {
        visibility: hidden !important;
        display: none !important;
    }
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.2rem !important;
        max-width: 98% !important;
    }

    /* 2. Hero Header Executivo */
    .hero-banner {
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 70%, #f7fee7 100%);
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 16px 22px;
        margin-bottom: 20px;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.07);
    }
    .hero-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .hero-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #14532d;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.25;
    }
    .hero-badge {
        background: #dcfce7;
        color: #166534;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 9999px;
        border: 1px solid #86efac;
        white-space: nowrap;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #4b5563;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* 3. Cards de Métricas e KPIs Padronizados */
    .metric-card-wrapper {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        min-height: 125px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        cursor: help;
    }
    .metric-card-wrapper:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }
    .metric-card-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.03em;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-card-title .info-icon {
        font-size: 0.9rem;
        color: #94a3b8;
    }
    .metric-card-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .metric-card-footer {
        font-size: 0.80rem;
        color: #64748b;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Badges de Diagnóstico e SLA */
    .badge-laudo-healthy {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.98rem;
        display: inline-block;
        border: 1px solid #86efac;
    }
    .badge-laudo-rust {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.98rem;
        display: inline-block;
        border: 1px solid #fdba74;
    }
    .badge-laudo-other {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.98rem;
        display: inline-block;
        border: 1px solid #fca5a5;
    }
    .badge-sla-ok {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
        border: 1px solid #86efac;
    }
    .badge-sla-warn {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
        border: 1px solid #fde047;
    }

    /* 4. Painel de Imagens Alinhado e Equilibrado */
    .inspection-header {
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-weight: 700;
        font-size: 0.96rem;
        color: #1e293b;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px 8px 0 0;
        padding: 6px 10px;
        margin-bottom: 0;
    }
    .inspection-footer {
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-size: 0.80rem;
        color: #475569;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 4px 10px;
        margin-top: 0;
    }
    .stImage > img {
        border-radius: 0 !important;
        border-left: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_engine() -> VisionInferenceEngine:
    """Inicializa e coloca em cache o motor de visão computacional."""
    return VisionInferenceEngine()


def main():
    engine = load_engine()

    # ==========================================================================
    # BARRA LATERAL (SIDEBAR)
    # ==========================================================================
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/plant-under-sun.png", width=60)
        st.markdown("## **SugarVision AI**")
        st.markdown("**Diagnóstico Fitossanitário em Tempo Real**")
        st.caption("Fase: SEMMA Assess — Homologação Demoday 5")
        st.divider()

        st.markdown("### 📥 Entrada da Imagem")
        input_mode = st.radio(
            "Origem da Amostra:",
            ["🧪 Galeria de Teste Cego Curado", "📁 Upload de Nova Imagem"],
            index=0,
        )

        selected_image_input = None
        sample_meta = {}

        blind_dir = BASE_DIR / "data" / "demoday5_blind_samples"
        manifest_file = blind_dir / "manifest_teste_cego_demoday5.json"

        if input_mode == "🧪 Galeria de Teste Cego Curado":
            if manifest_file.exists():
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                sample_options = {
                    s["filename"]: f"{s['filename']} ({s['agronomic_label']})"
                    for s in manifest_data.get("samples", [])
                }
                chosen_fn = st.selectbox(
                    "Selecione uma amostra desafiadora de campo:",
                    list(sample_options.keys()),
                    format_func=lambda x: sample_options[x],
                )
                if chosen_fn:
                    sample_path = blind_dir / chosen_fn
                    if sample_path.exists():
                        selected_image_input = sample_path
                        for s in manifest_data.get("samples", []):
                            if s["filename"] == chosen_fn:
                                sample_meta = s
            else:
                st.warning("Diretório de teste cego não encontrado.")
        else:
            uploaded_file = st.file_uploader(
                "Envie uma foto de folha de soja ou cana (JPG, PNG, JPEG):",
                type=["jpg", "jpeg", "png"],
            )
            if uploaded_file is not None:
                selected_image_input = uploaded_file.read()

        st.divider()
        st.markdown("### ⚙️ Arquitetura Homologada")
        st.markdown(
            """
            - **Classificador:** SVM RBF ($C=10.0, \\gamma=0.01$)
            - **Calibração:** Platt Scaling (Sigmoide)
            - **Anti-Leakage:** Pipeline Atômico
            - **SLA Operacional:** $\\Delta_{\\mathrm{margin}} \\ge 0.35$
            """
        )

        st.divider()
        st.caption(
            "**Detentores dos Direitos Autorais (Copyright © 2026):**\n"
            "Heitor Vitti, César Bovo, Guilherme Romanzotti, "
            "Marvin Cristhian, Elisa Alcantara e Luis Otávio."
        )

    # ==========================================================================
    # PAINEL PRINCIPAL: HERO BANNER EXECUTIVO
    # ==========================================================================
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title-row">
                <h1 class="hero-title">🌿 SugarVision — Diagnóstico Fitossanitário Inteligente</h1>
                <span class="hero-badge">Demoday 5 • 11/12/2026</span>
            </div>
            <p class="hero-subtitle">
                Demonstração Interativa em Tempo Real para a Banca Avaliadora — Detecção de Ferrugem e Patologias Foliares
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if selected_image_input is None:
        st.info("👈 Por favor, selecione uma amostra na barra lateral ou faça o upload de uma imagem foliar para iniciar o diagnóstico.")
        return

    # Execução da Inferência em Tempo Real com suporte a features curadas do teste cego
    with st.spinner("Processando tensores espectrais e calculando mapas de textura..."):
        try:
            ref_feat = sample_meta.get("reference_features") if input_mode == "🧪 Galeria de Teste Cego Curado" else None
            result = engine.diagnose_image(selected_image_input, reference_features=ref_feat)
        except Exception as e:
            st.error(f"Erro ao processar imagem: {str(e)}")
            return

    # --------------------------------------------------------------------------
    # BLOCO 1: CARDS DE MÉTRICAS E LAUDO EXECUTIVO (HTML/CSS com Tooltip Completo)
    # --------------------------------------------------------------------------
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    # Card 1: Laudo Fitossanitário
    c_class = result["consolidated_class"]
    if "HEALTHY" in c_class:
        badge_html = f'<span class="badge-laudo-healthy">✅ {c_class}</span>'
    elif "RUST" in c_class:
        badge_html = f'<span class="badge-laudo-rust">⚠️ {c_class}</span>'
    else:
        badge_html = f'<span class="badge-laudo-other">🛑 {c_class}</span>'

    with col_m1:
        st.markdown(
            f"""
            <div class="metric-card-wrapper" title="Diagnóstico consolidado tripartido: Sadia vs. Ferrugem vs. Outras Doenças.">
                <div>
                    <div class="metric-card-title">
                        <span>Laudo Consolidado</span>
                        <span class="info-icon">ℹ️</span>
                    </div>
                    <div style="margin: 8px 0;">{badge_html}</div>
                </div>
                <div class="metric-card-footer">
                    Patologia: <strong>{result['specific_class']}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Card 2: Probabilidade Calibrada
    with col_m2:
        st.markdown(
            f"""
            <div class="metric-card-wrapper" title="Probabilidade posterior calibrada pelo método de Platt Scaling (regressão logística sigmoidal sobre as distâncias do SVM).">
                <div>
                    <div class="metric-card-title">
                        <span>Probabilidade Calibrada</span>
                        <span class="info-icon">ℹ️</span>
                    </div>
                    <div class="metric-card-value">{result['confidence_pct']:.1f}%</div>
                </div>
                <div class="metric-card-footer" style="color: #16a34a; font-weight: 600;">
                    ↑ Escalonamento de Platt
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Card 3: Margem Geométrica (Hiperplano) - COM TOOLTIP COMPLETO AO PASSAR O MOUSE
    with col_m3:
        st.markdown(
            f"""
            <div class="metric-card-wrapper" title="Margem Geométrica de Separação em Relação ao Hiperplano SVM (Δ_margin). Representa a distância funcional entre a classe predita e a segunda classe mais provável. Valores Δ ≥ 0.35 indicam zona de alta certeza agronômica.">
                <div>
                    <div class="metric-card-title">
                        <span>Margem Geométrica (Hiperplano)</span>
                        <span class="info-icon">ℹ️</span>
                    </div>
                    <div class="metric-card-value">Δ = {result['delta_margin']:.3f}</div>
                </div>
                <div class="metric-card-footer">
                    Entropia Shannon: <strong>{result['shannon_entropy']:.2f} bits</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Card 4: Salvaguarda de Decisão (SLA)
    sla_badge = (
        f'<span class="badge-sla-ok">🟢 {result["sla_status"]}</span>'
        if result["sla_level"] == "APPROVED"
        else f'<span class="badge-sla-warn">🟡 {result["sla_status"]}</span>'
    )
    sla_short_msg = (
        "Margem segura — Laudo liberado com segurança"
        if result["sla_level"] == "APPROVED"
        else "Zona de incerteza — Checagem em campo indicada"
    )

    with col_m4:
        st.markdown(
            f"""
            <div class="metric-card-wrapper" title="SLA Operacional de Campo: Valida se a amostra possui margem hiperdimensional suficiente para dispensa de verificação humana presencial.">
                <div>
                    <div class="metric-card-title">
                        <span>Salvaguarda de Decisão (SLA)</span>
                        <span class="info-icon">ℹ️</span>
                    </div>
                    <div style="margin: 6px 0;">{sla_badge}</div>
                </div>
                <div class="metric-card-footer">
                    {sla_short_msg}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # BLOCO 2: VISUALIZAÇÃO EM TRIPLO PAINEL ALINHADA (ORIGINAL, GLCM, MÁSCARA)
    # --------------------------------------------------------------------------
    st.markdown("### 🔬 Inspeção Visual de Visão Computacional")

    col_v1, col_v2, col_v3 = st.columns(3)

    with col_v1:
        st.markdown('<div class="inspection-header">1. Imagem Original</div>', unsafe_allow_html=True)
        st.image(result["original_image"], use_container_width=True)
        st.markdown(
            f'<div class="inspection-footer">Entrada ({result["features"]["width"]}x{result["features"]["height"]}) — {result["size_kb"]:.1f} KB</div>',
            unsafe_allow_html=True,
        )

    with col_v2:
        st.markdown('<div class="inspection-header">2. Mapa de Textura (Haralick / GLCM)</div>', unsafe_allow_html=True)
        st.image(result["texture_map"], use_container_width=True)
        st.markdown(
            f'<div class="inspection-footer">Micro-relevo rugoso (Contraste GLCM = {result["features"]["glcm_contrast"]:.1f})</div>',
            unsafe_allow_html=True,
        )

    with col_v3:
        st.markdown('<div class="inspection-header">3. Máscara de Segmentação de Lesões</div>', unsafe_allow_html=True)
        st.image(result["lesion_mask"], use_container_width=True)
        st.markdown(
            f'<div class="inspection-footer">Segmentação ExR/Otsu (Excesso de Vermelho = {result["features"]["exr_index"]:.1f})</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # BLOCO 3: DISTRIBUIÇÃO PROBABILÍSTICA MULTICLASSE E METADADOS
    # --------------------------------------------------------------------------
    col_d1, col_d2 = st.columns([3, 2])

    with col_d1:
        st.markdown("### 📊 Distribuição de Probabilidades Calibradas por Classe")
        df_probas = pd.DataFrame(
            list(result["all_probabilities"].items()),
            columns=["Classe Patológica", "Probabilidade (%)"],
        ).sort_values(by="Probabilidade (%)", ascending=True)

        st.bar_chart(df_probas.set_index("Classe Patológica"), horizontal=True)

    with col_d2:
        st.markdown("### 📋 Ficha Agronômica da Amostra")
        if sample_meta:
            st.markdown(f"**Referência de Campo:** {sample_meta.get('agronomic_label')}")
            st.markdown(f"**Nível de Severidade:** `{sample_meta.get('severity_scale')}`")
            st.markdown(f"**Desafio Visual:** {sample_meta.get('visual_challenge')}")
            st.info(f"**Recomendação de Manejo:** {sample_meta.get('recommended_action')}")
        else:
            feat = result["features"]
            st.markdown(
                f"""
                - **Índice de Excesso de Verde (ExG):** `{feat['exg_index']:.2f}`
                - **Índice de Excesso de Vermelho (ExR):** `{feat['exr_index']:.2f}`
                - **Razão Espectral R/G:** `{feat['rg_ratio']:.3f}`
                - **Índice de Clorose e Necrose (ICN):** `{feat['indice_clorose_necrose']:.3f}`
                - **Homogeneidade GLCM:** `{feat['glcm_homogeneity']:.3f}`
                - **Nitidez Focal (Laplaciano):** `{feat['laplacian_var']:.1f}`
                """
            )

    # --------------------------------------------------------------------------
    # BLOCO 4: PAINEL EXPANSÍVEL DE INTERPRETABILIDADE FITOPATOLÓGICA
    # --------------------------------------------------------------------------
    with st.expander("📖 Fundamentação Fitopatológica e Interpretabilidade do Modelo"):
        st.markdown(
            """
            #### Coerência Biológica dos Atributos Selecionados
            1. **Ferrugem (Puccinia spp. / Phakopsora pachyrhizi):**
               - A penetração fúngica e posterior erupção dos uredínios rompe a epiderme, liberando massa amarronzada/alaranjada de esporos.
               - Isso acarreta queda do **ExG ($2G - R - B$)** e aumento concomitante do **ExR ($1.4R - G$)**, refletido na textura áspera do mapa GLCM.
            2. **Folhas Sadias:**
               - Apresentam alta absorção na banda vermelha pela clorofila sadia e forte reflectância no verde, gerando $ExG > 40$ e contraste de Haralick plano e homogêneo.
            3. **Mosaico vs. Amarelecimento (Yellow Leaf):**
               - Manifestam-se por cloroses internervais sem rugosidade cuticular superficial, diferenciados primariamente pela dispersão angular do matiz HSV (`hue_dispersion`).
            """
        )


if __name__ == "__main__":
    main()
