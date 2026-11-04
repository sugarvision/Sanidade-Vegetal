"""
Módulo de Inspeção Visual de Casos de Fronteira e Discordância no Conjunto de Teste
Projeto: Sanidade-Vegetal (SugarVision)
Sprint: 4 — Framework SEMMA (Fase: Assess — Diagnóstico de Fronteira Geométrica)
Responsável: Cesar (Lead Técnico & Visão Computacional)

Atividades e Checklist Coberto:
1. Mapear as imagens com menor margem geométrica de separação em relação ao hiperplano do SVM.
2. Identificar padrões fotográficos recorrentes nos erros (iluminação inadequada, sobreposição de folhas, necrose senescente).
3. Documentar visualmente os casos limítrofes entre ferrugem inicial e pequenas manchas foliares.
4. Consolidar os dados e artefatos diagnósticos para subsidiar a seção de limitações visuais do Relato da Solução Vol. 4.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy

# Compatibilidade reversa para ColumnTransformer deserializado
import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _ct._RemainderColsList = _RemainderColsList

# Logger estruturado
logger = logging.getLogger("SugarVision.BoundaryInspection")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

CLASS_NAMES_MAP: Dict[int, str] = {
    0: "GRASSY SHOOT",
    1: "HEALTHY",
    2: "LEAF SCALD",
    3: "MOSAIC",
    4: "RED ROT",
    5: "RUST",
    6: "YELLOW LEAF",
}


class VisualBoundaryInspector:
    """
    Executa a análise de casos de fronteira, distâncias geométricas ao hiperplano,
    e taxonomia fotográfica de erros no conjunto de teste.
    """

    def __init__(
        self,
        model_path: str = "models/svm_pipeline_optimal.joblib",
        data_path: str = "data/processed/abt_sanidade_vegetal.csv",
        misclassified_path: str = "models/svm_misclassified_test_instances.csv",
    ) -> None:
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.misclassified_path = Path(misclassified_path)

        self.model = None
        self.df_test = None
        self.feature_cols: List[str] = []
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None

    def load_resources(self) -> None:
        """Carrega modelo serializado e prepara partições de teste."""
        logger.info(f"Carregando estimador de: {self.model_path}")
        self.model = joblib.load(self.model_path)

        logger.info(f"Carregando dados tabulares de: {self.data_path}")
        df = pd.read_csv(self.data_path)
        self.df_test = df[df["split_partition"] == "test"].copy().reset_index(drop=True)

        ignored_cols = [
            "sample_id", "filepath", "filename", "relative_path",
            "split_partition", "class_label", "target_binary", "target_multiclass"
        ]
        self.feature_cols = [c for c in df.columns if c not in ignored_cols]
        self.X_test = self.df_test[self.feature_cols].copy()
        self.y_test = self.df_test["target_multiclass"].copy()

    def compute_geometric_margins(self) -> pd.DataFrame:
        """
        Calcula as distâncias geométricas ao hiperplano e margens relativas (Δ_margin)
        para cada amostra do conjunto de teste independente.
        """
        if self.model is None or self.X_test is None:
            self.load_resources()

        # decision_function shape: (N, 7) para 7 classes OvR ou (N, 21) se OvO puro
        dfunc = self.model.decision_function(self.X_test)
        probas = self.model.predict_proba(self.X_test)
        y_preds = self.model.predict(self.X_test)

        df_margins = self.df_test.copy()
        df_margins["y_pred"] = y_preds
        df_margins["y_pred_label"] = df_margins["y_pred"].map(CLASS_NAMES_MAP)
        df_margins["is_correct"] = (df_margins["target_multiclass"] == df_margins["y_pred"]).astype(int)

        # Se multiclasse com pontuação por classe (N, 7):
        if dfunc.ndim == 2 and dfunc.shape[1] == 7:
            # Ordena scores de cada amostra decrescentemente
            sorted_scores = np.sort(dfunc, axis=1)[:, ::-1]
            top1_scores = sorted_scores[:, 0]
            top2_scores = sorted_scores[:, 1]
            delta_margin = top1_scores - top2_scores

            # Identifica segunda classe mais provável
            sorted_classes = np.argsort(dfunc, axis=1)[:, ::-1]
            second_pred = sorted_classes[:, 1]

            df_margins["decision_score_top1"] = top1_scores
            df_margins["decision_score_top2"] = top2_scores
            df_margins["delta_geometric_margin"] = delta_margin
            df_margins["runner_up_class_id"] = second_pred
            df_margins["runner_up_class_label"] = df_margins["runner_up_class_id"].map(CLASS_NAMES_MAP)
        else:
            # Caso 1D binário ou formato alternativo
            delta_margin = np.abs(dfunc)
            df_margins["delta_geometric_margin"] = delta_margin
            df_margins["decision_score_top1"] = dfunc
            df_margins["decision_score_top2"] = 0.0
            df_margins["runner_up_class_id"] = -1
            df_margins["runner_up_class_label"] = "N/A"

        # Entropia de Shannon da distribuição de probabilidade calibrada
        # H = -sum(p * log2(p))
        eps = 1e-9
        shannon_entropy = entropy(probas + eps, base=2, axis=1)
        df_margins["shannon_entropy"] = shannon_entropy
        df_margins["max_probability"] = np.max(probas, axis=1)

        # Flag de proximidade da fronteira: menor margem geométrica
        margin_threshold = float(np.percentile(df_margins["delta_geometric_margin"], 20))
        df_margins["is_near_boundary"] = (df_margins["delta_geometric_margin"] <= margin_threshold).astype(int)

        # Ordena da menor para a maior margem (amostras mais limítrofes primeiro)
        df_margins = df_margins.sort_values(by="delta_geometric_margin", ascending=True).reset_index(drop=True)
        return df_margins

    def identify_photographic_error_patterns(self, df_margins: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Analisa as instâncias com erro de predição e categoriza os padrões fotográficos
        recorrentes (iluminação inadequada, sobreposição de folhas, necrose senescente, etc.).
        """
        df_errors = df_margins[df_margins["is_correct"] == 0].copy()

        # Limiares fotométricos baseados nos quantis da base de treino/teste
        sat_p15 = float(df_margins["std_saturation"].quantile(0.15))
        sat_p85 = float(df_margins["std_saturation"].quantile(0.85))
        lap_p15 = float(df_margins["laplacian_var"].quantile(0.15))
        icn_p85 = float(df_margins["indice_clorose_necrose"].quantile(0.85))

        patterns: List[str] = []

        for _, row in df_errors.iterrows():
            causes = []

            # 1. Iluminação Inadequada (Especularidade extrema ou subexposição/sombras fortes)
            if row["std_saturation"] < sat_p15 or row["std_saturation"] > sat_p85:
                causes.append("Iluminação Inadequada / Especularidade")

            # 2. Desfocagem / Macro Blur / Sobreposição fora de foco
            if row["laplacian_var"] < lap_p15:
                causes.append("Desfocagem / Sobreposição de Folhas")

            # 3. Necrose Senescente vs Patológica (Clorose avançada confundindo com manchas)
            if row["class_label"] in ["HEALTHY", "YELLOW LEAF"] and row["y_pred_label"] in ["LEAF SCALD", "RED ROT"]:
                causes.append("Necrose Senescente / Clorose Fisiológica")
            elif row["indice_clorose_necrose"] > icn_p85 and row["class_label"] != row["y_pred_label"]:
                causes.append("Necrose Senescente / Clorose Fisiológica")

            # 4. Ambiguidade de Micropústulas e Manchas Iniciais (Rust vs Leaf Scald / Red Rot)
            if (row["class_label"] == "RUST" and row["y_pred_label"] in ["LEAF SCALD", "RED ROT"]) or \
               (row["class_label"] in ["LEAF SCALD", "RED ROT"] and row["y_pred_label"] == "RUST"):
                causes.append("Ambiguidade Ferrugem Inicial vs Manchas")

            if not causes:
                causes.append("Ambiguidade Espectro-Textural Geral")

            patterns.append("; ".join(causes))

        df_errors["photographic_error_cause"] = patterns

        # Sumariza contagens por tipologia individual
        cause_counts = {
            "Iluminação Inadequada / Especularidade": 0,
            "Desfocagem / Sobreposição de Folhas": 0,
            "Necrose Senescente / Clorose Fisiológica": 0,
            "Ambiguidade Ferrugem Inicial vs Manchas": 0,
            "Ambiguidade Espectro-Textural Geral": 0,
        }

        for p_str in patterns:
            for c_key in cause_counts.keys():
                if c_key in p_str:
                    cause_counts[c_key] += 1

        n_errors = len(df_errors)
        summary_records = []
        for cause_name, count in cause_counts.items():
            pct = (count / n_errors * 100.0) if n_errors > 0 else 0.0
            summary_records.append({
                "failure_mode": cause_name,
                "misclassified_samples_count": count,
                "percentage_of_errors": round(pct, 2),
            })

        df_pattern_summary = pd.DataFrame(summary_records).sort_values(by="misclassified_samples_count", ascending=False)

        metadata = {
            "total_test_samples": len(df_margins),
            "total_misclassified": n_errors,
            "overall_error_rate_pct": round(n_errors / len(df_margins) * 100.0, 2),
            "photographic_failure_modes": cause_counts,
            "boundary_samples_count_20pct": int(df_margins["is_near_boundary"].sum()),
            "boundary_samples_error_rate_pct": round(
                float(1.0 - df_margins[df_margins["is_near_boundary"] == 1]["is_correct"].mean()) * 100.0, 2
            ),
        }

        return df_pattern_summary, metadata

    def plot_visual_boundary_cases_and_limitations(
        self,
        df_margins: pd.DataFrame,
        df_pattern_summary: pd.DataFrame,
        output_path: str = "docs/figures/sprint4_visual_boundary_cases_and_limitations.png",
    ) -> None:
        """
        Gera o painel visual diagnóstico de 4 quadrantes cobrindo:
        1. Distribuição da Margem Geométrica (Corretos vs Incorretos)
        2. Dispersão Margem Geométrica vs Entropia de Shannon (Zona Crítica)
        3. Padrões Fotográficos Recorrentes de Erro
        4. Casos Limítrofes Ferrugem vs Manchas Foliares (ExR vs ICN)
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

        # ----------------------------------------------------------------------
        # Quadrante 1: Distribuição da Margem Geométrica Δ_margin
        # ----------------------------------------------------------------------
        correct_margins = df_margins[df_margins["is_correct"] == 1]["delta_geometric_margin"]
        error_margins = df_margins[df_margins["is_correct"] == 0]["delta_geometric_margin"]

        bins = np.linspace(0, max(df_margins["delta_geometric_margin"].max(), 2.0), 30)
        ax1.hist(correct_margins, bins=bins, alpha=0.65, color="#2ca02c", label=f"Classificações Corretas (N={len(correct_margins)})", edgecolor="black")
        ax1.hist(error_margins, bins=bins, alpha=0.75, color="#d62728", label=f"Erros de Classificação (N={len(error_margins)})", edgecolor="black")

        median_err = float(error_margins.median())
        median_corr = float(correct_margins.median())
        ax1.axvline(median_err, color="#900C3F", linestyle="--", linewidth=2, label=f"Mediana Erros (Δ={median_err:.2f})")
        ax1.axvline(median_corr, color="#1E8449", linestyle="--", linewidth=2, label=f"Mediana Corretos (Δ={median_corr:.2f})")

        ax1.set_xlabel(r"Margem Geométrica de Separação ($\Delta_{\mathrm{margin}} = f_{\mathrm{top1}} - f_{\mathrm{top2}}$)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Frequência Absoluta (Amostras de Teste)", fontsize=11, fontweight="bold")
        ax1.set_title("1. Distribuição da Margem Geométrica do Hiperplano SVM", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        # ----------------------------------------------------------------------
        # Quadrante 2: Dispersão Margem Δ vs Entropia de Shannon (Zona Crítica)
        # ----------------------------------------------------------------------
        scatter_corr = ax2.scatter(
            correct_margins,
            df_margins[df_margins["is_correct"] == 1]["shannon_entropy"],
            c="#2ca02c",
            alpha=0.55,
            s=40,
            label="Corretos",
            edgecolors="none",
        )
        scatter_err = ax2.scatter(
            error_margins,
            df_margins[df_margins["is_correct"] == 0]["shannon_entropy"],
            c="#d62728",
            alpha=0.75,
            s=50,
            marker="x",
            label="Incorretos",
        )

        # Destaca a zona crítica de alta ambiguidade (Δ < 0.35 e H > 1.2)
        ax2.axvspan(0, 0.35, color="red", alpha=0.1, label="Zona de Ambiguidade Extrema (Fronteira)")
        ax2.set_xlabel(r"Margem Geométrica ($\Delta_{\mathrm{margin}}$)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Entropia de Decisão de Shannon (bits)", fontsize=11, fontweight="bold")
        ax2.set_title("2. Incerteza Informacional vs Proximidade da Fronteira", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # ----------------------------------------------------------------------
        # Quadrante 3: Padrões Fotográficos Recorrentes nos Erros
        # ----------------------------------------------------------------------
        y_pos = np.arange(len(df_pattern_summary))
        colors_bar = ["#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#7f7f7f"][:len(df_pattern_summary)]

        bars = ax3.barh(
            y_pos,
            df_pattern_summary["misclassified_samples_count"],
            color=colors_bar,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
        )
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(df_pattern_summary["failure_mode"], fontsize=10, fontweight="medium")
        ax3.invert_yaxis()
        ax3.set_xlabel("Número de Amostras Incorretas no Teste", fontsize=11, fontweight="bold")
        ax3.set_title("3. Padrões Fotográficos Recorrentes em Erros de Classificação", fontsize=12, fontweight="bold")
        ax3.grid(axis="x", linestyle="--", alpha=0.5)

        for bar in bars:
            w = bar.get_width()
            pct = (w / len(error_margins) * 100.0)
            ax3.text(
                w + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f"{int(w)} ({pct:.1f}%)",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
            )

        # ----------------------------------------------------------------------
        # Quadrante 4: Casos Limítrofes Ferrugem vs Manchas (ExR vs ICN)
        # ----------------------------------------------------------------------
        rust_cases = df_margins[df_margins["class_label"] == "RUST"]
        scald_cases = df_margins[df_margins["class_label"] == "LEAF SCALD"]
        redrot_cases = df_margins[df_margins["class_label"] == "RED ROT"]

        ax4.scatter(
            rust_cases["exr_index"],
            rust_cases["indice_clorose_necrose"],
            c="#ff7f0e",
            label="Ferrugem (RUST)",
            alpha=0.75,
            s=45,
            edgecolors="black",
            linewidth=0.5,
        )
        ax4.scatter(
            scald_cases["exr_index"],
            scald_cases["indice_clorose_necrose"],
            c="#1f77b4",
            label="Escaldadura (LEAF SCALD)",
            alpha=0.65,
            s=40,
            marker="s",
            edgecolors="black",
            linewidth=0.5,
        )
        ax4.scatter(
            redrot_cases["exr_index"],
            redrot_cases["indice_clorose_necrose"],
            c="#d62728",
            label="Podridão Vermelha (RED ROT)",
            alpha=0.55,
            s=35,
            marker="^",
            edgecolors="black",
            linewidth=0.5,
        )

        ax4.set_xlabel("Índice de Excesso de Vermelho (ExR)", fontsize=11, fontweight="bold")
        ax4.set_ylabel("Índice de Clorose e Necrose (ICN)", fontsize=11, fontweight="bold")
        ax4.set_title("4. Espaço Espectro-Textural Limítrofe (Ferrugem vs Manchas Foliares)", fontsize=12, fontweight="bold")
        ax4.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
        ax4.grid(True, linestyle="--", alpha=0.5)

        plt.suptitle(
            "Diagnóstico de Casos de Fronteira, Incerteza Geométrica e Limitações Fotográficas\n"
            "SugarVision Sanidade Vegetal — Conjunto de Teste Independente (N=380)",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )

        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Painel visual de fronteira e limitações salvo em: {output_path}")

    def save_artifacts(
        self,
        df_margins: pd.DataFrame,
        df_pattern_summary: pd.DataFrame,
        metadata: Dict[str, Any],
        margins_csv_path: str = "models/svm_boundary_geometric_margin_cases.csv",
        summary_csv_path: str = "models/svm_visual_error_patterns_summary.csv",
        metadata_json_path: str = "models/svm_visual_boundary_inspection_metadata.json",
    ) -> None:
        """Salva arquivos de dados e metadados formais."""
        Path(margins_csv_path).parent.mkdir(parents=True, exist_ok=True)

        cols_to_save = [
            "sample_id", "filename", "dataset_source", "class_label", "y_pred_label",
            "is_correct", "delta_geometric_margin", "decision_score_top1", "decision_score_top2",
            "runner_up_class_label", "shannon_entropy", "max_probability", "is_near_boundary",
            "std_saturation", "laplacian_var", "exr_index", "indice_clorose_necrose"
        ]
        available_cols = [c for c in cols_to_save if c in df_margins.columns]
        df_margins[available_cols].to_csv(margins_csv_path, index=False)
        df_pattern_summary.to_csv(summary_csv_path, index=False)

        metadata_full = {
            "task_name": "Inspeção Visual de Casos de Fronteira e Discordância no Conjunto de Teste",
            "project": "Sanidade-Vegetal (SugarVision)",
            "sprint": "Sprint 4 — Framework SEMMA (Fase: Assess)",
            "responsible": "Cesar (Lead Técnico & Visão Computacional)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_analyzed": str(self.model_path),
            **metadata,
            "top_10_most_ambiguous_test_samples": df_margins.head(10)[
                ["sample_id", "class_label", "y_pred_label", "delta_geometric_margin", "shannon_entropy"]
            ].to_dict(orient="records"),
            "artifacts_generated": [
                margins_csv_path,
                summary_csv_path,
                "docs/figures/sprint4_visual_boundary_cases_and_limitations.png",
            ],
        }

        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata_full, f, indent=2, ensure_ascii=False)

        logger.info(f"Artefatos salvos: {margins_csv_path}, {summary_csv_path}, {metadata_json_path}")


def run_visual_boundary_inspection() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ponto de entrada para execução completa da inspeção de fronteira."""
    inspector = VisualBoundaryInspector()
    inspector.load_resources()
    df_margins = inspector.compute_geometric_margins()
    df_pattern_summary, metadata = inspector.identify_photographic_error_patterns(df_margins)

    inspector.plot_visual_boundary_cases_and_limitations(df_margins, df_pattern_summary)
    inspector.save_artifacts(df_margins, df_pattern_summary, metadata)

    logger.info("Inspeção visual de fronteira e limitações executada com sucesso!")
    return df_margins, df_pattern_summary


if __name__ == "__main__":
    run_visual_boundary_inspection()

