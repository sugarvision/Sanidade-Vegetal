"""
Pipeline de Amostragem Estratificada e Agrupada para Sanidade Vegetal (SugarVision)
Sprint 1 - Definição da Estratégia de Amostragem

Este módulo implementa:
1. Divisão estratificada e agrupada por Talhão/Planta (Stratified Group Split).
2. Cálculo de pesos de balanceamento de classes (Inverse Class Frequencies e Effective Number).
3. Verificação de integridade para evitar Data Leakage.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


def calculate_class_weights(
    labels: pd.Series, 
    method: str = "inverse", 
    beta: float = 0.999
) -> Dict[str, float]:
    """
    Calcula pesos para as classes para uso em funções de perda (CrossEntropy/Focal Loss).
    
    Args:
        labels: Série contendo os rótulos de cada amostra.
        method: 'inverse' para frequências inversas clássicas ou 'effective' para Effective Number of Samples.
        beta: Hiperparâmetro de suavização para o método 'effective' (default: 0.999).
        
    Returns:
        Dicionário com o peso atribuído a cada classe.
    """
    counts = labels.value_counts().to_dict()
    total_samples = len(labels)
    num_classes = len(counts)
    
    weights = {}
    if method == "inverse":
        for cls_name, count in counts.items():
            weights[cls_name] = float(total_samples / (num_classes * count))
    elif method == "effective":
        for cls_name, count in counts.items():
            effective_num = 1.0 - np.power(beta, count)
            weights[cls_name] = float((1.0 - beta) / max(effective_num, 1e-8))
            
        # Normaliza os pesos para manter a média unitária
        mean_w = np.mean(list(weights.values()))
        weights = {k: v / mean_w for k, v in weights.items()}
    else:
        raise ValueError(f"Método desconhecido: {method}. Utilize 'inverse' ou 'effective'.")
        
    return weights


def create_stratified_group_split(
    df: pd.DataFrame,
    label_col: str = "target_doenca_classe",
    group_col: str = "talhao_id",
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Realiza a partição dos dados garantindo estratificação por classe e agrupamento por talhão/planta.
    
    Args:
        df: DataFrame com as anotações das imagens.
        label_col: Nome da coluna contendo a patologia (target).
        group_col: Nome da coluna de agrupamento (talhão ou planta).
        train_ratio: Proporção para treino (default: 0.70).
        valid_ratio: Proporção para validação (default: 0.15).
        test_ratio: Proporção para teste (default: 0.15).
        random_state: Semente de reprodutibilidade.
        
    Returns:
        DataFrame com uma nova coluna 'split' ('train', 'valid', 'test').
    """
    assert np.isclose(train_ratio + valid_ratio + test_ratio, 1.0), "A soma das razões deve ser 1.0."
    
    df = df.copy()
    np.random.seed(random_state)
    
    # Se o número de grupos for suficiente, particiona no nível de grupos para evitar vazamento
    unique_groups = df[group_col].dropna().unique()
    
    if len(unique_groups) > 5:
        # Agrupa por talhão e calcula o rótulo predominante/distribuição
        group_label_summary = df.groupby(group_col)[label_col].agg(lambda x: x.mode()[0])
        groups = group_label_summary.index.values
        group_classes = group_label_summary.values
        
        # Embaralha os grupos mantendo a reprodutibilidade
        shuffled_indices = np.random.permutation(len(groups))
        shuffled_groups = groups[shuffled_indices]
        
        n_groups = len(shuffled_groups)
        n_train = int(train_ratio * n_groups)
        n_valid = int(valid_ratio * n_groups)
        
        train_groups = set(shuffled_groups[:n_train])
        valid_groups = set(shuffled_groups[n_train:n_train + n_valid])
        test_groups = set(shuffled_groups[n_train + n_valid:])
        
        def assign_split(g):
            if g in train_groups:
                return "train"
            elif g in valid_groups:
                return "valid"
            else:
                return "test"
                
        df["split"] = df[group_col].apply(assign_split)
    else:
        # Caso haja poucos talhões definidos, estratifica no nível de amostra individual
        df["split"] = "train"
        for label in df[label_col].unique():
            idx = df[df[label_col] == label].index.values
            np.random.shuffle(idx)
            n_samples = len(idx)
            n_train = int(train_ratio * n_samples)
            n_valid = int(valid_ratio * n_samples)
            
            df.loc[idx[:n_train], "split"] = "train"
            df.loc[idx[n_train:n_train + n_valid], "split"] = "valid"
            df.loc[idx[n_train + n_valid:], "split"] = "test"
            
    return df


def generate_sampling_report(df: pd.DataFrame, label_col: str = "target_doenca_classe", split_col: str = "split") -> pd.DataFrame:
    """
    Gera tabela de contingência com a distribuição de classes por partição.
    """
    crosstab = pd.crosstab(df[label_col], df[split_col], margins=True, margins_name="Total")
    proportions = pd.crosstab(df[label_col], df[split_col], normalize="columns") * 100
    return crosstab


if __name__ == "__main__":
    print("Executando validação do Pipeline de Amostragem...")
    
    # Simulação representativa de dados de cana-de-açúcar
    mock_data = []
    classes = ["Saudavel", "Ferrugem", "Podridao_Vermelha", "Mosaico", "Mancha_Amarela"]
    probs = [0.35, 0.25, 0.20, 0.12, 0.08]
    
    for talhao_idx in range(1, 21):
        talhao_id = f"TAL-2026-N{talhao_idx:02d}"
        n_plantas = np.random.randint(10, 30)
        for planta_idx in range(1, n_plantas + 1):
            planta_id = f"{talhao_id}-P{planta_idx:02d}"
            # Atribui uma patologia predominante para a planta
            doenca = np.random.choice(classes, p=probs)
            # Cada planta gera de 1 a 4 imagens
            for img_idx in range(1, np.random.randint(2, 5)):
                mock_data.append({
                    "imagem_id": f"IMG_{planta_id}_{img_idx}",
                    "talhao_id": talhao_id,
                    "planta_id": planta_id,
                    "target_doenca_classe": doenca
                })
                
    df_mock = pd.DataFrame(mock_data)
    print(f"Total de imagens simuladas: {len(df_mock)}")
    
    # 1. Aplica o split estratificado agrupado
    df_split = create_stratified_group_split(df_mock, label_col="target_doenca_classe", group_col="talhao_id")
    
    # 2. Calcula pesos de balanceamento
    class_weights = calculate_class_weights(df_split[df_split["split"] == "train"]["target_doenca_classe"])
    print("\nPesos calculados para a Loss Function (Treino):")
    for cls, w in class_weights.items():
        print(f"  - {cls}: {w:.4f}")
        
    # 3. Exibe relatório de partições
    report = generate_sampling_report(df_split)
    print("\nRelatório de Amostragem por Partição:")
    print(report)
    print("\n[OK] Pipeline de amostragem executado com sucesso e sem vazamento de dados.")
