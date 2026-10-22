import os
import time
import psutil
import warnings
import joblib
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.exceptions import ConvergenceWarning

# Oculta os warnings de convergência do console durante a busca em grade
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def get_memory_usage():
    """Retorna o uso de memória RAM do processo atual em MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def run_parallel_svm_grid_search(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """
    Executa a malha de experimentos do SVM em paralelo, monitorando 
    eficiência computacional e salvando o modelo campeão.
    """
    print("="*80)
    print("INICIANDO EXECUÇÃO PARALELA EM LOTE (SVM GRID SEARCH)")
    print("="*80)

    # 1. Configuração de Diretórios e Salvamento
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "svm_best_model.joblib"

    # 2. Definição da Malha de Hiperparâmetros (Tratando Convergência)
    # max_iter e tol são ajustados para evitar ConvergenceWarnings em kernels complexos
    param_grid = [
        {
            'kernel': ['rbf'],
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto'],
            'max_iter': [2000, 5000],  # Limite de iterações para garantir convergência
            'tol': [1e-3, 1e-4]        # Tolerância de parada ajustada
        },
        {
            'kernel': ['linear'],
            'C': [0.1, 1.0, 10.0],
            'max_iter': [2000, 5000],
            'tol': [1e-3, 1e-4]
        }
    ]

    svm_estimator = SVC(probability=True, random_state=42, class_weight='balanced')
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. Configuração do GridSearch (n_jobs=-1 para múltiplos núcleos)
    grid_search = GridSearchCV(
        estimator=svm_estimator,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='f1_macro',
        n_jobs=-1,          # Habilita execução paralela usando todos os núcleos
        verbose=1,
        return_train_score=False
    )

    # 4. Monitoramento de Recursos (CPU e RAM)
    start_time = time.perf_counter()
    start_ram = get_memory_usage()
    
    print(f"[*] Iniciando ajuste do modelo... RAM Inicial: {start_ram:.2f} MB")
    print(f"[*] Utilizando todos os núcleos de processamento (n_jobs=-1).")
    
    # Execução da malha
    grid_search.fit(X_train, y_train)
    
    end_time = time.perf_counter()
    end_ram = get_memory_usage()

    # Cálculo de métricas de eficiência
    elapsed_time = end_time - start_time
    ram_consumed = end_ram - start_ram

    print("\n" + "="*80)
    print("RELATÓRIO DE EFICIÊNCIA COMPUTACIONAL E CONVERGÊNCIA")
    print("="*80)
    print(f"Tempo total de convergência da malha : {elapsed_time:.2f} segundos")
    print(f"Pico de consumo de memória RAM extra : {ram_consumed:.2f} MB (Total final: {end_ram:.2f} MB)")
    print(f"Melhor F1-Macro Score                : {grid_search.best_score_:.4f}")
    print(f"Melhor combinação de hiperparâmetros : {grid_search.best_params_}")

    # 5. Salvamento Seguro do Modelo Campeão em Disco
    joblib.dump(grid_search.best_estimator_, best_model_path, compress=3)
    print(f"\n[OK] Modelo campeão salvo com sucesso em: {best_model_path}")

    return grid_search

if __name__ == '__main__':
    # Exemplo de carregamento da ABT consolidada gerada na Sprint 2
    try:
        data_path = Path("data/processed/abt_features_modelagem.parquet")
        df_abt = pd.read_parquet(data_path)
        
        # Filtro da partição de treino
        df_train = df_abt[df_abt['split_partition'] == 'train']
        
        # Isolamento de features e target
        exclude_cols = ['sample_id', 'split_partition', 'class_label', 'target_binary', 'target_multiclass']
        feature_cols = [c for c in df_train.columns if c not in exclude_cols]
        
        X_train = df_train[feature_cols]
        y_train = df_train['target_multiclass'] # ou target_binary dependendo da abordagem
        
        # Executar a malha
        best_model = run_parallel_svm_grid_search(X_train, y_train)
        
    except FileNotFoundError:
        print("[Erro] Base de dados não encontrada. Verifique o caminho 'data/processed/abt_features_modelagem.parquet'.")