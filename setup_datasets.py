import os

def setup_datasets():
    datasets_dir = "datasets"
    
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
        print(f"Diretório '{datasets_dir}' criado com sucesso.")
    else:
        print(f"O diretório '{datasets_dir}' já existe.")

    print("\n--- Configuração dos Datasets ---")
    print("Devido ao tamanho desses datasets (5,28 GB no total), eles não são armazenados no Git.")
    print("Siga as instruções abaixo para obtê-los:")
    
    print("\n1. Dataset do Roboflow:")
    print("   URL: https://universe.roboflow.com/asad-unvar/sugarcane-disease-classification")
    print("   Ação: Baixe o dataset como arquivo ZIP e extraia-o dentro de: 'datasets/roboflow_sugarcane/'")
    
    print("\n2. Dataset do Mendeley Data:")
    print("   URL: https://data.mendeley.com/datasets/rzh99cj2rj/1")
    print("   Ação: Baixe os arquivos e extraia-os dentro de: 'datasets/mendeley_data/'")
    
    print("\nNota: Após o download e a extração, a estrutura da sua pasta 'datasets' deve ficar assim:")
    print("datasets/")
    print("├── roboflow_sugarcane/")
    print("└── mendeley_data/")

if __name__ == "__main__":
    setup_datasets()