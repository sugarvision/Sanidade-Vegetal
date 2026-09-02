import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

def test_notebook_execution(notebook_path):
    print(f"Iniciando execucao do notebook: {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    global_env = {}
    code_cell_count = 0
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            code_cell_count += 1
            code_str = "".join(cell['source'])
            print(f"  -> Executando celula de codigo {code_cell_count} (index {i})...")
            clean_code = code_str.replace("display(", "print(").replace("plt.show()", "plt.close('all')")
            exec(clean_code, global_env)
            
    print(f"\n[OK] SUCESSO: Todas as {code_cell_count} celulas de codigo foram executadas com perfeicao (0 erros)!")
    return True

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent
    nb_path = base_dir / 'notebooks' / '02_sprint1_master_pipeline_reprodutivel.ipynb'
    test_notebook_execution(nb_path)
