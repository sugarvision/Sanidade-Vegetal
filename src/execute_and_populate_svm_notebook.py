"""
Executor e renderizador de saídas do Notebook 05 de SVM
Executa cada célula de código, captura stdout e figuras geradas pelo matplotlib,
convertendo-as em saídas oficiais do Jupyter Notebook (Base64 PNG e Stream text).
"""

import json
import io
import sys
import base64
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def execute_and_populate(notebook_path: Path):
    print(f"[1/3] Lendo notebook de: {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    global_env = {'__name__': '__main__'}
    code_count = 0

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            code_count += 1
            code_lines = cell['source']
            code_str = "".join(code_lines)
            print(f"[2/3] Executando célula de código {code_count}...")
            
            # Intercepta stdout
            old_stdout = sys.stdout
            captured_stdout = io.StringIO()
            sys.stdout = captured_stdout
            
            cell_outputs = []
            
            # Substitui plt.show() por uma chamada customizada que não limpa a figura antes de capturar
            exec_code = code_str.replace("plt.show()", "# plt.show()")
            
            try:
                exec(exec_code, global_env)
            except Exception as e:
                sys.stdout = old_stdout
                print(f"[ERRO] Falha na execução da célula {code_count}: {e}")
                raise e
            finally:
                sys.stdout = old_stdout
            
            # Captura stdout se houver
            stdout_text = captured_stdout.getvalue()
            if stdout_text.strip():
                # Formata como lista de linhas com \n
                lines = [l + '\n' for l in stdout_text.splitlines()]
                cell_outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": lines
                })
            
            # Captura figuras geradas pelo matplotlib
            fig_nums = plt.get_fignums()
            for fig_num in fig_nums:
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=200)
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close(fig)
                
                cell_outputs.append({
                    "data": {
                        "image/png": img_b64,
                        "text/plain": ["<Figure size ... with ... Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
            
            cell['outputs'] = cell_outputs
            cell['execution_count'] = code_count

    print(f"[3/3] Gravando notebook atualizado com {code_count} células executadas...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"[OK] Notebook executado e saídas renderizadas com sucesso em: {notebook_path}")

if __name__ == '__main__':
    target = Path(__file__).resolve().parent.parent / 'notebooks' / '05_estudo_teorico_e_aplicado_svm.ipynb'
    execute_and_populate(target)
