# 🌿 Diagnóstico Inteligente na Cana-de-Açúcar (Transição da Fase Modify para Model)

**Projeto:** SugarVision — Sanidade Vegetal (*Saccharum officinarum*)  
**Metodologia:** Framework SEMMA (SAS Institute) — Transição Modify → Model  
**Arquivo Editável Gerado (.docx):** [`docs/Diagnostico_Inteligente_CanadeAcucar_Transicao_Modify_Model.docx`](Diagnostico_Inteligente_CanadeAcucar_Transicao_Modify_Model.docx)  
**Detentores dos Direitos Autorais (Copyright © 2026):**  
Heitor Vitti, César Bovo, Guilherme Romanzotti, Marvin Cristhian, Elisa Alcantara e Luis Otávio.

---

> ### 🎯 Objetivo
> Explicar como o projeto saiu do preparo dos dados foliares e entrou na fase de construção e validação da inteligência artificial para o campo.

---

## Passo 1: O Contexto e a Virada de Chave (Modify → Model)

* **Ideia central:** Entender o ciclo de evolução do projeto — de organizar os dados para ensinar a máquina a pensar.
* **O Ciclo do Projeto (Metodologia SEMMA):** O projeto segue uma trilha de engenharia de dados com cinco etapas: Amostragem, Exploração, Modificação (Modify), Modelar (Model) e Avaliar.
* **O que foi concluído na fase Modify:** Na etapa anterior (Modify), o time organizou, limpou e padronizou uma base com 6.571 fotos de folhas de cana. Dessas fotos, foram extraídas 35 medidas visuais (cor, brilho, rugosidade e textura) sem distorções estatísticas.
* **A Transição para a fase Model:** Este relato marca exatamente a virada de chave para a fase de Modelagem (Model). O foco deixou de ser arrumar a tabela e passou a ser: qual inteligência artificial aprende melhor a diagnosticar as 7 condições da lavoura (folha sadia e 6 doenças)?.

### 💻 Implementação no Código-Fonte do Projeto
```python
# src/atomic_pipeline_preprocessing.py — Mapeamento de Atributos da ABT Consolidada (6.571 instâncias)
NUMERICAL_FEATURES = [
    'mean_hue', 'std_saturation', 'exg_index', 'exr_index', 'rg_ratio',
    'indice_clorose_necrose', 'hue_dispersion', 'glcm_contrast',
    'glcm_homogeneity', 'glcm_dissimilarity', 'glcm_energy',
    'indice_rugosidade_pustula', 'laplacian_var', 'size_kb',
    'total_pixels', 'resolution_mp', 'aspect_ratio'
]

DISCRETIZED_FEATURES = [
    'size_kb_bin', 'laplacian_var_bin', 'resolution_bin', 'aspect_ratio_bin'
]

CATEGORICAL_NOMINAL_FEATURES = ['dataset_source', 'extension']
ALL_CATEGORICAL_FEATURES = CATEGORICAL_NOMINAL_FEATURES + DISCRETIZED_FEATURES

# src/vision_inference_engine.py — Extração determinística de índices bio-espectrais
# ExG: sensível à clorofila e tecido sadio | ExR: marcador fúngico de necrose e pústulas
exg_pixel = 2.0 * g - r - b
exr_pixel = 1.4 * r - g
exg_index = float(np.mean(exg_pixel))
exr_index = float(np.mean(exr_pixel))
indice_clorose_necrose = float(((np.mean(r) + 1.0) / (np.mean(g) + 1.0)) / (std_saturation + 1e-4))
```

---

## Passo 2: Estabelecendo a Régua de Comparação (Os Baselines)

* **Ideia central:** Antes de usar algoritmos avançados, precisamos provar se métodos simples já resolveriam o problema.
* **O "Chute Proporcional" (Dummy):** Se alguém apenas chutasse os diagnósticos respeitando a proporção em que cada doença aparece no campo, acertaria apenas 16,40% das vezes. Esse é o piso mínimo de partida.
* **As Regras Simples (Árvores e Fórmulas Clássicas):**
  * Criamos uma árvore de decisão com perguntas diretas (ex.: "A textura da folha está muito áspera? Há tons avermelhados de necrose?").
  * Esses métodos básicos alcançaram entre 60% e 68% de acurácia.
* **O Risco do Desbalanceamento:** Por padrão, essas fórmulas simples tendiam a ignorar doenças raras, focando apenas no que era comum. Ao ativar o balanceamento de classes (`class_weight='balanced'`), forçamos o sistema a enxergar ameaças raras, resgatando diagnósticos da patologia Grassy Shoot de 0% para 35% de sensibilidade.

### 💻 Implementação no Código-Fonte do Projeto
```python
# src/baseline_models_benchmark.py — Concepção dos Modelos Baselines e Ponderação de Classes
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 1. Classificador Ingênuo Proporcional (Piso estocástico de 16,40%)
def build_dummy_baseline(strategy: str = 'stratified', random_state: int = 42) -> DummyClassifier:
    return DummyClassifier(strategy=strategy, random_state=random_state)

# 2. Regressão Logística Multiclasse com Penalidade L2 (Ridge)
def build_logistic_regression_baseline(C: float = 1.0, class_weight: str = 'balanced', random_state: int = 42):
    return LogisticRegression(
        penalty='l2', C=C, class_weight=class_weight,
        max_iter=1000, random_state=random_state, solver='lbfgs'
    )

# 3. Árvore de Decisão Rasa (Profundidade Máxima = 4)
def build_decision_tree_baseline(max_depth: int = 4, class_weight: str = 'balanced', random_state: int = 42):
    return DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight=class_weight,   # Evita o colapso e omissão de patologias raras
        criterion='gini',
        random_state=random_state
    )
```

---

## Passo 3: O Modelo Campeão (SVM) e a Validação Agronômica

* **Ideia central:** Traçar uma cerca matemática segura e confirmar se a inteligência faz sentido na lavoura.
* **A Fronteira do SVM (Support Vector Machine):** Adotamos o SVM, um algoritmo que traça a fronteira de separação mais ampla e segura possível entre folhas doentes e sadias.
* **Superioridade Estatística Comprovada:** O SVM campeão alcançou 69,18% de acurácia. Por meio de testes estatísticos formais provamos que esse ganho não foi sorte, mas uma evolução real frente a todos os métodos simples.
* **Conferência com a Biologia da Planta:**
  * Quando a IA encontrava tons verdes intensos e textura uniforme, ela classificava a planta como sadia.
  * Quando encontrava manchas alaranjadas e quebra na película da folha, diagnosticava ferrugem.
  * Isso comprovou que o modelo aprendeu sintomas agronômicos reais, e não ruídos da imagem.

### 💻 Implementação no Código-Fonte do Projeto
```python
# src/atomic_svm_pipeline.py — Parametrização do Estimador SVM RBF Campeão
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

svm_champion = SVC(
    kernel='rbf',
    C=10.0,                      # Margem suave balanceando penalidade e generalização
    gamma=0.01,                  # Alcance ótimo de influência dos vetores de suporte
    probability=True,            # Habilita calibração de probabilidades a posteriori (Platt Scaling)
    class_weight='balanced',     # Ponderação agronômica para classes minoritárias
    random_state=42
)

# src/statistical_comparison_framework.py — Testes Estatísticos Formais Pareados (5 Folds)
from scipy.stats import ttest_rel, wilcoxon

t_stat, p_val_ttest = ttest_rel(svm_fold_scores, baseline_fold_scores)
w_stat, p_val_wilcoxon = wilcoxon(svm_fold_scores, baseline_fold_scores)
# Comprovação de superioridade com significância estatística rigorosa (p < 0.05)
is_statistically_superior = (p_val_ttest < 0.05) and (np.mean(svm_fold_scores) > np.mean(baseline_fold_scores))
```

---

## Passo 4: O "Truque dos Kernels" e o Teste de Sol e Sombra

* **Ideia central:** Projetar dados complexos e testar a resistência da IA contra o clima da fazenda.
* **Separando o que Parece Misturado (Kernel Trick):** Quando as lesões foliares se sobrepõem visualmente, o algoritmo utiliza funções matemáticas chamadas Kernels para encontrar ângulos onde a separação se torna clara.
* **O Teste de Estresse no Campo:** Na lavoura real há poeira, sol forte e nuvens gerando sombras. Simulamos essas perturbações no computador.
* **Descoberta Prática:**
  * O modelo matematicamente mais sofisticado (Kernel RBF) teve uma queda de 26,5 pontos percentuais sob sombra severa, pois depende excessivamente da iluminação local.
  * Versões lineares e polinomiais mostraram-se muito mais estáveis sob iluminação adversa.
  * Essa análise gera recomendações diretas para a captura de fotos no campo.

### 💻 Implementação no Código-Fonte do Projeto
```python
# src/svm_kernel_trick_study.py — Simulação de Ruído Fotométrico e Sombra Severa
CHROMA_FEATURES = ['mean_hue', 'std_saturation', 'exg_index', 'exr_index', 'rg_ratio', 'indice_clorose_necrose']

# 1. Perturbação de Poeira e Sol Forte (Ruído Gaussiano nos canais cromáticos)
def add_photometric_noise(X, train_std, level, rng):
    X_pert = X.copy()
    for col in CHROMA_FEATURES:
        X_pert[col] += rng.normal(0.0, level * train_std[col], size=len(X_pert))
    return X_pert

# 2. Perturbação de Sombra Severa (Fator multiplicativo f de atenuação exponencial)
SHADOW_POWERS = {
    'exg_index': 1, 'exr_index': 1,
    'laplacian_var': 2, 'glcm_contrast': 2,
    'glcm_dissimilarity': 1, 'indice_rugosidade_pustula': 3
}

def apply_shadow(X, factor: float):
    X_shadow = X.copy()
    for col, power in SHADOW_POWERS.items():
        X_shadow[col] = X_shadow[col] * (factor ** power)
    return X_shadow

# Resultado do estudo: sob sombra f=0.4, o Kernel RBF sofreu queda de 26,5 p.p.
# indicando a necessidade de protocolos rígidos de iluminação difusa para a captura.
```

---

## Passo 5: Blindagem, Segurança e Governança de Software

* **Ideia central:** Garantir que o modelo não cometa "fraude" e seja rastreável de ponta a ponta.
* **Blindagem Contra "Cola" (Anti-Leakage):** Construímos um duto de dados unificado (Pipeline) para garantir que a IA nunca tenha acesso prévio às imagens do teste cego, eliminando qualquer risco de aprendizado artificial viciado.
* **Assinatura Digital (Hashes SHA-256):** Todos os modelos e planilhas gerados receberam códigos criptográficos de integridade. Isso assegura que o arquivo levado a campo é idêntico ao auditado no laboratório.
* **Certeza Calibrada:** O sistema não apenas aponta um diagnóstico, mas calcula a probabilidade real de acerto (ex.: “88% de certeza para ferrugem”), fornecendo segurança para a tomada de decisão do agrônomo.

### 💻 Implementação no Código-Fonte do Projeto
```python
# 1. Blindagem Anti-Leakage: Encapsulamento Atômico do Pré-processador e Classificador
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), NUMERICAL_FEATURES),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ALL_CATEGORICAL_FEATURES)
    ],
    remainder='drop'
)
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('svm', svm_champion)])

# 2. Assinatura Digital Criptográfica (SHA-256) dos Artefatos de Produção
def calculate_file_hash(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

# 3. Certeza Calibrada via Platt Scaling e Margem Geométrica (src/vision_inference_engine.py)
probas = full_pipeline.predict_proba(df_feat)[0]
confidence_pct = float(round(np.max(probas) * 100.0, 2))  # Ex: 88,4% de probabilidade
dfunc = full_pipeline.decision_function(df_feat)[0]
delta_margin = float(np.sort(dfunc)[-1] - np.sort(dfunc)[-2])  # Margem hiperdimensional Δ
```

---

## Passo 6: Conclusão da Fase Model e os Próximos Passos (Fase Assess)

* **Ideia central:** O dever de casa da modelagem foi concluído com sucesso; agora o foco é o valor econômico e o uso no campo.
* **Critérios Atendidos na Modelagem:**
  * Engenharia sólida: Códigos reprodutíveis, blindados e auditados.
  * Eficácia matemática: SVM validado estatisticamente como o melhor modelo.
  * Coerência prática: Decisões alinhadas à fitopatologia da cana.
* **O Próximo Passo — Fase Assess (Avaliação de Negócio e Campo):**
  * **Matriz de Custo Financeiro:** Calcular o impacto econômico de não identificar a tempo doenças de quarentena, nas quais um erro custa muito mais caro que um alarme falso.
  * **Microsserviço de Diagnóstico:** Transformar esse modelo campeão em um serviço conectado a aplicativo móvel para leitura fotográfica em tempo real nas fazendas.

### 💻 Implementação no Código-Fonte do Projeto
```python
# 1. Matriz de Custo Financeiro Assimétrico (src/svm_agronomic_error_audit.py)
@dataclass(frozen=True)
class SugarcaneEconomicParameters:
    expected_yield_tons_per_ha: float = 85.0
    price_per_ton_brl: float = 145.0             # Receita bruta: R$ 12.325,00/ha
    rust_yield_loss_fraction: float = 0.35       # 35% de quebra por ferrugem omitida
    rust_epidemic_spread_multiplier: float = 1.80# Disseminação eólica para talhões vizinhos
    fungicide_cost_per_ha: float = 485.00        # Custo do insumo + aplicação

    @property
    def cost_fn_rust(self) -> float:
        # Falso Negativo de Ferrugem: R$ 10.017,25/ha (perda de safra + reforma precoce de soqueira)
        return (self.expected_yield_tons_per_ha * self.price_per_ton_brl * self.rust_yield_loss_fraction * self.rust_epidemic_spread_multiplier) + 2252.50

    @property
    def cost_fp_fungicide(self) -> float:
        # Falso Positivo: R$ 485,00/ha (aplicação desnecessária de fungicida químico)
        return self.fungicide_cost_per_ha

    # Razão de Assimetria = 20,65x: Omitir um foco de ferrugem é 20 vezes mais custoso que um alarme falso!

# 2. Microsserviço de Diagnóstico em Tempo Real (src/app_demoday5_sugarvision.py)
import streamlit as st
from src.vision_inference_engine import VisionInferenceEngine

engine = VisionInferenceEngine("models/svm_pipeline_optimal.joblib")
uploaded_leaf = st.file_uploader("Capture a foto da folha no campo:", type=["jpg", "png"])
if uploaded_leaf:
    result = engine.diagnose_image(uploaded_leaf.read())
    st.write(f"Diagnóstico: {result['consolidated_class']}")
    st.write(f"Certeza Calibrada: {result['confidence_pct']:.1f}%")
    st.write(f"Status do SLA: {result['sla_status']}")
```

---

> ### 💡 Síntese da Transição Modify → Model
> Com o encerramento da fase Modify, a equipe transformou 6.571 imagens fotográficas em uma matriz analítica blindada com 35 descritores. Na fase Model, provou-se a superioridade do SVM RBF ótimo (69,18% de acurácia com 100% de sensibilidade para Ferrugem), estabelecendo as bases para a fase Assess com matriz de custo financeiro e microsserviço de campo para o Demoday.

