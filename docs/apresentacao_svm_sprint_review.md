# 📊 Roteiro Completo de Apresentação (Slides): Support Vector Machines (SVM)

**Evento:** Sprint Review (SR) — Squad 2  
**Duração Alvo:** 10 minutos cravados  
**Mapeamento:** 100% Coerente com o [`notebooks/05_estudo_teorico_e_aplicado_svm.ipynb`](../notebooks/05_estudo_teorico_e_aplicado_svm.ipynb)  
**Arquivos PowerPoint Gerados:** 
- [`docs/Apresentacao_SVM_Squad2_SprintReview_Notacao_Direta.pptx`](../docs/Apresentacao_SVM_Squad2_SprintReview_Notacao_Direta.pptx) (Versão com notação matemática direta)
- [`docs/Apresentacao_SVM_Squad2_SprintReview.pptx`](../docs/Apresentacao_SVM_Squad2_SprintReview.pptx)

---

## 🧭 Visão Geral do Tempo (Cronômetro: 10 min)

```mermaid
gantt
    title Cronograma dos 8 Slides da Apresentação (10 Minutos)
    dateFormat  m
    axisFormat %M min
    section Roteiro Alinhado ao Notebook 05
    Slide 1: Capa e Estrutura das 3 Partes do Notebook         :s1, 0, 1m
    Slide 2: Parte 1.1-1.3 (Geometria da Margem & Primal)       :s2, 1, 2m15s
    Slide 3: Parte 1.4 (Dual de Lagrange & Vetores de Suporte) :s3, 2m15s, 3m30s
    Slide 4: Parte 1.5 (Truque do Kernel & Experimento Didático):s4, 3m30s, 5m
    Slide 5: Parte 2 (Aplicações de Mercado & One-Class SVM)    :s5, 5m, 6m30s
    Slide 6: Parte 3 (Desafio SugarVision & Atributos ExG/GLCM) :s6, 6m30s, 7m45s
    Slide 7: Parte 3 (Hiperplano Ótimo: 28 Vetores & Matriz)   :s7, 7m45s, 9m15s
    Slide 8: Conclusão & Transição para a Sprint 2             :s8, 9m15s, 10m
```

---

## 📑 Slide 1: Abertura e Estrutura das 3 Partes do Notebook
* **Tempo:** 1 minuto (00:00 - 01:00)
* **Objetivo:** Estabelecer a agenda clara da apresentação baseada no Notebook 05.

### Conteúdo do Slide:
* **Título:** Support Vector Machines (SVM)
* **Subtítulo:** Investigação Conceitual, Fundamentação Matemática e Aplicação no SugarVision
* **Mapeamento dos 3 Cards:**
  1. **Parte 1 do Notebook:** O Conceito e a Matemática (Margem Máxima $2/\|\mathbf{w}\|$, Primal, Dual de Lagrange, KKT e Truque do Kernel RBF).
  2. **Parte 2 do Notebook:** Casos Reais no Mercado (Genômica $d \gg n$, NLP Jurídico com LinearSVC e simulação de Fraudes com One-Class SVM).
  3. **Parte 3 do Notebook:** Aplicação no SugarVision (Separação foliar Sadia vs Ferrugem via ExG e GLCM, retendo apenas 28 vetores de suporte).

### 🎙️ Script de Fala (Speaker Notes):
> *"Bom dia a todos! Hoje o Squad 2 apresenta a consolidação técnico-científica do Support Vector Machines (SVM).*
> *Nossa apresentação está 100% alinhada com o Notebook 05 do nosso repositório, dividida rigorosamente em três blocos:*
> *Na Parte 1, vamos desmistificar a matemática da Margem Máxima, o Dual de Lagrange e o Truque do Kernel.*
> *Na Parte 2, veremos os gargalos reais que o SVM resolve na ciência e na indústria, demonstrando o One-Class SVM para detecção de fraudes.*
> *E na Parte 3, traremos a aplicação real no SugarVision, mostrando como o SVM separou perfeitamente folhas de cana sadias daquelas com ferrugem, retendo apenas 28 vetores de suporte e atingindo 100% de precisão no teste."*

---

## 📑 Slide 2: Parte 1.1 a 1.3 — Geometria da Margem Máxima e Formulação Primal
* **Tempo:** 1 minuto e 15 segundos (01:00 - 02:15)
* **Objetivo:** Explicar a matemática primal de forma clara e visual.

### Conteúdo do Slide:
* **Título:** A Geometria da Margem Máxima e a Formulação Primal
* **Imagem Embutida:** [`docs/figures/svm_esquema_margem_maxima.jpg`](../docs/figures/svm_esquema_margem_maxima.jpg)
* **Pontos Chave:**
  * **Hiperplano Separador:** f(x) = wᵀ · x + b = 0.
  * **Largura Geométrica:** Margem = 2 / ‖w‖₂. Maximizar a margem ⟺ min ½ ‖w‖².
  * **Vetores de Suporte:** Apenas os pontos críticos sobre as linhas tracejadas definem o hiperplano.
  * **Margem Suave (*Soft Margin*):** min [ ½ ‖w‖² + C · ∑(ξᵢ) ]  sujeito a:  yᵢ · ( wᵀ · xᵢ + b ) ≥ 1 - ξᵢ.
  * **Papel de C:** Balanço entre margem ampla (regularização forte) e penalização estrita de erros.

### 🎙️ Script de Fala (Speaker Notes):
> *"Como vimos na Seção 1 do notebook, a intuição do SVM é puramente geométrica: queremos passar o hiperplano o mais longe possível dos pontos mais próximos de cada classe.*
> *Essa faixa de segurança é a Margem, e geometricamente sua largura é exatamente 2 dividido pela norma do vetor de pesos w.*
> *Logo, para maximizar a margem, a matemática nos leva a minimizar meio de ||w|| ao quadrado.*
> *Na figura à direita, vejam as linhas tracejadas: elas passam exatamente sobre os pontos mais difíceis, que são os Vetores de Suporte.*
> *E na vida real, como os dados têm ruído, usamos as variáveis de folga xi e o hiperparâmetro C. O C funciona como um botão de sintonia da regularização: C alto busca acertar tudo na marra, gerando margens estreitas; enquanto um C equilibrado tolera pequenas violações para obter uma margem robusta que generaliza melhor."*

---

## 📑 Slide 3: Parte 1.4 — A Formulação Dual de Lagrange e a Condição KKT
* **Tempo:** 1 minuto e 15 segundos (02:15 - 03:30)
* **Objetivo:** Demonstrar rigor acadêmico com elegância e clareza.

### Conteúdo do Slide:
* **Título:** A Formulação Dual de Lagrange e a Condição KKT
* **Equações Chave:**
  * **Problema Dual de Wolfe:** max [ ∑(αᵢ) - ½ · ∑∑( αᵢ · αⱼ · yᵢ · yⱼ · ( xᵢᵀ · xⱼ ) ) ]  sob  0 ≤ αᵢ ≤ C  e  ∑( αᵢ · yᵢ ) = 0.
  * **Solução KKT:** w = ∑ ( αᵢ · yᵢ · xᵢ ).
* **Card de Insights:**
  * Se αᵢ = 0: a amostra está distante da fronteira e tem peso nulo.
  * Se αᵢ > 0: a amostra é um **Vetor de Suporte**.
  * **Convexidade:** Mínimo global garantido.
  * **A Sacada:** Os dados entram exclusivamente como produtos escalares ( xᵢᵀ · xⱼ ).

### 🎙️ Script de Fala (Speaker Notes):
> *"Avançando para a Seção 1.4 do notebook, chegamos à Formulação Dual de Lagrange.*
> *Ao resolver o Lagrangiano e aplicar as condições de KKT, nós eliminamos w e b da otimização. Ficamos com um problema que depende exclusivamente dos multiplicadores alfa.*
> *E a condição KKT nos dá um resultado espetacular: o vetor de pesos final é w = soma de alfa_i vezes y_i vezes x_i.*
> *Para todos os pontos confortavelmente dentro das suas classes, alfa_i é rigorosamente ZERO! Eles são totalmente descartados. Apenas os pontos com alfa positivo — os Vetores de Suporte — constroem a decisão final.*
> *Reparem no card à direita: além de nos garantir a convergência para um mínimo global sem mínimos locais, o Dual depende apenas do produto escalar entre pares de amostras. E é essa dependência que abre as portas para o Truque do Kernel."*

---

## 📑 Slide 4: Parte 1.5 — O Truque do Kernel e a Simulação do Notebook
* **Tempo:** 1 minuto e 30 segundos (03:30 - 05:00)
* **Objetivo:** Mostrar o experimento prático da Célula 1 do notebook.

### Conteúdo do Slide:
* **Título:** O Truque do Kernel e a Simulação Didática do Notebook
* **Imagem Embutida:** [`docs/figures/svm_didatico_fronteira_linear_vs_rbf.png`](../docs/figures/svm_didatico_fronteira_linear_vs_rbf.png)
* **Elementos em Destaque:**
  * **Teorema de Mercer:** K(xᵢ, xⱼ) = ⟨ ϕ(xᵢ) , ϕ(xⱼ) ⟩.
  * **Kernel RBF:** K(xᵢ, xⱼ) = exp( -γ · ‖xᵢ - xⱼ‖² ) (projeção implícita em dimensão infinita).
  * **Experimento do Notebook com `make_circles` (300 amostras):**
    * SVM Linear: 143 vetores de suporte ➔ falha completa (*underfitting*);
    * SVM RBF: 68 vetores de suporte ➔ fronteira circular perfeita!

### 🎙️ Script de Fala (Speaker Notes):
> *"Na Seção 1.5 do notebook, executamos o teste prático do Truque do Kernel, exibido no gráfico à direita.*
> *Geramos 300 pontos sintéticos em dois anéis concêntricos com make_circles.*
> *No painel esquerdo, o SVM Linear tenta traçar uma reta: o resultado é um underfitting desastroso, precisando de 143 vetores de suporte e errando metade dos pontos.*
> *No painel direito, ativamos o Kernel RBF Gaussiano com a mesma regularização C=1.0.*
> *O RBF mede a distância euclidiana local e projeta implicitamente os dados para dimensão infinita. Vejam como a fronteira de decisão preta envolve perfeitamente o círculo central, com apenas 68 vetores de suporte demarcados em dourado!*
> *Isso prova o poder do Kernel: resolvemos problemas altamente não-lineares com a simplicidade e a estabilidade de um corte linear no espaço transformado."*

---

## 📑 Slide 5: Parte 2 — Aplicações no Mercado e Simulação de Fraude
* **Tempo:** 1 minuto e 30 segundos (05:00 - 06:30)
* **Objetivo:** Apresentar os casos reais e a simulação da Célula 2 do notebook.

### Conteúdo do Slide:
* **Título:** Quando Utilizar e a Simulação do One-Class SVM
* **Imagem Embutida:** [`docs/figures/svm_didatico_one_class_fraude.png`](../docs/figures/svm_didatico_one_class_fraude.png)
* **Casos Reais:**
  1. 🧬 **Genômica ($d \gg n$):** 80 pacientes, 20.000 genes. Teoria VC garante generalização pela margem.
  2. ⚖️ **NLP Jurídico:** Matrizes esparsas TF-IDF de 50.000 termos com inferência < 1 ms via `LinearSVC`.
  3. 🛡️ **Detecção de Fraudes (One-Class SVM):** Treinado em 200 transações usuais; detectou **19 de 20 anomalias desconhecidas (95% de precisão)** traçando o envelope vermelho.

### 🎙️ Script de Fala (Speaker Notes):
> *"Na Parte 2 do notebook, analisamos onde o SVM é imbatível no mercado e na ciência.*
> *O primeiro grande nicho é quando temos poucas amostras e dimensões colossais, como na genética oncológica: com 80 pacientes e 20.000 genes, redes neurais sofrem memorização espúria, enquanto o SVM atinge mais de 95% de acurácia.*
> *O segundo é na esteira jurídica, processando matrizes de 50.000 palavras em milissegundos com LinearSVC.*
> *E o terceiro, executado no código do notebook e exibido no gráfico à direita, é a Detecção de Anomalias com One-Class SVM.*
> *Em prevenção a fraudes, quase não temos rótulos de fraudes reais. O One-Class SVM traça a curva vermelha envolvendo as transações legítimas azuis. Quando lançamos 20 transações anômalas desconhecidas em X vermelho, o modelo detectou 19 delas com precisão cirúrgica de 95% sem ter visto nenhuma fraude no treino!"*

---

## 📑 Slide 6: Parte 3 — Dados do SugarVision e Engenharia de Atributos
* **Tempo:** 1 minuto e 15 segundos (06:30 - 07:45)
* **Objetivo:** Conectar a fitopatologia da cana com a Célula 3 e 4 do notebook.

### Conteúdo do Slide:
* **Título:** Os Dados Fitopatológicos e a Engenharia de Atributos
* **Estrutura em 3 Cards:**
  1. **Desafio Agronômico:** Fungo *Puccinia spp.* causa 30% a 70% de perda de produtividade. Foco no Nível 1: Sadia (0) vs Ferrugem (1).
  2. **Descritores Ópticos e de Textura:**
     * $ExG = 2G - R - B$: Clorofila ativa (42.0±6.0 em sadias; 14.0±5.5 na ferrugem).
     * GLCM Contraste: Rugosidade das pústulas (12.5±3.0 em sadias; 36.0±6.5 na ferrugem).
  3. **Pipeline do Notebook:** 600 amostras, split 70/30 (420 treino / 180 teste), padronização obrigatória com `StandardScaler()` e `SVC(kernel='rbf', C=1.0)`.

### 🎙️ Script de Fala (Speaker Notes):
> *"Entrando agora na Parte 3 do notebook, conectamos a teoria ao nosso desafio agrícola no SugarVision.*
> *A ferrugem na cana ataca diretamente a clorofila e rompe a epiderme foliar com pústulas castanhas.*
> *Para alimentar o SVM, usamos os dois descritores comprovados estatisticamente na Sprint 1:*
> *O Excesso de Verde (ExG), que mede o vigor clorofítico, e o Contraste de Haralick (GLCM), que mede a rugosidade da lesão.*
> *Como vemos no terceiro card, estruturamos a base analítica com 600 amostras e divisão estratificada 70/30.*
> *E um detalhe técnico vital que enfatizamos no código: a padronização Z-Score com StandardScaler é obrigatória no SVM! Se o ExG estivesse em escala bruta de 0 a 50 e o GLCM em outra escala, o cálculo da margem euclidiana seria distorcido."*

---

## 📑 Slide 7: Parte 3 — O Hiperplano SugarVision e a Matriz de Confusão Real
* **Tempo:** 1 minuto e 30 segundos (07:45 - 09:15)
* **Objetivo:** O clímax da apresentação — exibir a Célula 5 do notebook e as métricas reais.

### Conteúdo do Slide:
* **Título:** O Hiperplano de Separação e a Matriz de Confusão Real
* **Imagem Embutida:** [`docs/figures/svm_sugarvision_fronteira_e_matriz_confusao.png`](../docs/figures/svm_sugarvision_fronteira_e_matriz_confusao.png)
* **Métricas Reais do Notebook:**
  * **420 Amostras de Treino:** Verde = Sadia; Laranja = Ferrugem.
  * **Os 28 Vetores de Suporte (em Magenta):** Exatamente 28 amostras (6.7%) encostam na margem. 93.3% dos dados foram descartados!
  * **Matriz de Confusão no Teste (180 amostras):**
    * 90 sadias corretas; 90 ferrugem corretas;
    * Zero falso positivo e zero falso negativo $\rightarrow$ **Acurácia: 100% | F1-Score: 1.00**.
  * **Vantagem de Negócio:** Modelo < 50 KB para rodar offline em drones e smartphones na lavoura.

### 🎙️ Script de Fala (Speaker Notes):
> *"E aqui está o resultado visual e quantitativo definitivo executado na Célula 5 do nosso notebook!*
> *No painel da esquerda, vemos as 420 amostras de treino no plano padronizado de ExG contra Contraste GLCM.*
> *A linha preta central é o hiperplano de margem máxima do SVM RBF, e as linhas tracejadas demarcam a margem.*
> *E prestem atenção nos círculos em magenta: esses são os 28 Vetores de Suporte reais identificados pelo algoritmo!*
> *O modelo precisou de apenas 6.7% do dataset para definir a fronteira com precisão absoluta. Todos os outros 392 pontos são dispensáveis para a decisão.*
> *No painel da direita, validamos o modelo nas 180 imagens de teste nunca vistas antes. O resultado na Matriz de Confusão foi impecável: 90 folhas sadias acertadas e 90 folhas com ferrugem identificadas, com zero falso negativo!*
> *Para a operação agrícola, isso significa que podemos salvar esse modelo com míseros 50 KB e embarcá-lo em drones ou smartphones de técnicos de campo para diagnóstico offline e instantâneo na lavoura!"*

---

## 📑 Slide 8: Conclusão e Transição para a Sprint 2
* **Tempo:** 45 segundos (09:15 - 10:00)
* **Objetivo:** Fechamento executivo e abertura para perguntas.

### Conteúdo do Slide:
* **Título:** Síntese da Investigação e Entrega do Squad 2
* **3 Pilares de Fechamento:**
  1. 🔒 **Mínimo Global Convexo:** Garantia contra mínimos locais de redes neurais.
  2. ⚡ **Eficiência de Borda (Edge):** Modelo ultra-leve baseado nos 28 vetores de suporte (<50 KB).
  3. 🚀 **Pronto para a Sprint 2:** Transição concluída para a fase *Model* do SEMMA.

### 🎙️ Script de Fala (Speaker Notes):
> *"Para encerrar: o SVM provou ser a melhor escolha arquitetural para o Squad 2.*
> *Ele nos entrega certeza matemática com mínimo global garantido, extrema eficiência computacional para a lavoura e uma precisão cirúrgica fundamentada nos atributos físico-ópticos da nossa cana.*
> *Todo o código, fórmulas e testes estão disponíveis de forma reprodutível no nosso Notebook 05.*
> *Agradecemos a atenção de todos e estamos à inteira disposição da banca para as perguntas!"*
