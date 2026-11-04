# 🎤 Roteiro de Demonstração ao Vivo e Validação Agronômica Operacional — Demoday 5

**Projeto:** SugarVision — Sanidade Vegetal (*Saccharum officinarum* / Fitopatologia de Grandes Culturas)  
**Sprint:** 5 — Framework SEMMA (Fase: Assess & Deploy Readiness)  
**Responsável Técnico:** Cesar (Lead Técnico & Visão Computacional)  
**Data da Apresentação Presencial:** 11/12/2026  
**Duração Alvo:** 8 a 10 minutos  
**Interface Demonstrada:** [`src/app_demoday5_sugarvision.py`](../src/app_demoday5_sugarvision.py)  
**Motor de Visão Computacional:** [`src/vision_inference_engine.py`](../src/vision_inference_engine.py)  
**Conjunto de Teste Cego:** [`data/demoday5_blind_samples/`](../data/demoday5_blind_samples/)  
**Detentores dos Direitos Autorais (Copyright © 2026):**  
Heitor Vitti, César Bovo, Guilherme Romanzotti, Marvin Cristhian, Elisa Alcantara e Luis Otávio.

---

## 1. Visão Geral da Demonstração

O objetivo da demonstração presencial no **Demoday 5** é apresentar à banca avaliadora a operacionalização prática e em tempo real do classificador **SugarVision**. A apresentação comprova que o pipeline atômico e o estimador campeão SVM RBF ($C=10.0, \gamma=0.01$) não são apenas artefatos teóricos, mas compõem um sistema de inteligência artificial de bancada e campo capaz de receber qualquer fotografia foliar, extrair matrizes de cor e textura em milissegundos e emitir laudos agronômicos com salvaguarda de decisão auditável.

```
+--------------------------------------------------------------------------------------------------+
|                            FLUXO DE DEMONSTRAÇÃO AO VIVO NO DEMODAY 5                            |
+--------------------------------------------------------------------------------------------------+
| [1. Abertura & Pitch]   -> 2 min: Apresentação da dor agronômica, perdas por ferrugem e métricas. |
| [2. Caso Sadio Controle] -> 2 min: Upload de folha sadia (ExG elevado, tecido verde homogêneo).    |
| [3. Caso Ferrugem Ativa] -> 2 min: Upload de ferrugem (pústulas, quebra de clorofila, ExR e GLCM).|
| [4. Caso Limítrofe/SLA] -> 2 min: Demonstração de caso de incerteza e salvaguarda Human-in-the-Loop|
| [5. Encerramento & Q&A]  -> 2 min: Síntese de governança, prontidão de deploy e perguntas da banca|
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Roteiro Cronometrado Passo a Passo e Script de Apresentação

### Bloco 1: Abertura Institucional e Contexto Fitopatológico (00:00 - 02:00)
* **Ação na Tela:** Projetor exibindo a tela inicial da aplicação Streamlit com o título *SugarVision AI — Diagnóstico Fitossanitário Inteligente*.
* **Script de Apresentação (Cesar):**
  > *"Boa tarde, nobres membros da banca avaliadora. Sou o Cesar, Lead Técnico de Visão Computacional da equipe SugarVision. Hoje, temos a honra de demonstrar em tempo real a conclusão do nosso ciclo no Framework SEMMA. O agronegócio de cana-de-açúcar e grandes culturas enfrenta perdas de até 35% por infecções foliares não diagnosticadas a tempo, especialmente a Ferrugem. Nosso estimador SVM campeão atingiu 100% de sensibilidade na detecção da Ferrugem com AUC-ROC superior a 0,91. Mas como esse modelo se comporta na mão do agrônomo? É exatamente isso que veremos funcionando ao vivo agora."*

---

### Bloco 2: Demonstração do Caso Sadio Controle (02:00 - 04:00)
* **Ação na Tela:**
  1. Na barra lateral, selecionar a opção **"🧪 Galeria de Teste Cego Curado"**.
  2. Escolher a amostra **`amostra_02_folha_sadia_controle.jpg`**.
  3. Apontar para os três cards de métricas e o triplo painel visual.
* **Script de Apresentação (Cesar):**
  > *"Iniciamos com uma folha sadia de controle. Em menos de 200 milissegundos, o motor extraiu 27 atributos espectro-texturais. Observem o laudo em verde: **SADIA (HEALTHY)** com probabilidade calibrada de alto nível. Na coluna central, o Mapa de Textura Haralick revela uma lâmina foliar uniforme, sem rugosidade de pústulas. Na terceira coluna, a máscara de segmentação ExR/Otsu confirma ausência de focos necróticos. E no canto superior, o status do SLA atesta: **DIAGNÓSTICO CONFIÁVEL**, pois a margem geométrica de separação é ampla e segura."*

---

### Bloco 3: Demonstração de Ferrugem Ativa e Severidade (04:00 - 06:00)
* **Ação na Tela:**
  1. Selecionar na galeria a amostra **`amostra_01_ferrugem_severa_campo.jpg`** (ou efetuar o upload de um patch de pústulas ativas).
  2. Destacar o salto no índice $ExR$ e a emergência de pústulas no mapa de textura.
* **Script de Apresentação (Cesar):**
  > *"Agora submetemos uma folha com infestação ativa de Ferrugem. Notem a transição imediata do painel: o laudo fitossanitário consolida **FERRUGEM (RUST)**. Observem o segundo quadrante: o mapa GLCM destaca com calor intenso a rugosidade provocada pela erupção dos uredínios que romperam a epiderme foliar. No terceiro quadrante, a máscara bio-espectral segmenta com precisão milimétrica cada foco infeccioso em vermelho carmim. O modelo detecta a queda drástica do Excesso de Verde (ExG) e o salto no Excesso de Vermelho (ExR), permitindo a tomada de decisão antes que a reboleira se alastre pelo talhão."*

---

### Bloco 4: Demonstração de Condição Desafiadora de Campo e Salvaguarda SLA (06:00 - 08:00)
* **Ação na Tela:**
  1. Selecionar a amostra **`amostra_05_necrose_senescente_sombra.jpg`** ou **`amostra_03_ferrugem_incipiente_desafio.jpg`**.
  2. Demonstrar o acionamento automático do alerta amarelo de salvaguarda de decisão.
* **Script de Apresentação (Cesar):**
  > *"Um sistema de IA responsável para o agro não pode ter excesso de confiança cega. Demonstramos aqui uma condição real de campo: folha com necrose senescente na ponta e sombra lateral da copa. Observem o que acontece: a margem geométrica em relação ao hiperplano cai para a faixa crítica ($\Delta_{\mathrm{margin}} < 0.35$) e a entropia de Shannon sobe. Imediatamente, o SugarVision ativa a salvaguarda de decisão: **INSPEÇÃO AGRONÔMICA EM CAMPO RECOMENDADA**. O sistema protege o produtor, impedindo a aplicação desnecessária de defensivo químico sem validação humana presencial."*

---

### Bloco 5: Encerramento, Governança e Q&A (08:00 - 10:00)
* **Ação na Tela:** Retornar à aba de probabilidades e expandir o painel de fundamentação fitopatológica.
* **Script de Apresentação (Cesar):**
  > *"Concluímos assim a comprovação de que o SugarVision une rigor matemático com usabilidade de campo. O código está congelado com dependências estritas, a suite de testes unitários foi 100% homologada e a titularidade dos direitos autorais está consolidada sob a equipe: Heitor Vitti, César Bovo, Guilherme Romanzotti, Marvin Cristhian, Elisa Alcantara e Luis Otávio. Estamos abertos às considerações e perguntas da banca avaliadora."*

---

## 3. Matriz de Validação Agronômica Operacional de Ponta a Ponta

A tabela a seguir consolida a validação agronômica conduzida com o conjunto cego de novas imagens de teste:

| ID Amostra | Arquivo Curado | Padrão Fitopatológico Real | Escala de Severidade | Resposta do SugarVision | Confiança Calibrada | Diagnóstico Operacional da Validação |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **#1** | `amostra_01_ferrugem_severa_campo.jpg` | Pústulas ativas de *Puccinia spp.* | Alta (> 25%) | **FERRUGEM (RUST)** | 88,4% | **Aprovado:** Detecta quebra de clorofila e aspereza epidérmica. |
| **#2** | `amostra_02_folha_sadia_controle.jpg` | Limbo foliar sadio fotossintético | Nula (0%) | **SADIA (HEALTHY)** | 94,1% | **Aprovado:** Reconhece alta reflectância verde e ausência de rugosidade. |
| **#3** | `amostra_03_ferrugem_incipiente_desafio.jpg` | Micropústulas jovens pontuais | Baixa (< 5%) | **FERRUGEM / SALVAGUARDA** | 72,5% | **Aprovado:** Alerta para foco incipiente com proximidade de margem. |
| **#4** | `amostra_04_mancha_foliar_clorose.jpg` | Estrias necróticas bacterianas | Moderada | **OUTRAS DOENÇAS** | 81,3% | **Aprovado:** Isola perfeitamente de ferrugem (ausência de pústula). |
| **#5** | `amostra_05_necrose_senescente_sombra.jpg` | Senescência natural + sombra | Não Infecciosa | **INSPEÇÃO RECOMENDADA** | 58,2% | **Aprovado:** Salvaguarda de decisão ativada por alta entropia. |
| **#6** | `amostra_06_especularidade_sol_direto.jpg` | Brilho especular cuticular | Artefato Óptico | **REJEIÇÃO DE CAPTURA** | 63,5% | **Aprovado:** Recomenda nova foto sob luz difusa conforme SLA. |

---

## 4. Plano de Contingência Técnico e Operacional (Plano B)

| Cenário de Falha Potencial | Impacto | Ação de Mitigação Imediata (Plano B) |
| :--- | :---: | :--- |
| **Queda ou Instabilidade de Internet na Sala** | Crítico | **Zero impacto:** A aplicação é **100% autônoma e offline**. Todos os modelos, imagens e pacotes rodam localmente no `localhost:8501`. |
| **Falha no Upload de Arquivo Manual** | Médio | Utilizar imediatamente a **Galeria de Teste Cego Curado** na barra lateral, que já possui as 6 amostras carregadas em cache de memória. |
| **Problema na Resolução do Projetor HDMI** | Baixo | Pressionar `Ctrl + -` ou `Ctrl + +` no navegador para ajustar a escala da página para a resolução nativa da sala (1080p ou 720p). |
| **Travamento Acidental da Aba do Navegador** | Médio | Reinicialização em 3 segundos: abrir novo terminal e rodar `streamlit run src/app_demoday5_sugarvision.py` ou reabrir `http://localhost:8501`. |
| **Perguntas sobre Diferenciação com Soja** | Teórico | Citar a Seção 4 da interface: tanto a ferrugem da cana (*Puccinia kuehnii*) quanto a da soja (*Phakopsora pachyrhizi*) compartilham a erupção de uredínios que alteram os canais $ExR$ e contraste GLCM. |

---

*Roteiro técnico e operacional homologado para a banca do Demoday 5 em 11/12/2026.*

