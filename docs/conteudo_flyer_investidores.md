# 💼 Dossiê e Roteiro do Flyer Comercial & Investidores (One-Pager A4) — SugarVision™

Este documento contém a estruturação executiva, dados de mercado, unit economics e roteiro de argumentação do **Flyer Comercial de 1 Página (A4 Oficial)** do **SugarVision (Sanidade Vegetal AI)**.

O material foi desenhado especificamente para **despertar interesse de investidores (Venture Capital / AgTech Funds)**, **usinas sucroenergéticas**, **cooperativas** e **parceiros estratégicos de M&A e OEM**.

---

## 📌 Links de Acesso Rápido

* **Flyer Comercial Interativo (Pronto para PDF / Impressão A4):** [`docs/flyer_comercial_investidores_sugarvision.html`](./flyer_comercial_investidores_sugarvision.html)
* **Flyer Institucional de Campo (Versão Produtor Rural):** [`docs/flyer_divulgacao_sugarvision.html`](./flyer_divulgacao_sugarvision.html)
* **Ilustração / Mockup de Alta Resolução do App em Campo:** [`docs/figures/sugarvision_app_mockup.jpg`](./figures/sugarvision_app_mockup.jpg)

> 💡 **Como Salvar em PDF de Alta Qualidade:**
> 1. Abra o arquivo [`docs/flyer_comercial_investidores_sugarvision.html`](./flyer_comercial_investidores_sugarvision.html) no Chrome ou Edge.
> 2. Pressione `Ctrl + P` (ou clique no botão **"Salvar em PDF / Imprimir"** no topo).
> 3. Em *Destino*, selecione **Salvar como PDF**.
> 4. Em *Layout*, selecione **Retrato** (A4). Em *Mais definições*, marque **Gráficos de segundo plano**.
> 5. Clique em **Salvar**. O arquivo terá exatamente 1 página A4 sem quebras.

---

## 🎯 Tese de Investimento & Resumo Executivo (Elevator Pitch)

| Eixo Estratégico | Detalhamento para Comitê de Investimentos |
| :--- | :--- |
| **Setor / Categoria** | AgTech • Inteligência Artificial Aplicada • SaaS B2B • Visão Computacional Edge |
| **Problema Crítico** | A Ferrugem da Cana (*Puccinia spp.*) e patologias foliares causam perdas anuais de 30% a 70% em TCH e ATR. As abordagens tradicionais dependem de amostragem humana lenta e pulverização preventiva em área total, desperdiçando defensivos. |
| **Solução SugarVision** | Diagnóstico fitossanitário instantâneo (< 0.8s) via smartphone ou câmeras em drones/maquinário, operando **100% offline** diretamente no talhão. |
| **Moat Tecnológico** | Classificador SVM RBF otimizado com índices espectrais físicos ($ExG, ExR, HSV$) e matriz de textura GLCM Haralick, treinado sobre **6.571 imagens de campo**, com zero vazamento de dados e conformidade MLOps (SEMMA). |
| **Proposta de Valor** | Redução de até **28% no consumo de defensivos agrícolas** por pulverização dirigida e preservação de até **R$ 2.400 / hectare** em biomassa e sacarose. |
| **Estágio** | Homologado (Sprint 5) com 29 testes de software aprovados; abrindo captação Seed para pilotos em usinas sucroalcooleiras de SP, MG e GO. |

---

## 📊 Estrutura dos Blocos do Flyer

### 1. Cabeçalho Corporativo
* **Marca:** `SugarVision™`
* **Assinatura:** `Inteligência Artificial & Sanidade Vegetal de Precisão`
* **Badges:** `AgTech B2B SaaS` • `Rodada Seed • 2026`

### 2. Hero Headline (Gancho Comercial)
* **Super-título:** `Diagnóstico Fitopatológico em Tempo Real`
* **Título Principal:**
  > *"Transformando perdas de safra em **economia milionária** com Visão Computacional no canavial."*
* **Subtítulo:**
  > *"Detectamos Ferrugem e fitopatologias na cana-de-açúcar em menos de 1 segundo, direto pelo smartphone ou maquinário, 100% offline, reduzindo em até 28% o custo de defensivos."*

### 3. Vitrine Visual do Produto (Ilustração & Mockup)
* **Mockup Fotográfico:** Smartphone premium escaneando folha de cana com pústulas de ferrugem em alta definição, interface com bounding box luminosa, score de confiança `98.7%`, cálculo de severidade e mapa cromático NDVI/ExG.
* **Badges Flutuantes:**
  * `0.78s de Resposta (Offline):` Sem latência de rede no canavial.
  * `Pulverização Cirúrgica:` Tratamento focado nos focos iniciais.
  * `Proteção de ATR e TCH:` Preserva até R$ 2.400/ha.

### 4. Faixa de Tração & Métricas Chave (Key Metrics)
* **6.571** Imagens Auditadas na Base de Campo.
* **> 91%** AUC-ROC Validado em 15 Folds (Repeated Stratified K-Fold).
* **28%** Economia Média Projetada em Insumos e Combustível.
* **< 1 Safra** Payback Estimado para Grupos Sucroalcooleiros.

### 5. Os 4 Pilares de Defesa para Investidores
1. **🌾 Tamanho de Mercado (TAM / SAM / SOM):**
   * *TAM:* Setor canavieiro brasileiro movimenta mais de R$ 120 Bilhões/ano (+650M toneladas e 9,5M ha).
   * *SAM:* 360 usinas industriais e mais de 65.000 fornecedores de cana.
   * *SOM:* Foco geográfico nos estados de SP, MG e GO (72% da moagem nacional).
2. **🧠 Moat Tecnológico & Defensibilidade:**
   * Pipeline atômico determinístico SEMMA (`ColumnTransformer` + `SVC`).
   * Índices bio-espectrais auditáveis ($ExG, ExR, R/G$) e textura microscópica Haralick.
   * Inferência edge leve: roda em microprocessadores e smartphones de baixo custo sem GPU industrial.
3. **💰 ROI & Unit Economics:**
   * Custo da inação: cada dia sem controle custa de R$ 80 a R$ 140/ha em sacarose degradada.
   * Eliminação de 2 a 3 pulverizações desnecessárias em área total.
   * LTV/CAC projetado superior a $6.5\times$ devido ao ciclo recorrente de safras.
4. **🚀 Go-To-Market & Modelos de Receita:**
   * *SaaS por Hectare:* Assinatura anual de R$ 14 a R$ 22/ha para usinas e cooperativas.
   * *OEM API:* Licenciamento de SDK embarcado para fabricantes de drones e barras de pulverização.

### 6. Espectro Fitopatológico Coberto (7 Classes)
* ✓ Saudável (*Healthy Leaf*)
* ⚠ Ferrugem da Cana (*Puccinia melanocephala* e *Puccinia kuehnii*)
* ⚠ Escaldadura (*Xanthomonas albilineans*)
* ⚠ Podridão Vermelha (*Colletotrichum falcatum*)
* ⚠ Mosaico e Amarelecimento Foliar (SCMV e SCYLV)

### 7. Modelo de Operação B2B
1. **Mobile App Agrônomo:** Laudos rápidos com GPS para equipes de monitoramento.
2. **Cloud Ops Dashboard:** Mapas de calor de dispersão para gerentes agrícolas.
3. **Edge SDK Maquinário:** Integração com controladores de bico inteligente (pulverização seletiva).

### 8. Rodapé de Contato & Deal Room
* **Call to Action:** *"Participe da Revolução da Sanidade Vegetal — Buscamos investidores e parceiros estratégicos para acelerar a expansão comercial em usinas piloto na safra 2026/2027."*
* **Canal Investidor:** `dealroom@sugarvision.agr.br`
* **Repositório do Projeto:** `github.com/Sanidade-Vegetal`
* **QR Code:** Direcionamento direto para deck e materiais adicionais.
