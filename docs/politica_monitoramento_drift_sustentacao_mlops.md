# 🛡️ Política de Monitoramento Contínuo de Drift, Sustentação e Retreinamento (MLOps)
## Arquitetura Estatística de Detecção de Data/Concept Drift e Governança Pós-Implantação (Sprint 5)

**Projeto:** Sanidade Vegetal (SugarVision) — Diagnóstico Fitopatológico em Cana-de-Açúcar (*Saccharum officinarum*)  
**Framework Metodológico:** SEMMA (*Fase Assess & Sustain — MLOps Governance*)  
**Sprint:** 5 — Monitoramento Contínuo em Produção e Sustentação do Modelo  
**Status Operacional:** **HOMOLOGADO E AUDITADO EM PRODUÇÃO (`v1.0.0`)**  
**Artefatos Vinculados:**
- Módulo Estatístico Core: [`../src/mlops_drift_monitoring.py`](../src/mlops_drift_monitoring.py)
- Suíte de Testes Unitários (`pytest`): [`../src/test_mlops_drift_monitoring.py`](../src/test_mlops_drift_monitoring.py)
- Configuração de Fontes e Limiares: [`../models/mlops_drift_monitoring_config.json`](../models/mlops_drift_monitoring_config.json)
- Relatório de Telemetria de Produção: [`../models/mlops_drift_monitoring_report.json`](../models/mlops_drift_monitoring_report.json)

---

## 1. Visão Geral e Arquitetura de Monitoramento Contínuo

Modelos de visão computacional implantados em ecossistemas agrícolas abertos estão sujeitos à **não-estacionariedade espaço-temporal**: a distribuição conjunta das variáveis ópticas de entrada e das classes fitopatológicas $P(X, Y)$ evolui continuamente devido ao clima, sensores de captura e fisiologia da lavoura.

Para evitar a degradação silenciosa (*silent model failure*) do classificador campeão ([`models/sanidade_vegetal_pipeline_final.joblib`](../models/sanidade_vegetal_pipeline_final.joblib)), o **SugarVision** implementa uma arquitetura em camadas de observabilidade estatística:

```mermaid
flowchart TD
    A["Lotes de Inferência em Campo (Imagens Foliares)"] --> B["Camada 1: SchemaValidator\n(Validação Física & Quarentena)"]
    B -->|Amostras Conformes| C["Extração das 17 Features Visuais\n(Cromáticas HSV/ExG/ExR + Textura GLCM + Óptica)"]
    C --> D["Camada 2: Motor Estatístico de Drift\n(src/mlops_drift_monitoring.py)"]
    D --> E["Cálculo do Population Stability Index (PSI)\n(Discretização Quantílica em 10 Decis)"]
    D --> F["Teste Bicaudal de Kolmogorov-Smirnov (KS)\n(Distância D entre CDFs Empíricas)"]
    E --> G{"Motor de Regras e Gatilhos\n(DriftThresholds)"}
    F --> G
    G -->|PSI < 0.10 e KS p >= 0.05| H["STATUS: NORMAL\nOperação Nominal v1.0.0"]
    G -->|0.10 <= PSI < 0.25 ou KS p < 0.05| I["STATUS: WARNING_MONITORING\nAmostragem Ativa (Active Learning)"]
    G -->|PSI >= 0.25 ou KS p < 0.01 (D >= 0.10)| J["STATUS: CRITICAL_RETRAIN_REQUIRED\nDisparo do Pipeline de Retreinamento"]
    G -->|Drift Severo + Queda F1 < 0.50| K["STATUS: EMERGENCY_ROLLBACK_AND_RETRAIN\nComutação Imediata para v0.9.0 + Retreino"]
```

---

## 2. Mapeamento das Fontes Potenciais de Drift no Campo Agrícola

O catálogo formal de fontes de desvio foi codificado em [`FIELD_DRIFT_SOURCES_CATALOG`](../src/mlops_drift_monitoring.py) e persistido em [`models/mlops_drift_monitoring_config.json`](../models/mlops_drift_monitoring_config.json), cobrindo três vetores estruturais do domínio canavieiro:

| ID Fonte | Categoria de Campo | Taxonomia Estatística | Causas Ambientais e Operacionais | Features Visuais Impactadas | Mecanismo Físico/Biológico e Mitigação |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **`DRIFT-SRC-01`** | **Mudança de Iluminação Sazonal** | *Data Drift* (*Covariate Shift*: $P_{\text{prod}}(X) \neq P_{\text{train}}(X)$) | Transição entre safra de verão (alta irradiação zenital e reflexo especular) e inverno seco (luz difusa oblíqua); nebulosidade, poeira de colheita e capturas no crepúsculo. | `mean_hue`, `std_saturation`, `exg_index`, `exr_index`, `rg_ratio`, `hue_dispersion` | Reflexos especulares saturam o brilho e comprimem `std_saturation`, enquanto iluminação crepuscular inflaciona `rg_ratio` e `exr_index`. **Mitigação:** re-calibração sazonal do `StandardScaler` e balanço de branco na borda. |
| **`DRIFT-SRC-02`** | **Novas Câmeras e Sensores Ópticos** | *Data Drift* (*Covariate Shift*) | Adoção de novos smartphones ou câmeras acopladas a tratores/VANTs; alterações de resolução CMOS, pós-processamento ISP (*sharpening*, HDR) e abrasão/sujeira nas lentes. | `laplacian_var`, `glcm_contrast`, `glcm_homogeneity`, `glcm_dissimilarity`, `glcm_energy`, `resolution_mp`, `total_pixels`, `size_kb`, `aspect_ratio` | Filtros ISP de nitidez elevam artificialmente `laplacian_var` e `glcm_contrast`; lentes sujas suavizam pústulas urediniais, reduzindo `indice_rugosidade_pustula`. **Mitigação:** bloqueio de `laplacian_var < 20` no `SchemaValidator` e lotes de calibração por dispositivo. |
| **`DRIFT-SRC-03`** | **Estádios Fenológicos e Novas Cultivares** | *Concept Drift* ($P_{\text{prod}}(Y \mid X) \neq P_{\text{train}}(Y \mid X)$) & *Covariate Shift* | Transição entre brotação, perfilhamento, crescimento e senescência natural; plantio de novas variedades (RB, CTC, IAC) com cerosidade e teor basal de clorofila distintos; estresse hídrico/nutricional. | `indice_clorose_necrose`, `indice_rugosidade_pustula`, `exg_index`, `exr_index`, `rg_ratio`, `mean_hue`, `glcm_dissimilarity` | Na maturação/senescência, a queda fisiológica de clorofila reduz `exg_index` e eleva `indice_clorose_necrose` em folhas sadias, deslocando a fronteira ótima entre folhas sadias maduras e *Yellow Leaf* / *Red Rot*. **Mitigação:** retreinamento trimestral (90 dias) com curadoria fitopatológica. |

---

## 3. Formulação Matemática das Métricas de Monitoramento

O módulo [`src/mlops_drift_monitoring.py`](../src/mlops_drift_monitoring.py) avalia cada uma das **17 features numéricas contínuas** da Tabela Analítica Base (ABT) combinando uma métrica de divergência entrópica discretizada (PSI) e um teste de hipótese não-paramétrico sobre a função de distribuição acumulada contínua (KS).

### 3.1 Population Stability Index (PSI)
Para cada feature visual $x$, a distribuição de referência do conjunto de treinamento ($N_{\text{ref}} = 5.574$) é particionada em $B = 10$ decis empíricos ($q_0, q_{10}, \dots, q_{100}$), com extremos abertos $(-\infty, +\infty)$ para capturar *outliers* em produção. Seja $Q_b$ a fração de amostras de referência no *bin* $b$ e $P_b$ a fração de amostras do lote de produção no mesmo *bin* $b$ (regularizadas por suavização de Laplace $\epsilon = 10^{-6}$):

$$\text{PSI} = \sum_{b=1}^{B} \left( P_b - Q_b \right) \times \ln\left( \frac{P_b}{Q_b} \right)$$

O PSI corresponde à divergência de Kullback-Leibler simetrizada (Divergência de Jeffreys) entre as distribuições discretizadas:
- **$\text{PSI} < 0.10$ (`STABLE`):** Distribuição estatisticamente estável.
- **$0.10 \le \text{PSI} < 0.25$ (`WARNING`):** Deslocamento moderado; requer acompanhamento intensivo.
- **$\text{PSI} \ge 0.25$ (`CRITICAL`):** Drift populacional severo; aciona gatilho de retreinamento e avaliação de rollback.

### 3.2 Teste Bicaudal de Kolmogorov-Smirnov (KS de Duas Amostras)
Para detectar mudanças finas de forma, assimetria ou variância sem depender de discretização em *bins*, aplica-se o teste exato/assintótico de Kolmogorov-Smirnov sobre as funções de distribuição empírica acumulada (ECDF) de treino $F_{1,n}(x)$ e de produção $F_{2,m}(x)$:

$$D_{n,m} = \sup_{x \in \mathbb{R}} \left| F_{1,n}(x) - F_{2,m}(x) \right|$$

A hipótese nula $H_0: F_1(x) = F_2(x)$ é rejeitada em nível crítico quando $p\text{-value} < \alpha_{\text{crit}} = 0.01$ associado a uma magnitude de efeito mínimo $D_{n,m} \ge 0.10$, prevenindo falsos positivos decorrentes de tamanhos amostrais elevados.

---

## 4. Regras de Alerta e Gatilhos Automatizados de Retreinamento

A função [`evaluate_retraining_triggers`](../src/mlops_drift_monitoring.py) consolida 7 regras determinísticas parametrizadas pela classe [`DriftThresholds`](../src/mlops_drift_monitoring.py):

| Código da Regra | Métrica Monitorada | Condição de Disparo (*Threshold*) | Ação Automatizada do Sistema MLOps |
| :--- | :--- | :--- | :--- |
| **`RULE-PSI-CRITICAL`** | Population Stability Index ($\text{PSI}$) | $\text{PSI} \ge 0.25$ em $\ge 2$ features ou em $\ge 1$ feature agronômica crítica | Dispara pipeline de retreinamento e avalia rollback preventivo para `v0.9.0`. |
| **`RULE-KS-CONCEPT-DATA-DRIFT`** | Teste de Kolmogorov-Smirnov ($D, p$) | $p < 0.01$ e $D \ge 0.10$ em $\ge 2$ features visuais | Dispara pipeline de retreinamento com re-estimação do hiperplano SVM. |
| **`RULE-MULTIVARIATE-DRIFT-FRACTION`** | Fração Sistêmica de Features Críticas | $\ge 20\%$ das features monitoradas ($\ge 4$ de $17$) em estado `CRITICAL` | Dispara retreinamento completo da cadeia `ColumnTransformer` + `SVC`. |
| **`RULE-DATA-VOLUME-CADENCE`** | Volume de Novas Imagens Laudadas | $\ge 500$ novas imagens validadas por engenheiros agrônomos | Dispara retreinamento incremental para incorporação de novos padrões de campo. |
| **`RULE-PHENOLOGICAL-WINDOW-90D`** | Janela Temporal desde Último Treino | $\ge 90\text{ dias}$ corridos em produção sem requalificação | Força retreinamento sazonal alinhado ao ciclo fenológico trimestral da cana. |
| **`RULE-PERFORMANCE-DEGRADATION-F1`** | $F_1$-Score Macro Auditado em Campo | $F_1\text{-Macro} < 0.50$ em lotes amostrados | **Rollback imediato** para `v0.9.0` e abertura de incidente de retreinamento urgente. |
| **`RULE-QUARANTINE-SPIKE`** | Taxa de Quarentena (`SchemaValidator`) | $> 5.0\%$ das amostras do lote retidas por violação de contrato | Emite alerta crítico de falha óptica/sensor e bloqueia promoção automática. |

---

## 5. Resultados Experimentais da Telemetria nos Cenários Homologados

A execução do motor [`src/mlops_drift_monitoring.py`](../src/mlops_drift_monitoring.py) sobre a Tabela Analítica Base ([`data/processed/abt_sanidade_vegetal.csv`](../data/processed/abt_sanidade_vegetal.csv)) auditou quatro cenários operacionais documentados em [`models/mlops_drift_monitoring_report.json`](../models/mlops_drift_monitoring_report.json):

| Cenário Auditado | Lote Analisado ($N$) | $\text{PSI}_{\max}$ | $\text{PSI}_{\text{médio}}$ | $D_{\max}$ (KS) | Features Críticas | Nível de Alerta Emitido | Retreino? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Operação Nominal Estável** | $500$ | $0.0553$ | $0.0185$ | $0.0442$ | $0 / 17$ | `NORMAL` | ❌ Não |
| **2. Transição Sazonal de Iluminação** | $500$ | $0.1331$ | $0.0252$ | $0.1636$ | $1 / 17$ (`mean_hue`) | `WARNING_MONITORING` | ❌ Não |
| **3. Drift Severo (Novas Cultivares + Sensores)** | $500$ | $12.4313$ | $2.4735$ | $0.8805$ | $5 / 17$ (`exg`, `exr`, `icn`, `glcm`, `laplacian`) | `EMERGENCY_ROLLBACK_AND_RETRAIN` | ✅ **Sim** |
| **4. Auditoria Real Entre Partições (Teste vs. Treino)** | $380$ | $0.8218$ | $0.1462$ | $0.2747$ | $5 / 17$ (atributos de resolução óptica) | `EMERGENCY_ROLLBACK_AND_RETRAIN` | ✅ **Sim** |

> **Nota Analítica Relevante (Cenário 4):** A auditoria cruzada entre a partição de treino e a partição de teste cego demonstra a sensibilidade real do motor: enquanto os descritores biológicos/cromáticos permanecem estáveis (garantindo $F_1\text{-Macro} = 0.6048$), os descritores físicos de arquivo (`size_kb`, `total_pixels`, `resolution_mp`) acusam $\text{PSI}_{\max} = 0.8218$ devido à diferença proporcional entre imagens das fontes *Roboflow* e *Mendeley* no split de teste.

---

## 6. Política de Sustentação, Ciclo de Retreinamento e Governança

Quando qualquer gatilho crítico define `retrain_required = True`, o ciclo padronizado de sustentação MLOps executa 5 etapas auditáveis:

1. **Congelamento e Snapshot Forense:** O lote causador do drift e o relatório JSON (`mlops_drift_monitoring_report.json`) são arquivados com timestamp e hash SHA-256. Se `rollback_required = True`, o serviço de inferência comuta imediatamente o ponteiro de produção para o artefato estável `v0.9.0` (`models/svm_pipeline_optimal.joblib`).
2. **Curadoria Fitopatológica (*Human-in-the-Loop / Active Learning*):** As amostras que apresentaram maior distância estatística ou maior entropia de probabilidade Platt são enviadas para validação por engenheiros agrônomos, assegurando rótulos livres de ruído antes da ingestão na ABT.
3. **Re-execução Atômica do Pipeline SEMMA:**
   - Atualização da ABT consolidada e verificação estrita de zero valores ausentes (`0 NaNs`).
   - Re-ajuste completo do `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) exclusivamente sobre a nova partição de treinamento expandida (blindagem anti-leakage).
   - Otimização de hiperparâmetros via `GridSearchCV` com validação cruzada estratificada repetida (`RepeatedStratifiedKFold`).
4. **Validação nos Portões de Promoção (*Promotion Gates*):** O modelo candidato (`v1.1.0` em estágio `Staging`) só é promovido para `In-Production` se superar simultaneamente:
   - $F_1\text{-Macro} \ge 0.55$ e Acurácia Balanceada $\ge 0.58$ no conjunto de teste cego atualizado;
   - Recall na classe crítica de Ferrugem (`RUST`) $\ge 0.60$;
   - Aprovação em 100% da suíte automatizada (`pytest`).
5. **Registro Criptográfico no Model Registry:** O novo binário recebe assinatura SHA-256 em [`models/model_registry.json`](../models/model_registry.json), e a versão anterior (`v1.0.0`) é movida para o estágio `Staging` (novo alvo de rollback imediato).
