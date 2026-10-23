# 🛡️ Seção de Governança de MLOps, Model Registry e Integridade de Dados em Produção
## Capítulo Oficial para Composição do Relato da Solução Final — Squad 2

**Projeto:** Sanidade Vegetal AI (SugarVision) — Diagnóstico Fitopatológico em Cana-de-Açúcar  
**Instituição:** FATEC Rio Claro — Curso Superior de Tecnologia em Inteligência Artificial  
**Squad:** Squad 2 / Grupo 2 — Engenharia de Machine Learning & MLOps  
**Responsável Técnica pela Seção:** Elisa Almeida Alcântara (MLOps & Machine Learning Engineer)  
**Ciclo:** Sprint 5 — Fase Assess & Deploy Readiness (Framework SEMMA)  
**Status do Módulo:** **100% HOMOLOGADO E AUDITADO**

---

## 1. Contexto e Motivação Agronômica para a Governança

A transição de modelos de aprendizado de máquina do ambiente de laboratório para a operação agrícola real expõe o estimador a condições ambientais adversas: lentes de câmeras sujas, variações extremas de luminosidade solar, arquivos fotográficos corrompidos e formatos digitais heterogêneos.

Para blindar o classificador contra o princípio *garbage-in, garbage-out* e assegurar que decisões agronômicas de aplicação defensiva sejam tomadas com confiabilidade estocástica, a Squad 2 estruturou uma arquitetura completa de **Governança de MLOps**, alicerçada em três pilares fundamentais:
1. **Contrato Formal de Dados (Schema Enforcement):** Verificação pré-inferência de tipos, limites físicos e coerência estatística das 23 variáveis descritoras.
2. **Registro Centralizado e Auditado de Modelos (Model Registry):** Rastreabilidade criptográfica imutável (hashes SHA-256) com versionamento semântico.
3. **Plano de Continuidade Operacional:** Regras quantitativas de Rollback instantâneo e critérios objetivos para requalificação e retreino contínuo.

---

## 2. Contrato Formal de Entrada e Saída (Schema Contract)

O contrato de dados foi formalizado no arquivo [`models/model_schema_contract.json`](file:///c:/Users/elisa/OneDrive/Documentos/Sanidade-Vegetal/models/model_schema_contract.json). Ele estabelece os limites operacionais aceitos pelo sistema de inteligência artificial:

### 2.1 Especificação das Variáveis de Entrada
O sistema consome **23 descritores modelados**, organizados em duas naturezas estatísticas:

* **17 Descritores Numéricos Contínuos:**
  * *Cromáticos:* Matiz médio foliar (`mean_hue`, limite físico $[0, 180]$), desvio padrão de saturação (`std_saturation`, limite $[0, 1]$), índice de excesso verde (`exg_index`, $[-100, 150]$), índice de excesso vermelho (`exr_index`, $[-100, 150]$), razão espectral $R/G$ (`rg_ratio`, $[0, 10]$), índice de clorose/necrose (`indice_clorose_necrose`, $[0, 100]$) e dispersão angular do matiz (`hue_dispersion`, $[0, 90]$).
  * *Texturais Haralick (GLCM):* Contraste (`glcm_contrast`, $[0, 255]$), homogeneidade (`glcm_homogeneity`, $[0, 1]$), dissimilaridade (`glcm_dissimilarity`, $[0, 50]$) e energia (`glcm_energy`, $[0, 1]$).
  * *Morfologia da Pústula e Óptica:* Índice de Rugosidade Foliar de Pústula (`indice_rugosidade_pustula`, $[0, 5000]$), variância laplaciana de nitidez (`laplacian_var`, $[0, 2000]$), tamanho em disco (`size_kb`, $[1, 50000]$), contagem de pixels (`total_pixels`, $[10^4, 10^8]$), resolução em megapixels (`resolution_mp`, $[0.01, 100]$) e proporção dimensional (`aspect_ratio`, $[0.1, 10]$).

* **6 Descritores Categóricos e Discretizados:**
  * Proveniência agronômica (`dataset_source` $\in \{\text{'roboflow_sugarcane'}, \text{'mendeley_data'}\}$).
  * Extensão do formato (`extension` $\in \{\text{'.jpg'}, \text{'.jpeg'}, \text{'.png'}\}$).
  * Bins quantílicos e ordinais derivados na Fase Modify (`size_kb_bin`, `laplacian_var_bin`, `resolution_bin`, `aspect_ratio_bin`).

### 2.2 Política de Descarte de Metadados Não Preditivos
A arquitetura do pipeline utiliza a diretiva `remainder='drop'` no [`ColumnTransformer`](file:///c:/Users/elisa/OneDrive/Documentos/Sanidade-Vegetal/src/atomic_pipeline_preprocessing.py). Quaisquer colunas de rastreabilidade ou telemetria externa (ex.: coordenadas GPS, identificador do operador, UUID da foto) são descartadas de forma atômica e transparente antes de alimentar o classificador SVM, impedindo qualquer intervenção manual na inferência.

---

## 3. Arquitetura do Validador de Schema e Sistema de Quarentena

Para impedir que dados defeituosos causem falhas em tempo de execução ou gerem diagnósticos espúrios, foi desenvolvido o módulo [`src/schema_validator.py`](file:///c:/Users/elisa/OneDrive/Documentos/Sanidade-Vegetal/src/schema_validator.py).

```mermaid
flowchart TD
    A["Lote de Amostras de Imagem"] --> B["SchemaValidator.validate_batch()"]
    B --> C{"Análise de Conformidade"}
    C -->|Conforme (100% íntegro)| D["Subconjunto Válido"]
    C -->|Violação de Domínio / NaN / Coluna Ausente| E["Isolamento em Quarentena"]
    D --> F["Pipeline Final SVM (models/sanidade_vegetal_pipeline_final.joblib)"]
    F --> G["Predições Discretas + Probabilidades Platt"]
    E --> H["Log de Auditoria & Alerta ao Operador"]
```

### Funcionamento do Modo Quarentena (Resiliente):
Quando um lote agrícola misto é submetido à predição:
1. O validador separa as instâncias em dois subconjuntos: **Válidas** e **Quarentena**.
2. As amostras em conformidade prosseguem imediatamente para a inferência no modelo final serializado.
3. As amostras anômalas são retidas, gerando um relatório estruturado de violações contendo a linha exata, o atributo inconsistente e a razão técnica da rejeição (ex.: `"Valor 350.0 na feature 'std_saturation' acima do limite físico permitido (1.0)"`).

---

## 4. Model Registry e Rastreabilidade Criptográfica

Todos os modelos desenvolvidos e avaliados ao longo do ciclo SEMMA foram unificados no catálogo [`models/model_registry.json`](file:///c:/Users/elisa/OneDrive/Documentos/Sanidade-Vegetal/models/model_registry.json), garantindo a genealogia estrita da solução:

| Versão | Nome do Artefato | Estágio MLOps | Hash SHA-256 | Commit Git | $F_1$-Macro Teste |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0.1.0** | `baseline_dummy_model.joblib` | Archived | `4b7b29a2...` | `fbcd47f` | $0.0872$ |
| **0.2.0** | `svm_pipeline_baseline.joblib` | Archived | `7a353683...` | `7875b96` | $0.5421$ |
| **0.9.0** | `svm_pipeline_optimal.joblib` | Staging (Backup Rollback) | `be3df791...` | `c50dcf9` | $0.5747$ |
| **1.0.0** | `sanidade_vegetal_pipeline_final.joblib` | **In-Production** | `0100a66f...` | `c79a436` | **$0.6048$** |

### Garantias de Integridade Imutável:
* O arquivo binário oficial de produção ([`models/sanidade_vegetal_pipeline_final.joblib`](file:///c:/Users/elisa/OneDrive/Documentos/Sanidade-Vegetal/models/sanidade_vegetal_pipeline_final.joblib)) possui o hash SHA-256 `0100a66f15c5e7601ff578efaabaea080bebac00a9f9ec170cd8e31fb27e3e02`.
* Qualquer alteração não autorizada no binário causará falha imediata na checagem criptográfica em tempo de inicialização do serviço de predição.

---

## 5. Protocolo de Rollback e Critérios de Requalificação

A governança do projeto estabelece procedimentos determinísticos para manter a sustentabilidade da solução:

### 5.1 Regras de Rollback Automático
Se o sistema de monitoramento acusar:
1. **Divergência Populacional:** $PSI > 0.25$ nas características das imagens capturadas.
2. **Rejeição Crítica de Dados:** Taxa de quarentena superior a $5.0\%$ do volume diário de inferências.
3. **Queda de Performance Diagnóstica:** $F_1$-Macro caindo abaixo de $0.50$ em amostras checadas em campo.

O tráfego de inferência é comutado instantaneamente para o modelo **v0.9.0** (`svm_pipeline_optimal.joblib`), preservando a continuidade operacional da fazenda.

### 5.2 Gatilhos de Re-treinamento e Requalificação
O pipeline de re-treinamento é disparado sob três condições:
* Entrada de mais de **500 novas imagens rotuladas** por patologistas.
* Identificação de *Concept Drift* estatístico via teste de Kolmogorov-Smirnov ($p < 0.01$).
* Vencimento do ciclo fenológico trimestral (**90 dias** de operação contínua).

---

## 6. Validação e Homologação por Testes Automatizados

A conformidade desta governança é assegurada pela suíte [`src/test_schema_validation_and_registry.py`](file:///c:/Users/elisa/OneDrive/Documentos/Sanidade-Vegetal/src/test_schema_validation_and_registry.py), cobrindo:
* Validação de schema JSON contra a especificação formal.
* Rejeição e isolamento de amostras corrompidas e nulas.
* Homologação de lotes sadios sem falso-positivos de validação.
* Rastreabilidade e alinhamento do modelo em produção com o Model Registry.
