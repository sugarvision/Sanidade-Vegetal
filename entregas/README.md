# 📦 Catálogo de Entregas das Sprints — SugarVision

Neste diretório estão consolidadas todas as **16 pastas de entregas e relatórios técnicos** das tarefas de cada cartão Scrum/Jira, organizadas por Sprint e Fase do framework SEMMA:

| Sprint | Fase SEMMA | Diretório de Entrega | Descrição e Artefatos Principais |
| :---: | :---: | :--- | :--- |
| **Sprint 1** | `Sample` | [`Definição do Escopo do Problema/`](./Definição%20do%20Escopo%20do%20Problema/) | Escopo fitossanitário, hipóteses iniciais e formulação do problema. |
| **Sprint 1** | `Sample` | [`Definição da Estratégia de Amostragem/`](./Definição%20da%20Estratégia%20de%20Amostragem/) | Particionamento determinístico 70/15/15 e blindagem anti-leakage de talhões. |
| **Sprint 1** | `Explore` | [`Exploração de Imagens e Padrões Visuais/`](./Exploração%20de%20Imagens%20e%20Padrões%20Visuais/) | EDA de resolução, foco ótico (variância laplaciana) e histogramas cromáticos. |
| **Sprint 2** | `Modify` | [`Engenharia de Atributos e Descritores Visuais/`](./Engenharia%20de%20Atributos%20e%20Descritores%20Visuais/) | Extração de índices cromáticos ($ExG, ExR, R/G, ICN$) e textura Haralick ($GLCM$). |
| **Sprint 2** | `Modify` | [`Pipeline Atômico de Pré-processamento e Anti-Leakage/`](./Pipeline%20Atômico%20de%20Pré-processamento%20e%20Anti-Leakage/) | Construção do `ColumnTransformer` (imputação mediana + Z-Score + OHE). |
| **Sprint 2** | `Modify` | [`Consolidação e Exportação da ABT Final/`](./Consolidação%20e%20Exportação%20da%20ABT%20Final/) | Base consolidada com 0 NaNs em Parquet e CSV com hashes SHA-256. |
| **Sprint 2** | `Modify` | [`Modelagem Inicial e Dicionário de Dados/`](./Modelagem%20Inicial%20e%20Dicionário%20de%20Dados/) | Dicionário de dados v1.0.0 a v3.0.0 (`dicionario_de_dados.md` e `.csv`). |
| **Sprint 3** | `Model` | [`Integração Atômica do Estimador SVM ao Pipeline Anti-Leakage/`](./Integração%20Atômica%20do%20Estimador%20SVM%20ao%20Pipeline%20Anti-Leakage/) | Encapsulamento completo `Pipeline([ColumnTransformer, SVC])` com Platt Scaling. |
| **Sprint 3** | `Model` | [`Modelagem SVM com Kernel Trick (RBF e Polinomial)/`](./Modelagem%20SVM%20com%20Kernel%20Trick%20%28RBF%20e%20Polinomial%29/) | Estudo comparativo de fronteiras de decisão (Linear, Polinomial e RBF). |
| **Sprint 3** | `Model` | [`Configuração e Execução do Grid Search Exaustivo (GridSearchCV)/`](./Configuração%20e%20Execução%20do%20Grid%20Search%20Exaustivo%20%28GridSearchCV%29/) | Otimização de hiperparâmetros elegendo o estimador ótimo ($C=10, \gamma=0.01$). |
| **Sprint 3** | `Model` | [`Concepção, Treinamento e Benchmark de Modelos Baseline/`](./Concepção,%20Treinamento%20e%20Benchmark%20de%20Modelos%20Baseline/) | Modelos de referência (Dummy, Reg. Logística e Árvore rasa) e ganho estatístico do SVM. |
| **Sprint 4** | `Assess` | [`Auditoria de Particoes e Robustez Estocastica Repeated Stratified K-Fold/`](./Auditoria%20de%20Particoes%20e%20Robustez%20Estocastica%20Repeated%20Stratified%20K-Fold/) | Validação de 15 splits comprovando ausência de viés amostral (ANOVA $p > 0,05$). |
| **Sprint 4** | `Assess` | [`Otimizacao e Paralelizacao Computacional da Validacao Cruzada/`](./Otimizacao%20e%20Paralelizacao%20Computacional%20da%20Validacao%20Cruzada/) | Paralelização multi-processo `n_jobs=-1` com 1,64x de aceleração e telemetria. |
| **Sprint 4** | `Assess` | [`Catalogo Formal de Metadados e Dicionario com Saidas Calibradas/`](./Catalogo%20Formal%20de%20Metadados%20e%20Dicionario%20com%20Saidas%20Calibradas/) | Linhagem de ponta a ponta e auditoria de métricas com intervalos de confiança. |
| **Sprint 5** | `Assess` | [`Consolidacao Final do Dicionario de Dados e Catalogo de Producao/`](./Consolidacao%20Final%20do%20Dicionario%20de%20Dados%20e%20Catalogo%20de%20Producao/) | Consolidação do catálogo final de artefatos de produção e matriz SEMMA. |
| **Sprint 5** | `Assess` | [`Auditoria Final de Codigo Deploy e Congelamento de Dependencias/`](./Auditoria%20Final%20de%20Codigo%20Deploy%20e%20Congelamento%20de%20Dependencias/) | Congelamento de dependências, checklist de deploy e conformidade de licenças. |
