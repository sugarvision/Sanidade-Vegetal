# 📚 Documentação Técnica do Projeto — Sanidade Vegetal (SugarVision)

**Framework Metodológico:** SEMMA (*Sample, Explore, Modify, Model, Assess*)  
**Sprint:** 1 — Setup, Sample & Explore  

---

## 📂 Índice Geral de Entregas e Documentos Técnicos

### 1. Definição do Escopo e Domínio Agronômico (Cesar)
- **[`../Definição do Escopo do Problema/README.md`](../entregas/Definição%20do%20Escopo%20do%20Problema/README.md)**: Resumo executivo e cobertura do checklist.
- **[`../Definição do Escopo do Problema/definicao_do_escopo_sanidade_vegetal.md`](../entregas/Definição%20do%20Escopo%20do%20Problema/definicao_do_escopo_sanidade_vegetal.md)**: Delimitação técnica do problema (binário vs multiclasse), classes fitopatológicas e impacto agronômico.
- **[`../Definição do Escopo do Problema/hipoteses_iniciais_e_metadados.md`](../entregas/Definição%20do%20Escopo%20do%20Problema/hipoteses_iniciais_e_metadados.md)**: Formulação de 5 hipóteses agronômicas/computacionais e matriz de variáveis de interesse.

### 2. Exploração de Imagens e Padrões Visuais (Cesar)
- **[`../Exploração de Imagens e Padrões Visuais/README.md`](../entregas/Explora%C3%A7%C3%A3o%20de%20Imagens%20e%20Padr%C3%B5es%20Visuais/README.md)**: Visão geral da caracterização visual e cobertura do checklist.
- **[`../Exploração de Imagens e Padrões Visuais/analise_padroes_visuais_e_desafios.md`](../entregas/Explora%C3%A7%C3%A3o%20de%20Imagens%20e%20Padr%C3%B5es%20Visuais/analise_padroes_visuais_e_desafios.md)**: Estudo aprofundado dos padrões de textura (GLCM), cor (HSV/RGB), desafios fotométricos (iluminação, sombras, fundo) e disparidade de resolução.
- **[`./figures/`](./figures/)**: Gráficos gerados para diagnóstico e análise exploratória visual.

### 3. Modelagem de Dados e Dicionário (Heitor)
- **[`../Modelagem Inicial e Dicionário de Dados/dicionario_de_dados.md`](../entregas/Modelagem%20Inicial%20e%20Dicion%C3%A1rio%20de%20Dados/dicionario_de_dados.md)**: Dicionário completo de dados e tipagem.
- **[`../Modelagem Inicial e Dicionário de Dados/modelagem_conceitual_entidades.md`](../entregas/Modelagem%20Inicial%20e%20Dicion%C3%A1rio%20de%20Dados/modelagem_conceitual_entidades.md)**: Modelagem relacional e conceitual das entidades.

### 4. Estratégia de Amostragem (Heitor)
- **[`../Definição da Estratégia de Amostragem/estrategia_de_amostragem.md`](../entregas/Defini%C3%A7%C3%A3o%20da%20Estrat%C3%A9gia%20de%20Amostragem/estrategia_de_amostragem.md)**: Metodologia de particionamento estratificado e mitigação de leakage.
- **[`../Definição da Estratégia de Amostragem/distribuicao_e_balanceamento.md`](../entregas/Defini%C3%A7%C3%A3o%20da%20Estrat%C3%A9gia%20de%20Amostragem/distribuicao_e_balanceamento.md)**: Estudo de desbalanceamento e pesos de classe.

### 5. Pipelines e Execução Técnica (Cesar & Equipe - Sprint 1)
- **[`../notebooks/02A_sprint1_master_pipeline_reprodutivel.ipynb`](../notebooks/02A_sprint1_master_pipeline_reprodutivel.ipynb)**: Notebook mestre da Sprint 1 executável de ponta a ponta sem falhas.
- **[`../data/processed/abt_sanidade_vegetal.csv`](../data/processed/abt_sanidade_vegetal.csv)**: Tabela analítica consolidada preliminar (6.571 registros x 16 atributos).

### 🏆 Relatórios Executivos da Sprint Review (Sprint 1)
- **[`./Sprint_Review_1_Sanidade_Vegetal_Completa.docx`](./Sprint_Review_1_Sanidade_Vegetal_Completa.docx)**: Relatório executivo completo em formato Microsoft Word (`.docx`), com tabelas estritas, caixas de destaque e imagens diagnósticas incorporadas.
- **[`./Sprint_Review_1_Sanidade_Vegetal_Completa.pdf`](./Sprint_Review_1_Sanidade_Vegetal_Completa.pdf)**: Relatório executivo completo em formato PDF pronto para distribuição e apresentação.

---

## 📂 Entregas da Sprint 2 — Framework SEMMA (Fase: Modify)

### 6. Engenharia de Atributos e Descritores Visuais (Cesar)
- **[`../Engenharia de Atributos e Descritores Visuais/README.md`](../entregas/Engenharia%20de%20Atributos%20e%20Descritores%20Visuais/README.md)**: Resumo executivo da entrega e cobertura dos checklists das tarefas do Cesar.
- **[`../Engenharia de Atributos e Descritores Visuais/engenharia_descritores_cromaticos_e_indices.md`](../entregas/Engenharia%20de%20Atributos%20e%20Descritores%20Visuais/engenharia_descritores_cromaticos_e_indices.md)**: Documentação da Tarefa 1 (Espaços HSV/CIELAB, índices $ExG, ExR$, razões espectrais $R/G$ e testes estatísticos de separabilidade).
- **[`../Engenharia de Atributos e Descritores Visuais/extracao_textura_glcm_e_rugosidade.md`](../entregas/Engenharia%20de%20Atributos%20e%20Descritores%20Visuais/extracao_textura_glcm_e_rugosidade.md)**: Documentação da Tarefa 2 (Matriz GLCM multidirecional, propriedades de Haralick, invariância à rotação e validação da hipótese de rugosidade foliar).
- **[`../src/feature_engineering_visual.py`](../src/feature_engineering_visual.py)**: Módulo Python executável de transformação de features visuais e geração de figuras diagnósticas.
- **[`./figures/sprint2_analise_cromaticas_hsv_exg.png`](./figures/sprint2_analise_cromaticas_hsv_exg.png)**: Diagnóstico visual dos índices cromáticos e espectrais.
- **[`./figures/sprint2_analise_texturas_glcm_haralick.png`](./figures/sprint2_analise_texturas_glcm_haralick.png)**: Diagnóstico visual dos descritores texturais de Haralick.

### 7. Construção do Pipeline Atômico com ColumnTransformer e Anti-Leakage (Elisa)
- **[`../Pipeline Atômico de Pré-processamento e Anti-Leakage/README.md`](../entregas/Pipeline%20At%C3%B4mico%20de%20Pr%C3%A9-processamento%20e%20Anti-Leakage/README.md)**: Resumo executivo da entrega e cobertura de 100% do checklist da Sprint 2.
- **[`../Pipeline Atômico de Pré-processamento e Anti-Leakage/pipeline_atomico_columntransformer.md`](../entregas/Pipeline%20At%C3%B4mico%20de%20Pr%C3%A9-processamento%20e%20Anti-Leakage/pipeline_atomico_columntransformer.md)**: Documentação técnica completa da arquitetura atômica, sub-pipelines numérico/categórico, auditoria matemática anti-leakage e governança MLOps.
- **[`../src/atomic_pipeline_preprocessing.py`](../src/atomic_pipeline_preprocessing.py)**: Módulo Python executável de pré-processamento atômico, testes de conformidade e serialização.
- **[`../notebooks/06_pipeline_atomico_columntransformer.ipynb`](../notebooks/06_pipeline_atomico_columntransformer.ipynb)**: Notebook executável auditável de ponta a ponta com diagrama interativo e testes.
- **[`../models/preprocessor_pipeline.joblib`](../models/preprocessor_pipeline.joblib)**: Pipeline de pré-processamento serializado e comprimido (2,58 KB) pronto para a Sprint 3.
- **[`../models/preprocessor_metadata.json`](../models/preprocessor_metadata.json)**: Manifesto de governança MLOps com hash SHA-256 e schema das 35 features resultantes.
- **[`../data/processed/abt_features_modelagem.parquet`](../data/processed/abt_features_modelagem.parquet)**: Tabela Analítica Base consolidada com 6.571 registros e zero valores ausentes.

### 8. Consolidação e Exportação da ABT Final (Elisa)
- **[`../Consolidação e Exportação da ABT Final/README.md`](../entregas/Consolida%C3%A7%C3%A3o%20e%20Exporta%C3%A7%C3%A3o%20da%20ABT%20Final/README.md)**: Resumo executivo da entrega e cobertura de 100% do checklist da Sprint 2.
- **[`../Consolidação e Exportação da ABT Final/consolidacao_exportacao_abt_final.md`](../entregas/Consolida%C3%A7%C3%A3o%20e%20Exporta%C3%A7%C3%A3o%20da%20ABT%20Final/consolidacao_exportacao_abt_final.md)**: Documentação técnica detalhada, auditoria estrita de 0 NaNs, benchmark de I/O (Parquet vs CSV) e contrato de integração com GridSearchCV.
- **[`../src/export_final_modeling_abt.py`](../src/export_final_modeling_abt.py)**: Script executável de consolidação, auditoria automatizada, hashes SHA-256 e smoke test com GridSearchCV.
- **[`../notebooks/07_consolidacao_exportacao_abt_final.ipynb`](../notebooks/07_consolidacao_exportacao_abt_final.ipynb)**: Notebook executável demonstrando benchmark de leitura, integridade dos alvos e prontidão para a Sprint 3.
- **[`../data/processed/abt_features_modelagem.parquet`](../data/processed/abt_features_modelagem.parquet)**: Tabela Analítica Base final em Apache Parquet (916,92 KB, Snappy, 6.571 x 40).
- **[`../data/processed/abt_features_modelagem.csv`](../data/processed/abt_features_modelagem.csv)**: Espelho em texto plano CSV para interoperabilidade e auditoria (2.989,66 KB).
- **[`../data/processed/abt_features_modelagem_manifest.json`](../data/processed/abt_features_modelagem_manifest.json)**: Manifesto de integridade com hashes criptográficos SHA-256 e estatísticas.

### 🏆 Relatórios Executivos e Apresentação do Demoday 2 (Fase Modify)
- **[`./Relato_da_Solucao_Volume_2_Engenharia_de_Features.docx`](./Relato_da_Solucao_Volume_2_Engenharia_de_Features.docx)**: Relatório executivo e técnico oficial do **Volume 2** em formato Microsoft Word (`.docx`), com banner institucional, tabelas estilizadas, caixas de destaque e imagens diagnósticas incorporadas.
- **[`./relato_da_solucao_volume_2.md`](./relato_da_solucao_volume_2.md)**: Versão completa em Markdown do Relato da Solução — Volume 2 para leitura e versionamento Git.
- **[`./Apresentacao_Demoday_2_Executiva_Resumida.pptx`](./Apresentacao_Demoday_2_Executiva_Resumida.pptx)**: **Apresentação Executiva Ultra-Resumida (.pptx)**: Slide deck direto de 6 slides sintetizando os 4 blocos centrais para pitch rápido.
- **[`./Apresentacao_Demoday_2_Engenharia_de_Features.pptx`](./Apresentacao_Demoday_2_Engenharia_de_Features.pptx)**: Apresentação de slides oficial widescreen 16:9 (`.pptx`) completa com 9 slides, design premium, cards, métricas e figuras diagnósticas.
- **[`./apresentacao_demoday_2_engenharia_de_features.md`](./apresentacao_demoday_2_engenharia_de_features.md)**: Roteiro completo de apresentação com minutagem cravada (10 minutos) e Speaker Notes (scripts de fala detalhados) para cada slide.
- **[`./inventario_fase_modify_semma.md`](./inventario_fase_modify_semma.md)**: **Inventário Geral e Dossiê da Fase Modify**: Catálogo completo com mapeamento de todos os notebooks, scripts em `src/`, documentações técnicas, bases persistidas e guia de execução.
- **[`./guia_didatico_termos_demoday2.md`](./guia_didatico_termos_demoday2.md)**: **Guia Didático para Leigos**: Explicação descomplicada com analogias cotidianas de todos os termos técnicos (Features, GLCM, ExG, Data Leakage, ABT, Parquet e SVM).

---

## 📂 Entregas da Sprint 3 — Framework SEMMA (Fase: Model)

### 9. Diagnóstico Matemático dos Vetores de Suporte e Curvas de Aprendizado (Viés-Variância)
- **[`../entregas/Diagnóstico Matemático dos Vetores de Suporte (Support Vectors)/README.md`](../entregas/Diagn%C3%B3stico%20Matem%C3%A1tico%20dos%20Vetores%20de%20Suporte%20%28Support%20Vectors%29/README.md)**: Dossiê completo de densidade de suporte (`n_support_`), trade-off de $C$, variáveis de folga ($\xi_i > 0$), isolamento de patches ambíguos e projeções 2D/3D do hiperplano.
- **[`./relatorio_curvas_aprendizado_vies_variancia_sprint3.md`](./relatorio_curvas_aprendizado_vies_variancia_sprint3.md)**: Relatório técnico-estatístico de Curvas de Aprendizado (10% a 100%), convergência assintótica ($S_\infty$) e decomposição do Trade-off Viés-Variância comprovando a ausência de sobreajuste.
- **[`../entregas/Curvas de Aprendizado e Análise do Trade-off Viés-Variância/README.md`](../entregas/Curvas%20de%20Aprendizado%20e%20An%C3%A1lise%20do%20Trade-off%20Vi%C3%A9s-Vari%C3%A2ncia/README.md)**: Dossiê executivo da entrega da Sprint 3 com tabelas, diagnósticos matemáticos e figuras em 300 DPI.
- **[`../src/svm_learning_curves_bias_variance.py`](../src/svm_learning_curves_bias_variance.py)**: Módulo Python de geração de curvas de aprendizado, diagnóstico assintótico e plotagem estatística.
- **[`./figures/sprint3_svm_learning_curve_champion.png`](./figures/sprint3_svm_learning_curve_champion.png)**: Curva de aprendizado (10% a 100%) e contração do gap de generalização do modelo campeão.
- **[`./figures/sprint3_svm_bias_variance_regimes_comparison.png`](./figures/sprint3_svm_bias_variance_regimes_comparison.png)**: Comparativo experimental dos três regimes de Viés-Variância (Subajuste, Equilíbrio Campeão e Sobreajuste).
- **[`./figures/sprint3_svm_learning_curve_scalability_times.png`](./figures/sprint3_svm_learning_curve_scalability_times.png)**: Escalabilidade temporal de treinamento/inferência e retorno marginal por incremento amostral.

---

## 📂 Entregas da Sprint 4 — Framework SEMMA (Fase: Assess — Relato Vol. 4)

### 10. Matrizes de Confusão Normalizadas, Curvas ROC-AUC, Precision-Recall e Threshold Tuning
- **[`./relatorio_validacao_estatistica_matrizes_roc_pr_vol4.md`](./relatorio_validacao_estatistica_matrizes_roc_pr_vol4.md)**: Relatório estatístico completo do **Relato Vol. 4** contendo matrizes de confusão absolutas e normalizadas (7 patologias e agrupamento agronômico Saudável/Ferrugem/Outras), curvas ROC One-vs-Rest (Macro/Micro AUC), curvas Precision-Recall focadas na Ferrugem e calibração de limiares (*Threshold Tuning*).
- **[`../entregas/Matrizes de Confusão e Curvas de Discriminação Estatística (ROC e PR)/README.md`](../entregas/Matrizes%20de%20Confus%C3%A3o%20e%20Curvas%20de%20Discrimina%C3%A7%C3%A3o%20Estat%C3%ADstica%20%28ROC%20e%20PR%29/README.md)**: Dossiê executivo da entrega da Sprint 4.
- **[`../src/svm_statistical_validation_assess.py`](../src/svm_statistical_validation_assess.py)**: Módulo Python de validação estatística e geração das figuras em alta resolução.
- **[`./figures/sprint4_confusion_matrices_multiclass_abs_norm.png`](./figures/sprint4_confusion_matrices_multiclass_abs_norm.png)**: Matrizes de confusão absoluta e normalizada das 7 patologias foliares.
- **[`./figures/sprint4_confusion_matrices_agronomic_3class.png`](./figures/sprint4_confusion_matrices_agronomic_3class.png)**: Matrizes de confusão absoluta e normalizada do agrupamento agronômico (Saudável, Ferrugem e Outras Patologias).
- **[`./figures/sprint4_roc_curves_multiclass_ovr.png`](./figures/sprint4_roc_curves_multiclass_ovr.png)**: Curvas ROC multiclasse One-vs-Rest (Macro e Micro AUC-ROC).
- **[`./figures/sprint4_pr_curve_rust_and_critical_classes.png`](./figures/sprint4_pr_curve_rust_and_critical_classes.png)**: Curva Precision-Recall focada na classe crítica de Ferrugem (`RUST`) e comparativo multiclasse.
- **[`./figures/sprint4_threshold_tuning_sensitivity_calibration.png`](./figures/sprint4_threshold_tuning_sensitivity_calibration.png)**: Curvas de calibração de limiares de decisão (*Threshold Tuning*) para otimização de sensibilidade.

### 10b. Arcabouço de Comparação Estatística: Baselines vs. Configurações Otimizadas de SVM (Luis — Relato Vol. 4)
- **[`./relatorio_arcabouco_comparacao_estatistica_baselines_vs_svm.md`](./relatorio_arcabouco_comparacao_estatistica_baselines_vs_svm.md)**: Relatório estatístico executivo completo cobrindo a comparação unificada dos 10 modelos supervisionados (Sprints 3 e 4), testes de hipótese pareados (teste t de Student e Wilcoxon), intervalos de confiança a 95%, margens de erro entre folds e justificativa de governança analítica.
- **[`../entregas/Estrutura do Arcabouco de Comparacao Estatistica entre Baselines e SVM/README.md`](../entregas/Estrutura%20do%20Arcabouco%20de%20Comparacao%20Estatistica%20entre%20Baselines%20e%20SVM/README.md)**: Dossiê completo da entrega com checklist 100% homologado.
- **[`../src/statistical_comparison_framework.py`](../src/statistical_comparison_framework.py)**: Módulo Python executável de avaliação cruzada 5-fold, testes de hipótese estatística e cálculo de intervalos de confiança.
- **[`../src/test_statistical_comparison_framework.py`](../src/test_statistical_comparison_framework.py)**: Suíte de testes automatizados com 100% de cobertura dos 4 critérios de aceite do checklist.
- **[`../models/unified_models_benchmark_sprint3_sprint4.csv`](../models/unified_models_benchmark_sprint3_sprint4.csv)**: Tabela consolidada com 27 atributos de generalização para os 10 modelos supervisionados.
- **[`../models/statistical_hypothesis_tests_summary.csv`](../models/statistical_hypothesis_tests_summary.csv)**: Consolidação dos testes pareados de Student, Wilcoxon, Cohen's $d$, ganhos absolutos e relativos.
- **[`../models/confidence_intervals_metrics_95.csv`](../models/confidence_intervals_metrics_95.csv)**: Documentação formal dos intervalos de confiança (95%) e margens de erro entre os folds.
- **[`../models/statistical_comparison_framework_metadata.json`](../models/statistical_comparison_framework_metadata.json)**: Manifesto de governança analítica e metadados MLOps.
- **[`./figures/models_benchmark_unified_comparison.png`](./figures/models_benchmark_unified_comparison.png)**: Comparativo unificado de métricas ($F_1\text{-Macro}$ e Acurácia) por partição.
- **[`./figures/confidence_intervals_error_margins_95.png`](./figures/confidence_intervals_error_margins_95.png)**: Forest plot com intervalos de confiança a 95% e margens de erro.
- **[`./figures/statistical_gain_hypothesis_tests.png`](./figures/statistical_gain_hypothesis_tests.png)**: Tamanhos de efeito de Cohen ($d$) e ganhos relativos percentuais sobre os baselines.
- **[`./figures/pareto_efficiency_latency_vs_f1.png`](./figures/pareto_efficiency_latency_vs_f1.png)**: Mapeamento da fronteira de Pareto de latência vs desempenho preditivo.

### 11. Auditoria de Erros Críticos e Impacto Agronômico/Econômico (Falsos Positivos vs. Falsos Negativos)
- **[`./relatorio_auditoria_erros_impacto_agronomico_sprint4.md`](./relatorio_auditoria_erros_impacto_agronomico_sprint4.md)**: Relatório executivo e financeiro de auditoria das 149 instâncias classificadas incorretamente no conjunto de teste independente ($N=380$), quantificação do risco agronômico de Falso Negativo (`RUST` e `RED ROT`), impacto econômico de Falso Positivo (fungicida químico) e Matriz de Custo/Perda Agronômica $7 \times 7$ com Regra de Decisão de Risco Mínimo de Bayes.
- **[`../entregas/Auditoria de Erros Críticos e Impacto Agronômico (FP vs FN)/README.md`](../entregas/Auditoria%20de%20Erros%20Cr%C3%ADticos%20e%20Impacto%20Agron%C3%B4mico%20%28FP%20vs%20FN%29/README.md)**: Dossiê completo da entrega de Auditoria de Erros da Sprint 4.
- **[`../src/svm_agronomic_error_audit.py`](../src/svm_agronomic_error_audit.py)**: Módulo Python de isolamento de instâncias incorretas, modelagem econômico-agronômica da cana-de-açúcar e geração automatizada de relatórios e gráficos.
- **[`../models/svm_misclassified_test_instances.csv`](../models/svm_misclassified_test_instances.csv)**: Catálogo completo das 149 amostras classificadas incorretamente no conjunto de teste com metadados visuais e probabilidades preditas.
- **[`../models/svm_agronomic_cost_matrix_7x7.csv`](../models/svm_agronomic_cost_matrix_7x7.csv)**: Matriz de Custo/Perda Agronômica $7 \times 7$ ($\mathbf{L}$) em R$/ha.
- **[`./figures/sprint4_agronomic_cost_matrix_and_losses.png`](./figures/sprint4_agronomic_cost_matrix_and_losses.png)**: Mapa de calor da Matriz de Custo Agronômico ($\text{R\$/ha}$) e Perda Financeira Realizada ($\text{R\$}$) no conjunto de teste.
- **[`./figures/sprint4_fn_fp_economic_risk_curves.png`](./figures/sprint4_fn_fp_economic_risk_curves.png)**: Distribuição das 149 instâncias incorretas por par de transição e curvas de custo assimétrico de Falso Negativo vs. Falso Positivo.

### 12. Interpretabilidade Matemática, Importância de Features e Limitações Visuais (Cesar — Relato Vol. 4)
- **[./relatorio_interpretabilidade_features_e_limitacoes_visuais_vol4.md](./relatorio_interpretabilidade_features_e_limitacoes_visuais_vol4.md)**: Relatório técnico completo de **Interpretabilidade Matemática e Limitações Visuais (Vol. 4)** cobrindo Permutation Feature Importance estocástica (=380, K=15$), comparação das métricas de cor (, ExR, R/G$) vs. textura Haralick (GLCM), avaliação de variáveis de discretização e bordas (laplacian_var), validação fitopatológica de pústulas e clorose/necrose, mapeamento de margens geométricas do hiperplano SVM ($\Delta_{\\mathrm{margin}}$), taxonomia de padrões fotográficos recorrentes de falha e condições de contorno operacionais para o aplicativo SugarVision em campo.
- **[../src/svm_feature_importance_interpretability.py](../src/svm_feature_importance_interpretability.py)**: Módulo Python executável de Permutation Feature Importance, agregação por famílias funcionais e plotagem executiva.
- **[../src/svm_visual_boundary_inspection.py](../src/svm_visual_boundary_inspection.py)**: Módulo Python executável de inspeção geométrica de fronteira, cálculo de margens relativas, entropia de Shannon e taxonomia de modos de falha fotográfica.
- **[../src/test_svm_feature_importance_interpretability.py](../src/test_svm_feature_importance_interpretability.py)**: Suíte de testes unitários automatizada do módulo de importância de features.
- **[../src/test_svm_visual_boundary_inspection.py](../src/test_svm_visual_boundary_inspection.py)**: Suíte de testes unitários automatizada do módulo de inspeção visual de fronteira.
- **[../models/svm_permutation_feature_importance.csv](../models/svm_permutation_feature_importance.csv)**: Ranking formal auditado das 27 features no teste independente.
- **[../models/svm_feature_importance_by_group.csv](../models/svm_feature_importance_by_group.csv)**: Consolidação estatística por famílias funcionais de atributos.
- **[../models/svm_feature_importance_metadata.json](../models/svm_feature_importance_metadata.json)**: Metadados JSON com governança e validação fitopatológica.
- **[../models/svm_boundary_geometric_margin_cases.csv](../models/svm_boundary_geometric_margin_cases.csv)**: Tabela de distâncias geométricas ao hiperplano, incerteza informacional e amostras limítrofes.
- **[../models/svm_visual_error_patterns_summary.csv](../models/svm_visual_error_patterns_summary.csv)**: Distribuição quantitativa dos modos de falha fotográfica.
- **[../models/svm_visual_boundary_inspection_metadata.json](../models/svm_visual_boundary_inspection_metadata.json)**: Manifesto de conformidade e auditoria dos casos de fronteira.
- **[./figures/sprint4_permutation_feature_importance_ranking.png](./figures/sprint4_permutation_feature_importance_ranking.png)**: Gráfico executivo de barras com ranking de importância e erro padrão.
- **[./figures/sprint4_feature_importance_by_group.png](./figures/sprint4_feature_importance_by_group.png)**: Gráfico comparativo de impacto acumulado e share percentual por família funcional.
- **[./figures/sprint4_visual_boundary_cases_and_limitations.png](./figures/sprint4_visual_boundary_cases_and_limitations.png)**: Painel diagnóstico de 4 quadrantes (margem geométrica, entropia, falhas fotográficas e casos limítrofes  \\times ICN$).

---

## 📂 Entregas da Sprint 5 — Framework SEMMA (Fase: Assess & Deploy Readiness — Demoday 5)

### 13. Interface Interativa em Tempo Real, Validação Operacional e Roteiro de Apresentação (Cesar — Sprint 5)
- **[../src/app_demoday5_sugarvision.py](../src/app_demoday5_sugarvision.py)**: Aplicação visual interativa em Streamlit desenvolvida para a demonstração ao vivo da banca avaliadora, contendo upload drag-and-drop, galeria de teste cego, cards de probabilidade calibrada (Platt Scaling), triplo painel visual (Imagem Original, Mapa de Textura Haralick GLCM e Máscara de Segmentação ExR/Otsu) e salvaguarda operacional de SLA.
- **[../src/vision_inference_engine.py](../src/vision_inference_engine.py)**: Motor de visão computacional em tempo real que extrai os 27 descritores espectro-texturais de fotos brutas, gera os mapas de relevo rugoso e executa o estimador campeão com predição tripartida (Sadia vs Ferrugem vs Outras Doenças).
- **[../src/test_demoday5_interface.py](../src/test_demoday5_interface.py)**: Suíte de testes automatizados unitários homologando o motor de inferência, a extração de atributos, a consistência dos mapas visuais e as amostras do teste cego (5/5 testes aprovados).
- **[../data/demoday5_blind_samples/](../data/demoday5_blind_samples/)**: Conjunto curado de 6 imagens de teste cego representando condições desafiadoras de campo (ferrugem severa, folha sadia controle, início de pústulas, estrias cloróticas, sombra senescente e reflexo solar direto).
- **[../data/demoday5_blind_samples/manifest_teste_cego_demoday5.json](../data/demoday5_blind_samples/manifest_teste_cego_demoday5.json)**: Manifesto auditado com laudos agronômicos de referência, escalas de severidade e recomendações fitossanitárias.
- **[./roteiro_demonstracao_ao_vivo_demoday5.md](./roteiro_demonstracao_ao_vivo_demoday5.md)**: Roteiro minucioso passo a passo cronometrado (8 a 10 minutos) para a banca avaliadora presencial, com script de fala, matriz de validação agronômica e plano de contingência técnico operacional (Plano B).
- **[./checklist_preparacao_ambiente_demoday5.md](./checklist_preparacao_ambiente_demoday5.md)**: Protocolo rigoroso de checklist de hardware, conectividade de vídeo, dependências do ambiente Python e comando único de inicialização offline para o dia 11/12/2026.
- **[./especificidade_escala_e_limitacoes_operacionais_modelo.md](./especificidade_escala_e_limitacoes_operacionais_modelo.md)**: Nota técnica formal sobre especificidade de escala (*Leaf-Level Macro Inspection* vs *Canopy / Field Landscape*), detalhando a física de degradação espectro-textural ($ExG$, $ExR$, GLCM), limites de Nyquist e roadmap de expansão multi-escala (YOLO / Drones).
- **[./Diagnostico_Inteligente_CanadeAcucar_Transicao_Modify_Model.docx](./Diagnostico_Inteligente_CanadeAcucar_Transicao_Modify_Model.docx)**: Documento editável em Microsoft Word (.docx) contendo o relato completo da transição metodológica SEMMA (Modify → Model) com recortes de código-fonte de cada etapa.
- **[./diagnostico_inteligente_transicao_modify_model.md](./diagnostico_inteligente_transicao_modify_model.md)**: Versão em Markdown estruturada do relato de transição metodológica (Modify → Model) com blocos de código integrados.

### 14. Relato da Solução Final: Síntese de Viabilidade Técnica, Operacional, Econômica e Matriz SWOT (Sprint 5)
- **[./relato_solucao_final.md](./relato_solucao_final.md)**: Documento oficial consolidado do **Relato da Solução Final**, contendo a síntese de ponta a ponta do ciclo SEMMA, o cálculo detalhado de redução de perdas produtivas na lavoura de soja ($\text{R\$ } 2.182,10\text{/ha}$ preservados) e de cana-de-açúcar, a análise comparativa entre o diagnóstico automatizado por visão computacional ($< 0,80\text{ s}$) e a inspeção manual por lupa ($180\text{–}300\text{ s}$), a Matriz SWOT completa em tabelas Markdown e a conclusão integrada de viabilidade e ROI.
- **[../entregas/Sintese de Viabilidade Tecnica Operacional e Economica/README.md](../entregas/Sintese%20de%20Viabilidade%20Tecnica%20Operacional%20e%20Economica/README.md)**: Dossiê executivo da entrega da Sprint 5 com cobertura integral do checklist.
