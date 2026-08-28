# Entrega: Definição da Estratégia de Amostragem
**Projeto:** Sanidade-Vegetal (SugarVision)  
**Sprint:** 1  
**Responsável:** Equipe de Ciência de Dados  
**Data:** 28 de Agosto de 2026  

---

## 📌 Visão Geral da Entrega

Esta entrega formaliza a **Estratégia de Amostragem e Particionamento dos Dados** para o projeto **Sanidade-Vegetal (SugarVision)**. 

O foco central é garantir a representatividade estatística das classes fitopatológicas da cana-de-açúcar, com atenção especial aos casos de **Ferrugem** (*Puccinia melanocephala / Puccinia kuehnii*) e **Folhas Saudáveis**, mitigando riscos de sobreajuste (*overfitting*), viés amostral e vazamento de dados (*data leakage*).

---

## 📂 Arquivos Desta Entrega

1. **[`estrategia_de_amostragem.md`](./estrategia_de_amostragem.md)**: Documento técnico completo com metodologia de amostragem estratificada, controle de desbalanceamento, particionamento temporal/espacial por talhão e critérios de divisão de treino, validação e teste.
2. **[`distribuicao_e_balanceamento.md`](./distribuicao_e_balanceamento.md)**: Estudo analítico do perfil de distribuição de classes nos datasets de referência (*Mendeley Data* e *Roboflow*), cálculo do fator de desbalanceamento e estratégias de ponderação de classes (*Class Weights*).
3. **[`sampling_pipeline.py`](./sampling_pipeline.py)**: Script em Python contendo a implementação executável dos algoritmos de amostragem estratificada (`StratifiedGroupKFold`), cálculo de pesos e divisão determinística de partições.

---

## ✅ Cobertura do Checklist da Task

| Item do Checklist | Status | Onde Encontrar |
| :--- | :---: | :--- |
| **Analisar distribuição das classes** | Concluído | Seção 2 de [`distribuicao_e_balanceamento.md`](./distribuicao_e_balanceamento.md) |
| **Verificar desbalanceamento** | Concluído | Seção 3 de [`distribuicao_e_balanceamento.md`](./distribuicao_e_balanceamento.md) |
| **Definir amostragem estratificada** | Concluído | Seção 2 de [`estrategia_de_amostragem.md`](./estrategia_de_amostragem.md) e [`sampling_pipeline.py`](./sampling_pipeline.py) |
| **Definir partições temporais (se houver dados temporais)** | Concluído | Seção 4 de [`estrategia_de_amostragem.md`](./estrategia_de_amostragem.md) |
| **Separar dados em desenvolvimento e validação inicial** | Concluído | Seção 3 de [`estrategia_de_amostragem.md`](./estrategia_de_amostragem.md) |
| **Documentar critérios de amostragem** | Concluído | [`estrategia_de_amostragem.md`](./estrategia_de_amostragem.md) |
