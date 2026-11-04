# ✅ Checklist de Preparação do Ambiente para Apresentação Presencial (Demoday 5 — 11/12)

**Projeto:** SugarVision — Sanidade Vegetal (*Saccharum officinarum*)  
**Sprint:** 5 — Framework SEMMA (Fase: Assess & Deploy Readiness)  
**Responsável Técnico:** Cesar (Lead Técnico & Visão Computacional)  
**Data da Bancada Avaliadora:** 11/12/2026  
**Detentores Exclusivos dos Direitos Autorais (Copyright © 2026):**  
Heitor Vitti, César Bovo, Guilherme Romanzotti, Marvin Cristhian, Elisa Alcantara e Luis Otávio.

---

## 1. Objetivo do Checklist

Garantir que a infraestrutura computacional, o ambiente Python, a interface Streamlit e os artefatos de modelo estejam **100% testados, pré-aquecidos e operacionais** antes do início da sessão de avaliação presencial no dia **11/12/2026**, assegurando zero atrito e zero falha técnica diante da banca avaliadora.

---

## 2. Checklist de Hardware e Projeção

- [ ] **Fonte de Alimentação / Carregador:** Notebook conectado à tomada com 100% de carga (desabilitar modo economia de bateria para desempenho máximo de CPU).
- [ ] **Adaptadores de Vídeo:** Levar adaptador USB-C / Mini DisplayPort para HDMI testado previamente no projetor da sala.
- [ ] **Configuração de Resolução de Tela:**
  - Resolução do projetor configurada para **1920x1080 (Full HD)** ou **1366x768**.
  - Modo de projeção configurado em **"Duplicar Vídeo"** (`Windows + P` -> *Duplicar*).
- [ ] **Modo Não Perturbe e Notificações:**
  - Ativar o modo *Não Perturbe* no Windows para bloquear notificações de WhatsApp, e-mail e atualizações do sistema durante a fala.
  - Fechar programas pesados em segundo plano (Spotify, navegadores extras, Discord, etc.).

---

## 3. Checklist de Ambiente de Software e Repositório

- [ ] **Status do Repositório Git:**
  - Branch ativa: `main` limpa e sincronizada no commit final de homologação.
  - Verificar ausência de arquivos temporários ou conflitos:
    ```bash
    git status
    ```
- [ ] **Dependências Python:**
  - Ambiente virtual ativo com as versões estritas do `requirements.txt`:
    ```bash
    python -c "import streamlit, sklearn, cv2, PIL, joblib; print('Ambiente 100% OK!')"
    ```
- [ ] **Suíte de Testes Automatizados:**
  - Executar a validação dos testes da interface e do motor:
    ```bash
    python -m unittest src/test_demoday5_interface.py
    ```
    *Critério de Aceite:* **5/5 testes passando com status OK**.

---

## 4. Comando Único de Inicialização Offline e Pré-Aquecimento

Para garantir que a aplicação inicie instantaneamente sem depender de conexão de rede ou telemetria externa, utilize a seguinte linha de comando no terminal do projeto:

```bash
streamlit run src/app_demoday5_sugarvision.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

### Protocolo de Pré-Aquecimento de Cache (T-30 Minutos)
1. Abrir o navegador no endereço: `http://localhost:8501`.
2. Pressionar `F11` para colocar o navegador em modo tela cheia (*Fullscreen*).
3. Selecionar na galeria lateral a **Amostra #1 (`amostra_01_ferrugem_severa_campo.jpg`)** para carregar os pesos do SVM na memória RAM.
4. Selecionar a **Amostra #2 (`amostra_02_folha_sadia_controle.jpg`)** para garantir que o pipeline atômico já esteja totalmente instanciado no cache do Streamlit (`@st.cache_resource`).
5. Manter a aba aberta e posicionada na tela inicial.

---

## 5. Cronograma de Ações nos Minutos Finais

### T-10 Minutos antes de Subir ao Palco:
- [ ] Conectar o cabo HDMI do projetor ao notebook e testar a exibição da tela duplicada.
- [ ] Ajustar o zoom do navegador (`Ctrl + 0` para 100% ou `Ctrl + +` para 110% caso a sala seja ampla).
- [ ] Verificar se o som do sistema está mudo.
- [ ] Abrir o roteiro de fala impresso ou no celular de apoio ([`docs/roteiro_demonstracao_ao_vivo_demoday5.md`](roteiro_demonstracao_ao_vivo_demoday5.md)).

### T-1 Minuto (Chamada da Equipe SugarVision):
- [ ] Posicionar o cursor do mouse sobre o seletor de amostras na barra lateral.
- [ ] Confirmar que a janela da aplicação Streamlit está maximizada e nítida no projetor.
- [ ] Respirar fundo, cumprimentar a banca avaliadora e iniciar o pitch conforme o roteiro cronometrado!

---

*Checklist formal de engenharia homologado para a apresentação do Demoday 5 (11/12/2026).*

