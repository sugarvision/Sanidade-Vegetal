# Relatório Formal de Auditoria de Licenças de Código Aberto e Permissões de Dados
## Governança de Propriedade Intelectual e Conformidade Ética de Dados (Sprint 5)

**Projeto:** Sanidade Vegetal (SugarVision) — Diagnóstico Fitopatológico em Cana-de-Açúcar  
**Sprint:** 5 — Auditoria Final, Padronização de Deploy e Reprodutibilidade  
**Responsável Técnico:** Heitor Partezani (Engenharia de Machine Learning & IA SVM)  
**Status da Auditoria:** **100% HOMOLOGADO E CONFORME (COMPLIANT)**  

---

## 1. Sumário Executivo

A viabilidade técnica e comercial de qualquer iniciativa de inteligência artificial aplicada ao agronegócio depende estritamente da segurança jurídica quanto à proveniência dos dados (*Data Provenance*) e ao licenciamento de seus componentes de software. 

Esta auditoria tem por objetivo atestar e certificar que:
1. Todos os **conjuntos de dados fotográficos externos** utilizados para treinar e validar o estimador SVM possuem termos de licença permissivos que autorizam legalmente sua ingestão, extração de atributos, treinamento e eventual comercialização da solução.
2. Todas as **bibliotecas de software livre e de código aberto (*FOSS*)** integradas ao ambiente de produção em `requirements.txt` possuem licenças compatíveis entre si e com a licença do projeto (**MIT**), ausentes de restrições virais (*copyleft forte*, como GPLv3 ou AGPLv3).
3. Não há risco de contaminação por violação de propriedade intelectual ou violação de privacidade (as imagens registram exclusivamente lavouras e tecidos botânicos foliares, sem dados pessoais ou biométricos sujeitos a LGPD/GDPR).

---

## 2. Auditoria dos Conjuntos de Dados (Bases Externas)

A solução **SugarVision** consome duas bases públicas globais consolidadas:

### 2.1 Roboflow Universe — *Sugarcane Disease Classification*
- **Mantenedor / Autor:** Asad Unvar
- **Repositório Oficial:** [Roboflow Universe - Sugarcane Disease Classification](https://universe.roboflow.com/asad-unvar/sugarcane-disease-classification)
- **Licença Declarada:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**
- **Volumetria no Projeto:** 6.571 imagens organizadas em partições `train/`, `valid/` e `test/`.
- **Patologias Cobertas:** *Healthy*, *Rust*, *Red Rot*, *Mosaic* e *Yellow Leaf*.
- **Termos e Permissões Concedidas (CC BY 4.0):**
  - **Compartilhamento:** Copiar e redistribuir o material em qualquer suporte ou formato.
  - **Adaptação:** Remixar, transformar e criar a partir do material para qualquer fim, **inclusive comercial**.
  - **Condição Obrigatória:** Atribuição de crédito adequado, inclusão do link para a licença e indicação se alterações foram feitas.
- **Conformidade do Projeto SugarVision:** **TOTALMENTE CONFORME**. A atribuição foi devidamente formalizada no `README.md`, no arquivo `LICENSE` e nos dicionários de dados. As imagens originais não são reempacotadas como produto fechado, mas utilizadas para extração de matrizes matemáticas de atributos na ABT.

### 2.2 Mendeley Data — *Sugarcane Diseases*
- **Autores:** Ijaz Kalhoro, Rafaqat Hussain e Hidayatullah Shaikh
- **Identificador de Objeto Digital (DOI):** [10.17632/rzh99cj2rj.1](https://doi.org/10.17632/rzh99cj2rj.1)
- **Publicação:** Mendeley Data, Elsevier (V1, 2020)
- **Licença Declarada:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**
- **Patologias Cobertas:** *Red Rot*, *Smut*, *Leaf Scald* e *Healthy*.
- **Termos e Permissões Concedidas (CC BY 4.0):**
  - Uso irrestrito para pesquisa, desenvolvimento tecnológico, benchmarking e aplicações derivadas sob citação da publicação.
- **Conformidade do Projeto SugarVision:** **TOTALMENTE CONFORME**. A citação acadêmica formal com DOI consta em toda a documentação da solução.

---

## 3. Auditoria das Bibliotecas e Dependências de Software Livre

A tabela abaixo detalha cada uma das dependências congeladas em `requirements.txt`:

| Pacote / Biblioteca | Versão Estrita | Licença Declarada | Tipo de Licença | Permissão Comercial? | Restrição Copyleft Viral? | Compatibilidade com MIT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **scikit-learn** | `1.8.0` | BSD 3-Clause | Permissiva | **Sim** | **Não** | **100% Compatível** |
| **numpy** | `2.4.4` | BSD 3-Clause | Permissiva | **Sim** | **Não** | **100% Compatível** |
| **pandas** | `3.0.3` | BSD 3-Clause | Permissiva | **Sim** | **Não** | **100% Compatível** |
| **scipy** | `1.17.1` | BSD 3-Clause | Permissiva | **Sim** | **Não** | **100% Compatível** |
| **matplotlib** | `3.10.9` | PSF / BSD-like | Permissiva | **Sim** | **Não** | **100% Compatível** |
| **joblib** | `1.5.3` | BSD 3-Clause | Permissiva | **Sim** | **Não** | **100% Compatível** |
| **psutil** | `7.2.2` | BSD 3-Clause | Permissiva | **Sim** | **Não** | **100% Compatível** |
| **Pillow** | `12.2.0` | HPND (Historical) | Permissiva | **Sim** | **Não** | **100% Compatível** |
| **opencv-python** | `5.0.0` | Apache 2.0 | Permissiva | **Sim** | **Não** | **100% Compatível** |

### 3.1 Análise de Compatibilidade Jurídica das Licenças
1. **Ausência de Copyleft Viral:** Nenhuma dependência utiliza GPL, AGPL ou LGPD estrita que imponha a obrigação de disponibilizar publicamente o código de eventuais aplicações proprietárias integradas.
2. **Licenças BSD 3-Clause e MIT:** São historicamente interoperáveis e amplamente reconhecidas pela Open Source Initiative (OSI). A única exigência é a manutenção do aviso de copyright original dos autores nos cabeçalhos e documentação, o que foi 100% atendido.
3. **Apache 2.0 (OpenCV):** Confere concessão expressa de direitos de patente, garantindo que o pipeline de visão computacional pode ser executado em infraestruturas industriais e de nuvem sem litígios de propriedade intelectual.

---

## 4. Conformidade Regulatória e Privacidade de Dados (LGPD / GDPR)

- **Natureza dos Dados Coletados:** Imagens fotográficas digitais capturando exclusivamente a morfologia foliar, colmos e tecidos vegetais de cana-de-açúcar (*Saccharum officinarum*), bem como leituras de telemetria agroclimática (temperatura, umidade, índices cromáticos).
- **Dados Pessoais:** Não há captura de rostos humanos, dados de geolocalização residencial ou informações que permitam a identificação individual de colaboradores de campo.
- **Conclusão:** O projeto é classificado como **risco zero** no âmbito da Lei Geral de Proteção de Dados Pessoais (Lei nº 13.709/2018 - LGPD) e do General Data Protection Regulation (GDPR).

---

## 5. Parecer Conclusivo da Governança

> **HOMOLOGAÇÃO JURÍDICA E TÉCNICA:**
> Atesta-se formalmente que o projeto **Sanidade-Vegetal (SugarVision)** cumpre integralmente todas as exigências legais de licenciamento de software livre e permissões de dados das bases utilizadas. O repositório está apto para publicação em código aberto sob licença MIT e pronto para deploy comercial em ambientes de produção sucroalcooleiros.
