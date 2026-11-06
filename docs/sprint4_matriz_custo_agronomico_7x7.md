# 📊 Matriz de Custo e Perda Agronômica Assimétrica 7x7 (R$/ha)

**Projeto:** Sanidade-Vegetal (*SugarVision*) | **Sprint:** 4 | **Fase SEMMA:** `Assess 5.0 — Matriz de Custo Agronômico`  
**Arquivo CSV Pareado:** `sprint4_matriz_custo_agronomico_7x7.csv` | **Dimensões:** `7 linhas x 8 colunas`

> Formulação financeira canavieira em R$/hectare ponderando assimetricamente Falsos Negativos de Ferrugem (R$ 10.017,25/ha) e Podridão Vermelha (R$ 2.711,50 a R$ 3.200,00/ha) contra Falsos Positivos de pulverização fungicida (R$ 485,00/ha).

## Tabela de Dados Auditada

| Classe_Real_vs_Predita | GRASSY SHOOT | HEALTHY | LEAF SCALD | MOSAIC | RED ROT | RUST | YELLOW LEAF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GRASSY SHOOT | 0 | 2400 | 320 | 320 | 320 | 805 | 320 |
| HEALTHY | 390 | 0 | 390 | 390 | 390 | 485 | 390 |
| LEAF SCALD | 320 | 2400 | 0 | 320 | 320 | 805 | 320 |
| MOSAIC | 320 | 2400 | 320 | 0 | 320 | 805 | 320 |
| RED ROT | 2711.5000 | 3200 | 2711.5000 | 2711.5000 | 0 | 805 | 2711.5000 |
| RUST | 8514.6600 | 10017.2500 | 8514.6600 | 8514.6600 | 8514.6600 | 0 | 8514.6600 |
| YELLOW LEAF | 320 | 2400 | 320 | 320 | 320 | 805 | 0 |
