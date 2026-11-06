# 📦 Entrega Sprint 5 — Monitoramento Contínuo de Drift e Política de Retreinamento (MLOps)

**Projeto:** Sanidade Vegetal (SugarVision)  
**Fase SEMMA:** `Assess & Sustain` (Governança de MLOps em Produção)  
**Status:** ✅ **100% CONCLUÍDO E TESTADO (`pytest`)**

---

## ✅ Checklist de Requisitos Atendidos

- [x] **Mapeamento de Fontes Potenciais de Drift no Campo:** Estruturado em [`../../src/mlops_drift_monitoring.py`](../../src/mlops_drift_monitoring.py) e exportado em [`./mlops_drift_monitoring_config.json`](./mlops_drift_monitoring_config.json), contemplando:
  - Mudança de iluminação sazonal (`DRIFT-SRC-01`);
  - Introdução de novas câmeras, lentes e sensores ópticos (`DRIFT-SRC-02`);
  - Estádios fenológicos da cana-de-açúcar e novas cultivares (`DRIFT-SRC-03`).
- [x] **Métricas de Monitoramento das Features Visuais de Entrada (KS e PSI):** Implementadas com tipagem estrita e docstrings em [`calculate_psi`](../../src/mlops_drift_monitoring.py) e [`calculate_ks_drift`](../../src/mlops_drift_monitoring.py), monitorando as 17 features visuais contínuas (cromáticas HSV/ExG/ExR, textura Haralick GLCM, rugosidade de pústula e nitidez óptica).
- [x] **Lógicas de Gatilhos e Regras de Alerta (Thresholds):** Programadas em [`DriftThresholds`](../../src/mlops_drift_monitoring.py) e [`evaluate_retraining_triggers`](../../src/mlops_drift_monitoring.py), classificando lotes em `NORMAL`, `WARNING_MONITORING`, `CRITICAL_RETRAIN_REQUIRED` e `EMERGENCY_ROLLBACK_AND_RETRAIN`.
- [x] **Política Consolidada de Sustentação e Retreinamento:** Documentada em [`../../docs/politica_monitoramento_drift_sustentacao_mlops.md`](../../docs/politica_monitoramento_drift_sustentacao_mlops.md) e integrada ao capítulo final de governança [`../../docs/secao_governanca_mlops_relato_final.md`](../../docs/secao_governanca_mlops_relato_final.md).
- [x] **Testes Unitários Automatizados (`pytest`):** Validados em [`../../src/test_mlops_drift_monitoring.py`](../../src/test_mlops_drift_monitoring.py) (13/13 testes aprovados).

---

## 📂 Artefatos Gerados nesta Entrega

1. [`../../src/mlops_drift_monitoring.py`](../../src/mlops_drift_monitoring.py): Motor estatístico de cálculo de PSI, teste de Kolmogorov-Smirnov, avaliação multivariada de features e acionamento automático de retreinamento.
2. [`../../src/test_mlops_drift_monitoring.py`](../../src/test_mlops_drift_monitoring.py): Suíte de testes unitários em `pytest`.
3. [`./mlops_drift_monitoring_config.json`](./mlops_drift_monitoring_config.json): Catálogo JSON de fontes de drift em campo e thresholds operacionais.
4. [`./mlops_drift_monitoring_report.json`](./mlops_drift_monitoring_report.json): Relatório de telemetria dos 4 cenários avaliados sobre a ABT oficial.
5. [`../../docs/politica_monitoramento_drift_sustentacao_mlops.md`](../../docs/politica_monitoramento_drift_sustentacao_mlops.md): Documento consolidado da política de sustentação e retreinamento MLOps.
