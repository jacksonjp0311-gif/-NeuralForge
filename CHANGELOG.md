# Changelog

## v2.4.0 — 2026-06-22

### New Tools
- **neuralforge_smart** — Smart Engine: Universal decision maker with 6 modes (retry, optimize, predict, pattern, fix, analyze). The "always-called" tool that agents use for any data-driven decision.
- **neuralforge_analyze** — Workflow Analyzer: Predicts failures, detects anomalies, recommends optimizations using Pattern Engine + Data Learner.
- **neuralforge_evolve** — Evolution Engine: 5-phase self-improvement cycle (observe → learn → predict → optimize → evolve). Uses ALL NeuralForge tools internally.

### Fixes
- Pattern detector v2.3: Trend now correctly detected (was over-classifying as seasonal in v2.2). Uses R² + slope significance.
- Anomaly detection: Now uses Autoencoder (AnomalyNet) instead of ClassificationNet.
- Forecasting detection: Fixed to detect continuous sequential data (was misclassifying sine waves).
- Analyzer KeyError when n < 5 (anomalies key missing).
- Evolution engine _predict handling small workflow groups.

### AGNT Plugin
- 15 tools total (up from 8 in v2.0)
- Plugin size: 9.2 KB .agnt package
- All tools have full descriptions, input/output schemas
- Tested on real AGNT data: 2,947 executions across 10 workflows

### Test Results (all pass)
- DataLearner Regression R²=0.945, Classification 98.7%
- Pattern Engine: Trend r=0.77, Seasonal r=0.79
- Quality Predictor: acc r=0.999, lat r=0.974, mem r=0.993
- Evolution Engine: 76.7% health score, 3 recommendations
- Smart Engine: 6/6 decision modes pass

## v2.3.0 — 2026-06-22
- Added DataLearner v2.3 (4 problem types: regression, classification, forecasting, anomaly)
- Added Pattern Engine v2.3 (5 pattern types with fixed detector)
- Added Multi-Objective Quality Predictor (acc/latency/memory)
- Added Self-Test Suite tool

## v2.2.0 — 2026-06-22
- Added Pattern Engine v2.2 (5 neural architectures)

## v2.1.0 — 2026-06-22
- Clean rebuild, clean git history

## v2.0.0 — 2026-06-22
- Initial release: 8 AGNT tools, 102 tests, full Python library
