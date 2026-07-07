<div align="center">

# 🔥 NeuralForge v2.5

### *Neural Networks. Forged by Agents.*

[![Tests](https://img.shields.io/badge/tests-102%20passing-brightgreen?style=flat-square)](./tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](./pyproject.toml)
[![AGNT](https://img.shields.io/badge/AGNT-16%20tools-purple?style=flat-square)](#-agnt-plugin)

**Build, train, optimize, and deploy neural networks — all from natural language.**
**Analyze workflows, predict failures, and evolve AGNT itself.**

[Quick Start](#-quick-start) · [Architecture](#-supported-architectures) · [AGNT Plugin](#-agnt-plugin) · [Tools](#-all-16-agnt-tools) · [Evidence](#-live-evidence)

</div>

## ✨ Why NeuralForge?

You describe what you want. NeuralForge builds it.

> *"Build me a ResNet for CIFAR-10 with under 5M parameters that hits 92% accuracy."*

That's it. No boilerplate. No guesswork. No 47-step PyTorch tutorial.

But NeuralForge is more than a model builder. It's AGNT's **data intelligence layer**:

- **Predict workflow failures** before they happen
- **Learn from execution data** to optimize performance
- **Detect anomalies** in any time-series data
- **Evolve AGNT itself** through continuous learning
- **Make smart decisions** — retry, optimize, fix, predict

**For humans:** It's the fastest path from idea to trained model.
**For AI agents:** It's a first-class tool that turns neural engineering into a callable capability.

## 🚀 Quick Start

```bash
pip install neuralforge
```

```python
import neuralforge as nf

# One line from idea to model
model = nf.quick_build("ResNet for CIFAR-10 with <5M params")
print(f"Parameters: {model.count_parameters():,}")
# → Parameters: 677,642

# Full pipeline with training
spec = nf.NeuralForgeSpec.from_description(
    "Transformer for sentiment analysis with >90% accuracy"
)
model = nf.create_model(spec)
engine = nf.TrainingEngine(model, spec)
result = engine.train(train_loader, val_loader)

# Evaluate — get metrics, calibration, failure analysis, recommendations
evaluator = nf.ModelEvaluator(model)
report = evaluator.evaluate(test_loader)
print(f"Accuracy: {report.metrics['accuracy']:.2%}")
```

## 🏗️ Supported Architectures

| Family | Status | Best For |
|--------|--------|----------|
| **CNN** | ✅ | Fast image classification, edge devices |
| **ResNet** | ✅ | Deep image networks, transfer learning |
| **Transformer** | ✅ | NLP, sequence modeling, text generation |
| **Vision Transformer (ViT)** | ✅ | Image classification, multimodal |
| **MLP-Mixer** | ✅ | Alternative to attention-based vision |
| **KAN** | ✅ | Scientific ML, interpretable models |
| **Mamba / SSM** | ✅ | Long-sequence modeling |
| **MoE** | ✅ | Sparse expert routing |
| **Diffusion** | ✅ | Generative modeling |
| **Custom** | ✅ | Bring your own architecture |

## 📦 AGNT Plugin — 16 Tools

| # | Tool | Description |
|---|------|-------------|
| 1 | **neuralforge_auto** | Auto-Evolution — Continuous monitoring & self-improvement |
| 2 | **neuralforge_smart** | Smart Engine — Universal decision maker (6 modes) |
| 3 | **neuralforge_evolve** | Evolution Engine — 5-phase self-improvement cycle |
| 4 | **neuralforge_analyze** | Workflow Analyzer — Predict failures, detect anomalies |
| 5 | **neuralforge_learn** | Data Learner — 4 problem types |
| 6 | **neuralforge_pattern** | Pattern Engine — 5 pattern types |
| 7 | **neuralforge_test** | Self-Test Suite |
| 8 | **neuralforge_create** | Create models from natural language |
| 9 | **neuralforge_train** | Train with AMP, EMA, LR scheduling |
| 10 | **neuralforge_evaluate** | Comprehensive evaluation |
| 11 | **neuralforge_evaluate_enhanced** | Eval + Quality Predictor (acc/lat/mem) |
| 12 | **neuralforge_optimize** | Architecture search |
| 13 | **neuralforge_export** | Export to ONNX/TorchScript/Safetensors |
| 14 | **neuralforge_profile** | Profile params, latency, memory |
| 15 | **neuralforge_prune** | Model pruning |
| 16 | **neuralforge_quantize** | Model quantization |

## 📊 Live Evidence

Tested on real AGNT data (2,963 workflow executions, 10 workflows):

| Component | Test | Result | Time |
|-----------|------|--------|------|
| **DataLearner** | Regression R² | **0.972** | 1.22s |
| **DataLearner** | Classification | **98.7%** | 0.54s |
| **Pattern Engine** | Trend detection | **r=0.89** | 0.31s |
| **Pattern Engine** | Seasonal detection | **r=0.79** | 0.13s |
| **Quality Predictor** | Accuracy r | **0.999** | 4.7s |
| **Evolution Engine** | Health score | **55.6%** (real workflows) | — |
| **Evolution Engine** | Recommendations | **3 generated** | — |
| **Smart Engine** | Retry decision | **retry (conf=0.75)** | — |
| **Auto-Evolution** | Anomaly detection | **12 alerts** across 10 batches | — |

**Full test suite: 102 unit tests + 11 integration tests, all passing.**

## 🧠 How NeuralForge Evolves AGNT

1. **Observe** — Ingests every workflow execution (duration, success/failure, step count)
2. **Learn** — Pattern engine detects trends, seasonal cycles, degradation
3. **Predict** — Forecasts which workflows will fail before they run
4. **Act** — Auto-applies fixes (retry with backoff, timeout increase, monitoring)
5. **Verify** — Measures improvement after each action
6. **Accumulate** — Knowledge base grows with every execution

## Tesseract Pathway Network v0.4

TPN v0.4 adds durable local training weights and a `TesseractMindCore` runtime wrapper.

```text
train_tpn_checkpoint()
load_tpn_checkpoint()
TesseractMindCore.from_checkpoint()
artifacts/tpn/tpn_mind_core_v0_4.pt
artifacts/tpn/tpn_mind_core_v0_4.json
```

This is the first weight-bearing local mind-core seed. It performs no external calls.
See [`docs/TESSERACT_PATHWAY_NETWORK_v0_4.md`](./docs/TESSERACT_PATHWAY_NETWORK_v0_4.md).

## Tesseract Pathway Network v0.3

TPN v0.3 upgrades the registered tesseract architecture into a sparse pathway network with vertex-supervised routing, axis-gate loss, and JSON inference receipts.

```text
TesseractSparseDispatcher
build_tesseract_receipts()
vertex_logits / selected_vertex
topk_indices / topk_weights
expert_usage / sparse_ratio
```

See [`docs/TESSERACT_PATHWAY_NETWORK_v0_3.md`](./docs/TESSERACT_PATHWAY_NETWORK_v0_3.md).

## Tesseract Pathway Network v0.2

TPN v0.2 registers the tesseract router as a NeuralForge architecture family and adds synthetic route-governance training/evaluation.

```text
ArchitectureFamily.TESSERACT
SyntheticTesseractRouteDataset
train_tpn_synthetic()
evaluate_tpn_model()
```

See [`docs/TESSERACT_PATHWAY_NETWORK_v0_2.md`](./docs/TESSERACT_PATHWAY_NETWORK_v0_2.md).

## 📄 License

Apache 2.0

<div align="center">
<strong>NeuralForge</strong> — <em>Neural Networks. Forged by Agents.</em><br>
<a href="https://github.com/jacksonjp0311-gif/-NeuralForge">github.com/jacksonjp0311-gif/-NeuralForge</a>
</div>

## ðŸ”· Tesseract Pathway Network

NeuralForge now includes an experimental **Tesseract Pathway Network** scaffold for sparse geometric routing across four operational axes:

```text
Intent -> Evidence -> Authority -> Context
```

The module lives in `neuralforge/tesseract/` and tests whether routeable geometry can reduce attention pressure while preserving evidence, authority, coherence, and drift signals.

See [`docs/TESSERACT_PATHWAY_NETWORK.md`](./docs/TESSERACT_PATHWAY_NETWORK.md).

Claim boundary: TPN v0.1 is a research scaffold, not a production safety proof, AGI claim, or autonomous authority layer.
