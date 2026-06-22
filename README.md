<div align="center">

# 🔥 NeuralForge v2.4

### *Neural Networks. Forged by Agents.*

[![Tests](https://img.shields.io/badge/tests-102%20passing-brightgreen?style=flat-square)](./tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](./pyproject.toml)
[![AGNT](https://img.shields.io/badge/AGNT-15%20tools-purple?style=flat-square)](#-agnt-plugin)

**Build, train, optimize, and deploy neural networks — all from natural language.**
**Analyze workflows, predict failures, and evolve AGNT itself.**

*One line. One model. Zero guesswork.*

[Quick Start](#-quick-start) · [Architecture](#-supported-architectures) · [AGNT Plugin](#-agnt-plugin) · [Tools](#-all-15-agnt-tools) · [Evidence](#-live-evidence)

---

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

---

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

### From the CLI

```bash
neuralforge create "ResNet for CIFAR-10 with <5M params" --name my-model
neuralforge info
neuralforge list-models
```

---

## 🏗️ Supported Architectures

| Family | Status | Best For |
|--------|--------|----------|
| **CNN** | ✅ | Fast image classification, edge devices |
| **ResNet** | ✅ | Deep image networks, transfer learning |
| **Transformer** | ✅ | NLP, sequence modeling, text generation |
| **Vision Transformer (ViT)** | ✅ | Image classification, multimodal |
| **MLP-Mixer** | ✅ | Alternative to attention-based vision |
| **KAN** | ✅ | Scientific ML, interpretable models |
| **Mamba / SSM** | 🔜 | Long-sequence modeling |
| **MoE** | 🔜 | Sparse expert routing |
| **Diffusion / Flow Matching** | 🔜 | Generative modeling |
| **Custom** | ✅ | Bring your own architecture |

---

## 🧠 Core Capabilities

### 1. Natural Language → Neural Network
```python
spec = NeuralForgeSpec.from_description(
    "Build a vision transformer for medical image classification "
    "under 8GB VRAM with >85% accuracy"
)
# Automatically parses: architecture=ViT, task=image_classification,
#   constraint.max_memory_mb=8192, constraint.min_accuracy=0.85
```

### 2. Auto-Architecture Search
```python
from neuralforge.auto.architect import ArchitectAgent
agent = ArchitectAgent()
proposals = agent.propose(
    task_description="Image classification for satellite imagery",
    data_profile=DataProfile(
        task_type=TaskType.IMAGE_CLASSIFICATION,
        input_shape=(3, 224, 224), num_classes=17
    ),
    constraints=Constraints(max_parameters=10_000_000),
    num_proposals=5,
)
```

### 3. Production Training Engine
- Mixed Precision (FP16/BF16/Mixed) with automatic loss scaling
- Exponential Moving Average (EMA) for stable inference
- Distributed Training (DDP, FSDP, DeepSpeed)
- LR Scheduling (Cosine, OneCycle, ReduceOnPlateau, Polynomial)
- Early Stopping with configurable patience
- Full determinism, experiment tracking, config versioning

### 4. Hyper-Optimization Suite
```python
from neuralforge.optimize import MetaOptimizer, prune_model, quantize_model, distill_model

meta = MetaOptimizer()
critique = meta.critique(spec, training_result)
next_spec = meta.propose_next_spec(spec, training_result)

pruned = prune_model(model, PruningConfig(amount=0.3))
quantized = quantize_model(model, QuantizationConfig(method=QuantizationMethod.DYNAMIC_INT8))
student = distill_model(teacher, student, DistillationConfig(temperature=4.0))
```

### 5. Comprehensive Evaluation
- Overall accuracy, loss, macro-F1
- Per-class precision, recall, F1, support
- Confusion matrix
- Expected Calibration Error (ECE)
- Failure analysis (top confused pairs, high-confidence failures)
- Actionable recommendations

### 6. Export Pipeline
```python
from neuralforge.utils import export_model
from neuralforge.spec import ExportConfig, ExportFormat

export_model(model, ExportConfig(format=ExportFormat.ONNX, output_path="./export"))
export_model(model, ExportConfig(format=ExportFormat.TORCHSCRIPT, output_path="./export"))
export_model(model, ExportConfig(format=ExportFormat.SAFETENSORS, output_path="./export"))
```

---

## 📦 AGNT Plugin

NeuralForge is available as a **first-class AGNT plugin** with 15 tools.

### Install
```bash
# From AGNT marketplace
agnt plugins install neuralforge

# From local file
agnt plugins install-file ./neuralforge-agnt-plugin.zip
```

### All 15 AGNT Tools

| # | Tool | Description |
|---|------|-------------|
| 1 | **neuralforge_smart** | Smart Engine — Universal decision maker (retry/optimize/predict/pattern/fix/analyze) |
| 2 | **neuralforge_evolve** | Evolution Engine — 5-phase self-improvement cycle |
| 3 | **neuralforge_analyze** | Workflow Analyzer — Predict failures, detect anomalies |
| 4 | **neuralforge_learn** | Data Learner — 4 problem types (regression/classification/forecasting/anomaly) |
| 5 | **neuralforge_pattern** | Pattern Engine — 5 pattern types (trend/seasonal/stationary/chaotic/step) |
| 6 | **neuralforge_test** | Self-Test Suite — Run all tests, get pass/fail + metrics |
| 7 | **neuralforge_create** | Create models from natural language |
| 8 | **neuralforge_train** | Train with AMP, EMA, LR scheduling, early stopping |
| 9 | **neuralforge_evaluate** | Comprehensive evaluation + recommendations |
| 10 | **neuralforge_evaluate_enhanced** | Eval + Multi-Objective Quality Predictor (acc/lat/mem) |
| 11 | **neuralforge_optimize** | Architecture search via evolutionary methods |
| 12 | **neuralforge_export** | Export to ONNX/TorchScript/Safetensors |
| 13 | **neuralforge_profile** | Profile params, latency, throughput, GPU memory |
| 14 | **neuralforge_prune** | Model pruning (L1 unstructured/structured) |
| 15 | **neuralforge_quantize** | Model quantization (dynamic INT8) |

---

## 📊 Live Evidence

All tools tested on real AGNT data (2,947 workflow executions, 10 workflows):

| Component | Test | Result | Time |
|-----------|------|--------|------|
| **DataLearner** | Regression R² | **0.945** | 1.65s |
| **DataLearner** | Classification | **98.7%** | 0.54s |
| **Pattern Engine** | Trend detection | **r=0.77** | 0.26s |
| **Pattern Engine** | Seasonal detection | **r=0.79** | 0.13s |
| **Quality Predictor** | Accuracy r | **0.999** | 4.7s |
| **Quality Predictor** | Memory r | **0.993** | 4.7s |
| **Evolution Engine** | Health score | **76.7%** | — |
| **Evolution Engine** | Recommendations | **3 generated** | — |
| **Smart Engine** | Retry decision | **retry (conf=0.75)** | — |
| **Smart Engine** | Pattern detection | **seasonal (r=0.46)** | — |
| **Workflow Analyzer** | Anomaly detection | **2 anomalies (z>2.0)** | — |

**Full test suite: 102 unit tests + 11 integration tests, all passing.**

---

## 🤖 Agent Integration

```python
from neuralforge import as_tool

# One-line registration
tool = as_tool("neuralforge")

# Invoke from any agent
result = tool.invoke({
    "action": "full_pipeline",
    "description": "Build a ResNet for CIFAR-10 with <5M params reaching >92% accuracy"
})
```

### LangChain
```python
from neuralforge import get_all_langchain_tools
tools = get_all_langchain_tools()
```

### CrewAI
```python
from neuralforge import get_crewai_tools
```

### AutoGen
```python
from neuralforge import get_autogen_functions
```

---

## 📁 Project Structure

```
neuralforge/
├── neuralforge/
│   ├── __init__.py              # Main exports + quick_build()
│   ├── spec.py                  # NeuralForgeSpec + all Pydantic models
│   ├── cli.py                   # Typer CLI
│   ├── core/                    # Engine, registry, model builder
│   │   ├── forge.py
│   │   └── registry.py
│   ├── training/                # Training engine, callbacks, distributed
│   │   ├── engine.py
│   │   ├── callbacks.py
│   │   └── distributed.py
│   ├── evaluation/              # Evaluator with quality prediction
│   │   ├── evaluator.py
│   │   └── quality_predictor.py
│   ├── auto/                    # Architecture search, NAS, scaling
│   │   ├── architect.py
│   │   ├── nas.py
│   │   └── scaling.py
│   ├── optimize/                # Meta-optimizer, pruning, quantization, distillation
│   │   ├── meta_optimizer.py
│   │   ├── pruning.py
│   │   ├── quantization.py
│   │   └── distillation.py
│   ├── pattern_engine.py        # Pattern detection & prediction
│   ├── learner.py               # Data learner (4 problem types)
│   ├── analyzer.py              # Workflow analyzer
│   ├── smart_engine.py          # Smart decision engine
│   ├── evo_engine.py            # Evolution engine
│   ├── tools/                   # Agent tool wrappers
│   │   ├── agent_tool.py
│   │   ├── langchain_tools.py
│   │   └── multi_agent.py
│   ├── memory/                  # Insights store
│   └── utils/                   # Export, profiling, visualization
├── examples/                    # 3 complete examples
├── tests/                       # 102 tests — all passing ✅
├── agnt-plugin/                 # AGNT marketplace plugin
│   ├── manifest.json
│   └── tools/                   # 15 JS tool files
├── pyproject.toml
├── README.md
├── CHANGELOG.md
└── AGENT_GUIDE.md
```

---

## 🧪 Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
# 102 passed ✅
```

---

## 🗺️ Roadmap

- [x] **Data Learner** — Universal data intelligence (4 problem types)
- [x] **Pattern Engine** — Time-series prediction (5 pattern types)
- [x] **Quality Predictor** — Multi-objective (accuracy/latency/memory)
- [x] **Workflow Analyzer** — Failure prediction + anomaly detection
- [x] **Smart Engine** — Universal decision maker (6 modes)
- [x] **Evolution Engine** — 5-phase self-improvement cycle
- [ ] **JAX/Flax backend** — TPU-native training
- [ ] **Mamba/SSM** — State-space model support
- [ ] **Diffusion models** — DDPM, score-based generative models
- [ ] **Optuna integration** — Full Bayesian hyperparameter search
- [ ] **WebGPU export** — Browser-native inference
- [ ] **Gradio/Streamlit UI** — No-code interface

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/amazing-feature`
3. Run tests: `pytest tests/ -v`
4. Submit a PR

---

## 📄 License

Apache 2.0 — use it, fork it, ship it.

---

<div align="center">

**NeuralForge** — *Neural Networks. Forged by Agents.*

Built with 🔥 by [jacksonjp0311-gif](https://github.com/jacksonjp0311-gif)

</div>
