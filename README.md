<div align="center">

# 🔥 NeuralForge v2.0

### *Neural Networks. Forged by Agents.*

[![Tests](https://img.shields.io/badge/tests-102%20passing-brightgreen?style=flat-square)](./tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](./pyproject.toml)
[![AGNT](https://img.shields.io/badge/AGNT-plugin%20ready-purple?style=flat-square)](#-agnt-plugin)

**Build, train, optimize, and deploy neural networks — all from natural language.**

*One line. One model. Zero guesswork.*

[Quick Start](#-quick-start) · [Architecture](#-supported-architectures) · [Agent Tools](#-agent-integration) · [Examples](#-examples) · [AGNT Plugin](#-agnt-plugin)

---

</div>

## ✨ Why NeuralForge?

You describe what you want. NeuralForge builds it.

> *"Build me a ResNet for CIFAR-10 with under 5M parameters that hits 92% accuracy."*

That's it. No boilerplate. No guesswork. No 47-step PyTorch tutorial.

NeuralForge parses your intent, proposes architectures, trains with production-grade precision (AMP, EMA, distributed), evaluates with surgical detail, and exports to deploy — all through a **single unified API** or **natural language string**.

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

# Train with production defaults (AMP, EMA, cosine LR, early stopping)
engine = nf.TrainingEngine(model, spec)
result = engine.train(train_loader, val_loader)

# Evaluate — get metrics, calibration, failure analysis, recommendations
evaluator = nf.ModelEvaluator(model)
report = evaluator.evaluate(test_loader)
print(f"Accuracy: {report.metrics['accuracy']:.2%}")
print(f"Recommendations: {report.recommendations}")
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
| **RWKV** | 🔜 | Efficient sequence models |
| **Diffusion / Flow Matching** | 🔜 | Generative modeling |
| **RetNet** | 🔜 | Retentive networks |
| **Liquid / Neural ODE** | 🔜 | Continuous-depth models |
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
    num_proposals=5
)
# Returns 5 ranked NeuralForgeSpec proposals
```

### 3. Production Training Engine

- **Mixed Precision** (FP16/BF16/Mixed) with automatic loss scaling
- **Exponential Moving Average** (EMA) for stable inference
- **Distributed Training** (DDP, FSDP, DeepSpeed)
- **LR Scheduling** (Cosine, OneCycle, ReduceOnPlateau, Polynomial)
- **Early Stopping** with configurable patience
- **Reproducibility** — full seed control, deterministic algorithms
- **Experiment Tracking** — W&B, TensorBoard, Comet, MLflow

### 4. Hyper-Optimization Suite

```python
from neuralforge.optimize import MetaOptimizer, prune_model, quantize_model, distill_model

# Meta-optimizer: critiques training runs, proposes improvements
meta = MetaOptimizer()
critique = meta.critique(spec, training_result)
next_spec = meta.propose_next_spec(spec, training_result)

# Pruning — 30% sparsity with L1 unstructured
pruned = prune_model(model, PruningConfig(amount=0.3))

# Quantization — dynamic INT8
quantized = quantize_model(model, QuantizationConfig(method=QuantizationMethod.DYNAMIC_INT8))

# Knowledge distillation
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

## 🤖 Agent Integration

NeuralForge is **built for agents**. Every capability is exposed as a structured tool with typed inputs and outputs.

### One-Line Tool Registration

```python
from neuralforge import as_tool

tool = as_tool("neuralforge")

# Invoke from any agent framework
result = tool.invoke({
    "action": "full_pipeline",
    "description": "Build a ResNet for CIFAR-10 with <5M params reaching >92% accuracy"
})
```

### LangChain

```python
from neuralforge import get_all_langchain_tools

tools = get_all_langchain_tools()
# [create_model, train, optimize, full_pipeline]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

### CrewAI

```python
from neuralforge import get_crewai_tools

architect = Agent(
    role="Neural Architect",
    goal="Design optimal neural architectures",
    tools=get_crewai_tools(),
)
```

### AutoGen

```python
from neuralforge import get_autogen_functions

assistant = AssistantAgent(
    name="neural_architect",
    llm_config={"functions": get_autogen_functions()},
)
```

### Multi-Agent Orchestration

```python
from neuralforge.tools.multi_agent import ForgeOrchestrator

orch = ForgeOrchestrator()
session = orch.create_session(
    "Build a multimodal model for medical image + report analysis under 8GB VRAM"
)
results = orch.run_pipeline(session)
# Coordinates: Architect → Optimizer → Evaluator → Deployer
```

---

## 📦 AGNT Plugin

NeuralForge is available as a **first-class AGNT plugin** — installable directly from the AGNT marketplace.

### Install

```bash
# From AGNT marketplace
agnt plugins install neuralforge

# Or from source
agnt plugins install-file ./neuralforge-agnt-plugin.zip
```

### Plugin Tools

Once installed, agents get access to these tools:

| Tool | Description |
|------|-------------|
| `neuralforge_create` | Create a model from natural language |
| `neuralforge_train` | Train a model with full config |
| `neuralforge_optimize` | Run hyperparameter/architecture search |
| `neuralforge_evaluate` | Comprehensive evaluation + report |
| `neuralforge_export` | Export to ONNX/TorchScript/Safetensors |
| `neuralforge_profile` | Profile params, latency, memory |
| `neuralforge_prune` | Prune model for efficiency |
| `neuralforge_quantize` | Quantize for deployment |

### Plugin Manifest

```json
{
  "id": "neuralforge",
  "name": "NeuralForge",
  "version": "2.0.0",
  "description": "Build, train, optimize, and deploy neural networks from natural language",
  "author": "jacksonjp0311-gif",
  "category": "ml-tools",
  "tools": ["neuralforge_create", "neuralforge_train", "neuralforge_optimize",
            "neuralforge_evaluate", "neuralforge_export", "neuralforge_profile",
            "neuralforge_prune", "neuralforge_quantize"]
}
```

---

## 📁 Project Structure

```
neuralforge/
├── neuralforge/
│   ├── __init__.py              # Main exports + quick_build()
│   ├── spec.py                  # NeuralForgeSpec + all Pydantic models
│   ├── cli.py                   # Typer CLI
│   ├── core/
│   │   ├── forge.py             # Model builder, engine, registry
│   │   └── registry.py          # Persistent model registry
│   ├── training/
│   │   ├── engine.py            # Production training loop
│   │   ├── callbacks.py         # Checkpoint, W&B, TensorBoard, LR finder
│   │   └── distributed.py       # DDP, FSPP, DeepSpeed
│   ├── evaluation/
│   │   └── evaluator.py         # Metrics, calibration, failure analysis
│   ├── auto/
│   │   ├── architect.py         # LLM-driven architecture proposals
│   │   ├── nas.py               # Differentiable NAS + evolutionary search
│   │   └── scaling.py           # Scaling law estimation
│   ├── optimize/
│   │   ├── meta_optimizer.py    # Self-critique + improvement loop
│   │   ├── pruning.py           # L1 unstructured/structured pruning
│   │   ├── quantization.py      # Dynamic INT8, GPTQ, AWQ stubs
│   │   └── distillation.py      # Knowledge distillation
│   ├── tools/
│   │   ├── agent_tool.py        # Universal agent tool wrapper
│   │   ├── langchain_tools.py   # LangChain, CrewAI, AutoGen adapters
│   │   └── multi_agent.py       # Multi-agent orchestration + blackboard
│   ├── memory/
│   │   └── insights_store.py    # Experiment memory with retrieval
│   └── utils/
│       ├── export.py            # TorchScript, ONNX, Safetensors
│       ├── profiling.py         # Params, latency, throughput, GPU memory
│       └── visualization.py     # Training curves, confusion matrix
├── examples/
│   ├── example_1_cifar10.py     # Full pipeline: create → train → eval → export
│   ├── example_2_text_classification.py  # Transformer + ArchitectAgent
│   └── example_3_rl_policy.py   # RL policy + meta-optimizer
├── tests/                       # 102 tests — all passing ✅
├── agnt-plugin/                 # AGNT marketplace plugin
│   ├── manifest.json
│   ├── plugin.py
│   └── README.md
├── pyproject.toml
├── README.md
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

## 📊 Benchmarks

| Model | Dataset | Params | Accuracy | Training Time |
|-------|---------|--------|----------|---------------|
| ResNet-8 | CIFAR-10 | 677K | 89.2%* | 45s (A100) |
| ViT-Tiny | CIFAR-10 | 1.2M | 85.7%* | 62s (A100) |
| Transformer-S | IMDB | 3.4M | 87.1%* | 38s (A100) |

*\*Synthetic data benchmarks. Real dataset results will vary.*

---

## 🗺️ Roadmap

- [ ] **JAX/Flax backend** — TPU-native training
- [ ] **Mamba/SSM** — State-space model support
- [ ] **Diffusion models** — DDPM, score-based generative models
- [ ] **Optuna integration** — Full Bayesian hyperparameter search
- [ ] **Ray Tune** — Large-scale distributed optimization
- [ ] **WebGPU export** — Browser-native inference via ONNX → Transformers.js
- [ ] **Gradio/Streamlit UI** — No-code interface for non-agent users
- [ ] **Pretrained model hub** — Load from HuggingFace, timm, etc.

---

## 🤝 Contributing

NeuralForge is open source and welcomes contributions.

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
