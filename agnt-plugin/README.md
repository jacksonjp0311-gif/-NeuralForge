# 🔥 NeuralForge — AGNT Marketplace Plugin

**Build, train, optimize, and deploy neural networks from natural language — directly inside AGNT.**

## Install

```bash
# From AGNT marketplace
agnt plugins install neuralforge

# From local file
agnt plugins install-file neuralforge-agnt-plugin.zip
```

## Available Tools

| Tool | What It Does |
|------|-------------|
| `neuralforge_create` | Create a model from natural language |
| `neuralforge_train` | Train with AMP, EMA, LR scheduling |
| `neuralforge_optimize` | Hyperparameter + architecture search |
| `neuralforge_evaluate` | Full evaluation + recommendations |
| `neuralforge_export` | Export to ONNX/TorchScript/Safetensors |
| `neuralforge_profile` | Profile params, latency, memory |
| `neuralforge_prune` | Prune for efficiency |
| `neuralforge_quantize` | Quantize for deployment |

## How to Prompt It

```
"Use NeuralForge to build a ResNet for CIFAR-10 with under 5M parameters"
"Use NeuralForge to profile the model's latency and memory"
"Use NeuralForge to prune the model by 30%"
"Use NeuralForge to export the model to ONNX"
```

## Requirements

- Python 3.10+
- PyTorch 2.1+
- NeuralForge library (auto-installed with plugin)

## Marketplace Info

- **ID:** neuralforge
- **Category:** ml-tools
- **Version:** 2.0.0
- **License:** Apache-2.0
- **Author:** jacksonjp0311-gif
- **Repo:** https://github.com/jacksonjp0311-gif/-NeuralForge
