# Changelog

## v2.0.0 — 2026-06-22

### Added
- **NeuralForgeSpec** — Complete Pydantic v2 spec with YAML serialization, natural language parsing, and config hashing
- **TrainingEngine** — Production training loop with AMP (FP16/BF16/mixed), EMA, gradient accumulation, LR scheduling, early stopping, reproducibility seeds
- **ModelEvaluator** — Comprehensive evaluation with per-class metrics, confusion matrix, ECE calibration, failure analysis, recommendations
- **ArchitectAgent** — LLM-driven architecture proposal with 20+ templates across 8 architecture families
- **EvolutionarySearch** — Neuro-evolution search with configurable population, mutation, crossover
- **ScalingLawEstimator** — Fit and predict scaling laws from observations
- **MetaOptimizer** — Self-critique loop that analyzes training results and proposes improvements
- **InsightsStore** — Memory store for experiment insights with search and retrieval
- **Pruning** — L1 unstructured, structured pruning with sparsity tracking
- **Quantization** — Dynamic INT8 quantization with GPTQ/AWQ/BnB stubs
- **Distillation** — Knowledge distillation with soft targets, FitNet, attention transfer
- **Export** — TorchScript, ONNX, Safetensors export pipeline
- **Callbacks** — ModelCheckpoint, WandbCallback, TensorBoardCallback, ConsoleLogger, LearningRateFinder
- **Distributed** — DDP, FSDP, DeepSpeed setup utilities
- **Agent Tools** — LangChain, CrewAI, AutoGen compatible tool wrappers
- **Multi-Agent** — ForgeOrchestrator with blackboard system for collaborative architecture design
- **CLI** — Typer-based CLI with `create`, `info`, `list_models` commands
- **Examples** — CIFAR-10 classification, text classification, RL policy network
- **Tests** — 102 tests covering specs, models, training, evaluation, optimization, auto-architecture

### Architecture Support
- CNN, ResNet, Transformer, Vision Transformer, MLP-Mixer, KAN, Mamba/SSM, MoE, RWKV

### Quality
- Fully typed with Pydantic v2
- 102 passing tests
- Clean error messages for agent consumption
- Reproducible with seed control
