"""Example 2: Text Classification with Transformer.

This example demonstrates:
- Transformer architecture for NLP tasks
- Text classification on synthetic data
- Using the ArchitectAgent for proposal generation
- LoRA configuration

Usage:
    python examples/example_2_text_classification.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import TensorDataset, DataLoader

from neuralforge.spec import (
    NeuralForgeSpec, TrainingConfig, TaskType, DataProfile,
    LoRAConfig, ExportFormat, ExportConfig,
)
from neuralforge.core.forge import create_model, train, export_model
from neuralforge.training.engine import TrainingEngine
from neuralforge.evaluation.evaluator import ModelEvaluator
from neuralforge.auto.architect import ArchitectAgent


def main():
    print("=" * 64)
    print("NeuralForge v2.0 — Example 2: Text Classification")
    print("=" * 64)

    # ── Step 1: Define task ───────────────────────────────────────
    data_profile = DataProfile(
        task_type=TaskType.TEXT_CLASSIFICATION,
        input_shape=(128,),
        vocab_size=10000,
        num_classes=4,
        max_sequence_length=128,
        data_format="text",
        num_samples=5000,
    )

    constraints = type("Constraints", (), {
        "max_parameters": 10_000_000,
        "max_memory_mb": 4096,
        "target_hardware": ["cuda"],
        "must_support_batch_size": 1,
        "deterministic": False,
        "max_flops": None,
        "max_latency_ms": None,
        "min_accuracy": None,
        "required_export_formats": [],
        "max_model_size_mb": None,
    })()

    # ── Step 2: Use ArchitectAgent for proposals ──────────────────
    architect = ArchitectAgent()
    proposals = architect.propose(
        "Text classification for sentiment analysis on short reviews",
        data_profile=data_profile,
        constraints=constraints,
        num_proposals=3,
    )

    print(f"\n[Architect] Proposed {len(proposals)} architectures:")
    for i, p in enumerate(proposals):
        print(f"  {i+1}. {p.name} ({p.architecture.family.value})")
        model = create_model(p)
        print(f"     Parameters: {model.count_parameters():,}")

    # ── Step 3: Train the best proposal ───────────────────────────
    spec = proposals[0]
    spec.training = TrainingConfig(
        epochs=5,
        batch_size=32,
        learning_rate=2e-4,
        scheduler="cosine",
        warmup_ratio=0.1,
        seed=42,
        precision="mixed",
    )
    spec.lora = LoRAConfig(
        enabled=False,  # Set True if using a pretrained model
        rank=16,
        alpha=32,
        target_modules=["q_proj", "v_proj"],
    )

    model = create_model(spec)
    print(f"\n[Model] Architecture: {spec.architecture.family.value}")
    print(f"  Parameters: {model.count_parameters():,}")

    # ── Step 4: Create synthetic text data ────────────────────────
    torch.manual_seed(42)
    seq_len = 128
    vs = data_profile.vocab_size

    X_train = torch.randint(0, vs, (500, seq_len))
    y_train = torch.randint(0, 4, (500,))
    X_val = torch.randint(0, vs, (100, seq_len))
    y_val = torch.randint(0, 4, (100,))

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=32, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

    # ── Step 5: Train ─────────────────────────────────────────────
    engine = TrainingEngine(model, spec, spec.training)
    result = engine.train(train_loader, val_loader)

    print(f"\n[Training Result]")
    print(f"  Epochs: {result.epochs_completed}")
    print(f"  Best metric: {result.best_metric:.4f}")
    print(f"  Time: {result.training_time_seconds:.1f}s")

    # ── Step 6: Evaluate ──────────────────────────────────────────
    evaluator = ModelEvaluator(engine.model)
    report = evaluator.evaluate(val_loader)
    print(f"\n[Evaluation]")
    for k, v in report.metrics.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 64)
    print("Example 2 complete!")
    print("=" * 64)


if __name__ == "__main__":
    main()
