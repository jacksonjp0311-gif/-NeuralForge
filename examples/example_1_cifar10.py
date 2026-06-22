"""Example 1: CIFAR-10 Image Classification with ResNet.

This example demonstrates:
- Creating a model from a natural language description
- Training with AMP, EMA, and learning rate scheduling
- Evaluation with comprehensive metrics
- Export to multiple formats

Usage:
    python examples/example_1_cifar10.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralforge.spec import NeuralForgeSpec, TrainingConfig, ExportConfig, ExportFormat
from neuralforge.core.forge import create_model, train, evaluate_and_report, export_model
from neuralforge.training.engine import TrainingEngine
from neuralforge.evaluation.evaluator import ModelEvaluator
from neuralforge.utils.profiling import profile_model


def main():
    print("=" * 64)
    print("NeuralForge v2.0 — Example 1: CIFAR-10 Classification")
    print("=" * 64)

    # ── Step 1: Create spec from description ──────────────────────
    spec = NeuralForgeSpec.from_description(
        "Build a ResNet for CIFAR-10 image classification with <5M params"
    )
    print(f"\n[Spec] {spec.name}")
    print(f"  Architecture: {spec.architecture.family.value}")
    print(f"  Config hash: {spec.config_hash()}")

    # ── Step 2: Create model ──────────────────────────────────────
    model = create_model(spec)
    params = model.count_parameters()
    print(f"\n[Model] Parameters: {params:,}")
    assert params < 5_000_000, f"Model too large: {params:,}"

    # ── Step 3: Profile model ─────────────────────────────────────
    profile = profile_model(model, input_shape=(1, 3, 32, 32))
    print(f"\n[Profile]")
    for k, v in profile.items():
        print(f"  {k}: {v}")

    # ── Step 4: Create synthetic dataset ──────────────────────────
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(42)
    X_train = torch.randn(1000, 3, 32, 32)
    y_train = torch.randint(0, 10, (1000,))
    X_val = torch.randn(200, 3, 32, 32)
    y_val = torch.randint(0, 10, (200,))

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # ── Step 5: Train ─────────────────────────────────────────────
    config = TrainingConfig(
        epochs=5,
        batch_size=64,
        learning_rate=1e-3,
        precision="mixed",
        seed=42,
        experiment_name="cifar10-resnet",
    )

    engine = TrainingEngine(model, spec, config)
    result = engine.train(train_loader, val_loader)

    print(f"\n[Training Result]")
    print(f"  Epochs completed: {result.epochs_completed}")
    print(f"  Best metric: {result.best_metric:.4f} @ epoch {result.best_epoch}")
    print(f"  Training time: {result.training_time_seconds:.1f}s")
    print(f"  Status: {result.status}")

    # ── Step 6: Evaluate ──────────────────────────────────────────
    evaluator = ModelEvaluator(engine.model)
    report = evaluator.evaluate(val_loader)

    print(f"\n[Evaluation Report]")
    for k, v in report.metrics.items():
        print(f"  {k}: {v}")
    if report.recommendations:
        print("  Recommendations:")
        for r in report.recommendations:
            print(f"    • {r}")

    # ── Step 7: Export ────────────────────────────────────────────
    os.makedirs("./example_output", exist_ok=True)
    export_cfg = ExportConfig(
        format=ExportFormat.PYTORCH_STATE_DICT,
        output_path="./example_output/cifar10_model",
    )
    export_path = export_model(engine.model, export_cfg)
    print(f"\n[Export] Model saved to: {export_path}")

    print("\n" + "=" * 64)
    print("Example 1 complete!")
    print("=" * 64)


if __name__ == "__main__":
    main()
