"""Live end-to-end test: Create model -> Train -> Evaluate -> Neural Quality Predict"""
import neuralforge as nf
from neuralforge.spec import *
from neuralforge.core.forge import create_model
from neuralforge.training.engine import TrainingEngine
from neuralforge.evaluation.evaluator import ModelEvaluator
from neuralforge.evaluation.quality_predictor import ModelQualityPredictor
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

print("=" * 60)
print("[AI] NeuralForge v2.1 — Live End-to-End Test")
print("=" * 60)

# Step 1: Create a model from natural language
print("\n[1] Creating model from natural language...")
spec = nf.NeuralForgeSpec.from_description("Build a ResNet for CIFAR-10 with <5M params")
model = nf.create_model(spec)
print(f"    Model: {spec.name}")
print(f"    Architecture: {spec.architecture.family.value}")
print(f"    Parameters: {model.count_parameters():,}")

# Step 2: Generate synthetic CIFAR-10-like data
print("\n[2] Generating synthetic data...")
torch.manual_seed(42)
X_train = torch.randn(500, 3, 32, 32)
y_train = torch.randint(0, 10, (500,))
X_test = torch.randn(100, 3, 32, 32)
y_test = torch.randint(0, 10, (100,))
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

# Step 3: Train with history tracking
print("\n[3] Training model (10 epochs)...")
config = TrainingConfig(epochs=10, batch_size=64, learning_rate=1e-3, precision="mixed", seed=42)
engine = TrainingEngine(model, spec, config)
result = engine.train(train_loader)
print(f"    Epochs: {result.epochs_completed}")
print(f"    Final loss: {result.final_loss:.4f}")
print(f"    Best metric: {result.best_metric:.4f}")
print(f"    Training time: {result.training_time_seconds:.1f}s")

# Step 4: Standard evaluation
print("\n[4] Standard evaluation...")
evaluator = ModelEvaluator(model)
report = evaluator.evaluate(test_loader, num_classes=10)
print(f"    Accuracy: {report.metrics['accuracy']:.2%}")
print(f"    Loss: {report.metrics['loss']:.4f}")
print(f"    Macro F1: {report.metrics['macro_f1']:.4f}")
print(f"    Calibration Error: {report.calibration_error:.4f}")

# Step 5: Train neural quality predictor
print("\n[5] Training neural quality predictor...")
predictor = ModelQualityPredictor()

# Generate synthetic training histories
histories = []
accuracies = []
arch_metas = []
for i in range(10):
    np.random.seed(i + 100)
    base_loss = np.random.uniform(1.5, 2.5)
    decay = np.random.uniform(0.1, 0.4)
    noise = np.random.normal(0, 0.03, 10)
    hist = {
        "train_loss": list(base_loss * np.exp(-decay * np.linspace(0, 3, 10)) + noise),
        "val_loss": list(base_loss * 1.1 * np.exp(-decay * 0.9 * np.linspace(0, 3, 10)) + np.random.normal(0, 0.04, 10)),
        "lr": list(np.linspace(0.001, 0.0001, 10)),
    }
    histories.append(hist)
    accuracies.append(float(np.random.uniform(0.5, 0.95)))
    arch_metas.append({
        "num_params": 200000 + i * 100000,
        "depth": 2 + i,
        "width": 32 * (i + 1),
        "num_classes": 10,
    })

predictor_result = predictor.train_on_history(
    histories, accuracies, arch_metas,
    task_types=["image_classification"] * 10,
    family_types=["resnet"] * 10,
    epochs=100,
)
print(f"    Predictor training loss: {predictor_result['loss']:.4f}")
print(f"    Predictor correlation: {predictor_result['correlation']:.4f}")

# Step 6: Predict quality from actual training dynamics
print("\n[6] Predicting model quality from training dynamics...")
quality = predictor.predict(
    history=result.history,
    num_params=model.count_parameters(),
    depth=spec.architecture.depth or 4,
    width=spec.architecture.width or 64,
    num_classes=10,
    task_type="image_classification",
    family_type="resnet",
)
print(f"    Quality Score: {quality['quality_score']:.4f}")
print(f"    Confidence: {quality['confidence']:.2f}")
print(f"    Epochs Observed: {quality['epochs_observed']}")

# Step 7: Enhanced evaluation with quality prediction
print("\n[7] Enhanced evaluation (with quality predictor)...")
enhanced_evaluator = ModelEvaluator(model, quality_predictor=predictor)
enhanced_report = enhanced_evaluator.evaluate(
    test_loader, num_classes=10,
    training_history=result.history,
    arch_metadata={
        "num_params": model.count_parameters(),
        "depth": spec.architecture.depth or 4,
        "width": spec.architecture.width or 64,
        "num_classes": 10,
        "task_type": "image_classification",
        "family_type": "resnet",
    }
)
print(f"    Accuracy: {enhanced_report.metrics['accuracy']:.2%}")
print(f"    Recommendations:")
for rec in enhanced_report.recommendations:
    print(f"      → {rec}")

print("\n" + "=" * 60)
print("[OK] All systems operational!")
print("=" * 60)
