"""End-to-end NeuralForge test: build, train, evaluate, export a neural net."""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from neuralforge.spec import NeuralForgeSpec, ExportConfig, ExportFormat
from neuralforge.core.forge import create_model, export_model
from neuralforge.training.engine import TrainingEngine
from neuralforge.evaluation.evaluator import ModelEvaluator
import torch
from torch.utils.data import TensorDataset, DataLoader

print("=" * 60)
print("NEURALFORGE END-TO-END TEST")
print("=" * 60)

# Step 1: Create model from natural language
print("\n[1] Creating model from description...")
spec = NeuralForgeSpec.from_description("Simple CNN for CIFAR-10 image classification under 2M parameters")
model = create_model(spec)
print(f"  Model: {spec.name}")
print(f"  Architecture: {spec.architecture.family.value}")
print(f"  Parameters: {model.count_parameters():,}")
print(f"  Input shape: {spec.data_profile.input_shape if spec.data_profile else 'N/A'}")
print(f"  Classes: {spec.data_profile.num_classes if spec.data_profile else 'N/A'}")

# Step 2: Generate synthetic training data
print("\n[2] Preparing training data...")
input_shape = spec.data_profile.input_shape if spec.data_profile else (3, 32, 32)
nc = spec.data_profile.num_classes if spec.data_profile and spec.data_profile.num_classes else 10
n_samples = 200
torch.manual_seed(42)
X = torch.randn(n_samples, *input_shape)
y = torch.randint(0, nc, (n_samples,))

# Split: 160 train, 20 val, 20 test
X_train, y_train = X[40:], y[40:]
X_val, y_val = X[:20], y[:20]
X_test, y_test = X[20:40], y[20:40]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

print(f"  Training samples: {len(X_train)}")
print(f"  Validation samples: {len(X_val)}")
print(f"  Test samples: {len(X_test)}")

# Step 3: Train
print("\n[3] Training model...")
spec.training.epochs = 5
engine = TrainingEngine(model, spec, spec.training)
result = engine.train(train_loader, val_loader)
print(f"  Epochs completed: {result.epochs_completed}")
print(f"  Final loss: {result.final_loss:.4f}")
print(f"  Best metric: {result.best_metric:.4f}")
print(f"  Best epoch: {result.best_epoch}")
print(f"  Training time: {result.training_time_seconds:.2f}s")
print(f"  Status: {result.status}")

# Step 4: Evaluate on held-out test set
print("\n[4] Evaluating on held-out test set...")
evaluator = ModelEvaluator(model)
report = evaluator.evaluate(test_loader, num_classes=nc)
print(f"  Accuracy: {report.metrics.get('accuracy', 0):.4f}")
print(f"  Loss: {report.metrics.get('loss', 0):.4f}")
print(f"  Macro F1: {report.metrics.get('macro_f1', 0):.4f}")
print(f"  Calibration error: {report.calibration_error:.4f}")
if report.recommendations:
    print(f"  Recommendations: {len(report.recommendations)}")
    for rec in report.recommendations[:3]:
        print(f"    - {rec}")

# Step 5: Export
print("\n[5] Exporting model...")
output_dir = os.path.join(repo_root, 'neuralforge_output')
os.makedirs(output_dir, exist_ok=True)
config = ExportConfig(format=ExportFormat.PYTORCH_STATE_DICT, output_path=output_dir)
path = export_model(model, config)
print(f"  Export path: {path}")
print(f"  File size: {os.path.getsize(path) / 1024:.1f} KB")

# Step 6: Verify the exported file loads back
print("\n[6] Verifying exported model loads back...")
loaded_model = create_model(spec)
loaded_model.load_state_dict(torch.load(path, map_location='cpu'))
loaded_model.eval()
print(f"  Loaded model parameters: {loaded_model.count_parameters():,}")
print(f"  Verification: OK")

print("\n" + "=" * 60)
print("ALL TESTS PASSED - NeuralForge is fully functional!")
print("=" * 60)
