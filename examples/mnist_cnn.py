import torch
from torch.utils.data import DataLoader, TensorDataset
import neuralforge as nf
from neuralforge.spec import NeuralForgeSpec, TrainingConfig, ExportConfig

def main():
    print("NeuralForge v2.0 - MNIST CNN Example")
    spec = nf.NeuralForgeSpec.from_description("CNN image classifier for MNIST with < 100K params")
    model = nf.create_model(spec)
    print(f"Parameters: {model.count_parameters():,}")
    X = torch.randn(1000, 1, 28, 28); y = torch.randint(0, 10, (1000,))
    result = nf.train(model, DataLoader(TensorDataset(X, y), batch_size=64), TrainingConfig(epochs=3, batch_size=64))
    print(f"Final Loss: {result.final_loss:.4f}, Time: {result.training_time_seconds:.1f}s")
    X_test = torch.randn(200, 1, 28, 28); y_test = torch.randint(0, 10, (200,))
    report = nf.evaluate_and_report(model, DataLoader(TensorDataset(X_test, y_test), batch_size=32))
    print(f"Accuracy: {report.metrics['accuracy']:.2%}")
    print("Done!")

if __name__ == "__main__": main()
