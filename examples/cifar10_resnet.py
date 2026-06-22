import torch
from torch.utils.data import DataLoader, TensorDataset
import neuralforge as nf
from neuralforge.spec import NeuralForgeSpec, TrainingConfig

def main():
    print("NeuralForge v2.0 - CIFAR-10 ResNet Example")
    spec = nf.NeuralForgeSpec.from_description("state-of-the-art image classifier for 10 classes with < 5M params")
    model = nf.create_model(spec)
    print(f"Parameters: {model.count_parameters():,}")
    X = torch.randn(2000, 3, 32, 32); y = torch.randint(0, 10, (2000,))
    result = nf.train(model, DataLoader(TensorDataset(X, y), batch_size=64), TrainingConfig(epochs=5, batch_size=64))
    print(f"Best Loss: {result.best_metric:.4f}")
    print("Done!")

if __name__ == "__main__": main()
