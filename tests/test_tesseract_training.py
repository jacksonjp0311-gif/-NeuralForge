import torch

from neuralforge.spec import (
    ArchitectureFamily,
    ArchitectureSpec,
    DataProfile,
    NeuralForgeSpec,
    TaskType,
)
from neuralforge.core.forge import create_model
from neuralforge.tesseract.data import SyntheticTesseractRouteDataset, make_tesseract_loaders
from neuralforge.tesseract.evaluate import evaluate_tpn_model
from neuralforge.tesseract.network import TesseractPathwayNetwork
from neuralforge.tesseract.train import train_tpn_synthetic


def test_synthetic_tesseract_dataset_shapes():
    ds = SyntheticTesseractRouteDataset(n=16, seed=1)
    x, target = ds[0]
    assert x.shape == (16,)
    assert set(target.keys()) == {"route", "authority", "evidence", "coherence", "delta_phi", "vertex", "axis_scores"}


def test_tpn_evaluation_metrics():
    _, val_loader = make_tesseract_loaders(n=64, seed=2, batch_size=16)
    model = TesseractPathwayNetwork(input_dim=16, d_model=24)
    metrics = evaluate_tpn_model(model, val_loader, device="cpu")
    assert 0.0 <= metrics["route_accuracy"] <= 1.0
    assert metrics["samples"] > 0


def test_tpn_synthetic_training_smoke():
    report = train_tpn_synthetic(n=96, epochs=1, batch_size=24, d_model=24, seed=3)
    assert report["status"] == "completed"
    assert "route_accuracy" in report["final"]


def test_neuralforge_architecture_family_tesseract_create_model():
    spec = NeuralForgeSpec(
        name="tpn-registered",
        task_type=TaskType.CUSTOM,
        data_profile=DataProfile(
            task_type=TaskType.CUSTOM,
            input_shape=(16,),
            num_classes=5,
            data_format="tesseract",
        ),
        architecture=ArchitectureSpec(
            family=ArchitectureFamily.TESSERACT,
            width=24,
            embedding_dim=24,
            expert_capacity=5,
        ),
    )
    model = create_model(spec)
    out = model(torch.randn(3, 16))
    assert out.shape == (3, 5)
