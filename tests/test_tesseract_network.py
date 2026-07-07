import torch

from neuralforge.tesseract import TesseractPathwayNetwork, tesseract_compound_loss
from neuralforge.tesseract.benchmark import run_toy_benchmark


def test_tpn_forward_shapes():
    model = TesseractPathwayNetwork(input_dim=16, d_model=32, num_routes=5, top_k=5)
    x = torch.randn(4, 16)
    out = model(x)

    assert out["axis_scores"].shape == (4, 4)
    assert out["vertex_probs"].shape == (4, 16)
    assert out["route_logits"].shape == (4, 5)
    assert out["authority_logits"].shape == (4, 3)
    assert out["evidence_logits"].shape == (4, 3)
    assert out["coherence"].shape == (4,)
    assert out["delta_phi"].shape == (4,)


def test_tpn_compound_loss_backward():
    model = TesseractPathwayNetwork(input_dim=16, d_model=32, num_routes=5, top_k=5)
    x = torch.randn(3, 16)
    out = model(x)
    loss = tesseract_compound_loss(
        out,
        {
            "route": torch.tensor([0, 1, 2]),
            "authority": torch.tensor([0, 1, 2]),
            "evidence": torch.tensor([0, 1, 2]),
            "coherence": torch.tensor([0.2, 0.7, 0.9]),
            "delta_phi": torch.tensor([0.8, 0.2, 0.1]),
        },
    )
    loss.backward()
    assert loss.item() > 0


def test_toy_benchmark_report():
    report = run_toy_benchmark(items=256, axis_dim=8, top_k=8, seed=7)
    assert report["dense"]["recall"] == 1.0
    assert report["tesseract_sparse"]["candidate_count"] < report["dense"]["candidate_count"]
    assert 0.0 <= report["tesseract_sparse"]["recall"] <= 1.0
