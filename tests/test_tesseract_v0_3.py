
import torch

from neuralforge.tesseract.data import SyntheticTesseractRouteDataset
from neuralforge.tesseract.loss import tesseract_compound_loss
from neuralforge.tesseract.network import TesseractPathwayNetwork, TesseractSparseDispatcher
from neuralforge.tesseract.receipt import build_tesseract_receipts


def test_v0_3_sparse_dispatch_outputs_route_metadata():
    model = TesseractPathwayNetwork(input_dim=16, d_model=24, top_k=4)
    out = model(torch.rand(5, 16))
    assert out["topk_indices"].shape == (5, 4)
    assert out["topk_weights"].shape == (5, 4)
    assert out["vertex_logits"].shape == (5, 16)
    assert out["selected_vertex"].shape == (5,)
    assert 0.0 < float(out["sparse_ratio"]) <= 1.0


def test_v0_3_receipts_are_json_ready():
    model = TesseractPathwayNetwork(input_dim=16, d_model=24, top_k=3)
    out = model(torch.rand(2, 16))
    receipts = build_tesseract_receipts(out)
    assert len(receipts) == 2
    assert receipts[0]["receipt_version"] == "tpn.v0.3"
    assert len(receipts[0]["selected_vertex"]) == 4
    assert len(receipts[0]["topk_vertices"]) == 3


def test_v0_3_loss_accepts_vertex_and_axis_targets():
    ds = SyntheticTesseractRouteDataset(n=8, seed=9)
    x, target = ds[0]
    model = TesseractPathwayNetwork(input_dim=16, d_model=24, top_k=4)
    out = model(x.unsqueeze(0))
    batched = {k: v.unsqueeze(0) if v.dim() == 0 else v.unsqueeze(0) for k, v in target.items()}
    loss = tesseract_compound_loss(out, batched)
    assert torch.isfinite(loss)


def test_sparse_dispatcher_route_distribution_sums():
    dispatcher = TesseractSparseDispatcher(d_model=16, top_k=4)
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    route = dispatcher.route(scores)
    assert torch.allclose(route["vertex_probs"].sum(dim=-1), torch.ones(1), atol=1e-5)
    assert torch.allclose(route["topk_weights"].sum(dim=-1), torch.ones(1), atol=1e-5)
