
import torch

from neuralforge.tesseract.communication import (
    TesseractEnglishAdapter,
    outputs_to_english,
    receipt_to_english,
)
from neuralforge.tesseract.checkpoint import TesseractCheckpointConfig, train_tpn_checkpoint
from neuralforge.tesseract.network import TesseractPathwayNetwork


def test_v0_5_receipt_to_english_contains_route_and_vertex():
    receipt = {
        "route": "shadow",
        "selected_vertex": "1101",
        "coherence": 0.72,
        "delta_phi": 0.18,
        "missing_axes": ["authority"],
        "axis_scores": {"intent": 0.9, "evidence": 0.8, "authority": 0.2, "context": 0.7},
        "topk_vertices": ["1101", "1111"],
        "topk_weights": [0.6, 0.4],
    }
    msg = receipt_to_english(receipt)
    text = msg.as_text()
    assert "1101" in text
    assert "shadow" in text
    assert "authority" in text.lower()


def test_v0_5_outputs_to_english_from_model():
    model = TesseractPathwayNetwork(input_dim=16, d_model=24, top_k=4)
    out = model(torch.rand(1, 16))
    messages = outputs_to_english(out)
    assert len(messages) == 1
    assert messages[0].summary.startswith("TPN selected vertex")


def test_v0_5_weighted_english_adapter(tmp_path):
    artifact = train_tpn_checkpoint(
        output_dir=tmp_path,
        name="tpn_english_test",
        config=TesseractCheckpointConfig(samples=64, epochs=1, batch_size=16, d_model=24, top_k=4, seed=51),
        device="cpu",
    )
    adapter = TesseractEnglishAdapter.from_checkpoint(artifact["checkpoint_path"])
    answer = adapter.think_english([0.5] * 16)
    assert answer["text"]
    assert answer["receipts"]
    assert "Local deterministic English" in answer["claim_boundary"]
