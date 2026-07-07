
import torch

from neuralforge.tesseract.checkpoint import (
    TesseractCheckpointConfig,
    load_tpn_checkpoint,
    train_tpn_checkpoint,
)
from neuralforge.tesseract.mind import TesseractMindCore


def test_v0_4_trains_saves_and_loads_local_weights(tmp_path):
    result = train_tpn_checkpoint(
        output_dir=tmp_path,
        name="tpn_test_weights",
        config=TesseractCheckpointConfig(samples=64, epochs=1, batch_size=16, d_model=24, top_k=4, seed=31),
        device="cpu",
    )
    ckpt = tmp_path / "tpn_test_weights.pt"
    manifest = tmp_path / "tpn_test_weights.json"
    assert ckpt.exists()
    assert manifest.exists()

    model, payload = load_tpn_checkpoint(ckpt)
    assert payload["format"] == "neuralforge.tesseract.checkpoint.v0.4"
    out = model(torch.rand(2, 16))
    assert out["route_logits"].shape == (2, 5)
    assert out["selected_vertex"].shape == (2,)


def test_v0_4_mind_core_emits_receipts(tmp_path):
    result = train_tpn_checkpoint(
        output_dir=tmp_path,
        name="tpn_test_mind",
        config=TesseractCheckpointConfig(samples=64, epochs=1, batch_size=16, d_model=24, top_k=4, seed=41),
        device="cpu",
    )
    core = TesseractMindCore.from_checkpoint(result["checkpoint_path"])
    answer = core.think(torch.rand(2, 16))
    assert len(answer["receipts"]) == 2
    assert answer["receipts"][0]["receipt_version"] == "tpn.v0.3"
    assert len(answer["selected_vertices"][0]) == 4
