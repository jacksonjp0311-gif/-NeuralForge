
import warnings
from pathlib import Path

from neuralforge.tesseract.checkpoint import TesseractCheckpointConfig, load_tpn_checkpoint, train_tpn_checkpoint


def test_v1_0_1_checkpoint_load_does_not_emit_weights_only_future_warning(tmp_path):
    artifact = train_tpn_checkpoint(
        output_dir=tmp_path,
        name="tpn_hardened_load",
        config=TesseractCheckpointConfig(samples=32, epochs=1, batch_size=16, d_model=24, top_k=4, seed=131),
        device="cpu",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, payload = load_tpn_checkpoint(artifact["checkpoint_path"], map_location="cpu")
    assert model is not None
    assert payload["config"]["d_model"] == 24
    messages = [str(w.message) for w in caught]
    assert not any("weights_only=False" in msg for msg in messages)


def test_v1_0_1_runtime_scripts_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "start_tesseract_jarvis.ps1").exists()
    assert (root / "scripts" / "status_tesseract_jarvis.ps1").exists()
    assert (root / "scripts" / "stop_tesseract_jarvis.ps1").exists()
