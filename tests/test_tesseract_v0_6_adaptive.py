
from pathlib import Path

from neuralforge.tesseract.adaptive import (
    TesseractReplayLedger,
    append_operator_feedback,
    seed_replay_from_synthetic,
    train_tpn_from_replay,
)
from neuralforge.tesseract.checkpoint import TesseractCheckpointConfig, train_tpn_checkpoint
from neuralforge.tesseract.communication import TesseractEnglishAdapter


def test_v0_6_seed_replay_and_train_adaptive_checkpoint(tmp_path):
    base = train_tpn_checkpoint(
        output_dir=tmp_path,
        name="tpn_base",
        config=TesseractCheckpointConfig(samples=64, epochs=1, batch_size=16, d_model=24, top_k=4, seed=81),
        device="cpu",
    )
    replay = tmp_path / "replay.jsonl"
    seed = seed_replay_from_synthetic(replay, n=32, seed=82)
    assert seed["records_written"] == 32
    assert len(TesseractReplayLedger(replay).records()) == 32

    out = tmp_path / "tpn_adaptive.pt"
    result = train_tpn_from_replay(
        checkpoint_path=base["checkpoint_path"],
        replay_path=replay,
        output_checkpoint=out,
        epochs=1,
        batch_size=16,
        device="cpu",
    )
    assert out.exists()
    assert Path(result["manifest_path"]).exists()
    assert result["approved_records"] == 32

    adapter = TesseractEnglishAdapter.from_checkpoint(out)
    answer = adapter.think_english([0.5] * 16)
    assert answer["text"]


def test_v0_6_operator_feedback_append(tmp_path):
    replay = tmp_path / "operator.jsonl"
    append_operator_feedback(
        replay,
        vector=[0.5] * 16,
        route=1,
        authority=1,
        evidence=1,
        coherence=0.5,
        delta_phi=0.1,
        vertex=15,
        note="unit test feedback",
    )
    records = TesseractReplayLedger(replay).records()
    assert len(records) == 1
    assert records[0].approved is True
    assert records[0].targets["vertex"] == 15
