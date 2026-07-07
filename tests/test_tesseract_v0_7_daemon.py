
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from neuralforge.tesseract.checkpoint import TesseractCheckpointConfig, train_tpn_checkpoint
from neuralforge.tesseract.daemon import TesseractWarmRuntime, make_handler


def test_v0_7_warm_runtime_think(tmp_path):
    artifact = train_tpn_checkpoint(
        output_dir=tmp_path,
        name="tpn_daemon_base",
        config=TesseractCheckpointConfig(samples=64, epochs=1, batch_size=16, d_model=24, top_k=4, seed=91),
        device="cpu",
    )
    runtime = TesseractWarmRuntime(artifact["checkpoint_path"], replay_path=tmp_path / "replay.jsonl")
    result = runtime.think([0.5] * 16, style="compact")
    assert result["ok"] is True
    assert result["latency_ms"] >= 0
    assert result["text"]
    assert result["receipts"]
    assert runtime.health()["calls"] == 1


def test_v0_7_feedback_appends_replay(tmp_path):
    artifact = train_tpn_checkpoint(
        output_dir=tmp_path,
        name="tpn_feedback_base",
        config=TesseractCheckpointConfig(samples=64, epochs=1, batch_size=16, d_model=24, top_k=4, seed=92),
        device="cpu",
    )
    replay = tmp_path / "feedback.jsonl"
    runtime = TesseractWarmRuntime(artifact["checkpoint_path"], replay_path=replay)
    result = runtime.feedback({
        "vector": [0.5] * 16,
        "route": 1,
        "authority": 1,
        "evidence": 1,
        "coherence": 0.5,
        "delta_phi": 0.1,
        "vertex": 15,
        "note": "daemon unit test",
    })
    assert result["ok"] is True
    assert replay.exists()


def test_v0_7_http_health_and_think(tmp_path):
    artifact = train_tpn_checkpoint(
        output_dir=tmp_path,
        name="tpn_http_base",
        config=TesseractCheckpointConfig(samples=64, epochs=1, batch_size=16, d_model=24, top_k=4, seed=93),
        device="cpu",
    )
    runtime = TesseractWarmRuntime(artifact["checkpoint_path"], replay_path=tmp_path / "replay.jsonl")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as res:
            health = json.loads(res.read().decode("utf-8"))
        assert health["ok"] is True

        body = json.dumps({"vector": [0.5] * 16, "style": "compact"}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/think",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            answer = json.loads(res.read().decode("utf-8"))
        assert answer["ok"] is True
        assert answer["text"]
    finally:
        server.shutdown()
        thread.join(timeout=5)
