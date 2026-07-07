
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from neuralforge.tesseract.checkpoint import TesseractCheckpointConfig, train_tpn_checkpoint
from neuralforge.tesseract.command import (
    CommandVectorizer,
    TesseractCommandMind,
    make_command_handler,
)


def test_v0_8_command_vectorizer_shape():
    vec = CommandVectorizer().vectorize("plan the next safe local step")
    assert len(vec.vector) == 16
    assert 0.0 <= vec.coherence <= 1.0
    assert len(vec.route_prior) == 5


def test_v0_8_command_mind_routes_and_executes_plan(tmp_path):
    artifact = train_tpn_checkpoint(
        output_dir=tmp_path,
        name="tpn_command_base",
        config=TesseractCheckpointConfig(samples=64, epochs=1, batch_size=16, d_model=24, top_k=4, seed=101),
        device="cpu",
    )
    mind = TesseractCommandMind(artifact["checkpoint_path"], memory_path=tmp_path / "memory.jsonl")
    answer = mind.handle("plan the next local evolution step", execute=True)
    assert answer["ok"] is True
    assert answer["packet"]["skill_id"] == "tpn.plan"
    assert answer["packet"]["allowed"] is True
    assert answer["text"]


def test_v0_8_mutating_skill_blocks_without_authority(tmp_path):
    artifact = train_tpn_checkpoint(
        output_dir=tmp_path,
        name="tpn_command_block",
        config=TesseractCheckpointConfig(samples=64, epochs=1, batch_size=16, d_model=24, top_k=4, seed=102),
        device="cpu",
    )
    mind = TesseractCommandMind(artifact["checkpoint_path"], memory_path=tmp_path / "memory.jsonl")
    answer = mind.handle("remember this important note", execute=True, allow_mutation=False)
    assert answer["packet"]["skill_id"] == "tpn.memory_note"
    assert answer["packet"]["allowed"] is False


def test_v0_8_http_command_endpoint(tmp_path):
    artifact = train_tpn_checkpoint(
        output_dir=tmp_path,
        name="tpn_command_http",
        config=TesseractCheckpointConfig(samples=64, epochs=1, batch_size=16, d_model=24, top_k=4, seed=103),
        device="cpu",
    )
    mind = TesseractCommandMind(artifact["checkpoint_path"], memory_path=tmp_path / "memory.jsonl")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_command_handler(mind))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps({"command": "status", "execute": True}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/command",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            answer = json.loads(res.read().decode("utf-8"))
        assert answer["ok"] is True
        assert answer["packet"]["skill_id"] == "tpn.status"
    finally:
        server.shutdown()
        thread.join(timeout=5)
