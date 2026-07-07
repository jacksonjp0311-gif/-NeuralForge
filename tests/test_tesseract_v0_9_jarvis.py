
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from neuralforge.tesseract.checkpoint import TesseractCheckpointConfig, train_tpn_checkpoint
from neuralforge.tesseract.jarvis import (
    JarvisServiceConfig,
    TesseractJarvisRuntime,
    make_jarvis_handler,
)


def _runtime(tmp_path):
    artifact = train_tpn_checkpoint(
        output_dir=tmp_path,
        name="tpn_jarvis_base",
        config=TesseractCheckpointConfig(samples=64, epochs=1, batch_size=16, d_model=24, top_k=4, seed=111),
        device="cpu",
    )
    return TesseractJarvisRuntime(JarvisServiceConfig(
        checkpoint=artifact["checkpoint_path"],
        memory_path=str(tmp_path / "memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
    ))


def test_v0_9_jarvis_command_writes_ledger(tmp_path):
    runtime = _runtime(tmp_path)
    answer = runtime.command("plan the next local step", execute=True)
    assert answer["ok"] is True
    assert Path(runtime.config.ledger_path).exists()
    recent = runtime.ledger_recent(limit=1)
    assert recent["entries"]
    assert recent["entries"][0]["kind"] == "command"


def test_v0_9_skills_manifest(tmp_path):
    runtime = _runtime(tmp_path)
    skills = runtime.skills()["skills"]
    ids = {skill["skill_id"] for skill in skills}
    assert "tpn.plan" in ids
    assert "tpn.memory_note" in ids


def test_v0_9_memory_search(tmp_path):
    runtime = _runtime(tmp_path)
    memory_path = Path(runtime.config.memory_path)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(json.dumps({"command": "remember Tesseract online"}) + "\n", encoding="utf-8")
    result = runtime.memory_search("Tesseract")
    assert result["hits"]


def test_v0_9_http_endpoints(tmp_path):
    runtime = _runtime(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_jarvis_handler(runtime))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/skills", timeout=5) as res:
            skills = json.loads(res.read().decode("utf-8"))
        assert skills["ok"] is True

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

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ledger/recent?limit=1", timeout=5) as res:
            ledger = json.loads(res.read().decode("utf-8"))
        assert ledger["ok"] is True
    finally:
        server.shutdown()
        thread.join(timeout=5)
