
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from neuralforge.tesseract.integration import TesseractIntegrationBus
from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractJarvisRuntime, make_jarvis_handler


def test_v1_1_integration_bus_repo_status(tmp_path):
    bus = TesseractIntegrationBus(repo_root=Path(__file__).resolve().parents[1])
    packet = bus.execute("repo.status")
    assert packet["skill_id"] == "repo.status"
    assert packet["allowed"] is True
    assert packet["result"]["cmd"] == ["git", "status", "--short"]


def test_v1_1_file_read_stays_inside_repo(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    bus = TesseractIntegrationBus(repo_root=repo)
    packet = bus.execute("file.read", {"path": "README.md", "max_bytes": 1000})
    assert packet["allowed"] is True
    assert packet["result"]["path"] == "README.md"

    escaped = bus.execute("file.read", {"path": "../outside.txt"})
    assert escaped["allowed"] is False


def test_v1_1_jarvis_task_records_ledger(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    runtime = TesseractJarvisRuntime(JarvisServiceConfig(
        repo_root=str(repo),
        memory_path=str(tmp_path / "memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        contract_path=str(tmp_path / "contract.json"),
    ))
    answer = runtime.task("system.ping")
    assert answer["ok"] is True
    assert answer["task"]["allowed"] is True
    assert Path(runtime.config.ledger_path).exists()


def test_v1_1_http_task_endpoint(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    runtime = TesseractJarvisRuntime(JarvisServiceConfig(
        repo_root=str(repo),
        memory_path=str(tmp_path / "memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        contract_path=str(tmp_path / "contract.json"),
    ))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_jarvis_handler(runtime))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/integration/skills", timeout=5) as res:
            skills = json.loads(res.read().decode("utf-8"))
        assert skills["ok"] is True
        assert "integration_skills" in skills

        body = json.dumps({"skill_id": "repo.log", "params": {"limit": 2}}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/task",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            answer = json.loads(res.read().decode("utf-8"))
        assert answer["version"].startswith("tpn.")
        assert answer["task"]["skill_id"] == "repo.log"
    finally:
        server.shutdown()
        thread.join(timeout=5)
