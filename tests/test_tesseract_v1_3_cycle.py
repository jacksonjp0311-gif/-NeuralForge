import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from neuralforge.tesseract.cycle import TesseractCycleEngine
from neuralforge.tesseract.integration import TesseractIntegrationBus
from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractJarvisRuntime, make_jarvis_handler
from neuralforge.tesseract.planner import TesseractTaskPlanner


def test_v1_3_cycle_plan_only():
    repo = Path(__file__).resolve().parents[1]
    engine = TesseractCycleEngine(TesseractTaskPlanner(TesseractIntegrationBus(repo_root=repo)))
    report = engine.run_cycle("check repo status and recent git log", execute=False)
    assert report["cycle_version"] == "tpn.cycle.v1.3"
    assert report["executed"] is False
    assert report["plan"]["steps"]


def test_v1_3_cycle_execute_observes_results():
    repo = Path(__file__).resolve().parents[1]
    engine = TesseractCycleEngine(TesseractTaskPlanner(TesseractIntegrationBus(repo_root=repo)))
    report = engine.run_cycle("check repo status and recent git log", execute=True)
    assert report["executed"] is True
    assert report["observations"]
    assert "next_recommendation" in report


def test_v1_3_jarvis_cycle_records_ledger(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    runtime = TesseractJarvisRuntime(JarvisServiceConfig(
        repo_root=str(repo),
        memory_path=str(tmp_path / "memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        contract_path=str(tmp_path / "contract.json"),
    ))
    answer = runtime.cycle("check repo status and recent git log", execute=True)
    assert answer["ok"] is True
    assert answer["cycle"]["cycle_version"] == "tpn.cycle.v1.3"
    assert Path(runtime.config.ledger_path).exists()


def test_v1_3_http_cycle_endpoint(tmp_path):
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
        body = json.dumps({"objective": "check repo status", "execute": True}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/cycle",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            answer = json.loads(res.read().decode("utf-8"))
        assert answer["version"].startswith("tpn.")
        assert answer["cycle"]["executed"] is True
        assert answer["cycle"]["observations"]
    finally:
        server.shutdown()
        thread.join(timeout=5)
