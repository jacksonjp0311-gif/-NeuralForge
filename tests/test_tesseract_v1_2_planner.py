import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from neuralforge.tesseract.integration import TesseractIntegrationBus
from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractJarvisRuntime, make_jarvis_handler
from neuralforge.tesseract.planner import TesseractTaskPlanner


def test_v1_2_planner_builds_status_log_readme_plan():
    repo = Path(__file__).resolve().parents[1]
    planner = TesseractTaskPlanner(TesseractIntegrationBus(repo_root=repo))
    plan = planner.make_plan("check repo status, recent commit log, and read README.md")
    skills = [step["skill_id"] for step in plan["steps"]]
    assert "repo.status" in skills
    assert "repo.log" in skills
    assert "file.read" in skills
    assert plan["plan_version"] == "tpn.plan.v1.2"


def test_v1_2_planner_executes_plan(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    planner = TesseractTaskPlanner(TesseractIntegrationBus(repo_root=repo, ledger_path=tmp_path / "ledger.jsonl"))
    answer = planner.plan_and_optionally_execute("check repo status and recent git log", execute=True)
    assert answer["executed"] is True
    assert answer["execution"]["ok"] is True
    assert answer["execution"]["step_count"] >= 2


def test_v1_2_jarvis_plan_records_ledger(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    runtime = TesseractJarvisRuntime(JarvisServiceConfig(
        repo_root=str(repo),
        memory_path=str(tmp_path / "memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        contract_path=str(tmp_path / "contract.json"),
    ))
    answer = runtime.plan("check repo status", execute=True)
    assert answer["ok"] is True
    assert answer["plan"]["plan_version"] == "tpn.plan.v1.2"
    assert Path(runtime.config.ledger_path).exists()


def test_v1_2_http_plan_endpoint(tmp_path):
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
        body = json.dumps({"command": "check repo status", "execute": True}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/plan",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            answer = json.loads(res.read().decode("utf-8"))
        assert answer["version"].startswith("tpn.")
        assert answer["executed"] is True
        assert answer["execution"]["ok"] is True
    finally:
        server.shutdown()
        thread.join(timeout=5)
