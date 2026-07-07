import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractJarvisRuntime, make_jarvis_handler
from neuralforge.tesseract.memory_core import MEMORY_CORE_VERSION, TesseractEpisodicMemory


def test_v1_5_episodic_memory_append_search_consolidate(tmp_path):
    memory = TesseractEpisodicMemory(tmp_path / "episodes.jsonl")
    memory.append("cycle", "Cycle completed for repo status", {"ok": True}, tags=["cycle", "repo.status"])
    memory.append("benchmark", "Benchmark mean score 1.0", {"mean_score": 1.0}, tags=["benchmark"])
    recent = memory.recent()
    assert recent["memory_version"] == MEMORY_CORE_VERSION
    assert recent["count"] == 2
    hits = memory.search("benchmark")
    assert hits["count"] == 1
    summary = memory.consolidate(tmp_path / "summary.json")
    assert summary["episode_count"] == 2
    assert "cycle" in summary["kinds"]


def test_v1_5_jarvis_cycle_records_episode(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    runtime = TesseractJarvisRuntime(JarvisServiceConfig(
        repo_root=str(repo),
        memory_path=str(tmp_path / "command_memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        episodic_memory_path=str(tmp_path / "episodes.jsonl"),
        contract_path=str(tmp_path / "contract.json"),
    ))
    answer = runtime.cycle("check repo status", execute=True)
    assert answer["ok"] is True
    recent = runtime.episodic_recent()
    assert recent["count"] >= 1
    assert "Cycle for" in recent["episodes"][-1]["summary"]


def test_v1_5_http_memory_endpoints(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    runtime = TesseractJarvisRuntime(JarvisServiceConfig(
        repo_root=str(repo),
        memory_path=str(tmp_path / "command_memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        episodic_memory_path=str(tmp_path / "episodes.jsonl"),
        contract_path=str(tmp_path / "contract.json"),
    ))
    runtime.cycle("check repo status", execute=True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_jarvis_handler(runtime))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/memory/episodes?limit=5", timeout=5) as res:
            recent = json.loads(res.read().decode("utf-8"))
        assert recent["ok"] is True
        assert recent["count"] >= 1

        body = json.dumps({"query": "repo", "limit": 5}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/memory/episodic/search",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            hits = json.loads(res.read().decode("utf-8"))
        assert hits["ok"] is True

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/memory/consolidate",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            summary = json.loads(res.read().decode("utf-8"))
        assert summary["ok"] is True
    finally:
        server.shutdown()
        thread.join(timeout=5)
