from pathlib import Path

from neuralforge.tesseract.benchmark import TesseractBenchmarkHarness, record_benchmark_episode, run_full_benchmark
from neuralforge.tesseract.improvement import TesseractImprovementProposalEngine
from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractJarvisRuntime
from neuralforge.tesseract.memory_core import TesseractEpisodicMemory


def _runtime(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    return TesseractJarvisRuntime(JarvisServiceConfig(
        repo_root=str(repo),
        memory_path=str(tmp_path / "command_memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        episodic_memory_path=str(tmp_path / "episodes.jsonl"),
        contract_path=str(tmp_path / "contract.json"),
    ))


def test_v1_6_1_records_benchmark_into_episodic_memory(tmp_path):
    runtime = _runtime(tmp_path)
    harness = TesseractBenchmarkHarness(runtime)
    report = harness.run()
    episode = record_benchmark_episode(report, tmp_path / "episodes.jsonl")

    assert episode["kind"] == "benchmark"
    assert "benchmark" in episode["tags"]

    memory = TesseractEpisodicMemory(tmp_path / "episodes.jsonl")
    hits = memory.search("benchmark")
    assert hits["count"] >= 1


def test_v1_6_1_improvement_uses_recorded_benchmark_memory(tmp_path):
    runtime = _runtime(tmp_path)
    harness = TesseractBenchmarkHarness(runtime)
    benchmark = harness.run()
    record_benchmark_episode(benchmark, tmp_path / "episodes.jsonl")

    memory = TesseractEpisodicMemory(tmp_path / "episodes.jsonl")
    summary = memory.consolidate(tmp_path / "summary.json")
    engine = TesseractImprovementProposalEngine(memory=memory)
    proposals = engine.propose(benchmark_report=benchmark, memory_summary=summary)

    assert proposals["benchmark_used"] is True
    assert proposals["memory_used"] is True
    assert proposals["proposal_count"] >= 1
    assert all(p["allowed_to_mutate"] is False for p in proposals["proposals"])

def test_v1_6_1_run_full_benchmark_record_memory(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    report = run_full_benchmark(write=False, record_memory=True)
    assert "memory_episode" in report
    assert report["memory_episode"]["kind"] == "benchmark"
