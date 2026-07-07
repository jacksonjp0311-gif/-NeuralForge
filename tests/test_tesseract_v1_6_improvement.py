from pathlib import Path

from neuralforge.tesseract.benchmark import TesseractBenchmarkHarness
from neuralforge.tesseract.improvement import IMPROVEMENT_VERSION, TesseractImprovementProposalEngine
from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractJarvisRuntime
from neuralforge.tesseract.memory_core import TesseractEpisodicMemory


def test_v1_6_improvement_proposals_from_empty_evidence(tmp_path):
    memory = TesseractEpisodicMemory(tmp_path / "episodes.jsonl")
    engine = TesseractImprovementProposalEngine(memory=memory)
    report = engine.propose(benchmark_report={}, memory_summary=memory.consolidate(tmp_path / "summary.json"))
    assert report["improvement_version"] == IMPROVEMENT_VERSION
    assert report["proposal_count"] >= 1
    assert all(p["allowed_to_mutate"] is False for p in report["proposals"])


def test_v1_6_improvement_proposals_from_runtime_benchmark(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    runtime = TesseractJarvisRuntime(JarvisServiceConfig(
        repo_root=str(repo),
        memory_path=str(tmp_path / "command_memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        episodic_memory_path=str(tmp_path / "episodes.jsonl"),
        contract_path=str(tmp_path / "contract.json"),
    ))
    runtime.cycle("check repo status", execute=True)
    memory = runtime.episodic_memory
    harness = TesseractBenchmarkHarness(runtime)
    engine = TesseractImprovementProposalEngine(memory=memory, benchmark_harness=harness)
    report = engine.propose()
    assert report["ok"] is True
    assert report["proposal_count"] >= 1


def test_v1_6_improvement_write_report(tmp_path):
    memory = TesseractEpisodicMemory(tmp_path / "episodes.jsonl")
    engine = TesseractImprovementProposalEngine(memory=memory)
    report = engine.propose(benchmark_report={}, memory_summary=memory.consolidate(tmp_path / "summary.json"))
    paths = engine.write_report(report, tmp_path / "improvement")
    assert Path(paths["latest"]).exists()
    assert Path(paths["ledger"]).exists()
