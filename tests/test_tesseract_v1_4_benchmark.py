from pathlib import Path

from neuralforge.tesseract.benchmark import (
    BENCHMARK_VERSION,
    TesseractBenchmarkHarness,
    default_benchmark_cases,
)
from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractJarvisRuntime


def _runtime(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    return TesseractJarvisRuntime(JarvisServiceConfig(
        repo_root=str(repo),
        memory_path=str(tmp_path / "memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        contract_path=str(tmp_path / "contract.json"),
    ))


def test_v1_4_default_benchmark_cases():
    cases = default_benchmark_cases()
    assert len(cases) >= 5
    assert any("repo.status" in case.expected_skills for case in cases)
    assert any("file.read" in case.expected_skills for case in cases)


def test_v1_4_benchmark_report_scores(tmp_path):
    runtime = _runtime(tmp_path)
    harness = TesseractBenchmarkHarness(runtime)
    report = harness.run()
    assert report["benchmark_version"] == BENCHMARK_VERSION
    assert report["case_count"] >= 5
    assert report["safety_case_count"] >= 2
    assert 0.0 <= report["mean_score"] <= 1.0
    assert report["safety_score"] == 1.0


def test_v1_4_benchmark_writes_report(tmp_path):
    runtime = _runtime(tmp_path)
    harness = TesseractBenchmarkHarness(runtime)
    report = harness.run()
    paths = harness.write_report(report, tmp_path / "benchmarks")
    assert Path(paths["latest"]).exists()
    assert Path(paths["ledger"]).exists()
