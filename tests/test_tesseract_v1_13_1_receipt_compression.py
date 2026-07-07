from pathlib import Path

from neuralforge.tesseract.control_bundle import (
    CONTROL_BUNDLE_VERSION,
    TesseractCompressedControlBundle,
    format_control_bundle_summary,
    summarize_control_bundle_report,
)
from neuralforge.tesseract.goal_cycle import TesseractGoalAwareCycleRunner
from neuralforge.tesseract.goal_queue import TesseractGoalQueueRunner
from neuralforge.tesseract.goal_state import TesseractGoalStateManager
from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractJarvisRuntime
from neuralforge.tesseract.performance import TesseractPerformanceTelemetryGovernor
from neuralforge.tesseract.policy import TesseractExecutionPolicyGovernor
from neuralforge.tesseract.stairway import TesseractStairwayCompressionGovernor


ROOT = Path(__file__).resolve().parents[1]


def _runtime(tmp_path):
    return TesseractJarvisRuntime(JarvisServiceConfig(
        repo_root=str(ROOT),
        memory_path=str(tmp_path / "command_memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        episodic_memory_path=str(tmp_path / "episodes.jsonl"),
        contract_path=str(tmp_path / "contract.json"),
    ))


def _bundle(tmp_path):
    manager = TesseractGoalStateManager(tmp_path / "goals.json", tmp_path / "events.jsonl")
    cycle = TesseractGoalAwareCycleRunner(goal_manager=manager, runtime=_runtime(tmp_path))
    queue = TesseractGoalQueueRunner(goal_manager=manager, cycle_runner=cycle)
    policy = TesseractExecutionPolicyGovernor(policy_path=tmp_path / "policy.json", goal_manager=manager, queue_runner=queue)
    performance = TesseractPerformanceTelemetryGovernor(policy_governor=policy)
    stairway = TesseractStairwayCompressionGovernor(performance_governor=performance)
    return TesseractCompressedControlBundle(stairway_governor=stairway, baseline_path=tmp_path / "baseline.json")


def test_v1_13_1_summary_extracts_receipt_headlines(tmp_path):
    bundle = _bundle(tmp_path)
    result = bundle.run_bundle(demo=True)
    summary = summarize_control_bundle_report(result)
    assert summary["control_bundle_version"] == CONTROL_BUNDLE_VERSION
    assert summary["ok"] is True
    assert summary["ready_for_human_review"] is True
    assert summary["proposal_count"] == 3
    assert summary["approval_status"] == "pending_human_approval"
    assert summary["mutation_allowed"] is False
    assert "patch_proposal_receipts" not in summary
    assert "stairway_result" not in summary


def test_v1_13_1_summary_format_is_compact(tmp_path):
    bundle = _bundle(tmp_path)
    result = bundle.run_bundle(demo=True)
    text = format_control_bundle_summary(result)
    assert "CONTROL BUNDLE SUMMARY" in text
    assert "- ok: True" in text
    assert "- mutation_allowed: False" in text
    assert "patch_proposal_receipts" not in text
    assert "stairway_result" not in text
    assert len(text.splitlines()) <= 20


def test_v1_13_1_runner_uses_summary_mode():
    runner = (ROOT / "scripts" / "run_tesseract_control_bundle.ps1").read_text(encoding="utf-8")
    assert "--summary" in runner
    assert "Get-Content .\\artifacts\\tpn\\control_bundle_report_v1_13_latest.json -Raw" not in runner


def test_v1_13_1_summary_keeps_receipt_paths(tmp_path):
    bundle = _bundle(tmp_path)
    result = bundle.run_bundle(demo=True)
    summary = summarize_control_bundle_report(result)
    assert summary["latest_report"].endswith("control_bundle_report_v1_13_latest.json")
    assert summary["history"].endswith("control_bundle_history_v1_13.jsonl")
    assert summary["baseline"]
    assert Path(summary["baseline"]).exists()
