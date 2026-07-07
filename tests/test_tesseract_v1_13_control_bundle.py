from pathlib import Path

from neuralforge.tesseract.control_bundle import CONTROL_BUNDLE_VERSION, TesseractCompressedControlBundle
from neuralforge.tesseract.goal_cycle import TesseractGoalAwareCycleRunner
from neuralforge.tesseract.goal_queue import TesseractGoalQueueRunner
from neuralforge.tesseract.goal_state import TesseractGoalStateManager
from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractJarvisRuntime
from neuralforge.tesseract.performance import TesseractPerformanceTelemetryGovernor
from neuralforge.tesseract.policy import TesseractExecutionPolicyGovernor
from neuralforge.tesseract.stairway import TesseractStairwayCompressionGovernor


def _runtime(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    return TesseractJarvisRuntime(JarvisServiceConfig(
        repo_root=str(repo),
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


def test_v1_13_control_bundle_run_ready_for_review(tmp_path):
    bundle = _bundle(tmp_path)
    report = bundle.run_bundle(demo=True)
    assert report["control_bundle_version"] == CONTROL_BUNDLE_VERSION
    assert report["ready_for_human_review"] is True
    assert report["mutation_allowed"] is False
    assert report["approval_request"]["status"] == "pending_human_approval"
    assert report["approval_request"]["approved_by_human"] is False


def test_v1_13_regression_sentinel_detects_slowdown(tmp_path):
    bundle = _bundle(tmp_path)
    baseline = {
        "control_bundle_version": CONTROL_BUNDLE_VERSION,
        "total_duration_ms": 100.0,
        "queue_duration_ms": 100.0,
        "max_skill_duration_ms": 10.0,
        "mean_skill_duration_ms": 5.0,
        "completed_goal_count": 2,
        "created_at_unix": 1.0,
    }
    current = {
        "total_duration_ms": 1000.0,
        "queue_duration_ms": 1000.0,
        "completed_goal_count": 2,
        "warnings": [],
        "summary": {"max_skill_duration_ms": 100.0, "mean_skill_duration_ms": 50.0},
    }
    result = bundle.evaluate_regression(current, baseline)
    assert result["status"] == "regression_detected"
    assert result["reasons"]


def test_v1_13_proposal_receipts_require_approval(tmp_path):
    bundle = _bundle(tmp_path)
    receipts = bundle.build_proposal_receipts({
        "proposals": [
            {"proposal_id": "add_patch_proposal_receipts", "title": "Patch receipts", "rationale": "Needed."}
        ]
    })
    assert receipts
    assert receipts[0]["approval_required"] is True
    assert receipts[0]["mutation_allowed"] is False
    assert receipts[0]["required_tests"]


def test_v1_13_approval_request_is_pending(tmp_path):
    bundle = _bundle(tmp_path)
    req = bundle.create_approval_request([{"proposal_id": "a"}, {"proposal_id": "b"}])
    assert req["status"] == "pending_human_approval"
    assert req["approved_by_human"] is False
    assert req["mutation_allowed"] is False
    assert req["approval_id"].startswith("approval_")


def test_v1_13_write_report(tmp_path):
    bundle = _bundle(tmp_path)
    report = {"ok": True, "control_bundle_version": CONTROL_BUNDLE_VERSION}
    paths = bundle.write_report(report, latest_path=tmp_path / "latest.json", history_path=tmp_path / "history.jsonl")
    assert Path(paths["latest"]).exists()
    assert Path(paths["history"]).exists()
