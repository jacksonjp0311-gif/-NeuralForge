from pathlib import Path

from neuralforge.tesseract.goal_cycle import TesseractGoalAwareCycleRunner
from neuralforge.tesseract.goal_queue import TesseractGoalQueueRunner
from neuralforge.tesseract.goal_state import TesseractGoalStateManager
from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractJarvisRuntime
from neuralforge.tesseract.performance import TesseractPerformanceTelemetryGovernor
from neuralforge.tesseract.policy import TesseractExecutionPolicyGovernor
from neuralforge.tesseract.stairway import STAIRWAY_VERSION, TesseractStairwayCompressionGovernor


def _runtime(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    return TesseractJarvisRuntime(JarvisServiceConfig(
        repo_root=str(repo),
        memory_path=str(tmp_path / "command_memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        episodic_memory_path=str(tmp_path / "episodes.jsonl"),
        contract_path=str(tmp_path / "contract.json"),
    ))


def _stairway(tmp_path):
    manager = TesseractGoalStateManager(tmp_path / "goals.json", tmp_path / "events.jsonl")
    cycle = TesseractGoalAwareCycleRunner(goal_manager=manager, runtime=_runtime(tmp_path))
    queue = TesseractGoalQueueRunner(goal_manager=manager, cycle_runner=cycle)
    policy = TesseractExecutionPolicyGovernor(policy_path=tmp_path / "policy.json", goal_manager=manager, queue_runner=queue)
    performance = TesseractPerformanceTelemetryGovernor(policy_governor=policy)
    return TesseractStairwayCompressionGovernor(performance_governor=performance)


def test_v1_12_stairway_demo_ready(tmp_path):
    stairway = _stairway(tmp_path)
    result = stairway.run_assessment(demo=True)
    report = result["stairway_report"]
    assert result["stairway_version"] == STAIRWAY_VERSION
    assert report["ready_for_next_bundle"] is True
    assert report["approval_required"] is True
    assert report["compressed_bundle"]["safe_to_compress"] is True
    assert len(report["proposals"]) == 3


def test_v1_12_stairway_blocks_on_performance_warning(tmp_path):
    stairway = _stairway(tmp_path)
    report = stairway.build_report({
        "performance_ok": False,
        "policy_allowed": True,
        "queue_result_present": True,
        "completed_goal_count": 0,
        "blocked_goal_count": 0,
        "warnings": ["slow path"],
        "summary": {},
    })
    assert report["ready_for_next_bundle"] is False
    assert report["drift_status"] == "blocked"
    assert report["proposals"][0]["proposal_id"] == "close_performance_or_policy_blockers"


def test_v1_12_stairway_proposals_do_not_mutate(tmp_path):
    stairway = _stairway(tmp_path)
    report = stairway.build_report({
        "performance_ok": True,
        "policy_allowed": True,
        "queue_result_present": True,
        "completed_goal_count": 2,
        "blocked_goal_count": 0,
        "warnings": [],
        "total_duration_ms": 100.0,
        "summary": {"max_skill_duration_ms": 1.0, "mean_skill_duration_ms": 1.0},
    })
    for proposal in report["proposals"]:
        assert proposal["approval_required"] is True
        assert proposal["mutation_allowed"] is False


def test_v1_12_stairway_write_report(tmp_path):
    stairway = _stairway(tmp_path)
    report = stairway.build_report({
        "performance_ok": True,
        "policy_allowed": True,
        "queue_result_present": True,
        "completed_goal_count": 1,
        "blocked_goal_count": 0,
        "warnings": [],
        "summary": {},
    })
    paths = stairway.write_report(report, latest_path=tmp_path / "latest.json", history_path=tmp_path / "history.jsonl")
    assert Path(paths["latest"]).exists()
    assert Path(paths["history"]).exists()
