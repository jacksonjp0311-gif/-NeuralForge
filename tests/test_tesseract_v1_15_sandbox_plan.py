from pathlib import Path

from neuralforge.tesseract.sandbox_plan import (
    SANDBOX_PLAN_VERSION,
    TesseractSandboxedPatchPlanner,
)


def _approved_receipt():
    return {
        "approval_id": "approval_test",
        "decision_id": "decision_test",
        "decision": "approve",
        "human_id": "James",
        "approved_by_human": True,
        "mutation_allowed": False,
        "next_step_unlocked": "sandboxed_patch_plan_receipt",
        "proposal_ids": ["p1", "p2", "p3"],
    }


def _rejected_receipt():
    data = _approved_receipt()
    data["decision"] = "reject"
    data["approved_by_human"] = False
    data["next_step_unlocked"] = "none"
    return data


def test_v1_15_approved_receipt_builds_plan(tmp_path):
    planner = TesseractSandboxedPatchPlanner(
        latest_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )
    plan = planner.build_plan(_approved_receipt())
    assert plan["sandbox_plan_version"] == SANDBOX_PLAN_VERSION
    assert plan["ok"] is True
    assert plan["planning_allowed"] is True
    assert plan["mutation_allowed"] is False
    assert plan["apply_allowed"] is False
    assert plan["sandbox_branch"].startswith("sandbox/tpn-v1-15-")
    assert len(plan["planned_steps"]) == 4
    assert Path(tmp_path / "latest.json").exists()
    assert Path(tmp_path / "history.jsonl").exists()


def test_v1_15_rejected_receipt_blocks_plan(tmp_path):
    planner = TesseractSandboxedPatchPlanner(
        latest_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )
    plan = planner.build_plan(_rejected_receipt())
    assert plan["ok"] is False
    assert plan["planning_allowed"] is False
    assert plan["mutation_allowed"] is False
    assert plan["apply_allowed"] is False
    assert plan["blocked_reasons"]


def test_v1_15_missing_approval_receipt_blocks(tmp_path):
    planner = TesseractSandboxedPatchPlanner(
        latest_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )
    approval = planner.load_approval_receipt(tmp_path / "missing.json")
    plan = planner.build_plan(approval)
    assert plan["ok"] is False
    assert "approval receipt is not approved_by_human" in plan["blocked_reasons"]


def test_v1_15_summary_is_compact(tmp_path):
    planner = TesseractSandboxedPatchPlanner(
        latest_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )
    plan = planner.build_plan(_approved_receipt())
    summary = planner.format_summary(plan)
    assert "SANDBOXED PATCH PLAN SUMMARY" in summary
    assert "mutation_allowed: False" in summary
    assert "apply_allowed: False" in summary
    assert len(summary.splitlines()) <= 14
