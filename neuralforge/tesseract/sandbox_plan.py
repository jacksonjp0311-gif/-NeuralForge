"""Sandboxed patch plan receipts for the Tesseract Jarvis roadmap.

v1.15 converts an approved human approval receipt into a sandboxed patch plan
receipt. It does not create branches, edit files, apply patches, or grant
autonomous write authority.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuralforge.tesseract.approval import (
    DEFAULT_APPROVAL_REPORT_PATH,
    TesseractHumanApprovalLedger,
)


SANDBOX_PLAN_VERSION = "tpn.sandbox_plan.v1.15"
DEFAULT_SANDBOX_PLAN_REPORT_PATH = Path("artifacts") / "tpn" / "sandbox_plan_report_v1_15_latest.json"
DEFAULT_SANDBOX_PLAN_HISTORY_PATH = Path("artifacts") / "tpn" / "sandbox_plan_history_v1_15.jsonl"


@dataclass
class TesseractSandboxedPatchPlanReceipt:
    ok: bool
    approval_id: str
    decision_id: str
    approved_by_human: bool
    proposal_ids: list[str]
    sandbox_branch: str
    planned_steps: list[dict[str, Any]]
    required_tests: list[str]
    rollback_plan: str
    blocked_reasons: list[str] = field(default_factory=list)
    planning_allowed: bool = False
    mutation_allowed: bool = False
    apply_allowed: bool = False
    created_at_unix: float = field(default_factory=time.time)
    sandbox_plan_version: str = SANDBOX_PLAN_VERSION
    claim_boundary: str = "Sandboxed patch plan receipt only; no branch creation, code mutation, or patch application."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TesseractSandboxedPatchPlanner:
    def __init__(
        self,
        *,
        latest_path: str | Path = DEFAULT_SANDBOX_PLAN_REPORT_PATH,
        history_path: str | Path = DEFAULT_SANDBOX_PLAN_HISTORY_PATH,
    ) -> None:
        self.latest_path = Path(latest_path)
        self.history_path = Path(history_path)

    def load_approval_receipt(self, path: str | Path = DEFAULT_APPROVAL_REPORT_PATH) -> dict[str, Any]:
        approval_path = Path(path)
        if not approval_path.exists():
            return {
                "ok": False,
                "approval_id": "",
                "decision_id": "",
                "approved_by_human": False,
                "mutation_allowed": False,
                "proposal_ids": [],
                "blocked_reasons": [f"approval receipt not found: {approval_path}"],
                "claim_boundary": "Missing approval receipt; no planning authority.",
            }
        return json.loads(approval_path.read_text(encoding="utf-8"))

    def build_plan(self, approval_receipt: dict[str, Any]) -> dict[str, Any]:
        blocked = self.blocked_reasons(approval_receipt)
        approved = not blocked
        approval_id = str(approval_receipt.get("approval_id", ""))
        decision_id = str(approval_receipt.get("decision_id", ""))
        proposal_ids = [str(item) for item in approval_receipt.get("proposal_ids", []) or []]
        branch = self.sandbox_branch(decision_id or approval_id)

        receipt = TesseractSandboxedPatchPlanReceipt(
            ok=approved,
            approval_id=approval_id,
            decision_id=decision_id,
            approved_by_human=bool(approval_receipt.get("approved_by_human", False)),
            proposal_ids=proposal_ids,
            sandbox_branch=branch,
            planned_steps=self.planned_steps(proposal_ids, branch),
            required_tests=self.required_tests(),
            rollback_plan="Delete the sandbox branch or revert the sandbox commit; rerun the full segmented TPN verifier.",
            blocked_reasons=blocked,
            planning_allowed=approved,
            mutation_allowed=False,
            apply_allowed=False,
        ).to_dict()
        self.write_report(receipt)
        return receipt

    def blocked_reasons(self, approval_receipt: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        if not bool(approval_receipt.get("approved_by_human", False)):
            reasons.append("approval receipt is not approved_by_human")
        if str(approval_receipt.get("decision", "")).lower() != "approve":
            reasons.append("approval decision is not approve")
        if bool(approval_receipt.get("mutation_allowed", False)):
            reasons.append("approval receipt unexpectedly allows mutation")
        if str(approval_receipt.get("next_step_unlocked", "")) != "sandboxed_patch_plan_receipt":
            reasons.append("approval receipt did not unlock sandboxed_patch_plan_receipt")
        if not approval_receipt.get("proposal_ids", []):
            reasons.append("approval receipt has no proposal ids")
        return reasons

    def sandbox_branch(self, seed: str) -> str:
        safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in seed).strip("-")
        if not safe:
            safe = "unapproved"
        return f"sandbox/tpn-v1-15-{safe[:24]}"

    def planned_steps(self, proposal_ids: list[str], branch: str) -> list[dict[str, Any]]:
        return [
            {
                "step_id": "step_01",
                "title": "create sandbox branch",
                "detail": f"Plan to create branch {branch}; not executed by this module.",
                "status": "planned_not_executed",
            },
            {
                "step_id": "step_02",
                "title": "draft patch set from approved proposal receipts",
                "detail": "Plan patch files from proposal_ids only; no files are changed here.",
                "proposal_ids": proposal_ids,
                "status": "planned_not_executed",
            },
            {
                "step_id": "step_03",
                "title": "run segmented verifier in sandbox",
                "detail": "Plan compileall, segmented pytest, smoke checks, and contract check.",
                "status": "planned_not_executed",
            },
            {
                "step_id": "step_04",
                "title": "emit sandbox result receipt",
                "detail": "Plan to record pass/fail and rollback instructions before any merge.",
                "status": "planned_not_executed",
            },
        ]

    def required_tests(self) -> list[str]:
        return [
            "python -m compileall neuralforge tests examples",
            "python -m pytest tests/test_tesseract_v1_15_sandbox_plan.py -q",
            "powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\run_tesseract_sandbox_plan.ps1",
            "powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\check_tesseract_contract.ps1 -Base http://127.0.0.1:9",
        ]

    def write_report(self, receipt: dict[str, Any]) -> dict[str, str]:
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.latest_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(receipt, sort_keys=True) + "\n")
        return {"latest": str(self.latest_path), "history": str(self.history_path)}

    def format_summary(self, receipt: dict[str, Any]) -> str:
        lines = [
            "SANDBOXED PATCH PLAN SUMMARY",
            f"- ok: {receipt.get('ok', False)}",
            f"- planning_allowed: {receipt.get('planning_allowed', False)}",
            f"- approved_by_human: {receipt.get('approved_by_human', False)}",
            f"- proposal_count: {len(receipt.get('proposal_ids', []) or [])}",
            f"- sandbox_branch: {receipt.get('sandbox_branch', '')}",
            f"- planned_steps: {len(receipt.get('planned_steps', []) or [])}",
            f"- required_tests: {len(receipt.get('required_tests', []) or [])}",
            f"- blocked_reasons: {len(receipt.get('blocked_reasons', []) or [])}",
            f"- mutation_allowed: {receipt.get('mutation_allowed', False)}",
            f"- apply_allowed: {receipt.get('apply_allowed', False)}",
            "- boundary: plan receipt only; no branch creation, no file edits, no patch application",
        ]
        return "\n".join(lines)


def demo_approved_receipt() -> dict[str, Any]:
    ledger = TesseractHumanApprovalLedger()
    control_report = ledger.load_control_bundle_report()
    return ledger.decide(
        control_report,
        decision="approve",
        human_id="demo",
        reason="sandbox plan smoke approval only",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval-report", default=str(DEFAULT_APPROVAL_REPORT_PATH))
    parser.add_argument("--demo-approved", action="store_true")
    args = parser.parse_args()

    planner = TesseractSandboxedPatchPlanner()
    if args.demo_approved:
        approval = demo_approved_receipt()
    else:
        approval = planner.load_approval_receipt(args.approval_report)
    receipt = planner.build_plan(approval)
    print(planner.format_summary(receipt))


if __name__ == "__main__":
    main()
