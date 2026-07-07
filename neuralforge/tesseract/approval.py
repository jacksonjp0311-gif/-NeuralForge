"""Human approval ledger for the Tesseract Jarvis roadmap.

v1.14 records explicit human approval/rejection decisions for proposal receipts.
It does not apply patches, mutate code, or grant autonomous write authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuralforge.tesseract.control_bundle import (
    DEFAULT_CONTROL_BUNDLE_REPORT_PATH,
    TesseractCompressedControlBundle,
)
from neuralforge.tesseract.performance import build_default_governor
from neuralforge.tesseract.stairway import TesseractStairwayCompressionGovernor


APPROVAL_VERSION = "tpn.approval.v1.14"
DEFAULT_APPROVAL_REPORT_PATH = Path("artifacts") / "tpn" / "approval_report_v1_14_latest.json"
DEFAULT_APPROVAL_LEDGER_PATH = Path("artifacts") / "tpn" / "approval_ledger_v1_14.jsonl"


@dataclass
class TesseractApprovalDecision:
    approval_id: str
    decision_id: str
    decision: str
    human_id: str
    proposal_ids: list[str]
    reason: str = ""
    approved_by_human: bool = False
    mutation_allowed: bool = False
    next_step_unlocked: str = "none"
    created_at_unix: float = field(default_factory=time.time)
    approval_version: str = APPROVAL_VERSION
    claim_boundary: str = "Human approval receipt only; no code mutation."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TesseractHumanApprovalLedger:
    def __init__(
        self,
        *,
        latest_path: str | Path = DEFAULT_APPROVAL_REPORT_PATH,
        ledger_path: str | Path = DEFAULT_APPROVAL_LEDGER_PATH,
    ) -> None:
        self.latest_path = Path(latest_path)
        self.ledger_path = Path(ledger_path)

    def load_control_bundle_report(self, path: str | Path = DEFAULT_CONTROL_BUNDLE_REPORT_PATH) -> dict[str, Any]:
        report_path = Path(path)
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf-8"))
        performance = build_default_governor()
        stairway = TesseractStairwayCompressionGovernor(performance_governor=performance)
        bundle = TesseractCompressedControlBundle(stairway_governor=stairway)
        return bundle.run_bundle(demo=True)

    def pending_summary(self, control_report: dict[str, Any]) -> dict[str, Any]:
        approval = control_report.get("approval_request", {}) or {}
        receipts = control_report.get("patch_proposal_receipts", []) or []
        return {
            "ok": bool(control_report.get("ok", False)),
            "approval_version": APPROVAL_VERSION,
            "approval_id": approval.get("approval_id", ""),
            "approval_status": approval.get("status", "unknown"),
            "approved_by_human": bool(approval.get("approved_by_human", False)),
            "mutation_allowed": bool(control_report.get("mutation_allowed", False)),
            "proposal_count": len(receipts),
            "proposal_ids": [str(receipt.get("proposal_id", "")) for receipt in receipts],
            "claim_boundary": "Pending approval summary only; no decision recorded.",
        }

    def decide(
        self,
        control_report: dict[str, Any],
        *,
        decision: str,
        human_id: str,
        reason: str = "",
    ) -> dict[str, Any]:
        decision_normalized = decision.strip().lower()
        if decision_normalized not in {"approve", "reject"}:
            raise ValueError("decision must be approve or reject")
        if not human_id.strip():
            raise ValueError("human_id is required")

        approval = control_report.get("approval_request", {}) or {}
        receipts = control_report.get("patch_proposal_receipts", []) or []
        proposal_ids = [str(receipt.get("proposal_id", "")) for receipt in receipts]
        approval_id = str(approval.get("approval_id", "")) or self._approval_id(proposal_ids)

        approved = decision_normalized == "approve"
        next_step = "sandboxed_patch_plan_receipt" if approved else "none"
        decision_id = self._decision_id(approval_id, decision_normalized, human_id, proposal_ids)

        receipt = TesseractApprovalDecision(
            approval_id=approval_id,
            decision_id=decision_id,
            decision=decision_normalized,
            human_id=human_id.strip(),
            proposal_ids=proposal_ids,
            reason=reason,
            approved_by_human=approved,
            mutation_allowed=False,
            next_step_unlocked=next_step,
        ).to_dict()
        self.write_receipt(receipt)
        return receipt

    def write_receipt(self, receipt: dict[str, Any]) -> dict[str, str]:
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.latest_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(receipt, sort_keys=True) + "\n")
        return {"latest": str(self.latest_path), "ledger": str(self.ledger_path)}

    def format_pending_summary(self, summary: dict[str, Any]) -> str:
        lines = [
            "APPROVAL LEDGER SUMMARY",
            f"- ok: {summary.get('ok', False)}",
            f"- approval_id: {summary.get('approval_id', '')}",
            f"- approval_status: {summary.get('approval_status', 'unknown')}",
            f"- approved_by_human: {summary.get('approved_by_human', False)}",
            f"- proposal_count: {summary.get('proposal_count', 0)}",
            f"- mutation_allowed: {summary.get('mutation_allowed', False)}",
            "- next: run with explicit approval/rejection command only when ready",
            "- boundary: no automatic approval; no patch application; no mutation authority",
        ]
        return "\n".join(lines)

    def format_decision_summary(self, receipt: dict[str, Any]) -> str:
        lines = [
            "APPROVAL DECISION RECEIPT",
            f"- decision: {receipt.get('decision')}",
            f"- approval_id: {receipt.get('approval_id')}",
            f"- decision_id: {receipt.get('decision_id')}",
            f"- human_id: {receipt.get('human_id')}",
            f"- approved_by_human: {receipt.get('approved_by_human')}",
            f"- proposal_count: {len(receipt.get('proposal_ids', []) or [])}",
            f"- next_step_unlocked: {receipt.get('next_step_unlocked')}",
            f"- mutation_allowed: {receipt.get('mutation_allowed')}",
            "- boundary: approval receipt only; no code mutation",
        ]
        return "\n".join(lines)

    def _approval_id(self, proposal_ids: list[str]) -> str:
        seed = json.dumps({"proposal_ids": proposal_ids, "version": APPROVAL_VERSION}, sort_keys=True)
        return "approval_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

    def _decision_id(self, approval_id: str, decision: str, human_id: str, proposal_ids: list[str]) -> str:
        seed = json.dumps(
            {
                "approval_id": approval_id,
                "decision": decision,
                "human_id": human_id,
                "proposal_ids": proposal_ids,
                "version": APPROVAL_VERSION,
            },
            sort_keys=True,
        )
        return "decision_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-report", default=str(DEFAULT_CONTROL_BUNDLE_REPORT_PATH))
    parser.add_argument("--decision", choices=["approve", "reject"], default="")
    parser.add_argument("--human-id", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--pending", action="store_true")
    args = parser.parse_args()

    ledger = TesseractHumanApprovalLedger()
    control_report = ledger.load_control_bundle_report(args.control_report)

    if args.decision:
        receipt = ledger.decide(
            control_report,
            decision=args.decision,
            human_id=args.human_id,
            reason=args.reason,
        )
        print(ledger.format_decision_summary(receipt))
        return

    summary = ledger.pending_summary(control_report)
    print(ledger.format_pending_summary(summary))


if __name__ == "__main__":
    main()
