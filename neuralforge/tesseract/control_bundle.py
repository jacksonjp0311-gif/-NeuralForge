"""Compressed control bundle for the Tesseract Jarvis roadmap.

v1.13 compresses three safe adjacent control layers:
1. drift/regression sentinel,
2. patch proposal receipts,
3. human approval gate.

It observes, scores, proposes, and requires approval. It does not mutate code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuralforge.tesseract.goal_state import DEFAULT_GOAL_EVENTS_PATH, DEFAULT_GOAL_STATE_PATH
from neuralforge.tesseract.performance import build_default_governor
from neuralforge.tesseract.stairway import TesseractStairwayCompressionGovernor


CONTROL_BUNDLE_VERSION = "tpn.control_bundle.v1.13"
DEFAULT_CONTROL_BUNDLE_REPORT_PATH = Path("artifacts") / "tpn" / "control_bundle_report_v1_13_latest.json"
DEFAULT_CONTROL_BUNDLE_HISTORY_PATH = Path("artifacts") / "tpn" / "control_bundle_history_v1_13.jsonl"
DEFAULT_CONTROL_BUNDLE_BASELINE_PATH = Path("artifacts") / "tpn" / "control_bundle_baseline_v1_13.json"


@dataclass
class TesseractRegressionSentinelResult:
    status: str
    reasons: list[str]
    baseline: dict[str, Any]
    current: dict[str, Any]
    control_bundle_version: str = CONTROL_BUNDLE_VERSION
    claim_boundary: str = "Regression sentinel only; no mutation."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TesseractPatchProposalReceipt:
    proposal_id: str
    title: str
    rationale: str
    suggested_next_version: str
    expected_files: list[str]
    required_tests: list[str]
    risk: str = "medium"
    rollback_plan: str = "Revert the generated commit and rerun the full TPN verifier."
    approval_required: bool = True
    mutation_allowed: bool = False
    control_bundle_version: str = CONTROL_BUNDLE_VERSION
    claim_boundary: str = "Patch proposal receipt only; no code mutation."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TesseractApprovalRequest:
    approval_id: str
    status: str
    proposal_ids: list[str]
    approval_required: bool = True
    approved_by_human: bool = False
    mutation_allowed: bool = False
    created_at_unix: float = field(default_factory=time.time)
    control_bundle_version: str = CONTROL_BUNDLE_VERSION
    claim_boundary: str = "Human approval gate only; no automatic approval."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TesseractCompressedControlBundle:
    def __init__(
        self,
        *,
        stairway_governor: TesseractStairwayCompressionGovernor | None = None,
        baseline_path: str | Path = DEFAULT_CONTROL_BUNDLE_BASELINE_PATH,
    ) -> None:
        self.stairway_governor = stairway_governor or TesseractStairwayCompressionGovernor()
        self.baseline_path = Path(baseline_path)

    def run_bundle(self, *, demo: bool = True) -> dict[str, Any]:
        stairway_result = self.stairway_governor.run_assessment(demo=demo)
        stairway_report = stairway_result.get("stairway_report", {})
        performance_summary = stairway_report.get("performance_summary", {})
        baseline = self.load_or_create_baseline(performance_summary)
        sentinel = self.evaluate_regression(performance_summary, baseline)
        receipts = self.build_proposal_receipts(stairway_report)
        approval_request = self.create_approval_request(receipts)
        ready = (
            bool(stairway_report.get("ready_for_next_bundle", False))
            and sentinel["status"] in {"baseline_created", "stable", "stable_fast"}
            and bool(receipts)
            and approval_request["status"] == "pending_human_approval"
        )
        report = {
            "ok": ready,
            "ready_for_human_review": ready,
            "control_bundle_version": CONTROL_BUNDLE_VERSION,
            "stairway_result": stairway_result,
            "regression_sentinel": sentinel,
            "patch_proposal_receipts": receipts,
            "approval_request": approval_request,
            "mutation_allowed": False,
            "claim_boundary": "Compressed control bundle only; no code mutation or autonomous authority.",
        }
        report["paths"] = self.write_report(report)
        return report

    def load_or_create_baseline(self, performance_summary: dict[str, Any]) -> dict[str, Any]:
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        if self.baseline_path.exists():
            try:
                baseline = json.loads(self.baseline_path.read_text(encoding="utf-8"))
                if baseline.get("control_bundle_version") == CONTROL_BUNDLE_VERSION:
                    return baseline
            except json.JSONDecodeError:
                pass
        baseline = {
            "control_bundle_version": CONTROL_BUNDLE_VERSION,
            "created_at_unix": time.time(),
            "total_duration_ms": float(performance_summary.get("total_duration_ms", 0.0) or 0.0),
            "queue_duration_ms": float(performance_summary.get("queue_duration_ms", 0.0) or 0.0),
            "max_skill_duration_ms": float((performance_summary.get("summary") or {}).get("max_skill_duration_ms", 0.0) or 0.0),
            "mean_skill_duration_ms": float((performance_summary.get("summary") or {}).get("mean_skill_duration_ms", 0.0) or 0.0),
            "completed_goal_count": int(performance_summary.get("completed_goal_count", 0) or 0),
            "claim_boundary": "Baseline for regression detection only.",
        }
        self.baseline_path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return baseline

    def evaluate_regression(self, performance_summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
        current = {
            "total_duration_ms": float(performance_summary.get("total_duration_ms", 0.0) or 0.0),
            "queue_duration_ms": float(performance_summary.get("queue_duration_ms", 0.0) or 0.0),
            "max_skill_duration_ms": float((performance_summary.get("summary") or {}).get("max_skill_duration_ms", 0.0) or 0.0),
            "mean_skill_duration_ms": float((performance_summary.get("summary") or {}).get("mean_skill_duration_ms", 0.0) or 0.0),
            "completed_goal_count": int(performance_summary.get("completed_goal_count", 0) or 0),
            "warnings": list(performance_summary.get("warnings", []) or []),
        }
        reasons: list[str] = []
        if current["warnings"]:
            reasons.extend([f"performance warning: {w}" for w in current["warnings"]])
        for key in ["total_duration_ms", "queue_duration_ms", "max_skill_duration_ms", "mean_skill_duration_ms"]:
            base = float(baseline.get(key, 0.0) or 0.0)
            now = float(current.get(key, 0.0) or 0.0)
            if base > 0.0 and now > max(base * 1.75, base + 250.0):
                reasons.append(f"{key} regressed from {base:.3f} ms to {now:.3f} ms.")
        if current["completed_goal_count"] < int(baseline.get("completed_goal_count", 1) or 1):
            reasons.append("completed_goal_count dropped below baseline.")
        if reasons:
            status = "regression_detected"
        elif abs(float(baseline.get("created_at_unix", 0.0) or 0.0) - time.time()) < 5.0:
            status = "baseline_created"
        elif current["total_duration_ms"] <= 500.0 and current["max_skill_duration_ms"] <= 100.0:
            status = "stable_fast"
        else:
            status = "stable"
        return TesseractRegressionSentinelResult(
            status=status,
            reasons=reasons,
            baseline=baseline,
            current=current,
        ).to_dict()

    def build_proposal_receipts(self, stairway_report: dict[str, Any]) -> list[dict[str, Any]]:
        receipts: list[dict[str, Any]] = []
        for proposal in stairway_report.get("proposals", []) or []:
            proposal_id = str(proposal.get("proposal_id", "proposal"))
            title = str(proposal.get("title", proposal_id))
            rationale = str(proposal.get("rationale", "No rationale provided."))
            receipts.append(
                TesseractPatchProposalReceipt(
                    proposal_id=proposal_id,
                    title=title,
                    rationale=rationale,
                    suggested_next_version=str(proposal.get("suggested_next_version", "v1.13-compressed")),
                    expected_files=self.expected_files_for(proposal_id),
                    required_tests=[
                        "python -m compileall neuralforge tests examples",
                        "python -m pytest tests/test_tesseract_v1_13_control_bundle.py -q",
                        "powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\run_tesseract_control_bundle.ps1",
                    ],
                    risk="medium" if "patch" in proposal_id else "low",
                ).to_dict()
            )
        return receipts

    def expected_files_for(self, proposal_id: str) -> list[str]:
        mapping = {
            "add_drift_regression_sentinel": [
                "neuralforge/tesseract/control_bundle.py",
                "artifacts/tpn/control_bundle_baseline_v1_13.json",
            ],
            "add_patch_proposal_receipts": [
                "neuralforge/tesseract/control_bundle.py",
                "artifacts/tpn/control_bundle_report_v1_13_latest.json",
            ],
            "add_human_approval_gate": [
                "neuralforge/tesseract/control_bundle.py",
                "artifacts/tpn/control_bundle_history_v1_13.jsonl",
            ],
        }
        return mapping.get(proposal_id, ["neuralforge/tesseract/control_bundle.py"])

    def create_approval_request(self, receipts: list[dict[str, Any]]) -> dict[str, Any]:
        proposal_ids = [str(receipt.get("proposal_id", "")) for receipt in receipts]
        seed = json.dumps({"proposal_ids": proposal_ids, "version": CONTROL_BUNDLE_VERSION}, sort_keys=True)
        approval_id = "approval_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        return TesseractApprovalRequest(
            approval_id=approval_id,
            status="pending_human_approval",
            proposal_ids=proposal_ids,
        ).to_dict()

    def write_report(
        self,
        report: dict[str, Any],
        *,
        latest_path: str | Path = DEFAULT_CONTROL_BUNDLE_REPORT_PATH,
        history_path: str | Path = DEFAULT_CONTROL_BUNDLE_HISTORY_PATH,
    ) -> dict[str, str]:
        latest_path = Path(latest_path)
        history_path = Path(history_path)
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, sort_keys=True) + "\n")
        return {"latest": str(latest_path), "history": str(history_path), "baseline": str(self.baseline_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", default=str(DEFAULT_GOAL_STATE_PATH))
    parser.add_argument("--events-path", default=str(DEFAULT_GOAL_EVENTS_PATH))
    parser.add_argument("--baseline-path", default=str(DEFAULT_CONTROL_BUNDLE_BASELINE_PATH))
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    performance = build_default_governor(state_path=args.state_path, events_path=args.events_path)
    stairway = TesseractStairwayCompressionGovernor(performance_governor=performance)
    bundle = TesseractCompressedControlBundle(stairway_governor=stairway, baseline_path=args.baseline_path)
    print(json.dumps(bundle.run_bundle(demo=True), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
