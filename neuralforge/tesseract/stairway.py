"""Stairway Compression Governor for the Tesseract Jarvis roadmap.

v1.12 compresses safe adjacent control layers into one governed assessment:
performance probe, drift/regression judgment, proposal receipts, and explicit
approval requirement. It does not mutate code or grant autonomous authority.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuralforge.tesseract.goal_state import DEFAULT_GOAL_EVENTS_PATH, DEFAULT_GOAL_STATE_PATH
from neuralforge.tesseract.performance import (
    PERFORMANCE_VERSION,
    TesseractPerformanceTelemetryGovernor,
    build_default_governor,
)


STAIRWAY_VERSION = "tpn.stairway.v1.12"
DEFAULT_STAIRWAY_REPORT_PATH = Path("artifacts") / "tpn" / "stairway_report_v1_12_latest.json"
DEFAULT_STAIRWAY_HISTORY_PATH = Path("artifacts") / "tpn" / "stairway_history_v1_12.jsonl"


@dataclass
class TesseractStairwayProposal:
    proposal_id: str
    title: str
    rationale: str
    suggested_next_version: str
    compressed_with: list[str] = field(default_factory=list)
    approval_required: bool = True
    mutation_allowed: bool = False
    claim_boundary: str = "Proposal receipt only; no code mutation."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TesseractStairwayReport:
    ok: bool
    ready_for_next_bundle: bool
    drift_status: str
    blocked_reasons: list[str]
    proposals: list[dict[str, Any]]
    performance_summary: dict[str, Any]
    approval_required: bool = True
    created_at_unix: float = field(default_factory=time.time)
    stairway_version: str = STAIRWAY_VERSION
    claim_boundary: str = "Compressed roadmap governance only; no autonomous authority."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TesseractStairwayCompressionGovernor:
    def __init__(
        self,
        *,
        performance_governor: TesseractPerformanceTelemetryGovernor | None = None,
    ) -> None:
        self.performance_governor = performance_governor or build_default_governor()

    def run_assessment(
        self,
        *,
        max_goals: int = 2,
        max_steps: int = 4,
        demo: bool = True,
    ) -> dict[str, Any]:
        if demo:
            performance_report = self.performance_governor.demo()
        else:
            performance_report = self.performance_governor.run_probe(max_goals=max_goals, max_steps=max_steps)
        report = self.build_report(performance_report)
        wrapped = {
            "ok": report["ok"],
            "stairway_report": report,
            "performance_report": performance_report,
            "paths": self.write_report(report),
            "stairway_version": STAIRWAY_VERSION,
            "claim_boundary": "Assessment, proposals, and approval requirement only.",
        }
        return wrapped

    def build_report(self, performance_report: dict[str, Any]) -> dict[str, Any]:
        blocked_reasons = self._blocked_reasons(performance_report)
        drift_status = self._drift_status(performance_report, blocked_reasons)
        ready = not blocked_reasons and bool(performance_report.get("performance_ok", False))
        proposals = self._proposals(performance_report, ready, blocked_reasons)
        summary = {
            "performance_version": performance_report.get("performance_version", PERFORMANCE_VERSION),
            "performance_ok": performance_report.get("performance_ok", False),
            "policy_allowed": performance_report.get("policy_allowed", False),
            "queue_result_present": performance_report.get("queue_result_present", False),
            "completed_goal_count": performance_report.get("completed_goal_count", 0),
            "blocked_goal_count": performance_report.get("blocked_goal_count", 0),
            "total_duration_ms": performance_report.get("total_duration_ms", 0.0),
            "queue_duration_ms": performance_report.get("queue_duration_ms", 0.0),
            "summary": performance_report.get("summary", {}),
            "warnings": performance_report.get("warnings", []),
        }
        report = TesseractStairwayReport(
            ok=ready,
            ready_for_next_bundle=ready,
            drift_status=drift_status,
            blocked_reasons=blocked_reasons,
            proposals=[proposal.to_dict() for proposal in proposals],
            performance_summary=summary,
        ).to_dict()
        report["compressed_bundle"] = {
            "bundle_id": "stairway_control_bundle_v1",
            "contains": [
                "drift_regression_sentinel",
                "proposal_receipt_engine",
                "human_approval_gate",
            ],
            "safe_to_compress": True,
            "reason": "These layers observe, score, and require approval; they do not apply code changes.",
        }
        return report

    def _blocked_reasons(self, performance_report: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        if not performance_report.get("performance_ok", False):
            reasons.append("performance_ok is false")
        if not performance_report.get("policy_allowed", False):
            reasons.append("policy did not allow execution")
        if not performance_report.get("queue_result_present", False):
            reasons.append("queue result missing")
        if performance_report.get("blocked_goal_count", 0):
            reasons.append("one or more goals blocked")
        for warning in performance_report.get("warnings", []) or []:
            reasons.append(f"performance warning: {warning}")
        return list(dict.fromkeys(reasons))

    def _drift_status(self, performance_report: dict[str, Any], blocked_reasons: list[str]) -> str:
        if blocked_reasons:
            return "blocked"
        summary = performance_report.get("summary", {}) or {}
        max_skill = float(summary.get("max_skill_duration_ms", 0.0) or 0.0)
        mean_skill = float(summary.get("mean_skill_duration_ms", 0.0) or 0.0)
        total = float(performance_report.get("total_duration_ms", 0.0) or 0.0)
        if total <= 500.0 and max_skill <= 100.0 and mean_skill <= 50.0:
            return "stable_fast"
        return "stable"

    def _proposals(
        self,
        performance_report: dict[str, Any],
        ready: bool,
        blocked_reasons: list[str],
    ) -> list[TesseractStairwayProposal]:
        if not ready:
            return [
                TesseractStairwayProposal(
                    proposal_id="close_performance_or_policy_blockers",
                    title="Close performance or policy blockers before compression",
                    rationale="The stairway cannot compress safely while performance or policy blockers exist: " + "; ".join(blocked_reasons),
                    suggested_next_version="v1.11.x",
                    compressed_with=[],
                )
            ]

        return [
            TesseractStairwayProposal(
                proposal_id="add_drift_regression_sentinel",
                title="Add drift and regression sentinel",
                rationale="Performance telemetry is stable; the next safe layer is a sentinel that compares future runs to recorded baselines.",
                suggested_next_version="v1.13-compressed",
                compressed_with=["proposal_receipt_engine", "human_approval_gate"],
            ),
            TesseractStairwayProposal(
                proposal_id="add_patch_proposal_receipts",
                title="Add patch proposal receipts",
                rationale="The system should be able to describe proposed code changes, tests, risks, and rollback plan without applying them.",
                suggested_next_version="v1.13-compressed",
                compressed_with=["drift_regression_sentinel", "human_approval_gate"],
            ),
            TesseractStairwayProposal(
                proposal_id="add_human_approval_gate",
                title="Add human approval gate",
                rationale="Compressed future layers must preserve explicit human approval before any mutation-capable patch step.",
                suggested_next_version="v1.13-compressed",
                compressed_with=["drift_regression_sentinel", "proposal_receipt_engine"],
            ),
        ]

    def write_report(
        self,
        report: dict[str, Any],
        *,
        latest_path: str | Path = DEFAULT_STAIRWAY_REPORT_PATH,
        history_path: str | Path = DEFAULT_STAIRWAY_HISTORY_PATH,
    ) -> dict[str, str]:
        latest_path = Path(latest_path)
        history_path = Path(history_path)
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, sort_keys=True) + "\n")
        return {"latest": str(latest_path), "history": str(history_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", default=str(DEFAULT_GOAL_STATE_PATH))
    parser.add_argument("--events-path", default=str(DEFAULT_GOAL_EVENTS_PATH))
    parser.add_argument("--max-goals", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--assess", action="store_true")
    args = parser.parse_args()

    performance = build_default_governor(state_path=args.state_path, events_path=args.events_path)
    stairway = TesseractStairwayCompressionGovernor(performance_governor=performance)
    result = stairway.run_assessment(max_goals=args.max_goals, max_steps=args.max_steps, demo=args.demo or not args.assess)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
