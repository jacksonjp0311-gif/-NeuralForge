"""Self-improvement proposal core for the Tesseract Jarvis runtime.

v1.6 adds evidence-based improvement proposals. It does not patch files,
execute shell commands, or mutate the repository. It converts benchmark and
memory evidence into proposed next actions with gates.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuralforge.tesseract.benchmark import TesseractBenchmarkHarness
from neuralforge.tesseract.memory_core import DEFAULT_EPISODE_PATH, TesseractEpisodicMemory

IMPROVEMENT_VERSION = "tpn.improvement.v1.6"
DEFAULT_PROPOSAL_DIR = Path("artifacts") / "tpn" / "improvement"


@dataclass(frozen=True)
class TesseractImprovementProposal:
    proposal_id: str
    title: str
    rationale: str
    risk: str
    expected_impact: str
    required_gates: list[str]
    suggested_next_version: str
    tags: list[str] = field(default_factory=list)
    allowed_to_mutate: bool = False
    improvement_version: str = IMPROVEMENT_VERSION
    claim_boundary: str = "Proposal only; no autonomous code mutation."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TesseractImprovementProposalEngine:
    def __init__(self, memory: TesseractEpisodicMemory | None = None, benchmark_harness: TesseractBenchmarkHarness | None = None) -> None:
        self.memory = memory or TesseractEpisodicMemory()
        self.benchmark_harness = benchmark_harness

    def propose(
        self,
        *,
        benchmark_report: dict[str, Any] | None = None,
        memory_summary: dict[str, Any] | None = None,
        max_proposals: int = 8,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if benchmark_report is None and self.benchmark_harness is not None:
            benchmark_report = self.benchmark_harness.run()
        if benchmark_report is None:
            benchmark_report = self._load_latest_benchmark()
        if memory_summary is None:
            memory_summary = self.memory.consolidate()

        proposals: list[TesseractImprovementProposal] = []
        proposals.extend(self._benchmark_proposals(benchmark_report))
        proposals.extend(self._memory_proposals(memory_summary))
        proposals.extend(self._roadmap_proposals(benchmark_report, memory_summary))

        seen: set[str] = set()
        unique: list[TesseractImprovementProposal] = []
        for proposal in proposals:
            if proposal.proposal_id in seen:
                continue
            seen.add(proposal.proposal_id)
            unique.append(proposal)
            if len(unique) >= max(1, int(max_proposals)):
                break

        report = {
            "ok": True,
            "improvement_version": IMPROVEMENT_VERSION,
            "proposal_count": len(unique),
            "proposals": [proposal.to_dict() for proposal in unique],
            "benchmark_used": bool(benchmark_report),
            "memory_used": bool(memory_summary),
            "duration_ms": (time.perf_counter() - started) * 1000.0,
            "claim_boundary": "Improvement proposals only; no autonomous repository mutation.",
        }
        return report

    def write_report(self, report: dict[str, Any], out_dir: str | Path = DEFAULT_PROPOSAL_DIR) -> dict[str, str]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        latest = out_dir / "tesseract_improvement_proposals_v1_6_latest.json"
        ledger = out_dir / "tesseract_improvement_proposals_history.jsonl"
        latest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, sort_keys=True) + "\n")
        return {"latest": str(latest), "ledger": str(ledger)}

    def _load_latest_benchmark(self) -> dict[str, Any]:
        path = Path("artifacts") / "tpn" / "benchmarks" / "tesseract_benchmark_v1_4_latest.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _benchmark_proposals(self, report: dict[str, Any]) -> list[TesseractImprovementProposal]:
        proposals: list[TesseractImprovementProposal] = []
        if not report:
            proposals.append(TesseractImprovementProposal(
                proposal_id="improve_benchmark_baseline",
                title="Generate a fresh benchmark baseline",
                rationale="No benchmark report was available for the proposal engine.",
                risk="low",
                expected_impact="Establishes measurable intelligence baseline before new autonomy layers.",
                required_gates=["run compileall", "run full TPN tests", "run benchmark smoke"],
                suggested_next_version="v1.6.1",
                tags=["benchmark", "baseline"],
            ))
            return proposals

        plan_accuracy = float(report.get("plan_accuracy", 0.0) or 0.0)
        safety_score = float(report.get("safety_score", 0.0) or 0.0)
        mean_score = float(report.get("mean_score", 0.0) or 0.0)

        if plan_accuracy < 0.95:
            proposals.append(TesseractImprovementProposal(
                proposal_id="improve_planner_skill_mapping",
                title="Improve planner skill mapping",
                rationale=f"Benchmark plan_accuracy is {plan_accuracy:.3f}; the planner may be missing expected skills.",
                risk="medium",
                expected_impact="Higher objective-to-skill selection accuracy.",
                required_gates=["add failing benchmark fixture", "run planner tests", "run benchmark smoke"],
                suggested_next_version="v1.6.1",
                tags=["planner", "benchmark"],
            ))

        if safety_score < 1.0:
            proposals.append(TesseractImprovementProposal(
                proposal_id="harden_safety_blocks",
                title="Harden safety block correctness",
                rationale=f"Benchmark safety_score is {safety_score:.3f}; safety gates must remain at 1.0 before autonomy expands.",
                risk="high",
                expected_impact="Restores safety invariant before broader action authority.",
                required_gates=["add regression safety test", "run all tests", "run safety benchmark"],
                suggested_next_version="v1.6.1",
                tags=["safety", "gate"],
            ))

        if mean_score >= 0.95 and safety_score == 1.0:
            proposals.append(TesseractImprovementProposal(
                proposal_id="expand_benchmark_suite",
                title="Expand benchmark suite with harder objectives",
                rationale=f"Mean benchmark score is {mean_score:.3f}; current tasks may be too easy.",
                risk="low",
                expected_impact="Prevents benchmark theater by adding harder memory, planning, and blocked-action cases.",
                required_gates=["add at least 5 new benchmark cases", "preserve safety_score 1.0", "record benchmark delta"],
                suggested_next_version="v1.7",
                tags=["benchmark", "evaluation"],
            ))

        return proposals

    def _memory_proposals(self, summary: dict[str, Any]) -> list[TesseractImprovementProposal]:
        proposals: list[TesseractImprovementProposal] = []
        if not summary:
            return proposals

        episode_count = int(summary.get("episode_count", 0) or 0)
        kinds = summary.get("kinds", {}) or {}
        text = json.dumps(summary, sort_keys=True).lower()

        if episode_count < 5:
            proposals.append(TesseractImprovementProposal(
                proposal_id="collect_more_episodes",
                title="Collect more cycle and benchmark episodes",
                rationale=f"Episodic memory contains only {episode_count} episode(s).",
                risk="low",
                expected_impact="Improves evidence quality for future self-improvement proposals.",
                required_gates=["run cycle tests", "run benchmark", "consolidate memory"],
                suggested_next_version="v1.6.1",
                tags=["memory", "evidence"],
            ))

        if "benchmark" not in kinds:
            proposals.append(TesseractImprovementProposal(
                proposal_id="record_benchmark_episodes",
                title="Record benchmark reports into episodic memory",
                rationale="Memory summary does not show benchmark episodes yet.",
                risk="low",
                expected_impact="Links measured intelligence changes to durable memory.",
                required_gates=["add benchmark-to-memory test", "run benchmark smoke", "verify memory search"],
                suggested_next_version="v1.7",
                tags=["memory", "benchmark"],
            ))

        if "fail" in text or "blocked" in text:
            proposals.append(TesseractImprovementProposal(
                proposal_id="close_failed_or_blocked_episodes",
                title="Close failed or blocked episodes first",
                rationale="Memory summary contains failure/blocking language.",
                risk="medium",
                expected_impact="Prevents compounding unresolved wounds.",
                required_gates=["inspect episodic search", "write close script", "run full tests"],
                suggested_next_version="v1.6.1",
                tags=["wound", "close"],
            ))

        return proposals

    def _roadmap_proposals(self, benchmark_report: dict[str, Any], memory_summary: dict[str, Any]) -> list[TesseractImprovementProposal]:
        return [
            TesseractImprovementProposal(
                proposal_id="add_goal_state_manager",
                title="Add goal-state manager",
                rationale="The system can plan, cycle, benchmark, and remember; the next bounded AGI-roadmap layer is explicit goal state.",
                risk="medium",
                expected_impact="Allows objectives to persist across cycles without uncontrolled autonomy.",
                required_gates=["goal schema", "goal ledger", "stop conditions", "full tests"],
                suggested_next_version="v1.7",
                tags=["goal", "roadmap", "autonomy"],
            ),
            TesseractImprovementProposal(
                proposal_id="add_patch_proposal_receipts",
                title="Add patch proposal receipts",
                rationale="Self-improvement should propose patches as artifacts without applying them automatically.",
                risk="medium",
                expected_impact="Moves toward governed self-improvement while preserving human approval.",
                required_gates=["proposal schema", "risk scoring", "human approval flag", "no auto-mutation test"],
                suggested_next_version="v1.7",
                tags=["self-improvement", "governance"],
            ),
        ]


def run_improvement_proposals(*, write: bool = False, out_dir: str | Path = DEFAULT_PROPOSAL_DIR) -> dict[str, Any]:
    engine = TesseractImprovementProposalEngine()
    report = engine.propose()
    if write:
        report["paths"] = engine.write_report(report, out_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--out-dir", default=str(DEFAULT_PROPOSAL_DIR))
    args = parser.parse_args()
    print(json.dumps(run_improvement_proposals(write=args.write, out_dir=args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
