"""Performance telemetry governor for the Tesseract Jarvis roadmap.

v1.11 measures the speed of policy-approved bounded execution. It records
policy, queue, goal-cycle, and skill-level latency receipts, then evaluates
them against explicit thresholds.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuralforge.tesseract.goal_cycle import TesseractGoalAwareCycleRunner
from neuralforge.tesseract.goal_queue import TesseractGoalQueueRunner
from neuralforge.tesseract.goal_state import DEFAULT_GOAL_EVENTS_PATH, DEFAULT_GOAL_STATE_PATH, TesseractGoalStateManager
from neuralforge.tesseract.policy import DEFAULT_POLICY_CONFIG_PATH, TesseractExecutionPolicyGovernor


PERFORMANCE_VERSION = "tpn.performance.v1.11"
DEFAULT_PERFORMANCE_REPORT_PATH = Path("artifacts") / "tpn" / "performance_report_v1_11_latest.json"
DEFAULT_PERFORMANCE_HISTORY_PATH = Path("artifacts") / "tpn" / "performance_history_v1_11.jsonl"

DEFAULT_PERFORMANCE_THRESHOLDS = {
    "max_total_duration_ms": 2000.0,
    "max_queue_duration_ms": 1000.0,
    "max_goal_cycle_duration_ms": 750.0,
    "max_skill_duration_ms": 500.0,
    "max_mean_skill_duration_ms": 250.0,
    "min_completed_goals": 1,
}


@dataclass
class TesseractSkillLatency:
    goal_id: str
    skill_id: str
    duration_ms: float
    ok: bool
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TesseractPerformanceReport:
    ok: bool
    performance_ok: bool
    total_duration_ms: float
    queue_duration_ms: float
    goal_cycle_durations_ms: list[float]
    skill_latencies: list[dict[str, Any]]
    warnings: list[str]
    thresholds: dict[str, Any]
    queue_result_present: bool
    policy_allowed: bool
    completed_goal_count: int
    blocked_goal_count: int
    created_at_unix: float = field(default_factory=time.time)
    performance_version: str = PERFORMANCE_VERSION
    claim_boundary: str = "Performance telemetry only; no autonomous authority."

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["summary"] = {
            "goal_cycle_count": len(self.goal_cycle_durations_ms),
            "skill_count": len(self.skill_latencies),
            "max_goal_cycle_duration_ms": max(self.goal_cycle_durations_ms) if self.goal_cycle_durations_ms else 0.0,
            "max_skill_duration_ms": max([s["duration_ms"] for s in self.skill_latencies]) if self.skill_latencies else 0.0,
            "mean_skill_duration_ms": statistics.fmean([s["duration_ms"] for s in self.skill_latencies]) if self.skill_latencies else 0.0,
        }
        return data


class TesseractPerformanceTelemetryGovernor:
    def __init__(
        self,
        *,
        policy_governor: TesseractExecutionPolicyGovernor | None = None,
        thresholds: dict[str, Any] | None = None,
    ) -> None:
        self.policy_governor = policy_governor or TesseractExecutionPolicyGovernor()
        self.thresholds = dict(DEFAULT_PERFORMANCE_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

    def run_probe(
        self,
        *,
        max_goals: int = 2,
        max_steps: int = 4,
        execute: bool = True,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        guarded = self.policy_governor.guarded_run(max_goals=max_goals, max_steps=max_steps, execute=execute)
        total_ms = (time.perf_counter() - started) * 1000.0
        telemetry = self.collect_from_guarded_report(guarded, total_duration_ms=total_ms)
        telemetry["guarded_result"] = guarded
        telemetry["paths"] = self.write_report(telemetry)
        return telemetry

    def collect_from_guarded_report(self, guarded: dict[str, Any], *, total_duration_ms: float | None = None) -> dict[str, Any]:
        total_ms = float(total_duration_ms if total_duration_ms is not None else 0.0)
        queue_result = guarded.get("queue_result") or {}
        queue_report = queue_result.get("report") or {}
        queue_ms = float(queue_report.get("duration_ms", 0.0) or 0.0)
        results = queue_report.get("results", []) or []

        goal_cycle_durations: list[float] = []
        skill_latencies: list[dict[str, Any]] = []

        for result in results:
            report = result.get("report") or {}
            goal_id = str(report.get("goal_id", ""))
            goal_cycle_durations.append(float(report.get("duration_ms", 0.0) or 0.0))
            observations = result.get("cycle_answer", {}).get("cycle", {}).get("observations", []) or []
            for obs in observations:
                skill_latencies.append(
                    TesseractSkillLatency(
                        goal_id=goal_id,
                        skill_id=str(obs.get("skill_id", "")),
                        duration_ms=float(obs.get("duration_ms", 0.0) or 0.0),
                        ok=bool(obs.get("ok", False)),
                        summary=str(obs.get("summary", "")),
                    ).to_dict()
                )

        completed = list(queue_report.get("completed_goal_ids", []) or [])
        blocked = list(queue_report.get("blocked_goal_ids", []) or [])
        policy_allowed = bool((guarded.get("policy_decision") or {}).get("allowed", False))
        ok = bool(guarded.get("ok", False))
        warnings = self.evaluate_warnings(
            total_duration_ms=total_ms,
            queue_duration_ms=queue_ms,
            goal_cycle_durations_ms=goal_cycle_durations,
            skill_latencies=skill_latencies,
            completed_goal_count=len(completed),
            blocked_goal_count=len(blocked),
            policy_allowed=policy_allowed,
            queue_result_present=bool(queue_result),
        )
        report = TesseractPerformanceReport(
            ok=ok,
            performance_ok=not warnings and ok,
            total_duration_ms=total_ms,
            queue_duration_ms=queue_ms,
            goal_cycle_durations_ms=goal_cycle_durations,
            skill_latencies=skill_latencies,
            warnings=warnings,
            thresholds=dict(self.thresholds),
            queue_result_present=bool(queue_result),
            policy_allowed=policy_allowed,
            completed_goal_count=len(completed),
            blocked_goal_count=len(blocked),
        ).to_dict()
        return report

    def evaluate_warnings(
        self,
        *,
        total_duration_ms: float,
        queue_duration_ms: float,
        goal_cycle_durations_ms: list[float],
        skill_latencies: list[dict[str, Any]],
        completed_goal_count: int,
        blocked_goal_count: int,
        policy_allowed: bool,
        queue_result_present: bool,
    ) -> list[str]:
        warnings: list[str] = []
        t = self.thresholds
        if not policy_allowed:
            warnings.append("Policy did not allow execution; no performance run occurred.")
        if not queue_result_present:
            warnings.append("Queue result missing.")
        if total_duration_ms > float(t["max_total_duration_ms"]):
            warnings.append(f"Total duration {total_duration_ms:.3f} ms exceeded threshold.")
        if queue_duration_ms > float(t["max_queue_duration_ms"]):
            warnings.append(f"Queue duration {queue_duration_ms:.3f} ms exceeded threshold.")
        for idx, duration in enumerate(goal_cycle_durations_ms, start=1):
            if duration > float(t["max_goal_cycle_duration_ms"]):
                warnings.append(f"Goal cycle {idx} duration {duration:.3f} ms exceeded threshold.")
        skill_values = [float(s.get("duration_ms", 0.0) or 0.0) for s in skill_latencies]
        for skill in skill_latencies:
            if float(skill.get("duration_ms", 0.0) or 0.0) > float(t["max_skill_duration_ms"]):
                warnings.append(
                    f"Skill {skill.get('skill_id')} for {skill.get('goal_id')} took {float(skill.get('duration_ms', 0.0)):.3f} ms."
                )
        if skill_values and statistics.fmean(skill_values) > float(t["max_mean_skill_duration_ms"]):
            warnings.append("Mean skill duration exceeded threshold.")
        if completed_goal_count < int(t["min_completed_goals"]):
            warnings.append("Completed goal count below threshold.")
        if blocked_goal_count:
            warnings.append("One or more goals blocked during performance probe.")
        return warnings

    def write_report(
        self,
        report: dict[str, Any],
        *,
        latest_path: str | Path = DEFAULT_PERFORMANCE_REPORT_PATH,
        history_path: str | Path = DEFAULT_PERFORMANCE_HISTORY_PATH,
    ) -> dict[str, str]:
        latest_path = Path(latest_path)
        history_path = Path(history_path)
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, sort_keys=True) + "\n")
        return {"latest": str(latest_path), "history": str(history_path)}

    def demo(self) -> dict[str, Any]:
        manager = self.policy_governor.goal_manager
        manager.create_goal(
            "Performance goal: repo status",
            "check repo status and recent git log",
            success_criteria=["repo.status", "repo.log"],
            stop_conditions=["safety_score below 1.0", "autonomous mutation requested"],
            risk="medium",
            max_cycles=1,
        )
        manager.create_goal(
            "Performance goal: README read",
            "read README.md",
            success_criteria=["file.read"],
            stop_conditions=["path escapes repo root", "autonomous mutation requested"],
            risk="low",
            max_cycles=1,
        )
        return self.run_probe(max_goals=2, max_steps=4, execute=True)


def build_default_governor(
    *,
    state_path: str | Path = DEFAULT_GOAL_STATE_PATH,
    events_path: str | Path = DEFAULT_GOAL_EVENTS_PATH,
    policy_path: str | Path = DEFAULT_POLICY_CONFIG_PATH,
) -> TesseractPerformanceTelemetryGovernor:
    manager = TesseractGoalStateManager(state_path, events_path)
    cycle = TesseractGoalAwareCycleRunner(goal_manager=manager)
    queue = TesseractGoalQueueRunner(goal_manager=manager, cycle_runner=cycle)
    policy = TesseractExecutionPolicyGovernor(policy_path=policy_path, goal_manager=manager, queue_runner=queue)
    return TesseractPerformanceTelemetryGovernor(policy_governor=policy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", default=str(DEFAULT_GOAL_STATE_PATH))
    parser.add_argument("--events-path", default=str(DEFAULT_GOAL_EVENTS_PATH))
    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_CONFIG_PATH))
    parser.add_argument("--max-goals", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    governor = build_default_governor(
        state_path=args.state_path,
        events_path=args.events_path,
        policy_path=args.policy_path,
    )
    if args.demo:
        print(json.dumps(governor.demo(), indent=2, sort_keys=True))
        return
    if args.probe:
        print(json.dumps(governor.run_probe(max_goals=args.max_goals, max_steps=args.max_steps), indent=2, sort_keys=True))
        return
    print(json.dumps({"ok": True, "performance_version": PERFORMANCE_VERSION, "thresholds": governor.thresholds}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
