"""External source intake governance for the Tesseract Jarvis roadmap.

v1.16 registers telemetry/API/scraping source candidates and emits provenance,
rate-limit, and compliance receipts. It does not perform network calls, scrape
websites, pull APIs, collect raw data, or grant autonomous write authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


SOURCE_INTAKE_VERSION = "tpn.source_intake.v1.16"
DEFAULT_SOURCE_INTAKE_REPORT_PATH = Path("artifacts") / "tpn" / "source_intake_report_v1_16_latest.json"
DEFAULT_SOURCE_INTAKE_HISTORY_PATH = Path("artifacts") / "tpn" / "source_intake_history_v1_16.jsonl"

SourceType = Literal["api", "scrape", "telemetry", "file", "manual"]


@dataclass
class TesseractSourceSpec:
    source_id: str
    source_type: SourceType
    purpose: str
    endpoint_or_location: str = ""
    allowed: bool = False
    compliance_note: str = ""
    provenance_required: bool = True
    rate_limit_per_minute: int = 0
    requires_auth: bool = False
    contains_personal_data: bool = False
    raw_collection_allowed: bool = False
    created_at_unix: float = field(default_factory=time.time)
    source_intake_version: str = SOURCE_INTAKE_VERSION
    claim_boundary: str = "Source registry entry only; no live network pull."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TesseractSourceIntakeReceipt:
    ok: bool
    intake_id: str
    source_count: int
    allowed_source_count: int
    blocked_source_count: int
    approved_sources: list[dict[str, Any]]
    blocked_sources: list[dict[str, Any]]
    planned_steps: list[dict[str, Any]]
    required_tests: list[str]
    blocked_reasons: list[str] = field(default_factory=list)
    registry_allowed: bool = True
    live_pull_allowed: bool = False
    scraping_allowed: bool = False
    raw_collection_allowed: bool = False
    mutation_allowed: bool = False
    created_at_unix: float = field(default_factory=time.time)
    source_intake_version: str = SOURCE_INTAKE_VERSION
    claim_boundary: str = "External source intake registry only; no network calls, scraping, raw collection, or mutation."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TesseractSourceIntakeGovernor:
    def __init__(
        self,
        *,
        latest_path: str | Path = DEFAULT_SOURCE_INTAKE_REPORT_PATH,
        history_path: str | Path = DEFAULT_SOURCE_INTAKE_HISTORY_PATH,
    ) -> None:
        self.latest_path = Path(latest_path)
        self.history_path = Path(history_path)

    def demo_sources(self) -> list[TesseractSourceSpec]:
        return [
            TesseractSourceSpec(
                source_id="local_tpn_runtime_telemetry",
                source_type="telemetry",
                purpose="Read local TPN runtime receipts for drift/performance context.",
                endpoint_or_location="artifacts/tpn/*.json",
                allowed=True,
                compliance_note="Local project-generated telemetry only.",
                rate_limit_per_minute=0,
                raw_collection_allowed=False,
            ),
            TesseractSourceSpec(
                source_id="github_repo_metadata_api",
                source_type="api",
                purpose="Plan future GitHub repository metadata pulls through authenticated API policy.",
                endpoint_or_location="https://api.github.com/repos/jacksonjp0311-gif/-NeuralForge",
                allowed=True,
                compliance_note="API-only candidate; respect authentication, rate limits, and platform terms before live pull.",
                rate_limit_per_minute=30,
                requires_auth=False,
                raw_collection_allowed=False,
            ),
            TesseractSourceSpec(
                source_id="public_docs_scrape_candidate",
                source_type="scrape",
                purpose="Candidate public documentation crawl, blocked until explicit robots/terms review receipt exists.",
                endpoint_or_location="https://example.com/docs",
                allowed=False,
                compliance_note="Blocked example: scraping requires explicit source compliance receipt before live use.",
                rate_limit_per_minute=1,
                raw_collection_allowed=False,
            ),
        ]

    def build_receipt(self, sources: list[TesseractSourceSpec]) -> dict[str, Any]:
        approved: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        all_reasons: list[str] = []

        for source in sources:
            reasons = self.validate_source(source)
            item = source.to_dict()
            item["blocked_reasons"] = reasons
            if reasons:
                blocked.append(item)
                all_reasons.extend([f"{source.source_id}: {reason}" for reason in reasons])
            else:
                approved.append(item)

        receipt = TesseractSourceIntakeReceipt(
            ok=True,
            intake_id=self.intake_id(sources),
            source_count=len(sources),
            allowed_source_count=len(approved),
            blocked_source_count=len(blocked),
            approved_sources=approved,
            blocked_sources=blocked,
            blocked_reasons=all_reasons,
            planned_steps=self.planned_steps(approved),
            required_tests=self.required_tests(),
            registry_allowed=True,
            live_pull_allowed=False,
            scraping_allowed=False,
            raw_collection_allowed=False,
            mutation_allowed=False,
        ).to_dict()
        self.write_report(receipt)
        return receipt

    def validate_source(self, source: TesseractSourceSpec) -> list[str]:
        reasons: list[str] = []
        if not source.source_id.strip():
            reasons.append("source_id is required")
        if source.source_type not in {"api", "scrape", "telemetry", "file", "manual"}:
            reasons.append("unsupported source_type")
        if not source.purpose.strip():
            reasons.append("purpose is required")
        if not source.allowed:
            reasons.append("source is not allowed by registry policy")
        if source.source_type in {"api", "scrape"} and not source.compliance_note.strip():
            reasons.append("api/scrape source requires compliance_note")
        if source.source_type == "scrape" and source.allowed:
            reasons.append("scrape sources require a future dedicated compliance receipt before being allowed")
        if source.rate_limit_per_minute < 0:
            reasons.append("rate_limit_per_minute cannot be negative")
        if source.contains_personal_data:
            reasons.append("sources containing personal data require a future privacy/redaction receipt")
        if source.raw_collection_allowed:
            reasons.append("raw_collection_allowed must remain false in v1.16")
        return reasons

    def planned_steps(self, approved_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "step_id": "step_01",
                "title": "register approved source candidates",
                "status": "planned_not_executed",
                "source_ids": [source.get("source_id", "") for source in approved_sources],
            },
            {
                "step_id": "step_02",
                "title": "attach provenance and compliance receipts",
                "status": "planned_not_executed",
                "detail": "Future layer must attach per-source provenance/compliance before any live pull.",
            },
            {
                "step_id": "step_03",
                "title": "build dry-run pull adapters",
                "status": "planned_not_executed",
                "detail": "Future layer may simulate pulls without network calls.",
            },
            {
                "step_id": "step_04",
                "title": "gate live API pulls behind policy",
                "status": "planned_not_executed",
                "detail": "Live API/scrape execution remains blocked in v1.16.",
            },
        ]

    def required_tests(self) -> list[str]:
        return [
            "python -m compileall neuralforge tests examples",
            "python -m pytest tests/test_tesseract_v1_16_source_intake.py -q",
            "powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\run_tesseract_source_intake.ps1",
            "powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\check_tesseract_contract.ps1 -Base http://127.0.0.1:9",
        ]

    def intake_id(self, sources: list[TesseractSourceSpec]) -> str:
        seed = json.dumps([source.to_dict() for source in sources], sort_keys=True)
        return "intake_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

    def write_report(self, receipt: dict[str, Any]) -> dict[str, str]:
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.latest_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(receipt, sort_keys=True) + "\n")
        return {"latest": str(self.latest_path), "history": str(self.history_path)}

    def format_summary(self, receipt: dict[str, Any]) -> str:
        lines = [
            "SOURCE INTAKE GOVERNOR SUMMARY",
            f"- ok: {receipt.get('ok', False)}",
            f"- registry_allowed: {receipt.get('registry_allowed', False)}",
            f"- source_count: {receipt.get('source_count', 0)}",
            f"- approved_sources: {receipt.get('allowed_source_count', 0)}",
            f"- blocked_sources: {receipt.get('blocked_source_count', 0)}",
            f"- planned_steps: {len(receipt.get('planned_steps', []) or [])}",
            f"- live_pull_allowed: {receipt.get('live_pull_allowed', False)}",
            f"- scraping_allowed: {receipt.get('scraping_allowed', False)}",
            f"- raw_collection_allowed: {receipt.get('raw_collection_allowed', False)}",
            f"- mutation_allowed: {receipt.get('mutation_allowed', False)}",
            "- boundary: source registry only; no network calls, scraping, raw collection, or mutation",
        ]
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-source-file", default="")
    args = parser.parse_args()

    governor = TesseractSourceIntakeGovernor()
    if args.json_source_file:
        raw = json.loads(Path(args.json_source_file).read_text(encoding="utf-8"))
        sources = [TesseractSourceSpec(**item) for item in raw]
    else:
        sources = governor.demo_sources()

    receipt = governor.build_receipt(sources)
    print(governor.format_summary(receipt))


if __name__ == "__main__":
    main()
