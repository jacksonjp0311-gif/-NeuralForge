"""Episodic memory core for the Tesseract Jarvis runtime.

v1.5 records bounded local experiences as durable JSONL episodes:
objectives, plans, observations, benchmark summaries, and recommendations.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MEMORY_CORE_VERSION = "tpn.memory.v1.5"
DEFAULT_EPISODE_PATH = Path("artifacts") / "tpn" / "episodic_memory_v1_5.jsonl"
DEFAULT_SUMMARY_PATH = Path("artifacts") / "tpn" / "episodic_memory_summary_v1_5.json"


@dataclass
class TesseractMemoryEpisode:
    kind: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_at_unix: float = field(default_factory=time.time)
    memory_version: str = MEMORY_CORE_VERSION
    claim_boundary: str = "Local episodic memory record; not consciousness."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TesseractEpisodicMemory:
    def __init__(self, path: str | Path = DEFAULT_EPISODE_PATH) -> None:
        self.path = Path(path)

    def append(self, kind: str, summary: str, payload: dict[str, Any] | None = None, tags: list[str] | None = None) -> dict[str, Any]:
        episode = TesseractMemoryEpisode(kind=kind, summary=summary, payload=payload or {}, tags=tags or [])
        record = episode.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"decode_error": True, "raw": line})
        return out

    def recent(self, limit: int = 10) -> dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        rows = self.rows()
        return {
            "ok": True,
            "memory_version": MEMORY_CORE_VERSION,
            "count": len(rows),
            "episodes": rows[-limit:],
            "path": str(self.path),
            "claim_boundary": "Local episodic memory retrieval only.",
        }

    def search(self, query: str, limit: int = 10) -> dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        q = query.lower().strip()
        rows = self.rows()
        hits = [row for row in rows if q in json.dumps(row, sort_keys=True).lower()]
        return {
            "ok": True,
            "memory_version": MEMORY_CORE_VERSION,
            "query": query,
            "count": len(hits),
            "episodes": hits[-limit:],
            "path": str(self.path),
            "claim_boundary": "Local episodic memory search only.",
        }

    def consolidate(self, out_path: str | Path = DEFAULT_SUMMARY_PATH) -> dict[str, Any]:
        rows = self.rows()
        kinds: dict[str, int] = {}
        tags: dict[str, int] = {}
        for row in rows:
            kind = str(row.get("kind", "unknown"))
            kinds[kind] = kinds.get(kind, 0) + 1
            for tag in row.get("tags", []) or []:
                tag = str(tag)
                tags[tag] = tags.get(tag, 0) + 1

        recent_summaries = [str(row.get("summary", "")) for row in rows[-10:]]
        summary = {
            "ok": True,
            "memory_version": MEMORY_CORE_VERSION,
            "episode_count": len(rows),
            "kinds": kinds,
            "tags": tags,
            "recent_summaries": recent_summaries,
            "next_recommendation": self._recommend(rows),
            "claim_boundary": "Compressed local memory summary; not autonomous self-awareness.",
        }
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary

    def _recommend(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No episodes yet. Run a bounded cycle or benchmark before drawing memory conclusions."
        failures = [row for row in rows if "fail" in json.dumps(row, sort_keys=True).lower() or "blocked" in json.dumps(row, sort_keys=True).lower()]
        if failures:
            return "Review blocked or failed episodes before adding broader autonomy."
        if len(rows) < 5:
            return "Collect more cycle and benchmark episodes before tuning behavior."
        return "Memory is accumulating; compare recent recommendations against benchmark scores before evolving."
