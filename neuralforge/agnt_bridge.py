"""
NeuralForge AGNT Runtime Bridge — Priority A

Wires RealtimeEvolutionEngine into AGNT's workflow/tool execution pipeline.
After every execution, call:

    from neuralforge.agnt_bridge import record_execution
    result = record_execution(event)

Event schema (all fields optional — sensible defaults applied):
- workflow_id: str
- workflow_name: str
- tool_name / action: str
- status: str ("success", "error", "failed", etc.)
- success: bool
- duration_ms: float
- step_count: int
- retry_count: int
- error_type: str
- recovery_action: str
- recovery_success: bool
- params: dict
- prompt / input: str (truncated for safety)
- response / output: str (truncated for safety)
- timestamp: float (unix epoch)
"""
from __future__ import annotations
import logging, os, time
from typing import Dict, Any, Optional
from pathlib import Path

from neuralforge.realtime_evo import RealtimeEvolutionEngine

logger = logging.getLogger("neuralforge.agnt_bridge")

# ── Singleton engine (lazy init) ──
_engine: Optional[RealtimeEvolutionEngine] = None
_event_log_path: Optional[Path] = None

# Safety limits for prompt/response capture
_MAX_TEXT_LEN = 2000  # truncate long text fields


def _resolve_event_log_path() -> Path:
    """Resolve the cold-storage event log path (Priority B)."""
    global _event_log_path
    if _event_log_path is not None:
        return _event_log_path

    # Priority: env var → cold_storage standard → repo-local fallback
    env_path = os.environ.get("NEURALFORGE_EVENT_LOG")
    if env_path:
        _event_log_path = Path(env_path)
    else:
        # AGNT-standard cold-storage path
        cold_storage = Path(os.environ.get("NEURALFORGE_COLD_STORAGE", "cold_storage"))
        _event_log_path = cold_storage / "neuralforge" / "execution_events.jsonl"

    _event_log_path.parent.mkdir(parents=True, exist_ok=True)
    return _event_log_path


def get_engine(
    window_size: int = 50,
    alert_threshold: float = 0.3,
    load_existing: bool = True,
) -> RealtimeEvolutionEngine:
    """Get or create the singleton RealtimeEvolutionEngine."""
    global _engine
    if _engine is None:
        path = _resolve_event_log_path()
        _engine = RealtimeEvolutionEngine(
            window_size=window_size,
            alert_threshold=alert_threshold,
            event_log_path=str(path),
            load_existing=load_existing,
        )
        logger.info("AGNT bridge engine initialized -- event log: %s", path)
    return _engine


def _sanitize_text(value: Any, max_len: int = _MAX_TEXT_LEN) -> str:
    """Safely truncate text fields to prevent log bloat."""
    if value is None:
        return ""
    text = str(value)
    if len(text) > max_len:
        return text[:max_len] + f"...[truncated {len(text)} chars]"
    return text


def normalize_agnt_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a raw AGNT execution event into the full realtime schema.

    Handles the complete event schema from the evolution plan:
    workflow_id, workflow_name, tool_name/action, status, success,
    duration_ms, step_count, retry_count, error_type, recovery_action,
    recovery_success, params, prompt/input, response/output, timestamp.
    """
    normalized = dict(event)

    # -- Identifiers --
    normalized["workflow_id"] = str(
        event.get("workflow_id")
        or event.get("workflowId")
        or event.get("workflow_name")
        or event.get("workflowName")
        or event.get("id")
        or "unknown"
    )
    normalized["workflow_name"] = str(
        event.get("workflow_name")
        or event.get("workflowName")
        or normalized["workflow_id"]
    )

    # -- Tool / action --
    normalized["tool_name"] = str(
        event.get("tool_name")
        or event.get("toolName")
        or event.get("tool")
        or event.get("action")
        or event.get("node_type")
        or ""
    )

    # -- Status & success --
    status = str(event.get("status", "")).lower()
    normalized["status"] = status
    if "success" in event:
        normalized["success"] = bool(event["success"])
    else:
        normalized["success"] = status not in {"error", "failed", "failure", "timeout", "cancelled"}

    # -- Numeric fields --
    normalized["duration_ms"] = float(
        event.get("duration_ms")
        or event.get("duration")
        or event.get("elapsed_ms")
        or event.get("latency_ms")
        or 0.0
    )
    normalized["step_count"] = int(
        event.get("step_count")
        or event.get("steps")
        or event.get("node_count")
        or len(event.get("nodes", []) or [])
        or 0
    )
    normalized["retry_count"] = int(
        event.get("retry_count")
        or event.get("retries")
        or event.get("attempt")
        or event.get("attempts")
        or 0
    )

    # -- Error info --
    if not event.get("error_type") and not normalized["success"]:
        normalized["error_type"] = (
            event.get("error")
            or event.get("exception_type")
            or event.get("errorType")
            or status
            or "unknown"
        )
    elif event.get("error_type"):
        normalized["error_type"] = str(event["error_type"])

    # -- Recovery info --
    normalized["recovery_action"] = str(
        event.get("recovery_action")
        or event.get("recoveryAction")
        or event.get("fix_action")
        or ""
    )
    if "recovery_success" in event:
        normalized["recovery_success"] = bool(event["recovery_success"])
    elif "resolved" in event:
        normalized["recovery_success"] = bool(event["resolved"])

    # -- Params --
    params = event.get("params") or event.get("parameters") or event.get("tool_params") or {}
    normalized["params"] = params if isinstance(params, dict) else {}

    # -- Text fields (sanitized) --
    normalized["prompt"] = _sanitize_text(
        event.get("prompt") or event.get("input") or event.get("query")
    )
    normalized["response"] = _sanitize_text(
        event.get("response") or event.get("output") or event.get("result")
    )

    # -- Timestamp --
    normalized["timestamp"] = event.get("timestamp") or event.get("ts") or time.time()

    return normalized


def record_execution(event: Dict[str, Any], persist: bool = True) -> Dict[str, Any]:
    """
    Main hook for AGNT runtime. Call this after every workflow/tool execution.

    Args:
        event: Raw execution event from AGNT runtime.
        persist: Whether to append to the JSONL event log.

    Returns:
        Analysis result dict with health_score, alerts, predictions, recommendations.
    """
    engine = get_engine()
    normalized = normalize_agnt_event(event)
    result = engine.ingest_execution(normalized, persist=persist)
    result["normalized_event"] = {
        "workflow_id": normalized["workflow_id"],
        "workflow_name": normalized["workflow_name"],
        "tool_name": normalized["tool_name"],
        "success": normalized["success"],
        "duration_ms": normalized["duration_ms"],
        "error_type": normalized.get("error_type", ""),
    }
    return result


def get_health_summary() -> Dict[str, Any]:
    """Get current system health summary."""
    engine = get_engine()
    return engine.get_knowledge_summary()


def rehydrate_from_log(path: Optional[str] = None) -> int:
    """
    Rehydrate engine state from an existing JSONL event log.
    Returns the number of events loaded.
    """
    engine = get_engine(load_existing=False)
    log_path = Path(path) if path else _resolve_event_log_path()
    if not log_path.exists():
        return 0
    events = RealtimeEvolutionEngine.load_event_log(log_path)
    if events:
        engine.process_batch(events, persist=False)
    return len(events)
