
"""Warm local runtime daemon for the Tesseract mind core.

v0.7 keeps the weighted TPN checkpoint loaded in memory so calls are fast after
startup. It uses only the Python standard library for the HTTP surface.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from neuralforge.tesseract.adaptive import append_operator_feedback
from neuralforge.tesseract.communication import TesseractEnglishAdapter

DEFAULT_CHECKPOINT = Path("artifacts") / "tpn" / "tpn_mind_core_v0_6.pt"
DEFAULT_REPLAY = Path("artifacts") / "tpn" / "replay" / "tpn_replay_v0_6.jsonl"


def _parse_vector(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, str):
        parts = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
        vector = [float(item) for item in parts]
    else:
        vector = [float(item) for item in value]
    if len(vector) != 16:
        raise ValueError(f"TPN vector must contain 16 floats, got {len(vector)}.")
    return vector


class TesseractWarmRuntime:
    """A warm in-memory local runtime around weighted TPN + English receipts."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        *,
        replay_path: str | Path = DEFAULT_REPLAY,
        map_location: str = "cpu",
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.replay_path = Path(replay_path)
        self.started_at = time.perf_counter()
        self.adapter = TesseractEnglishAdapter.from_checkpoint(self.checkpoint, map_location=map_location)
        self.calls = 0

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "runtime": "TesseractWarmRuntime",
            "version": "tpn.v0.7",
            "checkpoint": str(self.checkpoint),
            "replay_path": str(self.replay_path),
            "calls": self.calls,
            "uptime_seconds": time.perf_counter() - self.started_at,
            "claim_boundary": "Warm local TPN runtime; no external model call.",
        }

    def think(self, vector: str | list[float] | tuple[float, ...], *, style: str = "operator") -> dict[str, Any]:
        parsed = _parse_vector(vector)
        t0 = time.perf_counter()
        answer = self.adapter.think_english(parsed, style=style)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.calls += 1
        return {
            "ok": True,
            "version": "tpn.v0.7",
            "latency_ms": elapsed_ms,
            "text": answer["text"],
            "selected_vertices": answer["selected_vertices"],
            "route_ids": answer["route_ids"],
            "receipts": answer["receipts"],
            "claim_boundary": "Local weighted TPN inference; deterministic English; no external call.",
        }

    def feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        required = ["vector", "route", "authority", "evidence", "coherence", "delta_phi", "vertex"]
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError("feedback payload missing required keys: " + ", ".join(missing))
        result = append_operator_feedback(
            self.replay_path,
            vector=_parse_vector(payload["vector"]),
            route=int(payload["route"]),
            authority=int(payload["authority"]),
            evidence=int(payload["evidence"]),
            coherence=float(payload["coherence"]),
            delta_phi=float(payload["delta_phi"]),
            vertex=int(payload["vertex"]),
            axis_scores=payload.get("axis_scores"),
            approved=bool(payload.get("approved", True)),
            note=str(payload.get("note", "")),
        )
        return {
            "ok": True,
            "version": "tpn.v0.7",
            "feedback": result,
            "claim_boundary": "Feedback is appended to local replay only; it does not mutate weights until replay training runs.",
        }


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(runtime: TesseractWarmRuntime):
    class TesseractHandler(BaseHTTPRequestHandler):
        server_version = "TesseractWarmRuntime/0.7"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                _json_response(self, 200, runtime.health())
                return
            _json_response(self, 404, {"ok": False, "error": f"unknown endpoint: {path}"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(raw)
                if path == "/think":
                    result = runtime.think(payload.get("vector", []), style=str(payload.get("style", "operator")))
                    _json_response(self, 200, result)
                    return
                if path == "/feedback":
                    result = runtime.feedback(payload)
                    _json_response(self, 200, result)
                    return
                _json_response(self, 404, {"ok": False, "error": f"unknown endpoint: {path}"})
            except Exception as exc:
                _json_response(self, 400, {"ok": False, "error": str(exc)})

    return TesseractHandler


def run_server(
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    replay_path: str | Path = DEFAULT_REPLAY,
    host: str = "127.0.0.1",
    port: int = 8765,
    map_location: str = "cpu",
) -> None:
    runtime = TesseractWarmRuntime(checkpoint, replay_path=replay_path, map_location=map_location)
    server = ThreadingHTTPServer((host, int(port)), make_handler(runtime))
    print(json.dumps({
        "ok": True,
        "runtime": "TesseractWarmRuntime",
        "version": "tpn.v0.7",
        "url": f"http://{host}:{port}",
        "checkpoint": str(checkpoint),
        "endpoints": ["/health", "/think", "/feedback"],
        "claim_boundary": "Local daemon only. Stop with Ctrl+C.",
    }, indent=2, sort_keys=True))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTPN v0.7 daemon stopped.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--replay-path", default=str(DEFAULT_REPLAY))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--vector", default="0.5,0.5,0.5,0.5,0,0,0,0,0,0.5,0.1,0,1,0,0,0")
    parser.add_argument("--style", default="operator")
    args = parser.parse_args()

    if args.once:
        runtime = TesseractWarmRuntime(args.checkpoint, replay_path=args.replay_path, map_location=args.map_location)
        print(json.dumps(runtime.think(args.vector, style=args.style), indent=2, sort_keys=True))
        return

    run_server(
        checkpoint=args.checkpoint,
        replay_path=args.replay_path,
        host=args.host,
        port=args.port,
        map_location=args.map_location,
    )


if __name__ == "__main__":
    main()
