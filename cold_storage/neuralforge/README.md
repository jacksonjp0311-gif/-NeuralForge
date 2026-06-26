# NeuralForge Cold Storage

This directory holds persistent execution event data for NeuralForge's
real-time evolution engine.

## Files

- `execution_events.jsonl` — Append-only ledger of AGNT workflow/tool execution
  events. Each line is a JSON object conforming to the NeuralForge event schema.

## Event Schema

```json
{
  "workflow_id": "wf_001",
  "workflow_name": "Test Workflow",
  "tool_name": "web_search",
  "status": "success",
  "success": true,
  "duration_ms": 150.5,
  "step_count": 5,
  "retry_count": 0,
  "error_type": "none",
  "recovery_action": "",
  "recovery_success": false,
  "params": {"query": "AI news"},
  "prompt": "search for AI news",
  "response": "found 5 results",
  "timestamp": 1700000000.0
}
```

## Usage

```python
from neuralforge.agnt_bridge import record_execution, rehydrate_from_log

# Record a new execution
result = record_execution({
    "workflow_id": "wf_001",
    "status": "success",
    "duration_ms": 150.5,
})

# Rehydrate from existing log
count = rehydrate_from_log("cold_storage/neuralforge/execution_events.jsonl")
```

## Integration

The AGNT bridge (`neuralforge/agnt_bridge.py`) automatically resolves to this
path via the `NEURALFORGE_COLD_STORAGE` environment variable or the default
`cold_storage/` directory relative to the repo root.
