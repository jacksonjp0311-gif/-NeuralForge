"""Test the Evolution Engine — AGNT's self-improvement brain."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuralforge.evo_engine import EvolutionEngine

# Simulate 30 workflow executions across 3 workflows with realistic patterns
executions = [
    # Workflow 1: Email Pipeline — mostly healthy, slight degradation
    {"workflow_id": "email-pipeline", "workflow_name": "Email Pipeline", "duration_ms": 1200, "success": True, "step_count": 5, "timestamp": "2026-06-22T08:00:00Z"},
    {"workflow_id": "email-pipeline", "workflow_name": "Email Pipeline", "duration_ms": 1150, "success": True, "step_count": 5, "timestamp": "2026-06-22T09:00:00Z"},
    {"workflow_id": "email-pipeline", "workflow_name": "Email Pipeline", "duration_ms": 1300, "success": True, "step_count": 5, "timestamp": "2026-06-22T10:00:00Z"},
    {"workflow_id": "email-pipeline", "workflow_name": "Email Pipeline", "duration_ms": 1250, "success": True, "step_count": 5, "timestamp": "2026-06-22T11:00:00Z"},
    {"workflow_id": "email-pipeline", "workflow_name": "Email Pipeline", "duration_ms": 1400, "success": True, "step_count": 6, "timestamp": "2026-06-22T12:00:00Z"},
    {"workflow_id": "email-pipeline", "workflow_name": "Email Pipeline", "duration_ms": 1350, "success": True, "step_count": 5, "timestamp": "2026-06-22T13:00:00Z"},
    {"workflow_id": "email-pipeline", "workflow_name": "Email Pipeline", "duration_ms": 1500, "success": True, "step_count": 6, "timestamp": "2026-06-22T14:00:00Z"},
    {"workflow_id": "email-pipeline", "workflow_name": "Email Pipeline", "duration_ms": 1450, "success": True, "step_count": 5, "timestamp": "2026-06-22T15:00:00Z"},
    {"workflow_id": "email-pipeline", "workflow_name": "Email Pipeline", "duration_ms": 1600, "success": True, "step_count": 6, "timestamp": "2026-06-22T16:00:00Z"},
    {"workflow_id": "email-pipeline", "workflow_name": "Email Pipeline", "duration_ms": 1550, "success": True, "step_count": 5, "timestamp": "2026-06-22T17:00:00Z"},

    # Workflow 2: Data Sync — intermittent failures, increasing duration
    {"workflow_id": "data-sync", "workflow_name": "Data Sync", "duration_ms": 2000, "success": True, "step_count": 8, "timestamp": "2026-06-22T08:00:00Z"},
    {"workflow_id": "data-sync", "workflow_name": "Data Sync", "duration_ms": 2200, "success": True, "step_count": 8, "timestamp": "2026-06-22T09:00:00Z"},
    {"workflow_id": "data-sync", "workflow_name": "Data Sync", "duration_ms": 3500, "success": False, "step_count": 3, "error_type": "timeout", "timestamp": "2026-06-22T10:00:00Z"},
    {"workflow_id": "data-sync", "workflow_name": "Data Sync", "duration_ms": 2400, "success": True, "step_count": 8, "timestamp": "2026-06-22T11:00:00Z"},
    {"workflow_id": "data-sync", "workflow_name": "Data Sync", "duration_ms": 2600, "success": True, "step_count": 9, "timestamp": "2026-06-22T12:00:00Z"},
    {"workflow_id": "data-sync", "workflow_name": "Data Sync", "duration_ms": 4200, "success": False, "step_count": 2, "error_type": "oom", "timestamp": "2026-06-22T13:00:00Z"},
    {"workflow_id": "data-sync", "workflow_name": "Data Sync", "duration_ms": 2800, "success": True, "step_count": 9, "timestamp": "2026-06-22T14:00:00Z"},
    {"workflow_id": "data-sync", "workflow_name": "Data Sync", "duration_ms": 3000, "success": True, "step_count": 10, "timestamp": "2026-06-22T15:00:00Z"},
    {"workflow_id": "data-sync", "workflow_name": "Data Sync", "duration_ms": 3200, "success": True, "step_count": 10, "timestamp": "2026-06-22T16:00:00Z"},
    {"workflow_id": "data-sync", "workflow_name": "Data Sync", "duration_ms": 3400, "success": True, "step_count": 11, "timestamp": "2026-06-22T17:00:00Z"},

    # Workflow 3: Report Gen — mostly failing, needs attention
    {"workflow_id": "report-gen", "workflow_name": "Report Generator", "duration_ms": 5000, "success": False, "step_count": 2, "error_type": "timeout", "timestamp": "2026-06-22T08:00:00Z"},
    {"workflow_id": "report-gen", "workflow_name": "Report Generator", "duration_ms": 1800, "success": True, "step_count": 7, "timestamp": "2026-06-22T09:00:00Z"},
    {"workflow_id": "report-gen", "workflow_name": "Report Generator", "duration_ms": 5500, "success": False, "step_count": 2, "error_type": "timeout", "timestamp": "2026-06-22T10:00:00Z"},
    {"workflow_id": "report-gen", "workflow_name": "Report Generator", "duration_ms": 2000, "success": True, "step_count": 7, "timestamp": "2026-06-22T11:00:00Z"},
    {"workflow_id": "report-gen", "workflow_name": "Report Generator", "duration_ms": 6000, "success": False, "step_count": 1, "error_type": "oom", "timestamp": "2026-06-22T12:00:00Z"},
    {"workflow_id": "report-gen", "workflow_name": "Report Generator", "duration_ms": 2200, "success": True, "step_count": 8, "timestamp": "2026-06-22T13:00:00Z"},
    {"workflow_id": "report-gen", "workflow_name": "Report Generator", "duration_ms": 5800, "success": False, "step_count": 2, "error_type": "timeout", "timestamp": "2026-06-22T14:00:00Z"},
    {"workflow_id": "report-gen", "workflow_name": "Report Generator", "duration_ms": 2100, "success": True, "step_count": 7, "timestamp": "2026-06-22T15:00:00Z"},
    {"workflow_id": "report-gen", "workflow_name": "Report Generator", "duration_ms": 6200, "success": False, "step_count": 1, "error_type": "timeout", "timestamp": "2026-06-22T16:00:00Z"},
    {"workflow_id": "report-gen", "workflow_name": "Report Generator", "duration_ms": 2300, "success": True, "step_count": 8, "timestamp": "2026-06-22T17:00:00Z"},
]

engine = EvolutionEngine()
result = engine.evolve(executions, focus="all")

print("=" * 70)
print("  EVOLUTION ENGINE — Test Results")
print("=" * 70)
print(json.dumps(result, indent=2, default=str))
