"""Test the Workflow Analyzer tool with simulated execution data."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuralforge.analyzer import WorkflowAnalyzer

# Simulate 20 workflow executions with degradation trend + anomalies
executions = [
    {"duration_ms": 1200, "success": True, "step_count": 5},
    {"duration_ms": 1150, "success": True, "step_count": 5},
    {"duration_ms": 1300, "success": True, "step_count": 5},
    {"duration_ms": 1250, "success": True, "step_count": 5},
    {"duration_ms": 1400, "success": True, "step_count": 6},
    {"duration_ms": 3500, "success": False, "step_count": 3, "error_type": "timeout"},
    {"duration_ms": 1500, "success": True, "step_count": 6},
    {"duration_ms": 1600, "success": True, "step_count": 6},
    {"duration_ms": 1550, "success": True, "step_count": 6},
    {"duration_ms": 1700, "success": True, "step_count": 7},
    {"duration_ms": 1650, "success": True, "step_count": 7},
    {"duration_ms": 1800, "success": True, "step_count": 7},
    {"duration_ms": 4200, "success": False, "step_count": 2, "error_type": "oom"},
    {"duration_ms": 1900, "success": True, "step_count": 8},
    {"duration_ms": 2000, "success": True, "step_count": 8},
    {"duration_ms": 2100, "success": True, "step_count": 8},
    {"duration_ms": 2050, "success": True, "step_count": 9},
    {"duration_ms": 2200, "success": True, "step_count": 9},
    {"duration_ms": 2300, "success": True, "step_count": 9},
    {"duration_ms": 2400, "success": True, "step_count": 10},
]

analyzer = WorkflowAnalyzer()
result = analyzer.analyze(executions, predict_next=True)

print("=" * 60)
print("  Workflow Analyzer — Test Results")
print("=" * 60)
print(json.dumps(result, indent=2, default=str))
