"""Run Smart Engine on real AGNT execution data."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuralforge.smart_engine import SmartEngine

with open(os.path.join(os.path.dirname(__file__), 'exec_data.json'), 'r') as f:
    executions = json.load(f)

engine = SmartEngine()

# 1. Pattern on durations
durations = [float(e.get("duration_ms", 0)) for e in executions[:100]]
r = engine.decide("pattern", data=durations)
print("=== PATTERN ===")
print("Pattern:", r.get("pattern_type"), "Conf:", r.get("confidence"), "Corr:", r.get("correlation"))

# 2. Retry on recent history
recent = [{"success": e.get("success", True), "duration_ms": e.get("duration_ms", 0)} for e in executions[-20:]]
r = engine.decide("retry", history=recent)
print("\n=== RETRY ===")
print("Decision:", r.get("decision"), "Conf:", r.get("confidence"))
print("Success prob:", r.get("success_probability"))
print("Reasoning:", r.get("reasoning", [])[:2])

# 3. Auto-detect
r = engine.decide("auto", history=executions[-50:])
print("\n=== AUTO ===")
print("Decision:", r.get("decision"), "Conf:", r.get("confidence"))
