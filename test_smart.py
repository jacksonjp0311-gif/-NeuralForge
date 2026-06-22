"""Test the Smart Engine — the always-called tool."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuralforge.smart_engine import SmartEngine

engine = SmartEngine()
results = {}

# Test 1: Retry decision
print("\n[1] Smart: Should I retry?")
history = [
    {"success": True, "duration_ms": 1200},
    {"success": True, "duration_ms": 1100},
    {"success": False, "duration_ms": 5000, "error_type": "timeout"},
    {"success": True, "duration_ms": 1300},
    {"success": True, "duration_ms": 1250},
    {"success": False, "duration_ms": 4800, "error_type": "timeout"},
    {"success": True, "duration_ms": 1400},
    {"success": True, "duration_ms": 1350},
]
r = engine.decide("retry", history=history)
results["retry"] = r
print("  Decision:", r.get("decision"), "| Confidence:", r.get("confidence"))
print("  Reasoning:", r.get("reasoning", [])[:2])

# Test 2: Optimize decision
print("\n[2] Smart: What's the best option?")
history = [
    {"option": "gpt-4o", "score": 0.92}, {"option": "gpt-4o-mini", "score": 0.85},
    {"option": "gpt-4o", "score": 0.95}, {"option": "claude-sonnet", "score": 0.88},
    {"option": "gpt-4o", "score": 0.91}, {"option": "claude-sonnet", "score": 0.90},
]
r = engine.decide("optimize", history=history, options=["gpt-4o", "gpt-4o-mini", "claude-sonnet"])
results["optimize"] = r
print("  Decision:", r.get("decision"), "| Confidence:", r.get("confidence"))
print("  Ranked:", r.get("ranked_options", [])[:3])

# Test 3: Predict decision
print("\n[3] Smart: Will this succeed?")
history = [
    {"success": True}, {"success": True}, {"success": True},
    {"success": False}, {"success": True}, {"success": True},
    {"success": True}, {"success": True},
]
r = engine.decide("predict", history=history)
results["predict"] = r
print("  Decision:", r.get("decision"), "| Confidence:", r.get("confidence"))
print("  Success prob:", r.get("success_probability"))

# Test 4: Pattern detection
print("\n[4] Smart: What's the pattern?")
data = [10.0 * i + 50 + __import__('numpy').random.normal(0, 2) for i in range(30)]
r = engine.decide("pattern", data=data)
results["pattern"] = r
print("  Pattern:", r.get("pattern_type"), "| Correlation:", r.get("correlation"))

# Test 5: Fix recommendation
print("\n[5] Smart: How do I fix this?")
history = [
    {"success": True}, {"success": True},
    {"success": False, "error_type": "timeout"},
    {"success": True},
    {"success": False, "error_type": "timeout"},
    {"success": True}, {"success": True}, {"success": True},
    {"success": False, "error_type": "rate_limit"},
]
r = engine.decide("fix", history=history)
results["fix"] = r
print("  Decision:", r.get("decision"), "| Error rate:", r.get("error_rate"))
print("  Recommendations:", r.get("recommendations", [])[:3])

# Test 6: Auto-detect
print("\n[6] Smart: Auto-detect context")
r = engine.decide("auto", history=history)
results["auto"] = r
print("  Detected context:", r.get("decision"))

# Summary
print("\n" + "=" * 60)
print("  SMART ENGINE EVIDENCE SUMMARY")
print("=" * 60)
for name, r in results.items():
    status = "PASS" if r.get("status") == "success" else "FAIL"
    print("  %-12s | %s | decision=%s" % (name.upper(), status, r.get("decision","?")))
