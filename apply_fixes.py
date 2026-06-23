
import sys, os, json, time
sys.path.insert(0, r"C:\Users\jacks\OneDrive\Desktop\agnt-evo\neuralforge")

from neuralforge.evo_engine import EvolutionEngine
from neuralforge.smart_engine import SmartEngine
from neuralforge.realtime_evo import RealtimeEvolutionEngine
import numpy as np

# Load execution data
with open(r"C:\Users\jacks\OneDrive\Desktop\agnt-evo\neuralforge\exec_data.json", "r") as f:
    executions = json.load(f)

print("=" * 70)
print("  NEURALFORGE ACTION ENGINE — Fixing Detected Problems")
print("=" * 70)

# 1. Filter out test workflows
test_keywords = ['lss audit', 'smoke test', 'one-shot test', 'authorize button test']
real_execs = [e for e in executions if not any(kw in e.get("workflow_name","").lower() for kw in test_keywords)]
test_execs = [e for e in executions if any(kw in e.get("workflow_name","").lower() for kw in test_keywords)]

print(f"\n[1] FILTERING TEST WORKFLOWS")
print(f"    Total executions: {len(executions)}")
print(f"    Test executions filtered: {len(test_execs)}")
print(f"    Real executions: {len(real_execs)}")

# Credits wasted on test workflows
test_credits = sum(e.get("creditsUsed", 0) for e in test_execs)
real_credits = sum(e.get("creditsUsed", 0) for e in real_execs)
print(f"    Credits wasted on tests: {test_credits:.2f}")
print(f"    Credits used by real workflows: {real_credits:.2f}")

# 2. Run evolution on real workflows only
print(f"\n[2] EVOLUTION ON REAL WORKFLOWS")
evo = EvolutionEngine()
result = evo.evolve(real_execs, focus="all")

obs = result.get("observation", {})
print(f"    Health: {obs.get('health_score', '?')}%")
print(f"    Success Rate: {(obs.get('overall_success_rate',0)*100):.1f}%")
print(f"    Workflows: {obs.get('workflows_analyzed', '?')}")
print(f"    Failures: {obs.get('total_failures', '?')}")

# 3. Identify specific problems and fixes
print(f"\n[3] PROBLEM ANALYSIS & FIXES")

wf_stats = obs.get("workflow_stats", {})
for wid, stats in wf_stats.items():
    name = wid[:50]
    sr = (stats.get("success_rate", 0) * 100)
    fails = stats.get("failure_count", 0)
    
    if sr < 70:
        print(f"\n  ⚠ {name}")
        print(f"    Success: {sr:.1f}% | Fails: {fails}")
        
        # Diagnose the problem
        if "authorize" in name.lower() or "pause" in name.lower():
            print(f"    DIAGNOSIS: Requires manual authorization step")
            print(f"    FIX: Add auto-approve for known-safe operations")
            print(f"    ACTION: Set 'auto_approve=true' in workflow parameters")
        elif "webhook" in name.lower():
            print(f"    DIAGNOSIS: Webhook endpoint may be unavailable")
            print(f"    FIX: Add retry logic and fallback endpoint")
            print(f"    ACTION: Add 3 retries with exponential backoff")
        elif "promotion" in name.lower():
            print(f"    DIAGNOSIS: Promotion gate failing — geometry service may be down")
            print(f"    FIX: Add health check before promotion")
            print(f"    ACTION: Skip promotion if geometry service unhealthy")
        else:
            print(f"    DIAGNOSIS: General failure pattern")
            print(f"    FIX: Add comprehensive error handling and retry logic")

# 4. Generate action plan
print(f"\n[4] ACTION PLAN")
actions = [
    {
        "priority": "CRITICAL",
        "action": "Disable LSS Audit test workflow (3441 executions, 0% success, wastes credits)",
        "workflow": "d59de47e-d328-46d6-9b23-6692dd6dfc01",
        "savings": f"{test_credits:.2f} credits"
    },
    {
        "priority": "HIGH", 
        "action": "Add auto-approve to ASF Governed Evolution (50% success, manual auth required)",
        "workflow": "8a6abe6d-b69c-4866-91f1-6d26a4faf0d9",
        "fix": "Set auto_approve=true for known-safe operations"
    },
    {
        "priority": "HIGH",
        "action": "Add retry logic to ASF Promotion Gate (50% success, webhook failures)",
        "workflow": "141aebe7-3e7d-4dec-b710-981daef590b5",
        "fix": "3 retries with exponential backoff on webhook calls"
    },
    {
        "priority": "MEDIUM",
        "action": "Add health check before promotion gate execution",
        "workflow": "141aebe7-3e7d-4dec-b710-981daef590b5",
        "fix": "Check geometry service health before attempting promotion"
    }
]

for i, a in enumerate(actions, 1):
    print(f"\n  {i}. [{a['priority']}] {a['action']}")
    if 'savings' in a:
        print(f"     Savings: {a['savings']}")
    if 'fix' in a:
        print(f"     Fix: {a['fix']}")

# 5. Save results
output = {
    "timestamp": time.time(),
    "total_executions": len(executions),
    "test_executions_filtered": len(test_execs),
    "real_executions": len(real_execs),
    "credits_wasted_on_tests": round(test_credits, 2),
    "health_score": obs.get("health_score"),
    "success_rate": obs.get("overall_success_rate"),
    "workflows_analyzed": obs.get("workflows_analyzed"),
    "actions": actions,
    "recommendations": result.get("evolution", {}).get("recommendations", []),
}

with open(r"C:\Users\jacks\OneDrive\Desktop\agnt-evo\neuralforge\evo_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n{'='*70}")
print("  ACTIONS GENERATED — Apply fixes to improve system health")
print("=" * 70)
