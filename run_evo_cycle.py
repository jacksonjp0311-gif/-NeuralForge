"""Full evolution cycle: fetch → filter → analyze → act → save."""
import sys, os, json, time, urllib.request
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuralforge.evo_engine import EvolutionEngine
from neuralforge.smart_engine import SmartEngine

# 1. Fetch executions
print("=" * 60)
print("  NEURALFORGE EVOLUTION CYCLE")
print("=" * 60)

token = os.environ.get('AGNT_AUTH_TOKEN', '')
req = urllib.request.Request(
    'http://localhost:3333/api/executions/',
    headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/json'}
)
with urllib.request.urlopen(req, timeout=30) as resp:
    raw = json.loads(resp.read().decode())
executions = raw if isinstance(raw, list) else raw.get('executions', raw.get('data', []))

print(f"\n[1] Fetched {len(executions)} executions")

# 2. Transform
def transform(e):
    dur = 0
    if e.get('startTime') and e.get('endTime'):
        try: dur = float(e['endTime']) - float(e['startTime'])
        except: pass
    return {
        'workflow_id': e.get('workflowId', e.get('workflow_name', 'unknown')),
        'workflow_name': e.get('workflowName', 'unknown'),
        'duration_ms': dur,
        'success': e.get('status') in ('success', 'completed'),
        'step_count': e.get('nodeCount', 0),
        'error_type': None if e.get('status') in ('success','completed') else (e.get('errorType','unknown')),
        'timestamp': e.get('startTime',''),
    }

execs = [transform(e) for e in executions]

# 3. Filter test workflows  
test_kw = ['lss audit','smoke test','one-shot test','authorize button test','weekly']
before = len(execs)
execs = [e for e in execs if not any(k in e['workflow_name'].lower() for k in test_kw)]
print(f"[2] Filtered: {before} → {len(execs)} (removed {before-len(execs)} test executions)")

# 4. Save filtered data
with open(os.path.join(os.path.dirname(__file__), 'exec_data.json'), 'w') as f:
    json.dump(execs, f)
print(f"[3] Saved filtered data")

# 5. Run evolution
print(f"[4] Running evolution engine...")
evo = EvolutionEngine()
result = evo.evolve(execs, focus="all")
obs = result.get("observation", {})
evo_stage = result.get("evolution", {})

print(f"\n  ═══ RESULTS ═══")
print(f"  Health Score: {obs.get('health_score',0):.1f}%")
print(f"  Success Rate: {(obs.get('overall_success_rate',0)*100):.1f}%")
print(f"  Workflows: {obs.get('workflows_analyzed',0)}")
print(f"  Executions: {obs.get('total_executions',0)}")
print(f"  Failures: {obs.get('total_failures',0)}")
print(f"  Recommendations: {evo_stage.get('recommendation_count',0)}")
print(f"  High-Risk: {len(result.get('predictions',{}).get('high_risk',[]))}")

# 6. Smart decisions
smart = SmartEngine()
recent = [{"success": e.get("success",True), "duration_ms": e.get("duration_ms",0)} for e in execs[-20:]]
retry = smart.decide("retry", history=recent)
print(f"\n  Smart Engine:")
print(f"    Retry: {retry.get('decision')} (conf={retry.get('confidence',0):.2f})")

# 7. Print recommendations
print(f"\n  Recommendations:")
for r in evo_stage.get("recommendations",[]):
    print(f"    [{r['priority'].upper()}] {r['action']}")

# 8. Per-workflow stats
print(f"\n  Per-Workflow:")
wf_stats = obs.get("workflow_stats",{})
for wid, s in wf_stats.items():
    sr = (s.get('success_rate',0)*100)
    tag = '🔴' if sr < 50 else '🟡' if sr < 80 else '🟢'
    print(f"    {tag} {wid[:40]:40s} | {sr:5.1f}% | {s.get('executions',0)} execs | {s.get('failure_count',0)} fails")

# Save results
output = {
    "timestamp": time.time(),
    "executions_analyzed": len(execs),
    "health_score": obs.get("health_score"),
    "success_rate": obs.get("overall_success_rate"),
    "workflows_analyzed": obs.get("workflows_analyzed"),
    "total_failures": obs.get("total_failures"),
    "recommendations": evo_stage.get("recommendations",[]),
    "high_risk_workflows": result.get("predictions",{}).get("high_risk",[]),
    "retry_decision": retry.get("decision"),
    "retry_confidence": retry.get("confidence"),
    "workflow_stats": wf_stats,
    "evolution_stage": evo_stage.get("evolution_stage"),
}

with open(os.path.join(os.path.dirname(__file__), 'evo_results.json'), 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n{'='*60}")
print("  EVOLUTION CYCLE COMPLETE")
print("=" * 60)
