"""
NeuralForge Auto-Evolution — Runs continuously inside AGNT.
Fetches executions, runs evolution engine, stores results.
"""
import sys, os, json, time
import urllib.request

nf_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, nf_dir)

from neuralforge.evo_engine import EvolutionEngine
from neuralforge.realtime_evo import RealtimeEvolutionEngine
from neuralforge.smart_engine import SmartEngine

def fetch_executions():
    token = os.environ.get('AGNT_AUTH_TOKEN', '')
    try:
        req = urllib.request.Request(
            'http://localhost:3333/api/executions/',
            headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data if isinstance(data, list) else data.get('executions', data.get('data', []))
    except Exception as e:
        print(f"Warning: Could not fetch: {e}")
        cache = os.path.join(nf_dir, 'exec_data.json')
        if os.path.exists(cache):
            with open(cache, 'r') as f:
                return json.load(f)
        return []

def transform(executions):
    result = []
    for e in executions:
        dur = 0
        if e.get('startTime') and e.get('endTime'):
            try:
                dur = (float(e['endTime']) - float(e['startTime'])) if isinstance(e['startTime'], (int, float)) else 0
            except: pass
        result.append({
            'workflow_id': e.get('workflowId', e.get('workflow_name', 'unknown')),
            'workflow_name': e.get('workflowName', 'unknown'),
            'duration_ms': dur,
            'success': e.get('status') in ('success', 'completed'),
            'step_count': e.get('nodeCount', 0),
            'error_type': e.get('errorType', 'unknown') if e.get('status') == 'error' else None,
            'timestamp': e.get('startTime', ''),
        })
    return result

def main():
    print("=" * 60)
    print("  NEURALFORGE AUTO-EVOLUTION")
    print("=" * 60)
    
    print("\n[1] Fetching executions...")
    raw = fetch_executions()
    executions = transform(raw)
    print(f"    Got {len(executions)} executions")
    
    if len(executions) < 3:
        print("ERROR: Need at least 3 executions")
        return
    
    # Save
    with open(os.path.join(nf_dir, 'exec_data.json'), 'w') as f:
        json.dump(executions, f)
    
    # Filter test workflows
    real_execs = [e for e in executions if "LSS Audit" not in e.get("workflow_name", "") and "Smoke Test" not in e.get("workflow_name", "")]
    execs = real_execs if len(real_execs) >= 10 else executions
    print(f"\n[2] Using {len(execs)} executions (filtered: {len(real_execs) >= 10})")
    
    # Run evolution
    print("\n[3] Running evolution engine...")
    evo = EvolutionEngine()
    evo_result = evo.evolve(execs, focus="all")
    
    obs = evo_result.get('observation', {})
    print(f"    Health: {obs.get('health_score', '?')}%")
    print(f"    Workflows: {obs.get('workflows_analyzed', '?')}")
    print(f"    Failures: {obs.get('total_failures', '?')}")
    
    # Run realtime
    print("\n[4] Running real-time analysis...")
    rt = RealtimeEvolutionEngine(window_size=50)
    for i in range(0, len(execs), 50):
        rt.process_batch(execs[i:i+50])
    knowledge = rt.get_knowledge_summary()
    print(f"    Knowledge: {knowledge.get('knowledge_entries', 0)} entries")
    
    # Smart decisions
    print("\n[5] Smart engine decisions...")
    smart = SmartEngine()
    recent = [{"success": e.get("success", True), "duration_ms": e.get("duration_ms", 0)} for e in execs[-20:]]
    retry = smart.decide("retry", history=recent)
    print(f"    Retry: {retry.get('decision')} (conf={retry.get('confidence')})")
    
    # Save results
    output = {
        "timestamp": time.time(),
        "executions_analyzed": len(execs),
        "health_score": obs.get('health_score'),
        "success_rate": obs.get('overall_success_rate'),
        "workflows_analyzed": obs.get('workflows_analyzed'),
        "total_failures": obs.get('total_failures'),
        "recommendations": evo_result.get('evolution', {}).get('recommendations', []),
        "high_risk_workflows": evo_result.get('predictions', {}).get('high_risk', []),
        "knowledge_entries": knowledge.get('knowledge_entries'),
        "health_trend": knowledge.get('health_trend'),
        "retry_decision": retry.get('decision'),
        "retry_confidence": retry.get('confidence'),
    }
    
    with open(os.path.join(nf_dir, 'evo_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\n[6] Results saved")
    for rec in output.get('recommendations', []):
        print(f"  [{rec.get('priority', '?').upper()}] {rec.get('action', '')}")
    
    print("\n" + "=" * 60)
    print("  AUTO-EVOLUTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
