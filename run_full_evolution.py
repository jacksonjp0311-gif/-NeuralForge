"""
Full Evolution Analysis on real AGNT data.
Shows how NeuralForge improves the AGNT system.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuralforge.evo_engine import EvolutionEngine
from neuralforge.realtime_evo import RealtimeEvolutionEngine
from neuralforge.smart_engine import SmartEngine
from neuralforge.analyzer import WorkflowAnalyzer

# Load real AGNT execution data
exec_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exec_data.json')
with open(exec_data_path, 'r') as f:
    executions = json.load(f)

print("=" * 70)
print("  NEURALFORGE EVOLUTION — Real AGNT System Analysis")
print("=" * 70)
print(f"\nInput: {len(executions)} workflow executions")

# ── 1. Full Evolution Engine ──
print("\n" + "-" * 70)
print("[PHASE 1] Full Evolution Engine Analysis")
print("-" * 70)

evo = EvolutionEngine()
evo_result = evo.evolve(executions, focus="all")

obs = evo_result.get("observation", {})
print(f"\nSystem Health: {obs.get('health_score', '?')}%")
print(f"Total Executions: {obs.get('total_executions', '?')}")
print(f"Overall Success Rate: {obs.get('overall_success_rate', 0)*100:.2f}%")
print(f"Total Failures: {obs.get('total_failures', '?')}")
print(f"Workflows Analyzed: {obs.get('workflows_analyzed', '?')}")

print("\nPer-Workflow Stats:")
for wid, stats in obs.get("workflow_stats", {}).items():
    short_name = wid[:45]
    print(f"  {short_name:45s} | success={stats.get('success_rate',0)*100:5.1f}% | avg_dur={stats.get('avg_duration_ms',0):10.1f}ms | fails={stats.get('failure_count',0)}")

# ── 2. Real-Time Evolution Engine ──
print("\n" + "-" * 70)
print("[PHASE 2] Real-Time Evolution Engine")
print("-" * 70)

rt_evo = RealtimeEvolutionEngine(window_size=50)

# Feed executions in batches to simulate real-time
batch_size = 50
for i in range(0, min(len(executions), 500), batch_size):
    batch = executions[i:i+batch_size]
    rt_result = rt_evo.process_batch(batch)
    print(f"\nBatch {i//batch_size + 1} (executions {i}-{i+len(batch)}):")
    print(f"  Health: {rt_result.get('health_score', '?')}%")
    print(f"  Active Alerts: {rt_result.get('alert_count', 0)}")
    print(f"  Total Executions Processed: {rt_result.get('total_processed', 0)}")
    if rt_result.get('alerts'):
        for alert in rt_result['alerts'][:3]:
            print(f"  ⚠ {alert.get('type', '?')}: {alert.get('message', '')[:80]}")

# Knowledge summary
knowledge = rt_evo.get_knowledge_summary()
print(f"\nKnowledge Summary:")
print(f"  Total Processed: {knowledge.get('total_executions_processed', 0)}")
print(f"  Total Alerts: {knowledge.get('total_alerts', 0)}")
print(f"  Workflows Tracked: {knowledge.get('workflows_tracked', 0)}")
print(f"  Health Trend: {knowledge.get('health_trend', '?')}")

# ── 3. Smart Engine Decisions ──
print("\n" + "-" * 70)
print("[PHASE 3] Smart Engine Decisions")
print("-" * 70)

smart = SmartEngine()

# Retry decision based on recent history
recent = [{"success": e.get("success", True), "duration_ms": e.get("duration_ms", 0)} for e in executions[-20:]]
retry_result = smart.decide("retry", history=recent)
print(f"\nRetry Decision: {retry_result.get('decision')} (confidence: {retry_result.get('confidence')})")
print(f"Success Probability: {retry_result.get('success_probability')}")
if retry_result.get('reasoning'):
    for r in retry_result['reasoning'][:2]:
        print(f"  → {r}")

# Pattern detection on durations
durations = [float(e.get("duration_ms", 0)) for e in executions[:100]]
pattern_result = smart.decide("pattern", data=durations)
print(f"\nDuration Pattern: {pattern_result.get('pattern_type')} (correlation: {pattern_result.get('correlation'):.4f})")

# Auto-detect context
auto_result = smart.decide("auto", history=executions[-30:])
print(f"\nAuto-Detected Context: {auto_result.get('decision')} (confidence: {auto_result.get('confidence')})")

# ── 4. Evolution Recommendations ──
print("\n" + "-" * 70)
print("[PHASE 4] Evolution Recommendations")
print("-" * 70)

evo_stage = evo_result.get("evolution", {})
print(f"\nEvolution Stage: {evo_stage.get('evolution_stage', '?')}")
print(f"Recommendations: {evo_stage.get('recommendation_count', 0)}")
print(f"Critical: {evo_stage.get('critical_count', 0)} | High: {evo_stage.get('high_count', 0)}")

for rec in evo_stage.get("recommendations", []):
    print(f"\n  [{rec.get('priority', '?').upper()}] {rec.get('category', '?')}")
    print(f"  → {rec.get('action', '')}")

# ── 5. Improvement Summary ──
print("\n" + "-" * 70)
print("[PHASE 5] How NeuralForge Improves AGNT")
print("-" * 70)

improvements = [
    ("Failure Prediction", f"Identifies workflows likely to fail before they run ({len(evo_result.get('predictions', {}).get('high_risk', []))} high-risk workflows detected)"),
    ("Anomaly Detection", f"Finds outlier executions with z-score > 2.0 ({obs.get('total_failures', 0)} failures analyzed)"),
    ("Trend Analysis", f"Detects degradation patterns (duration trend: {evo_result.get('optimization', {}).get('duration_pattern', 'unknown')})"),
    ("Self-Healing", f"Generates {evo_stage.get('recommendation_count', 0)} actionable recommendations"),
    ("Real-Time Monitoring", f"Processes executions in batches of {batch_size} with rolling window analysis"),
    ("Smart Decisions", f"Retry/optimize/predict/pattern/fix/analyze — all from one tool"),
    ("Knowledge Accumulation", f"Every execution improves the system automatically ({knowledge.get('knowledge_entries', 0)} knowledge entries)"),
    ("Cross-Workflow Learning", f"Patterns learned from one workflow benefit all {knowledge.get('workflows_tracked', 0)} tracked workflows"),
]

for i, (title, desc) in enumerate(improvements, 1):
    print(f"\n  {i}. {title}")
    print(f"     {desc}")

print("\n" + "=" * 70)
print("  EVOLUTION COMPLETE")
print("=" * 70)
