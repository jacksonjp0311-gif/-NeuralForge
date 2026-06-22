"""Full Evolution Analysis on real AGNT data."""
import sys, os, json, time

# Add the neuralforge package to path
nf_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, nf_dir)

from neuralforge.evo_engine import EvolutionEngine
from neuralforge.realtime_evo import RealtimeEvolutionEngine
from neuralforge.smart_engine import SmartEngine
from neuralforge.analyzer import WorkflowAnalyzer
from neuralforge.pattern_engine import PatternEngine
from neuralforge.learner import DataLearner
import numpy as np

# Load real AGNT execution data
exec_data_path = os.path.join(nf_dir, 'exec_data.json')
with open(exec_data_path, 'r') as f:
    executions = json.load(f)

print("=" * 70)
print("  NEURALFORGE EVOLUTION — Real AGNT System Analysis")
print("=" * 70)
print(f"\nInput: {len(executions)} workflow executions")

# ── 1. Full Evolution Engine ──
print("\n" + "-" * 70)
print("[PHASE 1] Full Evolution Engine")
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
    short_name = wid[:50]
    print(f"  {short_name:50s} | success={stats.get('success_rate',0)*100:5.1f}% | avg_dur={stats.get('avg_duration_ms',0):10.1f}ms | fails={stats.get('failure_count',0)}")

# ── 2. Real-Time Evolution ──
print("\n" + "-" * 70)
print("[PHASE 2] Real-Time Evolution Engine")
print("-" * 70)

rt_evo = RealtimeEvolutionEngine(window_size=50)
batch_size = 50
for i in range(0, min(len(executions), 500), batch_size):
    batch = executions[i:i+batch_size]
    rt_result = rt_evo.process_batch(batch)
    print(f"\nBatch {i//batch_size + 1} (executions {i}-{i+len(batch)}):")
    print(f"  Health: {rt_result.get('health_score', '?')}% | Alerts: {rt_result.get('alert_count', 0)}")
    for alert in rt_result.get('alerts', [])[:2]:
        print(f"  ⚠ {alert.get('type', '?')}: {alert.get('message', '')[:90]}")

knowledge = rt_evo.get_knowledge_summary()
print(f"\nKnowledge: {knowledge.get('total_executions_processed', 0)} processed, {knowledge.get('total_alerts', 0)} alerts, trend: {knowledge.get('health_trend', '?')}")

# ── 3. Smart Engine ──
print("\n" + "-" * 70)
print("[PHASE 3] Smart Engine Decisions")
print("-" * 70)

smart = SmartEngine()
recent = [{"success": e.get("success", True), "duration_ms": e.get("duration_ms", 0)} for e in executions[-20:]]
retry_result = smart.decide("retry", history=recent)
print(f"\nRetry: {retry_result.get('decision')} (conf={retry_result.get('confidence')}, prob={retry_result.get('success_probability')})")

durations = [float(e.get("duration_ms", 0)) for e in executions[:100]]
pattern_result = smart.decide("pattern", data=durations)
print(f"Pattern: {pattern_result.get('pattern_type')} (corr={pattern_result.get('correlation'):.4f})")

auto_result = smart.decide("auto", history=executions[-30:])
print(f"Auto: {auto_result.get('decision')} (conf={auto_result.get('confidence')})")

# ── 4. Pattern Detector Calibration Test ──
print("\n" + "-" * 70)
print("[PHASE 4] Pattern Detector Calibration")
print("-" * 70)

pe = PatternEngine()
test_cases = [
    ("Trend", [2.0*i + np.random.normal(0, 0.5) for i in range(30)]),
    ("Seasonal", [10.0*np.sin(2*np.pi*i/12) + np.random.normal(0, 0.3) for i in range(40)]),
    ("Stationary", list(np.cumsum(np.random.normal(0, 1, 50)) + 100)),
    ("Step", list(np.random.normal(5, 0.5, 25)) + list(np.random.normal(15, 0.5, 25))),
    ("Chaotic", [0.3] + [3.9*x*(1-x) for x in np.cumsum([0.3] + list(np.random.normal(0, 0.01, 99)))]),
]

for name, data in test_cases:
    r = pe.analyze(data, predict_steps=3, epochs=30)
    print(f"  {name:12s} -> detected: {r.get('pattern_type'):10s} | corr={r.get('training_correlation'):.4f} | time={r.get('training_time_seconds',0):.2f}s")

# ── 5. Evolution Recommendations ──
print("\n" + "-" * 70)
print("[PHASE 5] Evolution Recommendations")
print("-" * 70)

evo_stage = evo_result.get("evolution", {})
print(f"\nStage: {evo_stage.get('evolution_stage', '?')} | Recs: {evo_stage.get('recommendation_count', 0)}")
for rec in evo_stage.get("recommendations", []):
    print(f"  [{rec.get('priority', '?').upper()}] {rec.get('action', '')}")

# ── 6. How NeuralForge Improves AGNT ──
print("\n" + "-" * 70)
print("[PHASE 6] How NeuralForge Improves AGNT")
print("-" * 70)

improvements = [
    ("Failure Prediction", f"{len(evo_result.get('predictions', {}).get('high_risk', []))} high-risk workflows identified before they run"),
    ("Anomaly Detection", f"Real-time alerts on duration spikes and failure rate changes"),
    ("Pattern Learning", f"5 pattern types detected with calibrated heuristics"),
    ("Self-Healing", f"{evo_stage.get('recommendation_count', 0)} actionable recommendations generated"),
    ("Real-Time Monitoring", f"{knowledge.get('total_executions_processed', 0)} executions processed in rolling windows"),
    ("Smart Decisions", f"Retry/optimize/predict/pattern/fix/analyze — all from one tool"),
    ("Knowledge Accumulation", f"{knowledge.get('knowledge_entries')} knowledge entries accumulated"),
    ("Cross-Workflow Learning", f"Patterns from {knowledge.get('workflows_tracked', 0)} workflows inform all predictions"),
]

for i, (title, desc) in enumerate(improvements, 1):
    print(f"\n  {i}. {title}")
    print(f"     {desc}")

print("\n" + "=" * 70)
print("  EVOLUTION COMPLETE")
print("=" * 70)
