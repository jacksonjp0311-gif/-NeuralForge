"""Quick smoke tests for all NeuralForge components."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

results = {}

t0 = time.time()
from neuralforge.learner import DataLearner
from neuralforge.pattern_engine import PatternEngine
from neuralforge.smart_engine import SmartEngine
from neuralforge.analyzer import WorkflowAnalyzer
from neuralforge.evo_engine import EvolutionEngine
import numpy as np
import torch

# 1. DataLearner
print("[1] DataLearner...")
dl = DataLearner(device=torch.device("cpu"))
X = np.random.randn(100, 3).tolist()
y = [2*x[0]+np.random.normal(0,0.2) for x in X]
r = dl.learn(X, y, epochs=30)
results["learner"] = {"status": r.get("status"), "metric": r.get("metric_value"), "time": round(r.get("training_time_seconds",0),2)}
print(f"  R²={r.get('metric_value'):.4f} | {r.get('training_time_seconds',0)}s")

# 2. Pattern Engine
print("[2] Pattern Engine...")
pe = PatternEngine()
data = [2.0*i+np.random.normal(0,0.5) for i in range(30)]
r = pe.analyze(data, predict_steps=3, epochs=30)
results["pattern"] = {"status": r.get("status"), "type": r.get("pattern_type"), "corr": r.get("training_correlation"), "time": round(r.get("training_time_seconds",0),2)}
print(f"  {r.get('pattern_type')} | r={r.get('training_correlation'):.4f} | {r.get('training_time_seconds',0)}s")

# 3. Smart Engine
print("[3] Smart Engine...")
sm = SmartEngine()
history = [{"success": True}, {"success": True}, {"success": False}, {"success": True}]
r = sm.decide("retry", history=history)
results["smart_retry"] = {"status": r.get("status"), "decision": r.get("decision"), "confidence": r.get("confidence")}
r = sm.decide("pattern", data=[10*np.sin(2*np.pi*i/12) for i in range(40)])
results["smart_pattern"] = {"status": r.get("status"), "type": r.get("pattern_type"), "corr": r.get("correlation")}
print(f"  retry={results['smart_retry']['decision']} | pattern={results['smart_pattern']['type']}")

# 4. Workflow Analyzer
print("[4] Workflow Analyzer...")
wa = WorkflowAnalyzer()
execs = [{"duration_ms": 1000+i*100+np.random.normal(0,50), "success": i%5!=0, "step_count": 5+i%3} for i in range(20)]
r = wa.analyze(execs)
results["analyzer"] = {"status": r.get("status"), "health": r.get("stats",{}).get("health_score"), "anomalies": r.get("anomalies",{}).get("count")}
print(f"  health={results['analyzer']['health']}% | anomalies={results['analyzer']['anomalies']}")

# 5. Evolution Engine
print("[5] Evolution Engine...")
ee = EvolutionEngine()
execs = []
for wid, name in [("wf-1","Email"),("wf-2","Sync"),("wf-3","Report")]:
    for i in range(10):
        execs.append({"workflow_id": wid, "workflow_name": name, "duration_ms": 1000+np.random.normal(0,200), "success": np.random.random()>0.2, "step_count": 5})
r = ee.evolve(execs)
results["evolution"] = {"status": r.get("status"), "health": r.get("observation",{}).get("health_score"), "recommendations": r.get("evolution",{}).get("recommendation_count")}
print(f"  health={results['evolution']['health']}% | recs={results['evolution']['recommendations']}")

results["total_time"] = round(time.time()-t0, 2)
results["all_pass"] = all(r.get("status")=="success" for r in results.values() if isinstance(r,dict))

print(f"\n{'='*60}")
print(f"ALL TESTS {'PASSED' if results['all_pass'] else 'SOME FAILED'} in {results['total_time']}s")
print(json.dumps(results, indent=2))
