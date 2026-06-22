"""Evidence: DataLearner v2.3 — all 4 problem types."""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuralforge.learner import DataLearner

print("=" * 70)
print("  NeuralForge DataLearner v2.3 — Full Evidence")
print("=" * 70)

results = {}

# 1. Regression
print("\n[1] Regression (y = 3x1 + 2x2 + noise)")
np.random.seed(42)
X = np.random.randn(200, 5).tolist()
y = [3*x[0]+2*x[1]+np.random.normal(0,0.3) for x in X]
dl = DataLearner(device=torch.device('cpu'))
t0 = time.time()
r = dl.learn(X, y, epochs=50)
results["regression"] = {"status": r.get("status"), "metric": r.get("metric_value"), "time": round(time.time()-t0,2), "pattern": r.get("problem_type"), "arch": r.get("architecture")}
print("  %s | %-14s | %s=%.4f | %.2fs" % ("PASS" if r.get("status")=="success" else "FAIL", r.get("problem_type","?"), r.get("metric_name","?"), r.get("metric_value",0), results["regression"]["time"]))

# 2. Classification
print("\n[2] Classification (3-class 2D blobs)")
np.random.seed(123)
n=150
X = np.vstack([np.random.randn(n//3,2)+[2,2],np.random.randn(n//3,2)+[-2,-2],np.random.randn(n//3,2)+[2,-2]]).tolist()
y = [0]*(n//3)+[1]*(n//3)+[2]*(n//3)
dl2 = DataLearner(device=torch.device('cpu'))
t0 = time.time()
r = dl2.learn(X, y, epochs=50)
results["classification"] = {"status": r.get("status"), "metric": r.get("metric_value"), "time": round(time.time()-t0,2), "pattern": r.get("problem_type"), "arch": r.get("architecture")}
print("  %s | %-14s | %s=%.4f | %.2fs" % ("PASS" if r.get("status")=="success" else "FAIL", r.get("problem_type","?"), r.get("metric_name","?"), r.get("metric_value",0), results["classification"]["time"]))

# 3. Forecasting
print("\n[3] Forecasting (sine wave)")
data = [float(np.sin(2*np.pi*i/20)) for i in range(100)]
X = [[data[i]] for i in range(80)]
y = [float(data[i+1]) for i in range(80)]
dl3 = DataLearner(device=torch.device('cpu'))
t0 = time.time()
r = dl3.learn(X, y, epochs=50)
results["forecasting"] = {"status": r.get("status"), "metric": r.get("metric_value"), "time": round(time.time()-t0,2), "pattern": r.get("problem_type"), "arch": r.get("architecture")}
print("  %s | %-14s | %s=%.4f | %.2fs" % ("PASS" if r.get("status")=="success" else "FAIL", r.get("problem_type","?"), r.get("metric_name","?"), r.get("metric_value",0), results["forecasting"]["time"]))

# 4. Anomaly
print("\n[4] Anomaly Detection (95% normal + 5% outliers)")
np.random.seed(99)
X = np.random.randn(190,3).tolist() + (np.random.randn(10,3)*5).tolist()
y = [0]*190+[1]*10
dl4 = DataLearner(device=torch.device('cpu'))
t0 = time.time()
r = dl4.learn(X, y, epochs=50)
results["anomaly"] = {"status": r.get("status"), "metric": r.get("metric_value"), "time": round(time.time()-t0,2), "pattern": r.get("problem_type"), "arch": r.get("architecture")}
print("  %s | %-14s | %s=%s | %.2fs" % ("PASS" if r.get("status")=="success" else "FAIL", r.get("problem_type","?"), r.get("metric_name","?"), str(r.get("metric_value","?")), results["anomaly"]["time"]))

# Summary
print("\n" + "=" * 70)
print("  EVIDENCE SUMMARY")
print("=" * 70)
all_ok = all(r.get("status")=="success" for r in results.values())
avg_time = np.mean([r["time"] for r in results.values()])
print("  All 4 tests pass: %s" % ("YES" if all_ok else "NO"))
print("  Avg training time: %.2fs" % avg_time)
for name, r in results.items():
    print("  %-14s -> %-14s | %s=%s" % (name, r.get("pattern","?"), r.get("metric_name","?"), str(r.get("metric_value","?"))))
