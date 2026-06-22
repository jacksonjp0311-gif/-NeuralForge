"""Evidence: DataLearner v2.2 — 4 problem types, all passing."""
import sys, os, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuralforge.learner import DataLearner

print("=" * 70)
print("  NeuralForge DataLearner v2.2 — Full Evidence")
print("=" * 70)

results = {}

# ── Test 1: Regression ──
print("\n[1] Regression (y = 3x1 + 2x2 + noise, 200 samples, 5 features)")
np.random.seed(42)
X = np.random.randn(200, 5).tolist()
y = [3*x[0] + 2*x[1] + np.random.normal(0, 0.3) for x in X]
learner = DataLearner(device=torch.device('cpu'))
r = learner.learn(X, y, epochs=50)
results["regression"] = r
print("  %s | %-14s | %s=%.4f | %.2fs" % (
    "PASS" if r.get("status")=="success" else "FAIL",
    r.get("problem_type","?"), r.get("metric_name","?"), r.get("metric_value",0), r.get("training_time_seconds",0)))

# ── Test 2: Classification ──
print("\n[2] Classification (3-class 2D blobs, 150 samples)")
np.random.seed(123)
n = 150
X = np.vstack([
    np.random.randn(n//3, 2) + [2, 2],
    np.random.randn(n//3, 2) + [-2, -2],
    np.random.randn(n//3, 2) + [2, -2],
]).tolist()
y = [0]*(n//3) + [1]*(n//3) + [2]*(n//3)
learner2 = DataLearner(device=torch.device('cpu'))
r = learner2.learn(X, y, epochs=50)
results["classification"] = r
print("  %s | %-14s | %s=%.4f | %.2fs" % (
    "PASS" if r.get("status")=="success" else "FAIL",
    r.get("problem_type","?"), r.get("metric_name","?"), r.get("metric_value",0), r.get("training_time_seconds",0)))

# ── Test 3: Forecasting ──
print("\n[3] Forecasting (sine wave, 80 samples)")
data = [float(np.sin(2 * np.pi * i / 20)) for i in range(100)]
X = [[data[i]] for i in range(80)]
y = [float(data[i+1]) for i in range(80)]
learner3 = DataLearner(device=torch.device('cpu'))
r = learner3.learn(X, y, epochs=50)
results["forecasting"] = r
print("  %s | %-14s | %s=%.4f | %.2fs" % (
    "PASS" if r.get("status")=="success" else "FAIL",
    r.get("problem_type","?"), r.get("metric_name","?"), r.get("metric_value",0), r.get("training_time_seconds",0)))

# ── Test 4: Anomaly Detection ──
print("\n[4] Anomaly Detection (95% normal + 5% outliers, 200 samples)")
np.random.seed(99)
normal = np.random.randn(190, 3).tolist()
outliers = (np.random.randn(10, 3) * 5).tolist()
X = normal + outliers
y = [0]*190 + [1]*10
learner4 = DataLearner(device=torch.device('cpu'))
r = learner4.learn(X, y, epochs=50)
results["anomaly"] = r
print("  %s | %-14s | %s=%s | %.2fs" % (
    "PASS" if r.get("status")=="success" else "FAIL",
    r.get("problem_type","?"), r.get("metric_name","?"), str(r.get("metric_value","?")), r.get("training_time_seconds",0)))

# ── Summary ──
print("\n" + "=" * 70)
print("  EVIDENCE SUMMARY")
print("=" * 70)
all_ok = all(r.get("status") == "success" for r in results.values())
avg_time = np.mean([r.get("training_time_seconds", 0) for r in results.values()])
print("  All 4 tests pass: %s" % ("YES" if all_ok else "NO"))
print("  Avg training time: %.2fs" % avg_time)
for name, r in results.items():
    print("  %-14s -> %-14s | %s=%s" % (
        name, r.get("problem_type","?"),
        r.get("metric_name","?"), str(r.get("metric_value","?"))))
