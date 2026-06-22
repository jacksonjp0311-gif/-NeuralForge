"""Evidence: Pattern Engine v2.2 — all 5 pattern types + real-world data."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuralforge.pattern_engine import PatternEngine, PatternType

print("=" * 70)
print("  NeuralForge Pattern Engine v2.2 — Evidence Test")
print("=" * 70)

engine = PatternEngine()
results = {}

# Test 1: Linear Trend
print("\n[1] Linear Trend (y = 2x + noise)")
data = [2.0 * i + np.random.normal(0, 0.5) for i in range(50)]
r = engine.analyze(data, predict_steps=5, epochs=50)
results["trend"] = r
ok = r.get("status") == "success"
print("  %s | pattern=%s | corr=%.4f | time=%.2fs" % ("OK" if ok else "FAIL", r.get("pattern_type","?"), r.get("training_correlation",0), r.get("training_time_seconds",0)))

# Test 2: Seasonal
print("\n[2] Seasonal (sine wave)")
data = [10.0 * np.sin(2 * np.pi * i / 12) + np.random.normal(0, 0.3) for i in range(60)]
r = engine.analyze(data, predict_steps=5, epochs=50)
results["seasonal"] = r
ok = r.get("status") == "success"
print("  %s | pattern=%s | corr=%.4f" % ("OK" if ok else "FAIL", r.get("pattern_type","?"), r.get("training_correlation",0)))

# Test 3: Stationary
print("\n[3] Stationary (mean-reverting)")
np.random.seed(42)
data = [100.0]
for _ in range(99):
    data.append(data[-1] + np.random.normal(0, 1) - 0.1 * (data[-1] - 100))
r = engine.analyze(data, predict_steps=5, epochs=50)
results["stationary"] = r
ok = r.get("status") == "success"
print("  %s | pattern=%s | corr=%.4f" % ("OK" if ok else "FAIL", r.get("pattern_type","?"), r.get("training_correlation",0)))

# Test 4: Step Change
print("\n[4] Step Change")
data = list(np.random.normal(5, 0.5, 30)) + list(np.random.normal(15, 0.5, 30))
r = engine.analyze(data, predict_steps=5, epochs=50)
results["step"] = r
ok = r.get("status") == "success"
print("  %s | pattern=%s | corr=%.4f" % ("OK" if ok else "FAIL", r.get("pattern_type","?"), r.get("training_correlation",0)))

# Test 5: Chaotic
print("\n[5] Chaotic (logistic map)")
x = 0.3
data = []
for _ in range(100):
    x = 3.9 * x * (1 - x)
    data.append(x + np.random.normal(0, 0.01))
r = engine.analyze(data, predict_steps=5, epochs=80)
results["chaotic"] = r
ok = r.get("status") == "success"
print("  %s | pattern=%s | corr=%.4f" % ("OK" if ok else "FAIL", r.get("pattern_type","?"), r.get("training_correlation",0)))

# Test 6: Stock-like
print("\n[6] Stock-like (random walk)")
np.random.seed(123)
price = 100.0
data = [price]
for _ in range(199):
    price *= (1 + np.random.normal(0.0005, 0.02))
    data.append(price)
r = engine.analyze(data, predict_steps=5, epochs=60)
results["stock"] = r
ok = r.get("status") == "success"
print("  %s | pattern=%s | corr=%.4f" % ("OK" if ok else "FAIL", r.get("pattern_type","?"), r.get("training_correlation",0)))

# Summary
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
all_ok = all(r.get("status") == "success" for r in results.values())
avg_corr = np.mean([r.get("training_correlation", 0) for r in results.values()])
avg_time = np.mean([r.get("training_time_seconds", 0) for r in results.values()])
print("  All tests pass: %s" % ("YES" if all_ok else "NO"))
print("  Avg correlation: %.4f" % avg_corr)
print("  Avg train time:  %.2fs" % avg_time)
for name, r in results.items():
    print("  %-12s -> %-10s (r=%.4f)" % (name, r.get("pattern_type","?"), r.get("training_correlation",0)))
