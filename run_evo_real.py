"""Run the Evolution Engine on real AGNT execution data."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuralforge.evo_engine import EvolutionEngine

# Load execution data
with open(os.path.join(os.path.dirname(__file__), 'exec_data.json'), 'r') as f:
    executions = json.load(f)

print(f"Loaded {len(executions)} executions")
print(f"Successes: {sum(1 for e in executions if e.get('success'))}")
print(f"Failures: {sum(1 for e in executions if not e.get('success'))}")

# Run evolution engine
engine = EvolutionEngine()
result = engine.evolve(executions, focus="all")

print("\n" + "=" * 70)
print("  EVOLUTION ENGINE — Real AGNT Data Analysis")
print("=" * 70)
print(json.dumps(result, indent=2, default=str))
