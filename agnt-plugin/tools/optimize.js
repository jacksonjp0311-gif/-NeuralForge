import { runPython } from "./_bridge.mjs";

class NeuralForgeOptimizeTool {
  constructor() {
    this.name = "neuralforge_optimize";
  }

  async execute(params) {
    const description = params.description || params._raw || "";
    if (!description) {
      return { error: "Missing required parameter: description" };
    }

    const objective = params.objective || "accuracy";
    const numTrials = params.num_trials || 10;

    const escaped = description.replace(/\\/g, "\\\\").replace(/"/g, '\\"');

    const code = `
spec = NeuralForgeSpec.from_description("${escaped}")
obj = OptimizationGoal(objective="${objective}", direction="maximize", budget=ComputeBudget(num_trials=${numTrials}))
result = nf_optimize(obj, base_spec=spec)
output = {
    "status": "success",
    "best_score": result.best_score,
    "num_trials": result.num_trials,
    "best_architecture": result.best_spec.architecture.family.value,
    "best_depth": result.best_spec.architecture.depth,
    "best_width": result.best_spec.architecture.width
}
`;
    const result = runPython(code, 180000);
    if (result.status === "error") return { error: result.error };
    return result;
  }
}

export default new NeuralForgeOptimizeTool();
