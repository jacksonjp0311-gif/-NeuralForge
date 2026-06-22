import { runPython } from "./_bridge.mjs";

class NeuralForgePruneTool {
  constructor() {
    this.name = "neuralforge_prune";
  }

  async execute(params) {
    const description = params.description || params._raw || "";
    if (!description) {
      return { error: "Missing required parameter: description" };
    }

    const amount = params.amount || 0.3;
    const method = params.method || "unstructured_global";

    const escaped = description.replace(/\\/g, "\\\\").replace(/"/g, '\\"');

    const code = `
spec = NeuralForgeSpec.from_description("${escaped}")
model = create_model(spec)
original_params = model.count_parameters()
config = PruningConfig(method=PruningMethod.${method.toUpperCase()}, amount=${amount})
pruned = prune_model(model, config)
total = sum(p.numel() for p in pruned.parameters())
zeros = sum((p.data == 0).sum().item() for p in pruned.parameters())
sparsity = zeros / total if total > 0 else 0
output = {
    "status": "success",
    "original_parameters": original_params,
    "sparsity": round(sparsity, 4),
    "sparsity_percent": f"{sparsity:.1%}"
}
`;
    const result = runPython(code);
    if (result.status === "error") return { error: result.error };
    return result;
  }
}

export default new NeuralForgePruneTool();
