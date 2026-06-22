import { runPython } from "./_bridge.mjs";

class NeuralForgeCreateTool {
  constructor() {
    this.name = "neuralforge_create";
  }

  async execute(params) {
    const description = params.description || params._raw || "";
    if (!description) {
      return { error: "Missing required parameter: description" };
    }

    const escaped = description.replace(/\\/g, "\\\\").replace(/"/g, '\\"');

    const code = `
spec = NeuralForgeSpec.from_description("${escaped}")
model = create_model(spec)
output = {
    "status": "success",
    "model_name": spec.name,
    "parameters": model.count_parameters(),
    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "architecture": spec.architecture.family.value,
    "depth": spec.architecture.depth,
    "width": spec.architecture.width,
    "task_type": spec.task_type.value,
    "config_hash": spec.config_hash()
}
`;
    const result = runPython(code);
    if (result.status === "error") return { error: result.error };
    return result;
  }
}

export default new NeuralForgeCreateTool();
