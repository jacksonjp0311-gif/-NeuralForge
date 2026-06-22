import { runPython } from "./_bridge.mjs";

class NeuralForgeEvaluateTool {
  constructor() {
    this.name = "neuralforge_evaluate";
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
nc = spec.data_profile.num_classes if spec.data_profile and spec.data_profile.num_classes else 10
input_shape = spec.data_profile.input_shape if spec.data_profile else (3, 32, 32)
if len(input_shape) == 1:
    X = torch.randint(0, 1000, (100, input_shape[0]))
else:
    X = torch.randn(100, *input_shape)
y = torch.randint(0, nc, (100,))
from torch.utils.data import TensorDataset, DataLoader
evaluator = ModelEvaluator(model)
report = evaluator.evaluate(DataLoader(TensorDataset(X, y), batch_size=32), num_classes=nc)
output = {
    "status": "success",
    "metrics": report.metrics,
    "calibration_error": report.calibration_error,
    "recommendations": report.recommendations
}
`;
    const result = runPython(code);
    if (result.status === "error") return { error: result.error };
    return result;
  }
}

export default new NeuralForgeEvaluateTool();
