import { runPython } from "./_bridge.mjs";

class NeuralForgeQuantizeTool {
  constructor() {
    this.name = "neuralforge_quantize";
  }

  async execute(params) {
    const description = params.description || params._raw || "";
    if (!description) {
      return { error: "Missing required parameter: description" };
    }

    const method = params.method || "dynamic_int8";

    const escaped = description.replace(/\\/g, "\\\\").replace(/"/g, '\\"');

    const code = `
spec = NeuralForgeSpec.from_description("${escaped}")
model = create_model(spec)
original_size = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024 / 1024
config = QuantizationConfig(method=QuantizationMethod.${method.upper()})
quantized = quantize_model(model, config)
quantized_size = sum(p.nelement() * p.element_size() for p in quantized.parameters()) / 1024 / 1024
output = {
    "status": "success",
    "method": "${method}",
    "original_size_mb": round(original_size, 2),
    "quantized_size_mb": round(quantized_size, 2),
    "compression_ratio": f"{original_size / max(quantized_size, 0.01):.1f}x"
}
`;
    const result = runPython(code);
    if (result.status === "error") return { error: result.error };
    return result;
  }
}

export default new NeuralForgeQuantizeTool();
