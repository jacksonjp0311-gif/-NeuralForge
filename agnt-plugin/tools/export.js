import { runPython } from "./_bridge.mjs";

class NeuralForgeExportTool {
  constructor() {
    this.name = "neuralforge_export";
  }

  async execute(params) {
    const description = params.description || params._raw || "";
    if (!description) {
      return { error: "Missing required parameter: description" };
    }

    const format = params.format || "pytorch_state_dict";
    const outputPath = params.output_path || "./neuralforge_output";

    const escaped = description.replace(/\\/g, "\\\\").replace(/"/g, '\\"');

    const code = `
spec = NeuralForgeSpec.from_description("${escaped}")
model = create_model(spec)
config = ExportConfig(format=ExportFormat.${format.toUpperCase()}, output_path="${outputPath}")
path = export_model(model, config)
output = {"status": "success", "export_path": str(path), "format": "${format}"}
`;
    const result = runPython(code);
    if (result.status === "error") return { error: result.error };
    return result;
  }
}

export default new NeuralForgeExportTool();
