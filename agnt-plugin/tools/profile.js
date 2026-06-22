import { runPython } from "./_bridge.mjs";

class NeuralForgeProfileTool {
  constructor() {
    this.name = "neuralforge_profile";
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
input_shape = spec.data_profile.input_shape if spec.data_profile else (3, 32, 32)
profile = profile_model(model, input_shape=(1, *input_shape))
output = {"status": "success", **profile}
`;
    const result = runPython(code);
    if (result.status === "error") return { error: result.error };
    return result;
  }
}

export default new NeuralForgeProfileTool();
