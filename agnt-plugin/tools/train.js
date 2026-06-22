import { runPython } from "./_bridge.mjs";

class NeuralForgeTrainTool {
  constructor() {
    this.name = "neuralforge_train";
  }

  async execute(params) {
    const description = params.description || params._raw || "";
    if (!description) {
      return { error: "Missing required parameter: description" };
    }

    const epochs = params.epochs || 10;
    const batchSize = params.batch_size || 32;
    const lr = params.learning_rate || 0.001;
    const precision = params.precision || "mixed";
    const seed = params.seed || 42;

    const escaped = description.replace(/\\/g, "\\\\").replace(/"/g, '\\"');

    const code = `
spec = NeuralForgeSpec.from_description("${escaped}")
spec.training = TrainingConfig(epochs=${epochs}, batch_size=${batchSize}, learning_rate=${lr}, precision="${precision}", seed=${seed})
model = create_model(spec)
nc = spec.data_profile.num_classes if spec.data_profile and spec.data_profile.num_classes else 10
input_shape = spec.data_profile.input_shape if spec.data_profile else (3, 32, 32)
torch.manual_seed(${seed})
if len(input_shape) == 1:
    X = torch.randint(0, 1000, (200, input_shape[0]))
else:
    X = torch.randn(200, *input_shape)
y = torch.randint(0, nc, (200,))
from torch.utils.data import TensorDataset, DataLoader
X_tr, y_tr = X[40:], y[40:]
X_va, y_va = X[:40], y[:40]
engine = TrainingEngine(model, spec, spec.training)
result = engine.train(DataLoader(TensorDataset(X_tr, y_tr), batch_size=${batchSize}, shuffle=True), DataLoader(TensorDataset(X_va, y_va), batch_size=${batchSize}))
output = {
    "status": "success",
    "epochs_completed": result.epochs_completed,
    "final_loss": result.final_loss,
    "best_metric": result.best_metric,
    "best_epoch": result.best_epoch,
    "training_time_seconds": round(result.training_time_seconds, 2),
    "status_flag": result.status
}
`;
    const result = runPython(code, 180000);
    if (result.status === "error") return { error: result.error };
    return result;
  }
}

export default new NeuralForgeTrainTool();
