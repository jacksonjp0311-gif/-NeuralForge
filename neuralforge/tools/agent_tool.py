from __future__ import annotations
import json, logging
from typing import Any, Dict, List, Optional
from neuralforge.spec import NeuralForgeSpec, TrainingConfig, OptimizationGoal, ExportConfig, ExportFormat
from neuralforge.core.forge import create_model, train, optimize, evaluate_and_report, evolve, auto_architecture, export_model
from neuralforge.training.engine import TrainingEngine
from neuralforge.evaluation.evaluator import ModelEvaluator
import torch
logger = logging.getLogger("neuralforge.tools.agent")

class NeuralForgeAgentTool:
    def __init__(self, name="neuralforge"):
        self.name = name
        self.description = "NeuralForge v2.5 - Build, train, optimize, and deploy neural networks."
        self._actions = {"create_model": self._action_create_model, "train": self._action_train,
                        "optimize": self._action_optimize, "evaluate": self._action_evaluate,
                        "evolve": self._action_evolve, "auto_architecture": self._action_auto_arch,
                        "export": self._action_export, "full_pipeline": self._action_full_pipeline}
    def invoke(self, params):
        action = params.get("action")
        if action not in self._actions: return {"status": "error", "error": f"Unknown action: {action}"}
        try: return {"status": "success", "action": action, "result": self._actions[action](params)}
        except Exception as e: return {"status": "error", "action": action, "error": str(e)}
    def get_available_actions(self):
        return [{"name": k, "description": v.__doc__ or ""} for k, v in self._actions.items()]

    def _action_create_model(self, params):
        desc = params.get("description", "")
        spec = NeuralForgeSpec.from_description(desc) if desc else NeuralForgeSpec(**params.get("spec", {}))
        model = create_model(spec)
        return {"model_name": spec.name, "parameters": model.count_parameters(), "architecture": spec.architecture.family.value, "config_hash": spec.config_hash()}

    def _action_train(self, params):
        """
        Train a neural network end-to-end.

        Params:
            description (str): Natural language model description.
            epochs (int): Number of training epochs. Default 10.
            batch_size (int): Training batch size. Default 32.
            learning_rate (float): Learning rate. Default 0.001.
            precision (str): Training precision (fp32, fp16, bf16, mixed). Default "mixed".
            seed (int): Random seed for reproducibility. Default 42.
            val_split (float): Fraction of data for validation. Default 0.2.

        Synthetic data is generated for demonstration when real data is not available.
        """
        desc = params.get("description", "simple CNN for image classification")
        spec = NeuralForgeSpec.from_description(desc)

        epochs = int(params.get("epochs", 10))
        batch_size = int(params.get("batch_size", 32))
        lr = float(params.get("learning_rate", 0.001))
        precision = str(params.get("precision", "mixed"))
        seed = int(params.get("seed", 42))
        val_split = float(params.get("val_split", 0.2))

        spec.training = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            precision=precision,
            seed=seed,
        )

        model = create_model(spec)
        input_shape = spec.data_profile.input_shape if spec.data_profile else (3, 32, 32)
        nc = spec.data_profile.num_classes if spec.data_profile and spec.data_profile.num_classes else 10
        n_samples = int(params.get("n_samples", 200))

        torch.manual_seed(seed)

        # Generate synthetic data for demonstration
        if len(input_shape) == 1:
            X = torch.randint(0, 1000, (n_samples, input_shape[0])).float()
        else:
            X = torch.randn(n_samples, *input_shape)
        y = torch.randint(0, nc, (n_samples,))

        # Split into train and validation
        n_val = int(n_samples * val_split)
        X_train = X[n_val:]
        y_train = y[n_val:]
        X_val = X[:n_val]
        y_val = y[:n_val]

        from torch.utils.data import TensorDataset, DataLoader
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
        )

        engine = TrainingEngine(model, spec, spec.training)
        result = engine.train(train_loader, val_loader)

        return {
            "model_name": spec.name,
            "epochs_completed": result.epochs_completed,
            "final_loss": result.final_loss,
            "best_metric": result.best_metric,
            "best_epoch": result.best_epoch,
            "training_time_seconds": round(result.training_time_seconds, 2),
            "status": result.status,
            "architecture": spec.architecture.family.value,
            "total_parameters": model.count_parameters(),
            "data_source": "synthetic",
            "note": "Trained on synthetic demo data. For real-data training, provide data tensors or load from a dataset file.",
        }

    def _action_optimize(self, params):
        obj_data = params.get("objective", {})
        obj = OptimizationGoal(**obj_data) if isinstance(obj_data, dict) else OptimizationGoal(objective=str(obj_data))
        result = optimize(obj)
        return result.model_dump()

    def _action_evaluate(self, params):
        """
        Evaluate a trained or freshly-created model on test data.

        Params:
            description (str): Natural language model description.
            model_weights (str, optional): Path to saved model weights. If None, uses a freshly initialized model.
            batch_size (int): Evaluation batch size. Default 32.
            num_classes (int): Number of output classes. Default 10.

        Returns accuracy, loss, macro-F1, per-class precision/recall, ECE, and recommendations.
        Uses synthetic data for demonstration when real test data is not provided.
        """
        desc = params.get("description", "simple CNN for image classification")
        spec = NeuralForgeSpec.from_description(desc)
        model = create_model(spec)

        batch_size = int(params.get("batch_size", 32))
        num_classes = int(params.get("num_classes", spec.data_profile.num_classes if spec.data_profile and spec.data_profile.num_classes else 10))
        n_test = int(params.get("n_test", 100))
        seed = int(params.get("seed", 42))

        # Load weights if provided
        weights_path = params.get("model_weights")
        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state_dict)
            except Exception as e:
                logger.warning(f"Could not load weights from {weights_path}: {e}. Using untrained model.")

        model.eval()
        input_shape = spec.data_profile.input_shape if spec.data_profile else (3, 32, 32)

        torch.manual_seed(seed)
        if len(input_shape) == 1:
            X_test = torch.randint(0, 1000, (n_test, input_shape[0])).float()
        else:
            X_test = torch.randn(n_test, *input_shape)
        y_test = torch.randint(0, num_classes, (n_test,))

        from torch.utils.data import TensorDataset, DataLoader
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=False,
        )

        evaluator = ModelEvaluator(model)
        report = evaluator.evaluate(test_loader, num_classes=num_classes)

        return {
            "model_name": spec.name,
            "architecture": spec.architecture.family.value,
            "metrics": report.metrics,
            "calibration_error": report.calibration_error,
            "recommendations": report.recommendations,
            "test_samples": n_test,
            "data_source": "synthetic",
            "note": "Evaluated on synthetic demo data. For real-data evaluation, provide a test data loader or dataset file.",
        }

    def _action_evolve(self, params):
        spec = NeuralForgeSpec(**params.get("spec", {})) if params.get("spec") else NeuralForgeSpec()
        best = evolve(spec, generations=params.get("generations", 20))
        return {"best_architecture": best.architecture.family.value, "config_hash": best.config_hash()}

    def _action_auto_arch(self, params):
        di = params.get("data_info", {})
        spec = auto_architecture(params.get("task", ""), di if isinstance(di, NeuralForgeSpec) else NeuralForgeSpec(**di) if di else NeuralForgeSpec())
        return {"architecture": spec.architecture.family.value, "config_hash": spec.config_hash()}

    def _action_export(self, params):
        """
        Export a trained model to a deployment format.

        Params:
            description (str): Natural language model description.
            format (str): Export format. One of: pytorch_state_dict, torchscript, onnx, safetensors.
                          Default "pytorch_state_dict".
            output_path (str): Output directory. Default "./neuralforge_output".
            model_weights (str, optional): Path to saved model weights to export.

        Returns the export path and format used.
        """
        desc = params.get("description", "simple CNN for image classification")
        spec = NeuralForgeSpec.from_description(desc)
        model = create_model(spec)

        export_format_str = params.get("format", "pytorch_state_dict")
        output_path = params.get("output_path", "./neuralforge_output")

        # Load weights if provided
        weights_path = params.get("model_weights")
        if weights_path:
            try:
                state_dict = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state_dict)
            except Exception as e:
                logger.warning(f"Could not load weights: {e}. Exporting untrained model state.")

        # Map string to ExportFormat enum using actual enum names
        format_map = {
            "pytorch_state_dict": ExportFormat.PYTORCH_STATE_DICT,
            "torchscript": ExportFormat.TORCHSCRIPT,
            "onnx": ExportFormat.ONNX,
            "safetensors": ExportFormat.SAFETENSORS,
        }
        fmt = format_map.get(export_format_str, ExportFormat.PYTORCH_STATE_DICT)

        config = ExportConfig(format=fmt, output_path=output_path)
        path = export_model(model, config)

        return {
            "model_name": spec.name,
            "architecture": spec.architecture.family.value,
            "export_path": str(path),
            "format": export_format_str,
            "total_parameters": model.count_parameters(),
            "note": f"Exported to {export_format_str}. For torchscript/onnx/safetensors, ensure required packages are installed.",
        }

    def _action_full_pipeline(self, params):
        desc = params.get("description", "")
        spec = NeuralForgeSpec.from_description(desc)
        model = create_model(spec)
        return {"status": "pipeline_initiated", "model_name": spec.name, "parameters": model.count_parameters(), "architecture": spec.architecture.family.value}

def as_tool(name="neuralforge"):
    return NeuralForgeAgentTool(name=name)
