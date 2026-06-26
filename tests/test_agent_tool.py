"""
Tests for NeuralForgeAgentTool wrapper (Priority E).
Tests that _action_train, _action_evaluate, _action_export are fully implemented.
"""
import pytest
from neuralforge.tools.agent_tool import NeuralForgeAgentTool, as_tool


@pytest.fixture
def tool():
    return as_tool()


class TestNeuralForgeAgentTool:
    def test_available_actions(self, tool):
        actions = tool.get_available_actions()
        action_names = [a["name"] for a in actions]
        assert "train" in action_names
        assert "evaluate" in action_names
        assert "export" in action_names
        assert "create_model" in action_names
        assert "optimize" in action_names
        assert "evolve" in action_names

    def test_unknown_action(self, tool):
        result = tool.invoke({"action": "nonexistent"})
        assert result["status"] == "error"
        assert "Unknown action" in result["error"]


class TestActionTrain:
    def test_train_returns_metrics(self, tool):
        result = tool.invoke({
            "action": "train",
            "description": "simple CNN for image classification",
            "epochs": 3,
            "batch_size": 16,
            "seed": 42,
        })
        assert result["status"] == "success"
        assert result["action"] == "train"
        train_result = result["result"]
        assert "epochs_completed" in train_result
        assert "final_loss" in train_result
        assert "training_time_seconds" in train_result
        assert train_result["status"] in ("completed", "early_stopped")

    def test_train_default_params(self, tool):
        result = tool.invoke({
            "action": "train",
            "description": "MLP for text classification",
        })
        assert result["status"] == "success"
        assert result["result"]["epochs_completed"] > 0

    def test_train_synthetic_labeled(self, tool):
        result = tool.invoke({
            "action": "train",
            "description": "ResNet for CIFAR-10",
            "epochs": 2,
        })
        assert result["result"]["data_source"] == "synthetic"
        assert "note" in result["result"]

    def test_train_with_val_split(self, tool):
        result = tool.invoke({
            "action": "train",
            "description": "CNN for image classification",
            "epochs": 2,
            "val_split": 0.3,
        })
        assert result["status"] == "success"


class TestActionEvaluate:
    def test_evaluate_returns_metrics(self, tool):
        result = tool.invoke({
            "action": "evaluate",
            "description": "simple CNN for image classification",
        })
        assert result["status"] == "success"
        assert result["action"] == "evaluate"
        eval_result = result["result"]
        assert "metrics" in eval_result
        assert "calibration_error" in eval_result
        assert "recommendations" in eval_result

    def test_evaluate_metrics_content(self, tool):
        result = tool.invoke({
            "action": "evaluate",
            "description": "ResNet for CIFAR-10",
        })
        metrics = result["result"]["metrics"]
        assert "accuracy" in metrics
        assert "loss" in metrics

    def test_evaluate_synthetic_labeled(self, tool):
        """Evaluate on CNN (image classification) — synthetic data is compatible."""
        result = tool.invoke({
            "action": "evaluate",
            "description": "simple CNN for image classification",
        })
        assert result["status"] == "success"
        assert result["result"]["data_source"] == "synthetic"

    def test_evaluate_custom_num_classes(self, tool):
        result = tool.invoke({
            "action": "evaluate",
            "description": "CNN for image classification",
            "num_classes": 5,
        })
        assert result["status"] == "success"


class TestActionExport:
    def test_export_default_format(self, tool):
        result = tool.invoke({
            "action": "export",
            "description": "simple CNN for image classification",
        })
        assert result["status"] == "success"
        assert result["action"] == "export"
        export_result = result["result"]
        assert "export_path" in export_result
        assert "format" in export_result
        assert export_result["format"] == "pytorch_state_dict"

    def test_export_torchscript(self, tool):
        result = tool.invoke({
            "action": "export",
            "description": "simple CNN for image classification",
            "format": "torchscript",
        })
        assert result["status"] == "success"
        assert result["result"]["format"] == "torchscript"

    def test_export_onnx(self, tool):
        result = tool.invoke({
            "action": "export",
            "description": "simple CNN for image classification",
            "format": "onnx",
        })
        assert result["status"] == "success"
        assert result["result"]["format"] == "onnx"

    def test_export_custom_output_path(self, tool, tmp_path):
        result = tool.invoke({
            "action": "export",
            "description": "simple CNN for image classification",
            "output_path": str(tmp_path / "my_model"),
        })
        assert result["status"] == "success"
        assert str(tmp_path / "my_model") in result["result"]["export_path"]

    def test_export_has_note(self, tool):
        result = tool.invoke({
            "action": "export",
            "description": "simple CNN for image classification",
        })
        assert "note" in result["result"]
