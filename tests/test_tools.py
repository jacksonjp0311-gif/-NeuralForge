"""Tests for agent tool wrappers."""
from __future__ import annotations
import pytest
import json
from neuralforge.tools.agent_tool import NeuralForgeAgentTool, as_tool
from neuralforge.core.forge import create_model
from neuralforge.spec import NeuralForgeSpec


class TestNeuralForgeAgentTool:
    def test_create(self):
        tool = as_tool()
        assert tool.name == "neuralforge"

    def test_available_actions(self):
        tool = as_tool()
        actions = tool.get_available_actions()
        assert len(actions) > 0
        action_names = [a["name"] for a in actions]
        assert "create_model" in action_names
        assert "train" in action_names
        assert "optimize" in action_names

    def test_invoke_create_model(self):
        tool = as_tool()
        result = tool.invoke({
            "action": "create_model",
            "description": "Build a CNN for CIFAR-10",
        })
        assert result["status"] == "success"
        assert result["action"] == "create_model"
        r = result["result"]
        assert "model_name" in r
        assert "parameters" in r

    def test_invoke_unknown_action(self):
        tool = as_tool()
        result = tool.invoke({"action": "nonexistent"})
        assert result["status"] == "error"

    def test_invoke_full_pipeline(self):
        tool = as_tool()
        result = tool.invoke({
            "action": "full_pipeline",
            "description": "Build a ResNet for CIFAR-10 with <5M params",
        })
        assert result["status"] == "success"
        r = result["result"]
        assert "model_name" in r
        assert "parameters" in r

    def test_invoke_optimize(self):
        tool = as_tool()
        result = tool.invoke({
            "action": "optimize",
            "objective": {"objective": "accuracy", "direction": "maximize"},
        })
        assert result["status"] == "success"

    def test_invoke_evolve(self):
        tool = as_tool()
        result = tool.invoke({
            "action": "evolve",
            "generations": 5,
        })
        assert result["status"] == "success"
