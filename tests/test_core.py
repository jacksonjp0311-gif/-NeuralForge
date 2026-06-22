import pytest
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import neuralforge as nf
from neuralforge.spec import *
from neuralforge.core.forge import create_model, NeuralForgeModule, train, evaluate_and_report
from neuralforge.core.registry import ModelRegistry

class TestNeuralForgeSpec:
    def test_basic_creation(self):
        spec = NeuralForgeSpec(name="test-model")
        assert spec.name == "test-model"
    def test_from_description_classifier(self):
        spec = NeuralForgeSpec.from_description("image classifier for CIFAR-10 with < 5M params that reaches >92% accuracy")
        assert spec.task_type == TaskType.IMAGE_CLASSIFICATION
        assert spec.constraints.max_parameters == 5_000_000
        assert spec.constraints.min_accuracy == 0.92
        assert spec.data_profile is not None and spec.data_profile.num_classes == 10
    def test_from_description_text(self):
        spec = NeuralForgeSpec.from_description("text classification model for sentiment analysis")
        assert spec.task_type == TaskType.TEXT_CLASSIFICATION
    def test_from_description_transformer(self):
        spec = NeuralForgeSpec.from_description("transformer language model for text generation")
        assert spec.architecture.family == ArchitectureFamily.TRANSFORMER
    def test_from_description_vit(self):
        spec = NeuralForgeSpec.from_description("vision transformer for image classification")
        assert spec.architecture.family == ArchitectureFamily.VISION_TRANSFORMER
    def test_from_description_memory(self):
        spec = NeuralForgeSpec.from_description("model within 8GB VRAM")
        assert spec.constraints.max_memory_mb == 8 * 1024
    def test_from_description_time(self):
        spec = NeuralForgeSpec.from_description("model trained within 30 minutes")
        assert spec.budget.max_hours == 0.5
    def test_hash_consistency(self):
        s1 = NeuralForgeSpec(name="test"); s2 = NeuralForgeSpec(name="test")
        assert s1.config_hash() == s2.config_hash()
    def test_hash_uniqueness(self):
        s1 = NeuralForgeSpec(name="a"); s2 = NeuralForgeSpec(name="b")
        assert s1.config_hash() != s2.config_hash()
    def test_yaml_roundtrip(self):
        spec = NeuralForgeSpec(name="yt", architecture=ArchitectureSpec(family=ArchitectureFamily.RESNET, depth=4, width=128))
        yaml_str = spec.to_yaml()
        restored = NeuralForgeSpec.from_yaml(yaml_str)
        assert restored.name == "yt" and restored.architecture.depth == 4

class TestModelCreation:
    def _make_spec(self, family=ArchitectureFamily.RESNET):
        return NeuralForgeSpec(name=f"test-{family.value}", task_type=TaskType.IMAGE_CLASSIFICATION,
            data_profile=DataProfile(task_type=TaskType.IMAGE_CLASSIFICATION, input_shape=(3,32,32), num_classes=10, data_format="image"),
            architecture=ArchitectureSpec(family=family, depth=3, width=64))
    def test_create_cnn(self):
        m = create_model(self._make_spec(ArchitectureFamily.CNN))
        assert isinstance(m, NeuralForgeModule)
    def test_create_resnet(self):
        m = create_model(self._make_spec(ArchitectureFamily.RESNET))
        assert isinstance(m, NeuralForgeModule)
    def test_create_vit(self):
        m = create_model(self._make_spec(ArchitectureFamily.VISION_TRANSFORMER))
        assert isinstance(m, NeuralForgeModule)
    def test_create_auto(self):
        m = create_model(self._make_spec(ArchitectureFamily.AUTO))
        assert isinstance(m, NeuralForgeModule)
    def test_forward_cnn(self):
        m = create_model(self._make_spec(ArchitectureFamily.CNN))
        x = torch.randn(2,3,32,32); out = m(x); assert out.shape == (2,10)
    def test_forward_resnet(self):
        m = create_model(self._make_spec(ArchitectureFamily.RESNET))
        x = torch.randn(2,3,32,32); out = m(x); assert out.shape == (2,10)
    def test_parameter_count(self):
        m = create_model(self._make_spec(ArchitectureFamily.RESNET))
        assert m.count_parameters() > 0

class TestTraining:
    def _make_data(self, n=200):
        X = torch.randn(n, 3, 32, 32); y = torch.randint(0, 10, (n,))
        return NeuralForgeSpec(name="tr", task_type=TaskType.IMAGE_CLASSIFICATION,
            data_profile=DataProfile(task_type=TaskType.IMAGE_CLASSIFICATION, input_shape=(3,32,32), num_classes=10, data_format="image"),
            architecture=ArchitectureSpec(family=ArchitectureFamily.CNN, depth=2, width=32)), TensorDataset(X, y)
    def test_basic_training(self):
        spec, dataset = self._make_data(); model = create_model(spec)
        result = train(model, DataLoader(dataset, batch_size=32), TrainingConfig(epochs=2, batch_size=32))
        assert result.epochs_completed == 2 and result.status == "completed"
    def test_training_history(self):
        spec, dataset = self._make_data(); model = create_model(spec)
        result = train(model, DataLoader(dataset, batch_size=32), TrainingConfig(epochs=3, batch_size=32))
        assert "train_loss" in result.history and len(result.history["train_loss"]) == 3

class TestEvaluation:
    def test_basic_eval(self):
        spec = NeuralForgeSpec(name="ev", task_type=TaskType.IMAGE_CLASSIFICATION,
            data_profile=DataProfile(task_type=TaskType.IMAGE_CLASSIFICATION, input_shape=(3,32,32), num_classes=10, data_format="image"),
            architecture=ArchitectureSpec(family=ArchitectureFamily.CNN, depth=2, width=32))
        model = create_model(spec)
        X = torch.randn(100, 3, 32, 32); y = torch.randint(0, 10, (100,))
        report = evaluate_and_report(model, DataLoader(TensorDataset(X, y), batch_size=32))
        assert "accuracy" in report.metrics and 0 <= report.metrics["accuracy"] <= 1

class TestToolWrappers:
    def test_as_tool(self):
        tool = nf.as_tool(); actions = tool.get_available_actions(); assert len(actions) > 0
    def test_tool_create(self):
        tool = nf.as_tool()
        r = tool.invoke({"action": "create_model", "description": "CNN for CIFAR-10"})
        assert r["status"] == "success" and "parameters" in r["result"]
    def test_tool_pipeline(self):
        tool = nf.as_tool()
        r = tool.invoke({"action": "full_pipeline", "description": "image classifier for CIFAR-10"})
        assert r["status"] == "success"
    def test_tool_unknown(self):
        tool = nf.as_tool()
        r = tool.invoke({"action": "nonexistent"})
        assert r["status"] == "error"

class TestIntegration:
    def test_full_pipeline(self):
        spec = NeuralForgeSpec.from_description("CNN image classifier for CIFAR-10")
        model = create_model(spec); assert model.count_parameters() > 0
        X = torch.randn(200, 3, 32, 32); y = torch.randint(0, 10, (200,))
        result = train(model, DataLoader(TensorDataset(X, y), batch_size=32), TrainingConfig(epochs=2, batch_size=32))
        assert result.status == "completed"
        X_test = torch.randn(50, 3, 32, 32); y_test = torch.randint(0, 10, (50,))
        report = evaluate_and_report(model, DataLoader(TensorDataset(X_test, y_test), batch_size=16))
        assert "accuracy" in report.metrics
    def test_meta_optimizer(self):
        from neuralforge.optimize.meta_optimizer import MetaOptimizer
        meta = MetaOptimizer(memory_path="./test_meta.json")
        critique = meta.critique(NeuralForgeSpec(name="t"),
            TrainingResult(model_name="t", spec_hash="a", epochs_completed=5, total_steps=100, final_loss=0.5,
                          history={"train_loss": [1.0,0.8,0.6,0.5,0.5]}))
        assert "score" in critique
    def test_insights_store(self):
        from neuralforge.memory.insights_store import InsightsStore
        store = InsightsStore(store_path="./test_insights")
        store.add_insight("ResNet d=4 w=128", "Acc 85%", {"accuracy": 0.85, "loss": 0.5})
        store.add_insight("CNN d=3 w=64", "Acc 80%", {"accuracy": 0.80, "loss": 0.7})
        assert store.get_statistics()["count"] == 2
    def test_profiling(self):
        from neuralforge.utils.profiling import profile_model
        spec = NeuralForgeSpec(name="pf", architecture=ArchitectureSpec(family=ArchitectureFamily.CNN, depth=2, width=32))
        model = create_model(spec)
        r = profile_model(model, input_shape=(1, 3, 32, 32), num_iterations=10)
        assert "total_parameters" in r and "latency_mean_ms" in r
