"""Tests for model creation and forward passes."""
from __future__ import annotations
import pytest
import torch
from neuralforge.spec import (
    NeuralForgeSpec, ArchitectureSpec, ArchitectureFamily,
    DataProfile, TaskType,
)
from neuralforge.core.forge import create_model, NeuralForgeModule


class TestModelCreation:
    def test_create_cnn(self):
        spec = NeuralForgeSpec(
            name="test-cnn",
            architecture=ArchitectureSpec(family=ArchitectureFamily.CNN, depth=3, width=32),
            data_profile=DataProfile(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                input_shape=(3, 32, 32),
                num_classes=10,
            ),
        )
        model = create_model(spec)
        assert isinstance(model, NeuralForgeModule)
        assert model.count_parameters() > 0

    def test_create_resnet(self):
        spec = NeuralForgeSpec(
            name="test-resnet",
            architecture=ArchitectureSpec(family=ArchitectureFamily.RESNET, depth=3, width=32),
            data_profile=DataProfile(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                input_shape=(3, 32, 32),
                num_classes=10,
            ),
        )
        model = create_model(spec)
        assert model.count_parameters() > 0

    def test_create_transformer(self):
        spec = NeuralForgeSpec(
            name="test-transformer",
            architecture=ArchitectureSpec(
                family=ArchitectureFamily.TRANSFORMER,
                depth=2, embedding_dim=64, num_heads=4,
            ),
            data_profile=DataProfile(
                task_type=TaskType.TEXT_CLASSIFICATION,
                input_shape=(64,),
                vocab_size=1000,
                num_classes=4,
                max_sequence_length=64,
                data_format="text",
            ),
        )
        model = create_model(spec)
        assert model.count_parameters() > 0

    def test_create_vit(self):
        spec = NeuralForgeSpec(
            name="test-vit",
            architecture=ArchitectureSpec(
                family=ArchitectureFamily.VISION_TRANSFORMER,
                depth=2, embedding_dim=64, num_heads=4,
            ),
            data_profile=DataProfile(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                input_shape=(3, 32, 32),
                num_classes=10,
            ),
        )
        model = create_model(spec)
        assert model.count_parameters() > 0

    def test_create_mlp_mixer(self):
        spec = NeuralForgeSpec(
            name="test-mixer",
            architecture=ArchitectureSpec(
                family=ArchitectureFamily.MLP_MIXER,
                depth=2, embedding_dim=64,
            ),
            data_profile=DataProfile(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                input_shape=(3, 32, 32),
                num_classes=10,
            ),
        )
        model = create_model(spec)
        assert model.count_parameters() > 0

    def test_create_kan(self):
        spec = NeuralForgeSpec(
            name="test-kan",
            architecture=ArchitectureSpec(
                family=ArchitectureFamily.KAN, depth=2, width=32,
            ),
            data_profile=DataProfile(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                input_shape=(3, 32, 32),
                num_classes=10,
            ),
        )
        model = create_model(spec)
        assert model.count_parameters() > 0


class TestForwardPass:
    def test_cnn_forward(self):
        spec = NeuralForgeSpec(
            name="fwd-cnn",
            architecture=ArchitectureSpec(family=ArchitectureFamily.CNN, depth=2, width=16),
            data_profile=DataProfile(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                input_shape=(3, 32, 32),
                num_classes=5,
            ),
        )
        model = create_model(spec)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 5)

    def test_resnet_forward(self):
        spec = NeuralForgeSpec(
            name="fwd-resnet",
            architecture=ArchitectureSpec(family=ArchitectureFamily.RESNET, depth=2, width=16),
            data_profile=DataProfile(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                input_shape=(3, 32, 32),
                num_classes=5,
            ),
        )
        model = create_model(spec)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 5)

    def test_transformer_forward(self):
        spec = NeuralForgeSpec(
            name="fwd-transformer",
            architecture=ArchitectureSpec(
                family=ArchitectureFamily.TRANSFORMER,
                depth=2, embedding_dim=32, num_heads=4,
            ),
            data_profile=DataProfile(
                task_type=TaskType.TEXT_CLASSIFICATION,
                input_shape=(32,),
                vocab_size=500,
                num_classes=3,
                max_sequence_length=32,
                data_format="text",
            ),
        )
        model = create_model(spec)
        x = torch.randint(0, 500, (2, 32))
        out = model(x)
        assert out.shape == (2, 3)

    def test_vit_forward(self):
        spec = NeuralForgeSpec(
            name="fwd-vit",
            architecture=ArchitectureSpec(
                family=ArchitectureFamily.VISION_TRANSFORMER,
                depth=2, embedding_dim=32, num_heads=4,
            ),
            data_profile=DataProfile(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                input_shape=(3, 32, 32),
                num_classes=5,
            ),
        )
        model = create_model(spec)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 5)

    def test_auto_inference(self):
        spec = NeuralForgeSpec(
            name="fwd-auto",
            architecture=ArchitectureSpec(family=ArchitectureFamily.AUTO),
            data_profile=DataProfile(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                input_shape=(3, 32, 32),
                num_classes=5,
            ),
        )
        model = create_model(spec)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 5)

    def test_gradient_flow(self):
        spec = NeuralForgeSpec(
            name="grad-test",
            architecture=ArchitectureSpec(family=ArchitectureFamily.CNN, depth=2, width=16),
            data_profile=DataProfile(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                input_shape=(3, 32, 32),
                num_classes=5,
            ),
        )
        model = create_model(spec)
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
