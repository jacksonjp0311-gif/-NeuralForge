"""Tests for evaluation module."""
from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from neuralforge.evaluation.evaluator import ModelEvaluator
from neuralforge.spec import EvaluationReport


class TestModelEvaluator:
    @pytest.fixture
    def simple_model(self):
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    @pytest.fixture
    def test_loader(self):
        X = torch.randn(50, 10)
        y = torch.randint(0, 3, (50,))
        return DataLoader(TensorDataset(X, y), batch_size=16)

    def test_evaluate_basic(self, simple_model, test_loader):
        evaluator = ModelEvaluator(simple_model, device=torch.device("cpu"))
        report = evaluator.evaluate(test_loader)
        assert isinstance(report, EvaluationReport)
        assert "accuracy" in report.metrics
        assert "loss" in report.metrics

    def test_accuracy_range(self, simple_model, test_loader):
        evaluator = ModelEvaluator(simple_model, device=torch.device("cpu"))
        report = evaluator.evaluate(test_loader)
        assert 0.0 <= report.metrics["accuracy"] <= 1.0

    def test_confusion_matrix(self, simple_model, test_loader):
        evaluator = ModelEvaluator(simple_model, device=torch.device("cpu"))
        report = evaluator.evaluate(test_loader, num_classes=3)
        assert report.confusion_matrix is not None
        cm = report.confusion_matrix
        assert len(cm) == 3
        assert len(cm[0]) == 3

    def test_per_class_metrics(self, simple_model, test_loader):
        evaluator = ModelEvaluator(simple_model, device=torch.device("cpu"))
        report = evaluator.evaluate(test_loader, num_classes=3)
        assert report.per_class_metrics is not None
        assert len(report.per_class_metrics) == 3

    def test_calibration_error(self, simple_model, test_loader):
        evaluator = ModelEvaluator(simple_model, device=torch.device("cpu"))
        report = evaluator.evaluate(test_loader)
        assert report.calibration_error is not None
        assert report.calibration_error >= 0.0

    def test_failure_analysis(self, simple_model, test_loader):
        evaluator = ModelEvaluator(simple_model, device=torch.device("cpu"))
        report = evaluator.evaluate(test_loader, detailed=True)
        assert report.failure_analysis is not None
        assert "failure_rate" in report.failure_analysis

    def test_recommendations(self, simple_model, test_loader):
        evaluator = ModelEvaluator(simple_model, device=torch.device("cpu"))
        report = evaluator.evaluate(test_loader)
        assert len(report.recommendations) > 0
