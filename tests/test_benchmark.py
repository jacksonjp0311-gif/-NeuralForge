"""
Tests for NeuralForge Held-Out Benchmark Flow (Priority C).
"""
import pytest
import json
import tempfile
from pathlib import Path

from neuralforge.benchmark import (
    split_data,
    run_learner_benchmark,
    run_full_benchmark,
    generate_synthetic_executions,
    BenchmarkResult,
)


class TestSplitData:
    def test_basic_split(self):
        data = [{"id": i} for i in range(100)]
        train, val, test = split_data(data, seed=42)
        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_split_reproducibility(self):
        data = [{"id": i} for i in range(100)]
        t1, v1, te1 = split_data(data, seed=42)
        t2, v2, te2 = split_data(data, seed=42)
        assert [d["id"] for d in t1] == [d["id"] for d in t2]
        assert [d["id"] for d in v1] == [d["id"] for d in v2]
        assert [d["id"] for d in te1] == [d["id"] for d in te2]

    def test_split_different_seeds(self):
        data = [{"id": i} for i in range(100)]
        t1, _, _ = split_data(data, seed=42)
        t2, _, _ = split_data(data, seed=99)
        assert [d["id"] for d in t1] != [d["id"] for d in t2]

    def test_split_no_overlap(self):
        data = [{"id": i} for i in range(100)]
        train, val, test = split_data(data, seed=42)
        train_ids = {d["id"] for d in train}
        val_ids = {d["id"] for d in val}
        test_ids = {d["id"] for d in test}
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_small_dataset(self):
        data = [{"id": i} for i in range(10)]
        train, val, test = split_data(data, seed=42)
        assert len(train) + len(val) + len(test) == 10

    def test_invalid_fractions(self):
        with pytest.raises(AssertionError):
            split_data([{"id": 0}], train_frac=0.5, val_frac=0.5, test_frac=0.5)


class TestBenchmarkResult:
    def test_to_dict(self):
        r = BenchmarkResult(
            component="DataLearner",
            metric_name="test_loss",
            metric_value=0.12345678,
            train_size=60,
            val_size=20,
            test_size=20,
            data_source="synthetic",
            seed=42,
            notes="test run",
        )
        d = r.to_dict()
        assert d["component"] == "DataLearner"
        assert d["value"] == 0.1235  # rounded
        assert d["split"]["train"] == 60
        assert d["data_source"] == "synthetic"
        assert "timestamp" in d


class TestRunLearnerBenchmark:
    def test_synthetic_benchmark(self):
        data = generate_synthetic_executions(n=100, seed=42)
        result = run_learner_benchmark(data, data_source="synthetic", seed=42)
        assert result["status"] == "success"
        assert "metrics" in result
        assert "train_loss" in result["metrics"]
        assert "val_loss" in result["metrics"]
        assert "test_loss" in result["metrics"]
        assert result["split"]["train_size"] > 0
        assert result["split"]["test_size"] > 0

    def test_insufficient_data(self):
        data = generate_synthetic_executions(n=5, seed=42)
        result = run_learner_benchmark(data, data_source="synthetic")
        assert result["status"] == "insufficient_data"

    def test_provenance_fields(self):
        data = generate_synthetic_executions(n=50, seed=42)
        result = run_learner_benchmark(data, data_source="synthetic", seed=42)
        assert result["data_source"] == "synthetic"
        assert result["seed"] == 42
        assert "timestamp" in result
        assert "wall_time_seconds" in result


class TestRunFullBenchmark:
    def test_full_benchmark_synthetic(self):
        data = generate_synthetic_executions(n=100, seed=42)
        report = run_full_benchmark(data, data_source="synthetic", seed=42)
        # Report has benchmark_version + components, not top-level status
        assert "components" in report
        assert "data_learner" in report["components"]
        assert "pattern_engine" in report["components"]
        assert "smart_engine" in report["components"]
        assert report["data_source"] == "synthetic"

    def test_full_benchmark_saves_report(self, tmp_path):
        data = generate_synthetic_executions(n=50, seed=42)
        output = tmp_path / "benchmark_report.json"
        report = run_full_benchmark(data, data_source="synthetic", output_path=str(output))
        assert output.exists()
        saved = json.loads(output.read_text())
        assert "components" in saved

    def test_full_benchmark_small_data(self):
        data = generate_synthetic_executions(n=5, seed=42)
        report = run_full_benchmark(data, data_source="synthetic")
        # Should still run but with insufficient_data for some components
        assert "components" in report


class TestGenerateSyntheticExecutions:
    def test_count(self):
        data = generate_synthetic_executions(n=50, seed=42)
        assert len(data) == 50

    def test_labeled_synthetic(self):
        data = generate_synthetic_executions(n=10, seed=42)
        for e in data:
            assert e["_data_source"] == "synthetic"

    def test_has_required_fields(self):
        data = generate_synthetic_executions(n=10, seed=42)
        for e in data:
            assert "workflow_id" in e
            assert "success" in e
            assert "duration_ms" in e
            assert "step_count" in e

    def test_reproducibility(self):
        d1 = generate_synthetic_executions(n=20, seed=42)
        d2 = generate_synthetic_executions(n=20, seed=42)
        assert [e["duration_ms"] for e in d1] == [e["duration_ms"] for e in d2]
