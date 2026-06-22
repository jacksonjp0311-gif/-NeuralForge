from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neuralforge.spec import EvaluationReport

logger = logging.getLogger("neuralforge.evaluation")


class ModelEvaluator:
    """Comprehensive model evaluation with multi-objective neural quality prediction.

    v2 enhancements:
      - Multi-objective quality prediction (accuracy + latency + memory)
      - Real-data trained quality predictor support
      - Pareto-front architecture recommendations
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        quality_predictor: Optional[Any] = None,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.quality_predictor = quality_predictor

    def evaluate(
        self,
        test_loader: DataLoader,
        detailed: bool = True,
        num_classes: Optional[int] = None,
        training_history: Optional[Dict[str, List[float]]] = None,
        arch_metadata: Optional[Dict] = None,
        predict_latency: bool = False,
    ) -> EvaluationReport:
        """Run full evaluation with optional multi-objective quality prediction."""
        self.model.eval()
        all_preds, all_targets, all_probs = [], [], []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = batch.to(self.device)
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                loss = F.cross_entropy(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        preds = np.array(all_preds)
        targets = np.array(all_targets)
        probs = np.array(all_probs)
        nc = num_classes or int(targets.max()) + 1
        accuracy = float((preds == targets).mean())
        avg_loss = total_loss / max(num_batches, 1)

        confusion = np.zeros((nc, nc), dtype=int)
        for t, p in zip(targets, preds):
            confusion[t][p] += 1

        per_class = {}
        for c in range(nc):
            tp = int(confusion[c, c])
            fp = int(confusion[:, c].sum() - tp)
            fn = int(confusion[c, :].sum() - tp)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            per_class[f"class_{c}"] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": int((targets == c).sum())}

        calibration_error = self._compute_ece(probs, targets, nc)
        robustness = {"clean_accuracy": accuracy} if detailed else None
        failure_analysis = self._analyze_failures(preds, targets, probs, nc) if detailed else None

        # Multi-objective quality prediction
        quality_prediction = None
        if training_history and self.quality_predictor:
            try:
                meta = arch_metadata or {}
                quality_prediction = self.quality_predictor.predict(
                    history=training_history,
                    num_params=meta.get("num_params", 1_000_000),
                    depth=meta.get("depth", 4),
                    width=meta.get("width", 64),
                    num_classes=nc,
                    task_type=meta.get("task_type", "image_classification"),
                    family_type=meta.get("family_type", "resnet"),
                )
            except Exception as e:
                logger.warning(f"Quality prediction failed: {e}")

        recommendations = self._generate_recommendations(accuracy, per_class, calibration_error, confusion, quality_prediction)

        return EvaluationReport(
            model_name="model",
            metrics={"accuracy": round(accuracy, 4), "loss": round(avg_loss, 4), "num_samples": len(targets),
                     "macro_f1": round(np.mean([v["f1"] for v in per_class.values()]), 4)},
            per_class_metrics=per_class,
            confusion_matrix=confusion.tolist(),
            robustness_scores=robustness,
            calibration_error=round(calibration_error, 4),
            failure_analysis=failure_analysis,
            recommendations=recommendations,
        )

    def _compute_ece(self, probs, targets, num_classes, n_bins=10):
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == targets).astype(float)
        ece = 0.0
        bin_edges = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if mask.sum() > 0:
                ece += mask.sum() / len(targets) * abs(accuracies[mask].mean() - confidences[mask].mean())
        return float(ece)

    def _analyze_failures(self, preds, targets, probs, num_classes):
        failures = preds != targets
        confusion_pairs = []
        for c in range(num_classes):
            for p in range(num_classes):
                if c != p and ((targets == c) & (preds == p)).sum() > 0:
                    confusion_pairs.append({"true": c, "predicted": p, "count": int(((targets == c) & (preds == p)).sum())})
        confusion_pairs.sort(key=lambda x: x["count"], reverse=True)
        max_probs = probs.max(axis=1)
        return {"failure_rate": round(float(failures.mean()), 4), "total_failures": int(failures.sum()),
                "top_confused_pairs": confusion_pairs[:5], "high_confidence_failures": int((failures & (max_probs > 0.9)).sum())}

    def _generate_recommendations(self, accuracy, per_class, ece, confusion, quality_prediction=None):
        recs = []
        if accuracy < 0.8:
            recs.append("Consider increasing model capacity or training longer.")
        supports = [v["support"] for v in per_class.values()]
        if supports and max(supports) > 3 * min(supports):
            recs.append("Class imbalance detected. Consider weighted loss or oversampling.")
        low_f1 = [k for k, v in per_class.items() if v["f1"] < 0.5]
        if low_f1:
            recs.append(f"Low F1 on {len(low_f1)} classes. Consider class-specific augmentation or focal loss.")
        if ece > 0.1:
            recs.append(f"High calibration error ({ece:.3f}). Consider temperature scaling or label smoothing.")
        if quality_prediction:
            qs = quality_prediction.get("quality_score", 0.5)
            ql = quality_prediction.get("predicted_latency_score", 0.5)
            qm = quality_prediction.get("predicted_memory_score", 0.5)
            conf = quality_prediction.get("confidence", 0)
            if conf > 0.5:
                if qs < 0.3:
                    recs.append(f"[QP] Low predicted quality (score={qs:.2f}). Consider changing architecture.")
                if ql > 0.7:
                    recs.append(f"[QP] High predicted latency (score={ql:.2f}). Consider a lighter architecture.")
                if qm > 0.7:
                    recs.append(f"[QP] High predicted memory (score={qm:.2f}). Consider pruning or distillation.")
                if qs > 0.8 and ql < 0.3 and qm < 0.3:
                    recs.append(f"[QP] Strong multi-objective profile: quality={qs:.2f}, latency={ql:.2f}, memory={qm:.2f}. Deployment-ready.")
        if not recs:
            recs.append("Model performance looks good. Consider pruning/quantization for deployment.")
        return recs
