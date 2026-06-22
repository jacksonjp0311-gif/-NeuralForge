from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

from neuralforge.spec import (
    ArchitectureFamily, ArchitectureSpec, Constraints, DataProfile,
    NeuralForgeSpec, TaskType,
)

logger = logging.getLogger("neuralforge.auto.architect")


# Architecture templates for different task families
ARCHITECTURE_TEMPLATES: Dict[ArchitectureFamily, List[Dict[str, Any]]] = {
    ArchitectureFamily.CNN: [
        {
            "name": "cnn-small",
            "depth": 3, "width": 64,
            "description": "Lightweight CNN for simple image tasks",
        },
        {
            "name": "cnn-medium",
            "depth": 5, "width": 128,
            "description": "Medium CNN for general image classification",
        },
        {
            "name": "cnn-large",
            "depth": 7, "width": 256,
            "description": "Deep CNN for complex image tasks",
        },
    ],
    ArchitectureFamily.RESNET: [
        {
            "name": "resnet-tiny",
            "depth": 3, "width": 32,
            "description": "Tiny ResNet for edge devices",
        },
        {
            "name": "resnet-small",
            "depth": 4, "width": 64,
            "description": "ResNet-18 style architecture",
        },
        {
            "name": "resnet-medium",
            "depth": 6, "width": 128,
            "description": "ResNet-34 style architecture",
        },
    ],
    ArchitectureFamily.TRANSFORMER: [
        {
            "name": "transformer-tiny",
            "depth": 2, "embedding_dim": 128, "num_heads": 4,
            "description": "Tiny transformer for simple NLP",
        },
        {
            "name": "transformer-small",
            "depth": 4, "embedding_dim": 256, "num_heads": 8,
            "description": "Small transformer for text classification",
        },
        {
            "name": "transformer-medium",
            "depth": 6, "embedding_dim": 512, "num_heads": 8,
            "description": "Medium transformer for generation tasks",
        },
    ],
    ArchitectureFamily.VISION_TRANSFORMER: [
        {
            "name": "vit-tiny",
            "depth": 4, "embedding_dim": 192, "num_heads": 3,
            "description": "ViT-Tiny for small images",
        },
        {
            "name": "vit-small",
            "depth": 6, "embedding_dim": 384, "num_heads": 6,
            "description": "ViT-Small for medium images",
        },
        {
            "name": "vit-base",
            "depth": 12, "embedding_dim": 768, "num_heads": 12,
            "description": "ViT-Base for standard image classification",
        },
    ],
    ArchitectureFamily.MLP_MIXER: [
        {
            "name": "mixer-small",
            "depth": 4, "embedding_dim": 256,
            "description": "MLP-Mixer small variant",
        },
        {
            "name": "mixer-medium",
            "depth": 8, "embedding_dim": 512,
            "description": "MLP-Mixer medium variant",
        },
    ],
    ArchitectureFamily.KAN: [
        {
            "name": "kan-small",
            "depth": 3, "width": 64,
            "description": "Small KAN for tabular/small data",
        },
        {
            "name": "kan-medium",
            "depth": 5, "width": 128,
            "description": "Medium KAN for scientific ML",
        },
    ],
    ArchitectureFamily.MAMBA: [
        {
            "name": "mamba-small",
            "depth": 4, "embedding_dim": 256,
            "description": "Small Mamba for sequence tasks",
        },
        {
            "name": "mamba-medium",
            "depth": 8, "embedding_dim": 512,
            "description": "Medium Mamba for long sequences",
        },
    ],
    ArchitectureFamily.MIXTURE_OF_EXPERTS: [
        {
            "name": "moe-small",
            "depth": 4, "embedding_dim": 256,
            "num_experts": 4, "expert_capacity": 2,
            "description": "Small MoE with 4 experts",
        },
    ],
}


# Task → recommended architecture families
TASK_FAMILY_MAP: Dict[TaskType, List[ArchitectureFamily]] = {
    TaskType.IMAGE_CLASSIFICATION: [
        ArchitectureFamily.RESNET, ArchitectureFamily.VISION_TRANSFORMER,
        ArchitectureFamily.CNN, ArchitectureFamily.MLP_MIXER,
    ],
    TaskType.TEXT_CLASSIFICATION: [
        ArchitectureFamily.TRANSFORMER, ArchitectureFamily.CNN,
        ArchitectureFamily.MAMBA,
    ],
    TaskType.TEXT_GENERATION: [
        ArchitectureFamily.TRANSFORMER, ArchitectureFamily.MAMBA,
        ArchitectureFamily.RWKV,
    ],
    TaskType.OBJECT_DETECTION: [
        ArchitectureFamily.RESNET, ArchitectureFamily.VISION_TRANSFORMER,
    ],
    TaskType.IMAGE_SEGMENTATION: [
        ArchitectureFamily.RESNET, ArchitectureFamily.VISION_TRANSFORMER,
    ],
    TaskType.TIME_SERIES: [
        ArchitectureFamily.TRANSFORMER, ArchitectureFamily.MAMBA,
        ArchitectureFamily.CNN,
    ],
    TaskType.ANOMALY_DETECTION: [
        ArchitectureFamily.CNN, ArchitectureFamily.TRANSFORMER,
    ],
    TaskType.MULTIMODAL: [
        ArchitectureFamily.TRANSFORMER, ArchitectureFamily.VISION_TRANSFORMER,
    ],
    TaskType.AUDIO_CLASSIFICATION: [
        ArchitectureFamily.CNN, ArchitectureFamily.TRANSFORMER,
    ],
    TaskType.REINFORCEMENT_LEARNING: [
        ArchitectureFamily.CNN, ArchitectureFamily.TRANSFORMER,
    ],
    TaskType.GRAPH_PREDICTION: [
        ArchitectureFamily.CUSTOM,
    ],
}


class ArchitectAgent:
    """LLM-driven architecture proposal agent.

    Proposes neural architectures based on task description, data profile,
    and constraints. Uses template matching + constraint satisfaction.
    """

    def __init__(self):
        self.templates = ARCHITECTURE_TEMPLATES
        self.task_map = TASK_FAMILY_MAP

    def propose(
        self,
        task_description: str,
        data_profile: DataProfile,
        constraints: Optional[Constraints] = None,
        num_proposals: int = 3,
    ) -> List[NeuralForgeSpec]:
        """Propose multiple architecture specs for the given task."""
        constraints = constraints or Constraints()
        task_type = data_profile.task_type

        # Get candidate families for this task
        families = self.task_map.get(task_type, [ArchitectureFamily.CNN])

        proposals = []
        for family in families:
            templates = self.templates.get(family, [])
            for template in templates:
                spec = self._template_to_spec(
                    template, family, task_description, data_profile, constraints
                )
                if spec is not None and self._check_constraints(spec, constraints):
                    proposals.append(spec)
                    if len(proposals) >= num_proposals:
                        break
            if len(proposals) >= num_proposals:
                break

        if not proposals:
            # Fallback: auto-infer
            proposals = [
                NeuralForgeSpec(
                    name="auto-fallback",
                    task_type=task_type,
                    data_profile=data_profile,
                    architecture=ArchitectureSpec(family=ArchitectureFamily.AUTO),
                    constraints=constraints,
                )
            ]

        return proposals

    def _template_to_spec(
        self,
        template: Dict[str, Any],
        family: ArchitectureFamily,
        description: str,
        data_profile: DataProfile,
        constraints: Constraints,
    ) -> Optional[NeuralForgeSpec]:
        """Convert a template to a NeuralForgeSpec."""
        arch = ArchitectureSpec(
            family=family,
            depth=template.get("depth", 4),
            width=template.get("width", 128),
            embedding_dim=template.get("embedding_dim"),
            num_heads=template.get("num_heads"),
            num_experts=template.get("num_experts"),
            expert_capacity=template.get("expert_capacity"),
        )

        return NeuralForgeSpec(
            name=template.get("name", f"arch-{family.value}"),
            description=template.get("description", description),
            task_type=data_profile.task_type,
            data_profile=data_profile,
            architecture=arch,
            constraints=constraints,
        )

    def _check_constraints(
        self, spec: NeuralForgeSpec, constraints: Constraints
    ) -> bool:
        """Check if a spec satisfies constraints (rough estimate)."""
        if constraints.max_parameters is not None:
            # Rough parameter estimate
            arch = spec.architecture
            d = arch.depth or 4
            w = arch.width or 128
            # Very rough: d * w^2 * 4 (conv layers)
            est_params = d * w * w * 4
            if est_params > constraints.max_parameters * 2:
                return False
        return True

    def describe_architecture(self, spec: NeuralForgeSpec) -> str:
        """Generate human-readable description of an architecture."""
        arch = spec.architecture
        parts = [f"Architecture: {arch.family.value}"]
        if arch.depth:
            parts.append(f"  Depth: {arch.depth} layers")
        if arch.width:
            parts.append(f"  Width: {arch.width} channels")
        if arch.embedding_dim:
            parts.append(f"  Embedding dim: {arch.embedding_dim}")
        if arch.num_heads:
            parts.append(f"  Attention heads: {arch.num_heads}")
        if arch.dropout:
            parts.append(f"  Dropout: {arch.dropout}")
        if arch.activation:
            parts.append(f"  Activation: {arch.activation}")
        return "\n".join(parts)
