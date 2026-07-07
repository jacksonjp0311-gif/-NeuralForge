# Tesseract Pathway Network v0.2

v0.2 turns the v0.1 scaffold into a trainable NeuralForge architecture.

## Additions

```text
ArchitectureFamily.TESSERACT
NeuralForgeModule._build_tesseract()
SyntheticTesseractRouteDataset
evaluate_tpn_model()
train_tpn_synthetic()
tests/test_tesseract_training.py
examples/tesseract_training_demo.py
```

## Training target

TPN learns five outputs from a 16-feature tesseract state packet:

```text
route_logits
authority_logits
evidence_logits
coherence
delta_phi
```

The synthetic training target is intentionally narrow:

```text
Can this architecture learn route-governance geometry?
Can it preserve evidence/authority/coherence labels?
Can NeuralForge build it through the standard create_model path?
```

## Claim boundary

TPN v0.2 proves only that the architecture is registered, trainable, testable, and benchmarkable inside NeuralForge.

It does not prove production efficiency, safety, AGI, correctness, or autonomous authority.
