# Tesseract Pathway Network v0.1

TPN is a sparse geometric routing experiment for NeuralForge.

It routes neural state through four operational axes:

```text
Intent
Evidence
Authority
Context
```

A route state becomes a four-bit tesseract vertex:

```text
0000 = no route-ready axes
1000 = intent only
1100 = intent + evidence
1110 = intent + evidence + authority
1111 = intent + evidence + authority + context
```

## Purpose

TPN tests whether a neural system can reduce compute and improve pathway discipline by shaping context before dense reasoning.

```text
Unshaped context creates attention pressure.
Tesseract-shaped context creates routeable intelligence.
```

## Components

```text
neuralforge/tesseract/geometry.py  - 16 vertices, 32 edges, path/distance utilities
neuralforge/tesseract/axes.py      - axis scores, delta_phi, coherence, route state
neuralforge/tesseract/router.py    - interpretable sparse expert selection
neuralforge/tesseract/network.py   - PyTorch TPN block and multi-head network
neuralforge/tesseract/loss.py      - compound route/evidence/authority/coherence loss
neuralforge/tesseract/benchmark.py - toy sparse-vs-dense benchmark
```

## Claim boundary

TPN v0.1 is a research scaffold.

It does not prove AGI, safety, correctness, production efficiency, or autonomous authority.
It is an experimental sparse-routing architecture for benchmark-driven development.
