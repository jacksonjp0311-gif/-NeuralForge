
# Tesseract Pathway Network v0.3

v0.3 upgrades TPN from a registered trainable architecture into a governed sparse pathway system.

## What changed

```text
TesseractSparseDispatcher
vertex_logits
selected_vertex
topk_indices
topk_weights
expert_usage
sparse_ratio
build_tesseract_receipts()
axis-supervised gate loss
vertex-supervised routing loss
```

## Routing model

The network predicts four axis scores:

```text
Intent
Evidence
Authority
Context
```

Those scores induce a probability distribution over the 16 tesseract vertices. The model selects top-k vertices and dispatches hidden state through the corresponding expert pathways.

## Receipts

Each inference can emit a receipt:

```text
selected_vertex
missing_axes
topk_vertices
topk_weights
axis_scores
route
coherence
delta_phi
```

A receipt is a route explanation artifact. It is not a proof of truth, safety, or correctness.

## Claim boundary

TPN v0.3 demonstrates trainable pathway-sparse routing and receipt production on synthetic route-governance tasks.

It does not claim production efficiency, external safety, AGI, or correctness on real-world tasks.
