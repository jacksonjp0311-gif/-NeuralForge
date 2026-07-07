
# Tesseract Pathway Network v0.5

v0.5 adds local English communication.

## What this means

TPN can now convert its route receipts into plain English without an external language model or API call.

```text
local vector
→ weighted TPN
→ receipt
→ deterministic English explanation
```

## New components

```text
receipt_to_english()
receipts_to_english()
outputs_to_english()
TesseractEnglishAdapter
examples/tesseract_english_demo.py
examples/tesseract_weighted_english_demo.py
```

## Boundary

This is deterministic English communication from receipts. It is not open-ended language generation, world knowledge, or a general chat model.

That boundary is intentional. The communication layer speaks the mind core's routing state clearly and quickly.
