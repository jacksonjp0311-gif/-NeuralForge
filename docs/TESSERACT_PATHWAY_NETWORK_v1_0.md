# Tesseract Pathway Network v1.0

v1.0 stabilizes the Tesseract Jarvis core contract.

## What changed

```text
JARVIS_VERSION = tpn.v1.0
API_CONTRACT_VERSION = jarvis.api.v1
ACTION_PACKET_VERSION = tpn.action.v1.0
GET /contract
stable contract manifest
runtime artifact policy
contract check script
```

## Stable endpoints

```text
GET  /health
GET  /contract
GET  /skills
POST /command
POST /memory/search
GET  /ledger/recent
POST /ledger/search
```

## Runtime artifact policy

Tracked:

```text
source code
tests
docs
promoted seed checkpoints
stable contract manifest
```

Ignored:

```text
runtime command memory JSONL
runtime action ledger JSONL
demo scratch ledgers
cache/process residue
```

## Boundary

This is a local governed Jarvis substrate over a weighted TPN. It does not provide arbitrary shell execution, external model calls, or autonomous authority.

v1.0 means the API contract is stable enough to build local tools against it.
