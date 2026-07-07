from pathlib import Path

from neuralforge.tesseract.approval import APPROVAL_VERSION, TesseractHumanApprovalLedger


def _control_report():
    return {
        "ok": True,
        "mutation_allowed": False,
        "approval_request": {
            "approval_id": "approval_test",
            "status": "pending_human_approval",
            "approved_by_human": False,
        },
        "patch_proposal_receipts": [
            {"proposal_id": "p1"},
            {"proposal_id": "p2"},
        ],
    }


def test_v1_14_pending_summary(tmp_path):
    ledger = TesseractHumanApprovalLedger(
        latest_path=tmp_path / "latest.json",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    summary = ledger.pending_summary(_control_report())
    assert summary["approval_version"] == APPROVAL_VERSION
    assert summary["approval_status"] == "pending_human_approval"
    assert summary["proposal_count"] == 2
    assert summary["mutation_allowed"] is False


def test_v1_14_approve_records_receipt_without_mutation(tmp_path):
    ledger = TesseractHumanApprovalLedger(
        latest_path=tmp_path / "latest.json",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    receipt = ledger.decide(_control_report(), decision="approve", human_id="James", reason="Proceed to planning.")
    assert receipt["decision"] == "approve"
    assert receipt["approved_by_human"] is True
    assert receipt["mutation_allowed"] is False
    assert receipt["next_step_unlocked"] == "sandboxed_patch_plan_receipt"
    assert Path(tmp_path / "latest.json").exists()
    assert Path(tmp_path / "ledger.jsonl").exists()


def test_v1_14_reject_records_receipt_without_unlock(tmp_path):
    ledger = TesseractHumanApprovalLedger(
        latest_path=tmp_path / "latest.json",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    receipt = ledger.decide(_control_report(), decision="reject", human_id="James", reason="Not yet.")
    assert receipt["decision"] == "reject"
    assert receipt["approved_by_human"] is False
    assert receipt["mutation_allowed"] is False
    assert receipt["next_step_unlocked"] == "none"


def test_v1_14_decision_requires_human_id(tmp_path):
    ledger = TesseractHumanApprovalLedger(
        latest_path=tmp_path / "latest.json",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    try:
        ledger.decide(_control_report(), decision="approve", human_id="")
    except ValueError as exc:
        assert "human_id is required" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_v1_14_format_pending_summary_compact(tmp_path):
    ledger = TesseractHumanApprovalLedger(
        latest_path=tmp_path / "latest.json",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    text = ledger.format_pending_summary(ledger.pending_summary(_control_report()))
    assert "APPROVAL LEDGER SUMMARY" in text
    assert "mutation_allowed: False" in text
    assert len(text.splitlines()) <= 12
