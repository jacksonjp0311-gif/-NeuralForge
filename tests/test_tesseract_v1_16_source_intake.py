from pathlib import Path

from neuralforge.tesseract.source_intake import (
    SOURCE_INTAKE_VERSION,
    TesseractSourceIntakeGovernor,
    TesseractSourceSpec,
)


def test_v1_16_demo_sources_emit_registry_receipt(tmp_path):
    governor = TesseractSourceIntakeGovernor(
        latest_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )
    receipt = governor.build_receipt(governor.demo_sources())
    assert receipt["source_intake_version"] == SOURCE_INTAKE_VERSION
    assert receipt["ok"] is True
    assert receipt["registry_allowed"] is True
    assert receipt["source_count"] == 3
    assert receipt["allowed_source_count"] == 2
    assert receipt["blocked_source_count"] == 1
    assert receipt["live_pull_allowed"] is False
    assert receipt["scraping_allowed"] is False
    assert receipt["raw_collection_allowed"] is False
    assert receipt["mutation_allowed"] is False
    assert Path(tmp_path / "latest.json").exists()
    assert Path(tmp_path / "history.jsonl").exists()


def test_v1_16_blocks_allowed_scrape_until_future_compliance_receipt(tmp_path):
    governor = TesseractSourceIntakeGovernor(
        latest_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )
    source = TesseractSourceSpec(
        source_id="scrape_test",
        source_type="scrape",
        purpose="test scrape candidate",
        endpoint_or_location="https://example.com",
        allowed=True,
        compliance_note="placeholder note",
        rate_limit_per_minute=1,
    )
    reasons = governor.validate_source(source)
    assert "scrape sources require a future dedicated compliance receipt before being allowed" in reasons


def test_v1_16_blocks_personal_data_until_redaction_receipt(tmp_path):
    governor = TesseractSourceIntakeGovernor(
        latest_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )
    source = TesseractSourceSpec(
        source_id="pii_api",
        source_type="api",
        purpose="personal-data candidate",
        endpoint_or_location="https://example.com/api",
        allowed=True,
        compliance_note="api note",
        contains_personal_data=True,
    )
    reasons = governor.validate_source(source)
    assert "sources containing personal data require a future privacy/redaction receipt" in reasons


def test_v1_16_summary_is_compact(tmp_path):
    governor = TesseractSourceIntakeGovernor(
        latest_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
    )
    receipt = governor.build_receipt(governor.demo_sources())
    text = governor.format_summary(receipt)
    assert "SOURCE INTAKE GOVERNOR SUMMARY" in text
    assert "live_pull_allowed: False" in text
    assert "scraping_allowed: False" in text
    assert "mutation_allowed: False" in text
    assert len(text.splitlines()) <= 14
