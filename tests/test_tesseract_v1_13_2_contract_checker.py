from pathlib import Path
import subprocess

from neuralforge.tesseract.contract import contract_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_v1_13_2_contract_manifest_version():
    manifest = contract_manifest()
    assert manifest["version"] == "tpn.v1.13.2"
    assert manifest["api_contract_version"] == "jarvis.api.v1"
    assert manifest["contract_checker_version"] == "tpn.contract_checker.v1.13.2"


def test_v1_13_2_contract_checker_has_offline_fallback():
    script = (ROOT / "scripts" / "check_tesseract_contract.ps1").read_text(encoding="utf-8")
    assert "Invoke-OfflineContractCheck" in script
    assert "contract_manifest" in script
    assert "RequireLive" in script
    assert "tpn.v1.13.2" in script


def test_v1_13_2_contract_checker_offline_mode_succeeds():
    script = ROOT / "scripts" / "check_tesseract_contract.ps1"
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Base",
            "http://127.0.0.1:9",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "mode=offline" in result.stdout
    assert "tpn.v1.13.2" in result.stdout
