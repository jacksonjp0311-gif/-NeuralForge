import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from neuralforge.tesseract.checkpoint import TesseractCheckpointConfig, train_tpn_checkpoint
from neuralforge.tesseract.contract import (
    ACTION_PACKET_VERSION,
    API_CONTRACT_VERSION,
    JARVIS_VERSION,
    contract_manifest,
)
from neuralforge.tesseract.jarvis import (
    JarvisServiceConfig,
    TesseractJarvisRuntime,
    make_jarvis_handler,
)


def _runtime(tmp_path):
    artifact = train_tpn_checkpoint(
        output_dir=tmp_path,
        name="tpn_v1_contract_base",
        config=TesseractCheckpointConfig(samples=64, epochs=1, batch_size=16, d_model=24, top_k=4, seed=121),
        device="cpu",
    )
    return TesseractJarvisRuntime(JarvisServiceConfig(
        checkpoint=artifact["checkpoint_path"],
        memory_path=str(tmp_path / "memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        contract_path=str(tmp_path / "contract.json"),
    ))


def test_v1_0_contract_manifest_constants():
    manifest = contract_manifest()
    assert manifest["version"] == JARVIS_VERSION
    assert manifest["api_contract_version"] == API_CONTRACT_VERSION
    assert manifest["action_packet_version"] == ACTION_PACKET_VERSION
    assert manifest["endpoint_count"] >= 7


def test_v1_0_command_response_versions(tmp_path):
    runtime = _runtime(tmp_path)
    answer = runtime.command("plan the next stable step", execute=True)
    assert answer["version"] == JARVIS_VERSION
    assert answer["api_contract_version"] == API_CONTRACT_VERSION
    assert answer["packet"]["packet_version"] == ACTION_PACKET_VERSION


def test_v1_0_http_contract_endpoint(tmp_path):
    runtime = _runtime(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_jarvis_handler(runtime))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/contract", timeout=5) as res:
            contract = json.loads(res.read().decode("utf-8"))
        assert contract["ok"] is True
        assert contract["version"] == JARVIS_VERSION

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as res:
            health = json.loads(res.read().decode("utf-8"))
        assert health["api_contract_version"] == API_CONTRACT_VERSION
    finally:
        server.shutdown()
        thread.join(timeout=5)
