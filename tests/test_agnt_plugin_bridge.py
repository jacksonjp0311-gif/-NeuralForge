"""Regression tests for the AGNT JavaScript bridge."""
from __future__ import annotations

import json
import subprocess


def test_bridge_resolves_repo_root_and_runs_python():
    script = """
import { runPython, REPO_ROOT, NEURALFORGE_ROOT } from './agnt-plugin/tools/_bridge.mjs';
const result = runPython('output={"status":"success","version":nf.__version__}', 30000);
console.log(JSON.stringify({ result, REPO_ROOT, NEURALFORGE_ROOT }));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["result"] == {"status": "success", "version": "2.5.0"}
    assert payload["REPO_ROOT"].replace("\\", "/").endswith("/NeuralForge")
    assert payload["NEURALFORGE_ROOT"].replace("\\", "/").endswith("/NeuralForge/neuralforge")

