/**
 * NeuralForge Python Bridge
 *
 * Each tool calls the neuralforge Python library via subprocess.
 * This bridge handles the communication between the AGNT JS runtime
 * and the neuralforge Python engine.
 */

import { spawnSync } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Find the neuralforge Python library location
const REPO_ROOT = path.resolve(__dirname, "../..");
const NEURALFORGE_ROOT = path.join(REPO_ROOT, "neuralforge");

/**
 * Execute a Python code string that uses NeuralForge and return parsed JSON result.
 * The Python code should set a variable named `output` to an object with at least a `status` field.
 */
export function runNeuralforge(code, timeoutMs = 120000) {
  const fullCode = `
import sys, json, os
sys.path.insert(0, r"${REPO_ROOT}")
os.chdir(r"${REPO_ROOT}")

try:
    import neuralforge as nf
    from neuralforge.spec import *
    from neuralforge.core.forge import create_model, train as nf_train, optimize as nf_optimize
    from neuralforge.core.forge import evaluate_and_report, evolve, auto_architecture, export_model
    from neuralforge.training.engine import TrainingEngine
    from neuralforge.evaluation.evaluator import ModelEvaluator
    from neuralforge.utils.profiling import profile_model
    from neuralforge.optimize import prune_model, quantize_model
    from neuralforge.optimize.meta_optimizer import MetaOptimizer
    from neuralforge.auto.architect import ArchitectAgent
    import torch
    import numpy as np
except ImportError as e:
    print("NF_RESULT:" + json.dumps({"status": "error", "error": f"Import error: {str(e)}"}))
    sys.exit(0)

try:
${code.split("\n").map(l => "    " + l).join("\n")}
    print("NF_RESULT:" + json.dumps(output, default=str, ensure_ascii=False))
except Exception as e:
    import traceback
    print("NF_RESULT:" + json.dumps({"status": "error", "error": str(e), "trace": traceback.format_exc()[-500:]}))
`;

  try {
    const result = spawnSync("python", ["-c", fullCode], {
      timeout: timeoutMs,
      encoding: "utf-8",
      maxBuffer: 10 * 1024 * 1024,
      cwd: REPO_ROOT,
    });

    const lines = (result.stdout || "").split("\n");
    for (const line of lines) {
      if (line.startsWith("NF_RESULT:")) {
        return JSON.parse(line.substring("NF_RESULT:".length));
      }
    }

    const errText = (result.stderr || "").slice(-500);
    return { status: "error", error: errText || "No result from NeuralForge" };
  } catch (err) {
    return { status: "error", error: err.message };
  }
}

export { NEURALFORGE_ROOT, REPO_ROOT };
