"""Tesseract Pathway Network.

Sparse geometric routing for intent, evidence, authority, and context.
"""

from neuralforge.tesseract.geometry import (
    AXES,
    VERTEX_COUNT,
    EDGE_COUNT,
    bits_to_vertex,
    vertex_to_bits,
    vertex_id,
    neighbors,
    hamming_distance,
    missing_axes,
    shortest_path,
    validate_tesseract,
)
from neuralforge.tesseract.axes import AxisScores, TesseractRouteState, build_route_state
from neuralforge.tesseract.router import TesseractRouter
from neuralforge.tesseract.network import TesseractPathwayNetwork, TesseractPathwayBlock, TesseractSparseDispatcher
from neuralforge.tesseract.loss import tesseract_compound_loss

from neuralforge.tesseract.data import SyntheticTesseractRouteDataset, make_tesseract_loaders
from neuralforge.tesseract.evaluate import evaluate_tpn_model
from neuralforge.tesseract.train import train_tpn_synthetic
from neuralforge.tesseract.receipt import build_tesseract_receipts

__all__ = [
    "AXES",
    "VERTEX_COUNT",
    "EDGE_COUNT",
    "bits_to_vertex",
    "vertex_to_bits",
    "vertex_id",
    "neighbors",
    "hamming_distance",
    "missing_axes",
    "shortest_path",
    "validate_tesseract",
    "AxisScores",
    "TesseractRouteState",
    "build_route_state",
    "TesseractRouter",
    "TesseractPathwayNetwork",
    "TesseractPathwayBlock",
    "tesseract_compound_loss",
    "SyntheticTesseractRouteDataset",
    "make_tesseract_loaders",
    "evaluate_tpn_model",
    "train_tpn_synthetic",
    "TesseractSparseDispatcher",
    "build_tesseract_receipts",
    "TesseractCheckpointConfig",
    "train_tpn_checkpoint",
    "save_tpn_checkpoint",
    "load_tpn_checkpoint",
    "TesseractMindCore",
    "receipt_to_english",
    "receipts_to_english",
    "outputs_to_english",
    "TesseractEnglishAdapter",
    "TesseractFeedbackRecord",
    "TesseractReplayLedger",
    "append_operator_feedback",
    "seed_replay_from_synthetic",
    "train_tpn_from_replay",
    "TesseractWarmRuntime",
    "make_handler",
    "run_server",
    "CommandVectorizer",
    "TesseractActionPacket",
    "TesseractSkillRegistry",
    "TesseractCommandMind",
    "make_command_handler",
    "run_command_server",
    "JarvisServiceConfig",
    "TesseractActionLedger",
    "TesseractJarvisRuntime",
    "make_jarvis_handler",
    "run_jarvis_server",
    "JARVIS_VERSION",
    "API_CONTRACT_VERSION",
    "ACTION_PACKET_VERSION",
    "TesseractJarvisContract",
    "contract_manifest",
    "write_contract_manifest",
    "load_contract_manifest",
    "IntegrationSkill",
    "IntegrationTaskPacket",
    "TesseractIntegrationBus",
    "TesseractPlanStep",
    "TesseractTaskPlan",
    "TesseractTaskPlanner",
    "TesseractCycleEngine",
    "TesseractCycleObservation",
    "TesseractCycleReport",
]
from neuralforge.tesseract.checkpoint import TesseractCheckpointConfig, train_tpn_checkpoint, save_tpn_checkpoint, load_tpn_checkpoint
from neuralforge.tesseract.mind import TesseractMindCore
from neuralforge.tesseract.communication import receipt_to_english, receipts_to_english, outputs_to_english, TesseractEnglishAdapter
from neuralforge.tesseract.adaptive import TesseractFeedbackRecord, TesseractReplayLedger, append_operator_feedback, seed_replay_from_synthetic, train_tpn_from_replay
from neuralforge.tesseract.daemon import TesseractWarmRuntime, make_handler, run_server
from neuralforge.tesseract.command import CommandVectorizer, TesseractActionPacket, TesseractSkillRegistry, TesseractCommandMind, make_command_handler, run_command_server
from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractActionLedger, TesseractJarvisRuntime, make_jarvis_handler, run_jarvis_server
from neuralforge.tesseract.contract import JARVIS_VERSION, API_CONTRACT_VERSION, ACTION_PACKET_VERSION, TesseractJarvisContract, contract_manifest, write_contract_manifest, load_contract_manifest
from neuralforge.tesseract.integration import IntegrationSkill, IntegrationTaskPacket, TesseractIntegrationBus
from neuralforge.tesseract.planner import TesseractPlanStep, TesseractTaskPlan, TesseractTaskPlanner
from neuralforge.tesseract.cycle import TesseractCycleEngine, TesseractCycleObservation, TesseractCycleReport
