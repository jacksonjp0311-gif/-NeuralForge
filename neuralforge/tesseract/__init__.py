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
from neuralforge.tesseract.network import TesseractPathwayNetwork, TesseractPathwayBlock
from neuralforge.tesseract.loss import tesseract_compound_loss

from neuralforge.tesseract.data import SyntheticTesseractRouteDataset, make_tesseract_loaders
from neuralforge.tesseract.evaluate import evaluate_tpn_model
from neuralforge.tesseract.train import train_tpn_synthetic

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
]
