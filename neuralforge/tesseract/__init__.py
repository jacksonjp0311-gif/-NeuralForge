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
]
