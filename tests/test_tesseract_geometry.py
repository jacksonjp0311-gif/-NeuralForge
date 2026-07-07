from neuralforge.tesseract.geometry import (
    all_edges,
    all_vertices,
    hamming_distance,
    missing_axes,
    neighbors,
    shortest_path,
    validate_tesseract,
)


def test_tesseract_counts():
    report = validate_tesseract()
    assert report["valid"] is True
    assert report["vertices"] == 16
    assert report["edges"] == 32
    assert report["degree_min"] == 4
    assert report["degree_max"] == 4
    assert report["diameter"] == 4


def test_neighbors_and_path():
    assert len(neighbors("0000")) == 4
    assert hamming_distance("0000", "1111") == 4
    assert missing_axes("1101") == ["authority"]
    assert shortest_path("1000", "1111") == ["1000", "1100", "1110", "1111"]


def test_edge_uniqueness():
    assert len(all_vertices()) == 16
    assert len(all_edges()) == 32
    assert len(set(all_edges())) == 32
