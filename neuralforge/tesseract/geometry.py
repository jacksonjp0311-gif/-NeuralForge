"""Pure tesseract geometry for pathway routing."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

AXES: tuple[str, str, str, str] = ("intent", "evidence", "authority", "context")
AXIS_INDEX = {name: i for i, name in enumerate(AXES)}
VERTEX_COUNT = 16
EDGE_COUNT = 32


def vertex_to_bits(vertex: int | str | Sequence[int]) -> tuple[int, int, int, int]:
    if isinstance(vertex, int):
        if vertex < 0 or vertex >= VERTEX_COUNT:
            raise ValueError(f"vertex int out of range 0..15: {vertex}")
        return tuple(int(ch) for ch in f"{vertex:04b}")  # type: ignore[return-value]
    if isinstance(vertex, str):
        cleaned = vertex.strip().replace("0b", "")
        if len(cleaned) != 4 or any(ch not in "01" for ch in cleaned):
            raise ValueError(f"vertex string must be four binary digits: {vertex!r}")
        return tuple(int(ch) for ch in cleaned)  # type: ignore[return-value]
    bits = tuple(int(bool(v)) for v in vertex)
    if len(bits) != 4:
        raise ValueError(f"vertex sequence must have 4 bits, got {len(bits)}")
    return bits  # type: ignore[return-value]


def bits_to_vertex(bits: Sequence[int], as_int: bool = False) -> str | int:
    clean = vertex_to_bits(bits)
    text = "".join(str(b) for b in clean)
    return int(text, 2) if as_int else text


def vertex_id(vertex: int | str | Sequence[int]) -> int:
    return int("".join(str(b) for b in vertex_to_bits(vertex)), 2)


def neighbors(vertex: int | str | Sequence[int]) -> list[str]:
    bits = list(vertex_to_bits(vertex))
    out: list[str] = []
    for i in range(4):
        nb = bits.copy()
        nb[i] = 1 - nb[i]
        out.append(bits_to_vertex(nb))  # type: ignore[arg-type]
    return out


def all_vertices() -> list[str]:
    return [f"{i:04b}" for i in range(VERTEX_COUNT)]


def all_edges() -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for v in all_vertices():
        for n in neighbors(v):
            a, b = sorted((v, n))
            edge = (a, b)
            if edge not in seen:
                edges.append(edge)
                seen.add(edge)
    return edges


def hamming_distance(a: int | str | Sequence[int], b: int | str | Sequence[int]) -> int:
    aa = vertex_to_bits(a)
    bb = vertex_to_bits(b)
    return sum(int(x != y) for x, y in zip(aa, bb))


def missing_axes(vertex: int | str | Sequence[int], target: int | str | Sequence[int] = "1111") -> list[str]:
    v = vertex_to_bits(vertex)
    t = vertex_to_bits(target)
    return [AXES[i] for i, (x, y) in enumerate(zip(v, t)) if x == 0 and y == 1]


def shortest_path(start: int | str | Sequence[int], target: int | str | Sequence[int] = "1111") -> list[str]:
    cur = list(vertex_to_bits(start))
    goal = list(vertex_to_bits(target))
    path = [bits_to_vertex(cur)]  # type: ignore[arg-type]
    for i in range(4):
        if cur[i] != goal[i]:
            cur[i] = goal[i]
            path.append(bits_to_vertex(cur))  # type: ignore[arg-type]
    return path  # type: ignore[return-value]


def faces() -> list[tuple[str, tuple[str, str], tuple[str, str, str, str]]]:
    result = []
    for active in combinations(range(4), 2):
        fixed = [i for i in range(4) if i not in active]
        for fixed_values in range(4):
            bits = [0, 0, 0, 0]
            fv = [(fixed_values >> 1) & 1, fixed_values & 1]
            for idx, value in zip(fixed, fv):
                bits[idx] = value
            verts = []
            for av in range(4):
                b = bits.copy()
                b[active[0]] = (av >> 1) & 1
                b[active[1]] = av & 1
                verts.append(bits_to_vertex(b))  # type: ignore[arg-type]
            result.append((
                "".join("*" if i in active else str(bits[i]) for i in range(4)),
                (AXES[active[0]], AXES[active[1]]),
                tuple(verts),  # type: ignore[arg-type]
            ))
    return result


def validate_tesseract() -> dict[str, int | bool]:
    vertices = all_vertices()
    edges = all_edges()
    degrees = {v: len(neighbors(v)) for v in vertices}
    diameter = max(hamming_distance(a, b) for a in vertices for b in vertices)
    return {
        "vertices": len(vertices),
        "edges": len(edges),
        "degree_min": min(degrees.values()),
        "degree_max": max(degrees.values()),
        "diameter": diameter,
        "square_faces": len(faces()),
        "valid": len(vertices) == 16 and len(edges) == 32 and min(degrees.values()) == 4 and max(degrees.values()) == 4 and diameter == 4,
    }
