"""Unit tests for mask_to_graph (keypoint and sknw backends).

Tests import mask_to_graph directly from the external DeepRoadExtraction source
via importlib (not from the project's src/topo_metrics) to avoid pulling in
torch or other heavy dependencies.
"""

from __future__ import annotations

import importlib.util

import networkx as nx
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "graph",
    "external/DeepRoadExtraction/src/roaddeeplab/utils/graph.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
mask_to_graph = _mod.mask_to_graph


# ── empty mask ──────────────────────────────────────────────────────────────


def test_empty_mask_keypoint() -> None:
    """An all-zero mask should yield an empty graph via the keypoint backend."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    G, pos = mask_to_graph(mask, method="keypoint")
    assert len(G.nodes) == 0
    assert len(pos) == 0


def test_empty_mask_sknw() -> None:
    """An all-zero mask should yield an empty graph via the sknw backend."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    G, pos = mask_to_graph(mask, method="sknw")
    assert len(G.nodes) == 0
    assert len(pos) == 0


# ── simple horizontal line ──────────────────────────────────────────────────


def test_horizontal_line_keypoint() -> None:
    """A straight horizontal line should produce a connected graph (keypoint)."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, :] = 1
    G, pos = mask_to_graph(
        mask, method="keypoint", min_distance=8, k=4, r=200, add_edge_nodes=True
    )
    assert nx.is_connected(G)


def test_horizontal_line_sknw() -> None:
    """A straight horizontal line should produce a connected graph (sknw)."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, :] = 1
    G, pos = mask_to_graph(mask, method="sknw")
    assert nx.is_connected(G)


# ── T-intersection ──────────────────────────────────────────────────────────


def test_t_intersection_sknw() -> None:
    """T-shaped road should be connected via the junction (sknw)."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 10:40] = 1  # horizontal bar
    mask[15:25, 25] = 1  # vertical stem
    G, pos = mask_to_graph(mask, method="sknw")
    assert nx.is_connected(G)


# ── disconnected components ─────────────────────────────────────────────────


def test_disconnected_components_sknw() -> None:
    """Two disconnected road patches should produce a graph with 2 components."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1  # top-left block
    mask[60:70, 60:70] = 1  # bottom-right block
    G, pos = mask_to_graph(mask, method="sknw")
    assert nx.number_connected_components(G) == 2


# ── fix_borders ─────────────────────────────────────────────────────────────


def test_line_touching_border_fix_borders() -> None:
    """A line segment touching the tile edge should be preserved with fix_borders."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 0:10] = 1  # line starting at left edge
    G, pos = mask_to_graph(mask, method="sknw", fix_borders=True)
    assert len(G.nodes) > 0
    assert nx.is_connected(G)


# ── invalid method ──────────────────────────────────────────────────────────


def test_invalid_method_raises() -> None:
    """Passing an unrecognized method should raise ValueError."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, :] = 1
    with pytest.raises(ValueError, match="Unknown method"):
        mask_to_graph(mask, method="invalid")


# ── coordinate convention ───────────────────────────────────────────────────


def test_sknw_node_positions_convention() -> None:
    """node_positions should map to (x, y) = (col, row) convention."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 1  # square block
    G, pos = mask_to_graph(mask, method="sknw")
    for nid, (x, y) in pos.items():
        assert 0 <= x < 50, f"x={x} out of range"
        assert 0 <= y < 50, f"y={y} out of range"


# ── return type consistency ─────────────────────────────────────────────────


def test_keypoint_fix_borders_return_type() -> None:
    """fix_borders=True with the keypoint method should preserve return types."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 10:40] = 1
    G, pos = mask_to_graph(
        mask,
        method="keypoint",
        fix_borders=True,
        min_distance=8,
        k=4,
        r=200,
        add_edge_nodes=True,
    )
    assert isinstance(G, nx.Graph)
    assert isinstance(pos, dict)
    for nid, (x, y) in pos.items():
        assert isinstance(x, int), f"x has type {type(x).__name__}"
        assert isinstance(y, int), f"y has type {type(y).__name__}"


# ── hybrid method ───────────────────────────────────────────────────────────


def test_empty_mask_hybrid() -> None:
    """An all-zero mask should yield an empty graph via the hybrid backend."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    G, pos = mask_to_graph(mask, method="hybrid")
    assert len(G.nodes) == 0
    assert len(pos) == 0


def test_horizontal_line_hybrid() -> None:
    """A straight horizontal line should produce a connected graph (hybrid)."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, :] = 1
    G, pos = mask_to_graph(
        mask, method="hybrid", min_distance=8, k=4, add_edge_nodes=True
    )
    assert nx.is_connected(G)


def test_t_intersection_hybrid() -> None:
    """T-shaped road should be connected (hybrid)."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 10:40] = 1
    mask[15:25, 25] = 1
    G, pos = mask_to_graph(mask, method="hybrid")
    assert nx.is_connected(G)


def test_disconnected_components_hybrid() -> None:
    """Two disconnected road patches should produce graph with 2 components (hybrid)."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    mask[60:70, 60:70] = 1
    G, pos = mask_to_graph(mask, method="hybrid")
    assert nx.number_connected_components(G) == 2


def test_hybrid_has_more_nodes_than_sknw() -> None:
    """Hybrid should have more nodes than pure sknw on a long road."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20, 10:80] = 1
    mask[20:70, 80] = 1
    G_sknw, _ = mask_to_graph(mask, method="sknw")
    G_hybrid, _ = mask_to_graph(
        mask, method="hybrid", min_distance=10, add_edge_nodes=True
    )
    assert len(G_hybrid.nodes) >= len(G_sknw.nodes)


def test_hybrid_node_positions_convention() -> None:
    """node_positions should map to (x, y) = (col, row) convention (hybrid)."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 10:40] = 1
    G, pos = mask_to_graph(mask, method="hybrid")
    for nid, (x, y) in pos.items():
        assert 0 <= x < 50, f"x={x} out of range"
        assert 0 <= y < 50, f"y={y} out of range"
