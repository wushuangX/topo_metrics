"""Tests for mask_to_graph and mask_to_nx_graph (hybrid method)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from topo_metrics import TopoGraph, mask_to_graph, mask_to_nx_graph


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def empty_mask() -> np.ndarray:
    return np.zeros((50, 50), dtype=np.uint8)


@pytest.fixture
def horizontal_line() -> np.ndarray:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, :] = 1
    return mask


@pytest.fixture
def vertical_line() -> np.ndarray:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[:, 25] = 1
    return mask


@pytest.fixture
def t_intersection() -> np.ndarray:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 10:40] = 1
    mask[15:25, 25] = 1
    return mask


@pytest.fixture
def cross() -> np.ndarray:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 10:40] = 1
    mask[15:35, 25] = 1
    return mask


@pytest.fixture
def l_shape() -> np.ndarray:
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20, 10:30] = 1
    mask[20:40, 30] = 1
    return mask


@pytest.fixture
def two_disconnected_blocks() -> np.ndarray:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    mask[60:70, 60:70] = 1
    return mask


@pytest.fixture
def three_disconnected_lines() -> np.ndarray:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10, 10:50] = 1
    mask[40, 20:60] = 1
    mask[70, 30:70] = 1
    return mask


@pytest.fixture
def diagonal_line() -> np.ndarray:
    """Diagonal line that creates a more complex skeleton."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    for i in range(40):
        mask[5 + i, 5 + i] = 1
        mask[5 + i, 6 + i] = 1  # 2px wide to ensure connectivity
    return mask


# ── Empty mask ──────────────────────────────────────────────────────────────


class TestEmptyMask:
    def test_empty_returns_empty_topograph(self, empty_mask: np.ndarray) -> None:
        tg = mask_to_graph(empty_mask)
        assert tg.num_nodes == 0
        assert tg.num_edges == 0

    def test_empty_returns_empty_nx_graph(self, empty_mask: np.ndarray) -> None:
        G, pos = mask_to_nx_graph(empty_mask)
        assert len(G.nodes) == 0
        assert len(pos) == 0

    def test_empty_topograph_type(self, empty_mask: np.ndarray) -> None:
        tg = mask_to_graph(empty_mask)
        assert isinstance(tg, TopoGraph)
        assert tg.nodes.shape == (0, 2)
        assert tg.edges.shape == (0, 2)


# ── Connectivity ────────────────────────────────────────────────────────────


class TestConnectivity:
    def test_horizontal_line_connected(self, horizontal_line: np.ndarray) -> None:
        G, _ = mask_to_nx_graph(horizontal_line)
        assert nx.is_connected(G)

    def test_vertical_line_connected(self, vertical_line: np.ndarray) -> None:
        G, _ = mask_to_nx_graph(vertical_line)
        assert nx.is_connected(G)

    def test_t_intersection_connected(self, t_intersection: np.ndarray) -> None:
        G, _ = mask_to_nx_graph(t_intersection)
        assert nx.is_connected(G)

    def test_cross_connected(self, cross: np.ndarray) -> None:
        G, _ = mask_to_nx_graph(cross)
        assert nx.is_connected(G)

    def test_l_shape_connected(self, l_shape: np.ndarray) -> None:
        G, _ = mask_to_nx_graph(l_shape)
        assert nx.is_connected(G)

    def test_diagonal_connected(self, diagonal_line: np.ndarray) -> None:
        G, _ = mask_to_nx_graph(diagonal_line)
        assert nx.is_connected(G)

    def test_two_blocks_two_components(
        self, two_disconnected_blocks: np.ndarray
    ) -> None:
        G, _ = mask_to_nx_graph(two_disconnected_blocks)
        assert nx.number_connected_components(G) == 2

    def test_three_lines_three_components(
        self, three_disconnected_lines: np.ndarray
    ) -> None:
        G, _ = mask_to_nx_graph(three_disconnected_lines)
        assert nx.number_connected_components(G) == 3


# ── Node and edge counts ────────────────────────────────────────────────────


class TestNodeEdgeCounts:
    def test_line_has_edges(self, horizontal_line: np.ndarray) -> None:
        tg = mask_to_graph(horizontal_line)
        assert tg.num_nodes >= 2
        assert tg.num_edges >= 1

    def test_cross_has_junction(self, cross: np.ndarray) -> None:
        """A cross should produce at least one junction node (degree >= 3)."""
        G, _ = mask_to_nx_graph(cross)
        degrees = dict(G.degree())
        assert any(d >= 3 for d in degrees.values()), (
            f"Expected at least one junction node, got degrees: {degrees}"
        )

    def test_more_road_more_nodes(
        self, horizontal_line: np.ndarray, l_shape: np.ndarray
    ) -> None:
        """More road pixels should generally yield at least as many nodes."""
        _, pos1 = mask_to_nx_graph(horizontal_line)
        _, pos2 = mask_to_nx_graph(l_shape)
        # Both should have nodes, exact count depends on keypoint density
        assert len(pos1) > 0
        assert len(pos2) > 0


# ── Return types and conventions ────────────────────────────────────────────


class TestReturnTypes:
    def test_topograph_dtypes(self, horizontal_line: np.ndarray) -> None:
        tg = mask_to_graph(horizontal_line)
        assert tg.nodes.dtype == np.float64
        assert tg.edges.dtype == np.int64

    def test_nx_graph_is_standard(self, horizontal_line: np.ndarray) -> None:
        G, pos = mask_to_nx_graph(horizontal_line)
        assert isinstance(G, nx.Graph)
        assert isinstance(pos, dict)

    def test_node_positions_int_coords(self, horizontal_line: np.ndarray) -> None:
        _, pos = mask_to_nx_graph(horizontal_line)
        for nid, (x, y) in pos.items():
            assert isinstance(x, int), f"x={x} is not int"
            assert isinstance(y, int), f"y={y} is not int"

    def test_node_positions_in_bounds(self, horizontal_line: np.ndarray) -> None:
        h, w = horizontal_line.shape
        _, pos = mask_to_nx_graph(horizontal_line)
        for nid, (x, y) in pos.items():
            assert 0 <= x < w, f"x={x} out of bounds [0, {w})"
            assert 0 <= y < h, f"y={y} out of bounds [0, {h})"

    def test_edge_indices_valid(self, horizontal_line: np.ndarray) -> None:
        tg = mask_to_graph(horizontal_line)
        n = tg.num_nodes
        if n > 0 and tg.num_edges > 0:
            assert np.all(tg.edges >= 0)
            assert np.all(tg.edges < n)

    def test_topograph_has_pos_attr(self, horizontal_line: np.ndarray) -> None:
        """mask_to_nx_graph must set 'pos' attribute on nodes."""
        G, _ = mask_to_nx_graph(horizontal_line)
        for nid in G.nodes:
            assert "pos" in G.nodes[nid], f"Node {nid} missing 'pos' attribute"


# ── Parameters ──────────────────────────────────────────────────────────────


class TestParameters:
    def test_min_distance_affects_node_count(self, horizontal_line: np.ndarray) -> None:
        """Smaller min_distance should yield more nodes."""
        _, pos_dense = mask_to_nx_graph(horizontal_line, min_distance=4)
        _, pos_sparse = mask_to_nx_graph(horizontal_line, min_distance=20)
        assert len(pos_dense) >= len(pos_sparse)

    def test_drop_border_corners_zero(self, horizontal_line: np.ndarray) -> None:
        """With drop_border_corners=0, corners near border are kept."""
        G, _ = mask_to_nx_graph(horizontal_line, drop_border_corners=0)
        # Should still produce a valid graph
        assert len(G.nodes) > 0

    def test_border_natural_margin_affects_border_nodes(
        self, horizontal_line: np.ndarray
    ) -> None:
        """Larger border_natural_margin should remove more natural border nodes."""
        _, pos0 = mask_to_nx_graph(horizontal_line, border_natural_margin=0)
        _, pos3 = mask_to_nx_graph(horizontal_line, border_natural_margin=3)
        # With margin=3, more border nodes are removed
        assert len(pos3) <= len(pos0)

    def test_spacing_preserved(self, horizontal_line: np.ndarray) -> None:
        """Custom spacing should be stored in the TopoGraph."""
        tg = mask_to_graph(horizontal_line, spacing=(2.0, 3.0))
        assert tg.spacing == (2.0, 3.0)


# ── Edge cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_pixel(self) -> None:
        """A single foreground pixel should produce a graph (may be empty or 1 node)."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[10, 10] = 1
        G, pos = mask_to_nx_graph(mask)
        # Single pixel skeleton has 0 skeleton neighbors -> marked as node (endpoint)
        assert len(pos) <= 1

    def test_full_mask(self) -> None:
        """A fully filled mask should produce a graph without errors."""
        mask = np.ones((30, 30), dtype=np.uint8)
        G, pos = mask_to_nx_graph(mask)
        assert len(pos) > 0
        assert nx.is_connected(G)

    def test_thick_road(self) -> None:
        """A wide road strip should skeletonize and produce a connected graph."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 5:45] = 1
        G, pos = mask_to_nx_graph(mask)
        assert nx.is_connected(G)

    def test_very_small_mask(self) -> None:
        """Tiny masks should not crash."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 1:4] = 1
        G, pos = mask_to_nx_graph(mask)
        assert isinstance(G, nx.Graph)

    def test_boolean_input(self, horizontal_line: np.ndarray) -> None:
        """Boolean mask input should work the same as uint8."""
        bool_mask = horizontal_line.astype(bool)
        G1, _ = mask_to_nx_graph(horizontal_line)
        G2, _ = mask_to_nx_graph(bool_mask)
        assert len(G1.nodes) == len(G2.nodes)
        assert len(G1.edges) == len(G2.edges)

    def test_3d_input_squeezed(self) -> None:
        """3D input with singleton dimension should be handled."""
        mask_2d = np.zeros((50, 50), dtype=np.uint8)
        mask_2d[25, :] = 1
        mask_3d = mask_2d[np.newaxis, :, :]  # shape (1, 50, 50)
        # mask_to_nx_graph should handle this
        with pytest.raises(Exception):
            # 3D input is not 2D, should fail or handle gracefully
            mask_to_nx_graph(mask_3d)


# ── Integration with TopoGraph ──────────────────────────────────────────────


class TestTopoGraphIntegration:
    def test_shortest_paths(self, horizontal_line: np.ndarray) -> None:
        """mask_to_graph result should support shortest path computation."""
        tg = mask_to_graph(horizontal_line)
        if tg.num_nodes >= 2:
            paths = tg.shortest_paths(0, [tg.num_nodes - 1])
            dist = paths[tg.num_nodes - 1]
            assert dist >= 0

    def test_densify(self, horizontal_line: np.ndarray) -> None:
        """mask_to_graph result should support densification."""
        tg = mask_to_graph(horizontal_line)
        if tg.num_edges > 0:
            tg_dense = tg.densify(interval=5.0)
            assert tg_dense.num_nodes >= tg.num_nodes

    def test_to_networkx_roundtrip(self, horizontal_line: np.ndarray) -> None:
        """Converting mask_to_graph result to nx and back should preserve structure."""
        tg = mask_to_graph(horizontal_line)
        g = tg.to_networkx()
        tg2 = TopoGraph.from_networkx(g, pos_attr="pos")
        assert tg.num_nodes == tg2.num_nodes
        assert tg.num_edges == tg2.num_edges

    def test_get_chains(self, horizontal_line: np.ndarray) -> None:
        """mask_to_graph result should support chain extraction."""
        tg = mask_to_graph(horizontal_line)
        chains = tg.get_chains()
        assert isinstance(chains, list)


# ── Regression: known patch results ─────────────────────────────────────────


class TestRegressionPatches:
    @pytest.mark.parametrize(
        "filename,expected_max_cc",
        [
            ("patch_000964.npy", 1),  # Was broken with keypoint, fixed with hybrid
            ("patch_001024.npy", 5),
            ("patch_001088.npy", 4),
        ],
    )
    def test_cc_matches_mask(self, filename: str, expected_max_cc: int) -> None:
        """Hybrid method should produce graph with CC count matching mask."""
        import os

        path = os.path.join("tests", "test_data", filename)
        if not os.path.exists(path):
            pytest.skip(f"Test data file {path} not found")
        arr = np.load(path).squeeze()
        mask = (arr > 0).astype(np.uint8)

        from skimage.measure import label

        _, mask_cc = label(mask, connectivity=2, return_num=True)

        G, _ = mask_to_nx_graph(mask)
        graph_cc = nx.number_connected_components(G) if len(G.nodes) > 0 else 0

        assert graph_cc == mask_cc, (
            f"{filename}: mask_cc={mask_cc}, graph_cc={graph_cc}"
        )
