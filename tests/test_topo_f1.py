"""Tests for the TOPO F1 metric (topological precision, recall, F1)."""

from __future__ import annotations

import numpy as np
import pytest

from topo_metrics import TopoGraph, topo_f1


def test_topo_f1_both_empty() -> None:
    g = TopoGraph(
        nodes=np.empty((0, 2)),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    assert topo_f1(g, g) == (1.0, 1.0, 1.0)


def test_topo_f1_pred_empty_gt_nonempty() -> None:
    empty = TopoGraph(
        nodes=np.empty((0, 2)),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    gt = TopoGraph(
        nodes=np.array([[0, 0], [200, 0]]),
        edges=np.array([[0, 1]]),
    )
    assert topo_f1(empty, gt) == (0.0, 0.0, 0.0)


def test_topo_f1_gt_empty_pred_nonempty() -> None:
    empty = TopoGraph(
        nodes=np.empty((0, 2)),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    pred = TopoGraph(
        nodes=np.array([[0, 0], [200, 0]]),
        edges=np.array([[0, 1]]),
    )
    assert topo_f1(pred, empty) == (0.0, 0.0, 0.0)


def test_topo_f1_identical() -> None:
    nodes = np.array([[0, 0], [200, 0], [400, 0]])
    edges = np.array([[0, 1], [1, 2]])
    g = TopoGraph(nodes=nodes, edges=edges)

    p, r, f1 = topo_f1(
        g, g,
        radius=30.0,
        sample_step=5.0,
        match_threshold=10.0,
        seed_density=200.0,
    )
    assert p == pytest.approx(1.0, abs=0.01)
    assert r == pytest.approx(1.0, abs=0.01)
    assert f1 == pytest.approx(1.0, abs=0.01)


def test_topo_f1_partial_match() -> None:
    gt = TopoGraph(
        nodes=np.array([[0, 0], [400, 0]]),
        edges=np.array([[0, 1]]),
    )
    pred = TopoGraph(
        nodes=np.array([[0, 0], [200, 0]]),
        edges=np.array([[0, 1]]),
    )

    p, r, f1 = topo_f1(
        pred, gt,
        radius=30.0,
        sample_step=5.0,
        match_threshold=10.0,
        seed_density=400.0,
    )
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 < f1 < 1.0


def test_topo_f1_completely_different() -> None:
    g1 = TopoGraph(
        nodes=np.array([[0, 0], [200, 0], [400, 0]]),
        edges=np.array([[0, 1], [1, 2]]),
    )
    g2 = TopoGraph(
        nodes=np.array([[10000, 10000], [10200, 10000]]),
        edges=np.array([[0, 1]]),
    )

    p, r, f1 = topo_f1(
        g1, g2,
        radius=30.0,
        sample_step=5.0,
        match_threshold=5.0,
        seed_density=20.0,
    )
    assert (p, r, f1) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "gt_nodes, gt_edges, pred_nodes, pred_edges, expected",
    [
        (
            np.array([[0, 0], [200, 0]]),
            np.array([[0, 1]]),
            np.empty((0, 2)),
            np.empty((0, 2), dtype=np.int64),
            (0.0, 0.0, 0.0),
        ),
        (
            np.empty((0, 2)),
            np.empty((0, 2), dtype=np.int64),
            np.array([[0, 0], [200, 0]]),
            np.array([[0, 1]]),
            (0.0, 0.0, 0.0),
        ),
    ],
)
def test_topo_f1_one_empty_parametrized(
    gt_nodes: np.ndarray,
    gt_edges: np.ndarray,
    pred_nodes: np.ndarray,
    pred_edges: np.ndarray,
    expected: tuple[float, float, float],
) -> None:
    gt = TopoGraph(nodes=gt_nodes, edges=gt_edges)
    pred = TopoGraph(nodes=pred_nodes, edges=pred_edges)
    assert topo_f1(pred, gt) == expected
