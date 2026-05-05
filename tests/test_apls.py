"""Tests for the APLS (Average Path Length Similarity) metric."""

from __future__ import annotations

import numpy as np
import pytest

from topo_metrics import TopoGraph, apls


def test_apls_both_empty() -> None:
    g = TopoGraph(
        nodes=np.empty((0, 2)),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    assert apls(g, g) == 1.0


def test_apls_pred_empty_gt_nonempty() -> None:
    empty = TopoGraph(
        nodes=np.empty((0, 2)),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    gt = TopoGraph(
        nodes=np.array([[0, 0], [200, 0]]),
        edges=np.array([[0, 1]]),
    )
    assert apls(empty, gt) == 0.0


def test_apls_gt_empty_pred_nonempty() -> None:
    empty = TopoGraph(
        nodes=np.empty((0, 2)),
        edges=np.empty((0, 2), dtype=np.int64),
    )
    pred = TopoGraph(
        nodes=np.array([[0, 0], [200, 0]]),
        edges=np.array([[0, 1]]),
    )
    assert apls(pred, empty) == 0.0


def test_apls_identical() -> None:
    nodes = np.array([[0, 0], [200, 0], [400, 0]])
    edges = np.array([[0, 1], [1, 2]])
    g = TopoGraph(nodes=nodes, edges=edges)

    score = apls(g, g, min_path_length=10.0, sample_interval=10.0)
    assert score == pytest.approx(1.0, abs=0.1)


def test_apls_partial_match() -> None:
    gt = TopoGraph(
        nodes=np.array([[0, 0], [200, 0], [400, 0], [600, 0]]),
        edges=np.array([[0, 1], [1, 2], [2, 3]]),
    )
    pred = TopoGraph(
        nodes=np.array([[0, 0], [200, 0]]),
        edges=np.array([[0, 1]]),
    )

    score = apls(
        pred, gt,
        snapping_tol=10.0,
        min_path_length=10.0,
        sample_interval=10.0,
    )
    assert 0.0 < score < 1.0


def test_apls_completely_different() -> None:
    g1 = TopoGraph(
        nodes=np.array([[0, 0], [200, 0]]),
        edges=np.array([[0, 1]]),
    )
    g2 = TopoGraph(
        nodes=np.array([[10000, 10000], [10200, 10000]]),
        edges=np.array([[0, 1]]),
    )

    score = apls(
        g1, g2,
        snapping_tol=10.0,
        min_path_length=10.0,
        sample_interval=10.0,
    )
    assert score == pytest.approx(0.0)


def test_apls_identical_branching() -> None:
    nodes = np.array([
        [0, 0],
        [200, 0],
        [400, 0],
        [200, 200],
    ])
    edges = np.array([[0, 1], [1, 2], [1, 3]])
    g = TopoGraph(nodes=nodes, edges=edges)

    score = apls(g, g, min_path_length=10.0, sample_interval=10.0)
    assert score == pytest.approx(1.0, abs=0.1)
