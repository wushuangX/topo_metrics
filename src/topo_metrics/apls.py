"""APLS (Average Path Length Similarity) metric implementation."""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from topo_metrics._graph import TopoGraph


def _sample_control_points(
    graph: TopoGraph,
    sample_interval: float,
) -> list[int]:
    """Sample control points on a graph for APLS computation.

    Traverses chains (degree-2 sequences between junctions/endpoints),
    places control points at regular physical intervals, and includes
    junction/endpoint nodes. Uses spatial grid deduplication.

    Args:
        graph: Densified TopoGraph to sample from.
        sample_interval: Physical distance between control points on chains.

    Returns:
        List of node indices selected as control points.
    """
    if graph.num_nodes == 0:
        return []

    adj = graph._build_adjacency()
    degree = {i: len(neighbors) for i, neighbors in adj.items()}

    control_points: list[int] = []
    covered: set[int] = set()
    grid_keys: set[str] = set()
    spacing_arr = np.array(graph.spacing)

    def _grid_key(node_idx: int) -> str:
        coord = graph.nodes[node_idx] * spacing_arr
        cell = sample_interval
        return f"{int(coord[0] / cell)}_{int(coord[1] / cell)}"

    def _mark_covered(node_idx: int, hop_range: int) -> None:
        queue: list[tuple[int, int]] = [(node_idx, 0)]
        visited_local: set[int] = {node_idx}
        while queue:
            current, dist = queue.pop(0)
            if dist > hop_range:
                continue
            covered.add(current)
            for nb in adj.get(current, []):
                if nb not in visited_local:
                    visited_local.add(nb)
                    queue.append((nb, dist + 1))

    def _try_add_control(node_idx: int) -> None:
        gk = _grid_key(node_idx)
        if gk not in grid_keys:
            grid_keys.add(gk)
            control_points.append(node_idx)
            _mark_covered(node_idx, 4)

    chains = graph.get_chains()
    for chain in chains:
        if len(chain) <= 2:
            continue

        total_dist = 0.0
        cumulative: list[float] = [0.0]
        for k in range(1, len(chain)):
            seg_len = graph.physical_distance(chain[k - 1], chain[k])
            total_dist += seg_len
            cumulative.append(total_dist)

        if total_dist <= sample_interval:
            continue

        n_samples = int(total_dist / sample_interval) + 1
        for i in range(1, n_samples):
            target_dist = total_dist * i / n_samples
            idx = 1
            while idx < len(cumulative) and cumulative[idx] < target_dist:
                idx += 1
            node_idx = chain[min(idx, len(chain) - 1)]

            if node_idx not in covered or degree.get(node_idx, 0) == 1:
                _try_add_control(node_idx)

    for node_id in range(graph.num_nodes):
        if degree.get(node_id, 0) != 2:
            if node_id not in covered or degree.get(node_id, 0) == 1:
                _try_add_control(node_id)

    return control_points


def _snap_control_points(
    control_indices: list[int],
    source_graph: TopoGraph,
    target_graph: TopoGraph,
    tolerance: float,
) -> np.ndarray:
    """Snap control points from source graph to nearest target graph nodes.

    Uses one-to-one matching with exclusion propagation: when a target
    node is matched, its graph neighborhood (4 hops) is excluded from
    further matching.

    Args:
        control_indices: Node indices in source_graph that are control points.
        source_graph: Graph containing the control points.
        target_graph: Graph to snap to.
        tolerance: Maximum physical distance for a valid match.

    Returns:
        (M,) array of matched target node indices. -1 means no match.
    """
    n_cp = len(control_indices)
    if n_cp == 0 or target_graph.num_nodes == 0:
        return np.full(n_cp, -1, dtype=np.int64)

    spacing_arr = np.array(source_graph.spacing)
    source_coords = source_graph.nodes[control_indices] * spacing_arr
    target_coords = target_graph.nodes * spacing_arr

    target_tree = KDTree(target_coords)
    matches = np.full(n_cp, -1, dtype=np.int64)
    used_targets: set[int] = set()

    distances, indices = target_tree.query(source_coords)

    target_adj = target_graph._build_adjacency()

    def _exclude_neighborhood(node_idx: int, hops: int) -> None:
        queue: list[tuple[int, int]] = [(node_idx, 0)]
        visited_local: set[int] = {node_idx}
        while queue:
            current, dist = queue.pop(0)
            if dist > hops:
                continue
            used_targets.add(current)
            for nb in target_adj.get(current, []):
                if nb not in visited_local:
                    visited_local.add(nb)
                    queue.append((nb, dist + 1))

    for i in range(n_cp):
        if distances[i] > tolerance:
            continue
        candidate = int(indices[i])
        if candidate in used_targets:
            continue
        matches[i] = candidate
        _exclude_neighborhood(candidate, 4)

    return matches


def _compute_apls_one_way(
    source_graph: TopoGraph,
    target_graph: TopoGraph,
    sample_interval: float,
    snapping_tol: float,
    min_path_length: float,
) -> float:
    """Compute APLS in one direction: source → target.

    Samples control points on source_graph, snaps them to target_graph,
    computes shortest paths on both graphs, and calculates path length
    similarity.

    Args:
        source_graph: Graph to sample control points from (densified).
        target_graph: Graph to snap to and compare paths (densified).
        sample_interval: Physical distance between control points.
        snapping_tol: Maximum physical distance for node matching.
        min_path_length: Ignore path pairs shorter than this.

    Returns:
        One-way APLS score in [0, 1].
    """
    control_indices = _sample_control_points(source_graph, sample_interval)
    if len(control_indices) == 0:
        return 1.0

    snapped = _snap_control_points(control_indices, source_graph, target_graph, snapping_tol)

    matched_cp_source: list[int] = []
    matched_cp_target: list[int] = []

    for i, cp_idx in enumerate(control_indices):
        if snapped[i] >= 0:
            matched_cp_source.append(cp_idx)
            matched_cp_target.append(int(snapped[i]))

    target_unique = list(dict.fromkeys(matched_cp_target))

    sp_source: dict[int, dict[int, float]] = {}
    for cp in matched_cp_source:
        sp_source[cp] = source_graph.shortest_paths(cp, matched_cp_source)

    sp_target: dict[int, dict[int, float]] = {}
    for cp in target_unique:
        sp_target[cp] = target_graph.shortest_paths(cp, target_unique)

    sum_penalty = 0.0
    num_pairs = 0

    for i in range(len(control_indices)):
        for j in range(i + 1, len(control_indices)):
            cp1_src = control_indices[i]
            cp2_src = control_indices[j]
            cp1_tgt = int(snapped[i])
            cp2_tgt = int(snapped[j])

            num_pairs += 1

            if cp1_tgt < 0 or cp2_tgt < 0:
                sum_penalty += 1.0
                continue

            d_src = sp_source[cp1_src].get(cp2_src, -1.0)
            if d_src < 0:
                num_pairs -= 1
                continue

            if d_src <= min_path_length:
                num_pairs -= 1
                continue

            d_tgt = sp_target.get(cp1_tgt, {}).get(cp2_tgt, -1.0)
            if d_tgt < 0:
                d_tgt = 0.0

            penalty = abs(d_src - d_tgt) / d_src
            penalty = min(penalty, 1.0)
            sum_penalty += penalty

    if num_pairs == 0:
        return 1.0

    return 1.0 - sum_penalty / num_pairs


def apls(
    prediction: TopoGraph,
    ground_truth: TopoGraph,
    snapping_tol: float = 10.0,
    min_path_length: float = 100.0,
    sample_interval: float = 50.0,
    densify_interval: float = 2.0,
) -> float:
    """Compute Average Path Length Similarity.

    Measures topological similarity by comparing shortest path lengths
    between matched control points on two graphs. Runs bidirectionally
    (GT→pred and pred→GT) and returns the average score.

    Args:
        prediction: Predicted graph.
        ground_truth: Ground truth graph.
        snapping_tol: Maximum physical distance for node matching.
        min_path_length: Ignore path pairs shorter than this.
        sample_interval: Physical distance between control points.
        densify_interval: Physical distance for edge densification.

    Returns:
        APLS score in [0, 1]. 1 means perfect topology match.
    """
    if prediction.num_nodes == 0 and ground_truth.num_nodes == 0:
        return 1.0
    if prediction.num_nodes == 0 or ground_truth.num_nodes == 0:
        return 0.0

    pred_dense = prediction.densify(densify_interval)
    gt_dense = ground_truth.densify(densify_interval)

    score_gt_to_pred = _compute_apls_one_way(
        gt_dense, pred_dense, sample_interval, snapping_tol, min_path_length
    )
    score_pred_to_gt = _compute_apls_one_way(
        pred_dense, gt_dense, sample_interval, snapping_tol, min_path_length
    )

    return (score_gt_to_pred + score_pred_to_gt) / 2.0
