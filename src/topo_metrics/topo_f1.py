"""TOPO F1 metric — topological precision, recall, and F1 score.

Clean rewrite of the Sat2Graph TOPO metric (Biagioni & Eriksson, 2012).
Evaluates local geometric and topological overlap between a predicted graph
and a ground-truth graph by sampling points along edges within a radius and
computing bipartite matching.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree

from topo_metrics._graph import TopoGraph


def _physical_coords(nodes: np.ndarray, spacing: tuple[float, float]) -> np.ndarray:
    return nodes * np.array(spacing)


def _edge_direction(graph: TopoGraph, n1: int, n2: int) -> np.ndarray:
    diff = (graph.nodes[n2] - graph.nodes[n1]) * np.array(graph.spacing)
    length = np.linalg.norm(diff)
    if length < 1e-12:
        return np.zeros(2)
    return diff / length


def _generate_seeds(
    gt: TopoGraph,
    density: float,
    margin: float = 0.0,
    region: tuple[float, float, float, float] | None = None,
) -> list[tuple[np.ndarray, int, int, float, float]]:
    """Sample seed points along GT chains.

    Returns list of (position_physical, edge_node1, edge_node2,
    dist_from_n1, dist_from_n2) for each seed.
    """
    spacing_arr = np.array(gt.spacing)
    chains = gt.get_chains()
    seeds: list[tuple[np.ndarray, int, int, float, float]] = []

    for chain in chains:
        if len(chain) < 2:
            continue

        cum_dists = [0.0]
        for i in range(len(chain) - 1):
            d = gt.physical_distance(chain[i], chain[i + 1])
            cum_dists.append(cum_dists[-1] + d)

        total_dist = cum_dists[-1]
        if total_dist < density / 2:
            continue

        n_seeds = max(int(total_dist / density), 1)
        alphas = [(k + 1) / (n_seeds + 1) for k in range(n_seeds)]

        for alpha in alphas:
            target_dist = alpha * total_dist
            for j in range(len(chain) - 1):
                if cum_dists[j] <= target_dist <= cum_dists[j + 1]:
                    seg_len = cum_dists[j + 1] - cum_dists[j]
                    if seg_len < 1e-12:
                        break
                    frac = (target_dist - cum_dists[j]) / seg_len
                    pos = gt.nodes[chain[j]] * (1 - frac) + gt.nodes[chain[j + 1]] * frac
                    pos_phys = pos * spacing_arr

                    if region is not None and margin > 0:
                        if (
                            pos_phys[0] - region[0] < margin
                            or region[2] - pos_phys[0] < margin
                            or pos_phys[1] - region[1] < margin
                            or region[3] - pos_phys[1] < margin
                        ):
                            break

                    dist_n1 = target_dist - cum_dists[j]
                    dist_n2 = seg_len - dist_n1
                    seeds.append((pos_phys, chain[j], chain[j + 1], dist_n1, dist_n2))
                    break

    return seeds


def _match_seed_to_pred(
    seed_pos: np.ndarray,
    pred: TopoGraph,
    threshold: float,
    cos_angle_threshold: float,
    gt_n1: int,
    gt_n2: int,
    gt: TopoGraph,
    used_pred_edges: set[tuple[int, int]],
) -> tuple[int, int, float, float] | None:
    """Find the nearest predicted edge for a GT seed point.

    Returns (pred_n1, pred_n2, dist_from_n1, dist_from_n2) or None.
    """
    if pred.num_nodes == 0 or pred.num_edges == 0:
        return None

    spacing_arr = np.array(pred.spacing)
    pred_phys = pred.nodes * spacing_arr
    tree = KDTree(pred_phys)
    dists, indices = tree.query(seed_pos, k=min(8, pred.num_nodes))

    if isinstance(dists, float):
        dists = np.array([dists])
        indices = np.array([indices], dtype=int)

    gt_dir = _edge_direction(gt, gt_n1, gt_n2)

    best_dist = threshold
    best_result: tuple[int, int, float, float] | None = None

    for idx in range(len(dists)):
        node_id = int(indices[idx])

        adj = pred._build_adjacency()
        for neighbor in adj.get(node_id, []):
            edge_key = (min(node_id, neighbor), max(node_id, neighbor))
            if edge_key in used_pred_edges:
                continue

            pred_dir = _edge_direction(pred, node_id, neighbor)
            cos_sim = abs(float(np.dot(gt_dir, pred_dir)))

            if cos_sim < cos_angle_threshold:
                continue

            p1 = pred.nodes[node_id] * spacing_arr
            p2 = pred.nodes[neighbor] * spacing_arr
            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-12:
                continue

            proj = float(np.dot(seed_pos - p1, edge_vec) / edge_len)
            proj = max(0.0, min(edge_len, proj))
            closest = p1 + (proj / edge_len) * edge_vec
            pt_dist = float(np.linalg.norm(seed_pos - closest))

            if pt_dist < best_dist:
                best_dist = pt_dist
                best_result = (node_id, neighbor, proj, edge_len - proj)

    if best_result is not None:
        return best_result
    return None


def _topo_walk(
    graph: TopoGraph,
    seed_n1: int,
    seed_n2: int,
    seed_dist1: float,
    seed_dist2: float,
    step: float,
    radius: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """BFS walk from seed edge, sampling points every *step* physical units.

    Returns list of (position_physical, direction_unit_vector).
    """
    spacing_arr = np.array(graph.spacing)
    adj = graph._build_adjacency()

    n1_pos = graph.nodes[seed_n1] * spacing_arr
    n2_pos = graph.nodes[seed_n2] * spacing_arr
    edge_vec = n2_pos - n1_pos
    edge_len = float(np.linalg.norm(edge_vec))
    direction = edge_vec / edge_len if edge_len > 1e-12 else np.zeros(2)

    visited_nodes: dict[int, float] = {}
    edge_covered: dict[tuple[int, int], float] = {}
    points: list[tuple[np.ndarray, np.ndarray]] = []

    alpha = 0.0
    while alpha <= 1.0:
        pt = n1_pos * (1 - alpha) + n2_pos * alpha
        d1 = alpha * edge_len
        d2 = (1 - alpha) * edge_len
        if seed_dist1 - d1 < radius or seed_dist2 - d2 < radius:
            points.append((pt, direction))
        alpha += step / edge_len if edge_len > 1e-12 else 1.0

    queue: deque[tuple[int, int, float]] = deque()
    queue.append((seed_n1, -1, seed_dist1))
    queue.append((seed_n2, -1, seed_dist2))

    while queue:
        node_cur, node_prev, dist_from_seed = queue.popleft()

        if node_cur in visited_nodes and visited_nodes[node_cur] <= dist_from_seed:
            continue
        if dist_from_seed > radius:
            continue

        visited_nodes[node_cur] = dist_from_seed

        for next_node in adj.get(node_cur, []):
            if next_node == node_prev:
                continue
            if next_node == seed_n1 or next_node == seed_n2:
                continue
            if next_node == node_cur:
                continue

            cur_pos = graph.nodes[node_cur] * spacing_arr
            next_pos = graph.nodes[next_node] * spacing_arr
            evec = next_pos - cur_pos
            elen = float(np.linalg.norm(evec))
            if elen < 1e-12:
                continue
            edir = evec / elen

            start_limit = 0.0
            end_limit = elen
            ek = (node_cur, next_node)
            if ek in edge_covered:
                start_limit = edge_covered[ek]
            ek_rev = (next_node, node_cur)
            if ek_rev in edge_covered:
                end_limit = elen - edge_covered[ek_rev]

            bias = step * math.ceil(dist_from_seed / step) - dist_from_seed
            cur = bias

            while cur < elen:
                a = cur / elen
                if dist_from_seed + elen * a > radius:
                    break
                if elen * a < start_limit:
                    cur += step
                    continue
                if elen * a > end_limit:
                    break

                pt = cur_pos * (1 - a) + next_pos * a
                points.append((pt, edir))
                cur += step

            if ek in edge_covered:
                edge_covered[ek] = max(edge_covered[ek], cur - step)
            else:
                edge_covered[ek] = cur - step

            queue.append((next_node, node_cur, dist_from_seed + elen))

    return points


def _bipartite_match(
    marbles: list[tuple[np.ndarray, np.ndarray]],
    holes: list[tuple[np.ndarray, np.ndarray]],
    dist_threshold: float,
    cos_angle_threshold: float,
) -> tuple[float, float]:
    """Compute precision and recall via bipartite matching.

    Uses scipy.optimize.linear_sum_assignment with a cost matrix.
    """
    n_m = len(marbles)
    n_h = len(holes)

    if n_m == 0 or n_h == 0:
        return 0.0, 0.0

    large_cost = max(dist_threshold * 2.0, 1.0)

    cost = np.full((n_m, n_h), large_cost)

    marble_pos = np.array([m[0] for m in marbles])
    marble_dir = np.array([m[1] for m in marbles])
    hole_pos = np.array([h[0] for h in holes])
    hole_dir = np.array([h[1] for h in holes])

    marble_tree = KDTree(marble_pos)
    pairs = marble_tree.query_ball_point(hole_pos, r=dist_threshold)

    for j in range(n_h):
        for i in pairs[j]:
            d = float(np.linalg.norm(marble_pos[i] - hole_pos[j]))
            if d >= dist_threshold:
                continue
            cos_sim = abs(float(np.dot(marble_dir[i], hole_dir[j])))
            if cos_sim < cos_angle_threshold:
                continue
            cost[i, j] = d

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_m = 0
    matched_h = 0
    matched_holes: set[int] = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < large_cost:
            matched_m += 1
            matched_holes.add(c)

    matched_h = len(matched_holes)

    precision = matched_m / n_m if n_m > 0 else 0.0
    recall = matched_h / n_h if n_h > 0 else 0.0
    return precision, recall


def _graph_distance_between_seeds(
    graph: TopoGraph,
    s1: tuple[int, int, float, float],
    s2: tuple[int, int, float, float],
    max_dist: float,
) -> float:
    """Estimate graph distance between two seed locations."""
    n1_a, n2_a, d1_a, d2_a = s1
    n1_b, n2_b, d1_b, d2_b = s2

    if n1_a == n1_b and n2_a == n2_b:
        return abs(d1_a - d1_b)
    if n1_a == n2_b and n2_a == n1_b:
        return abs(d1_a - d2_b)

    dist = graph.physical_distance(n1_a, n1_b)
    if dist < max_dist:
        return dist

    targets = [n1_b, n2_b]
    paths = graph.shortest_paths(n1_a, targets)
    best = max_dist * 2
    for t in targets:
        d = paths.get(t, -1.0)
        if 0 <= d < best:
            best = d
    paths2 = graph.shortest_paths(n2_a, targets)
    for t in targets:
        d = paths2.get(t, -1.0)
        if 0 <= d < best:
            adj_d = d + d2_a
            if adj_d < best:
                best = adj_d
    return best


def _deduplicate_seeds(
    gt: TopoGraph,
    results: list[tuple[float, float]],
    seed_data: list[tuple[int, int, float, float]],
    dedup_dist: float,
) -> list[tuple[float, float]]:
    """Remove TOPO results where seeds are too close (graph-distance based).

    Keeps the higher-F1 seed when two seeds are within dedup_dist.
    """
    if not results:
        return results

    f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 for p, r in results]
    order = sorted(range(len(results)), key=lambda i: f1_scores[i], reverse=True)

    keep: list[int] = []
    removed: set[int] = set()

    for idx in order:
        if idx in removed:
            continue
        keep.append(idx)
        for other_idx in order:
            if other_idx in removed or other_idx == idx:
                continue
            gd = _graph_distance_between_seeds(gt, seed_data[idx], seed_data[other_idx], dedup_dist)
            if gd < dedup_dist:
                removed.add(other_idx)

    return [results[i] for i in keep]


def topo_f1(
    prediction: TopoGraph,
    ground_truth: TopoGraph,
    radius: float = 300.0,
    sample_step: float = 5.0,
    match_threshold: float = 10.0,
    angle_threshold: float = 30.0,
    seed_density: float = 50.0,
) -> tuple[float, float, float]:
    """Compute TOPO metric (precision, recall, F1).

    Args:
        prediction: Predicted graph.
        ground_truth: Ground truth graph.
        radius: Subgraph sampling radius in physical units.
        sample_step: Sampling interval along edges in physical units.
        match_threshold: Maximum distance for marble-hole matching.
        angle_threshold: Maximum angle difference in degrees for matching.
        seed_density: Physical distance between seed points on GT graph.

    Returns:
        Tuple of (precision, recall, f1). All in [0, 1].
    """
    cos_angle_threshold = math.cos(math.radians(angle_threshold))

    if ground_truth.num_nodes == 0 and prediction.num_nodes == 0:
        return (1.0, 1.0, 1.0)
    if ground_truth.num_nodes == 0 or prediction.num_nodes == 0:
        return (0.0, 0.0, 0.0)
    if ground_truth.num_edges == 0 and prediction.num_edges == 0:
        return (1.0, 1.0, 1.0)
    if ground_truth.num_edges == 0 or prediction.num_edges == 0:
        return (0.0, 0.0, 0.0)

    seeds = _generate_seeds(ground_truth, seed_density)

    if not seeds:
        return (0.0, 0.0, 0.0)

    results: list[tuple[float, float]] = []
    seed_edge_data: list[tuple[int, int, float, float]] = []
    used_pred_edges: set[tuple[int, int]] = set()

    for seed_pos, gt_n1, gt_n2, gt_d1, gt_d2 in seeds:
        match = _match_seed_to_pred(
            seed_pos,
            prediction,
            match_threshold,
            cos_angle_threshold,
            gt_n1,
            gt_n2,
            ground_truth,
            used_pred_edges,
        )

        if match is None:
            continue

        pred_n1, pred_n2, pred_d1, pred_d2 = match
        edge_key = (min(pred_n1, pred_n2), max(pred_n1, pred_n2))
        used_pred_edges.add(edge_key)

        marbles = _topo_walk(
            prediction,
            pred_n1,
            pred_n2,
            pred_d1,
            pred_d2,
            sample_step,
            radius,
        )
        holes = _topo_walk(
            ground_truth,
            gt_n1,
            gt_n2,
            gt_d1,
            gt_d2,
            sample_step,
            radius + sample_step,
        )

        if not marbles or not holes:
            results.append((0.0, 0.0))
            seed_edge_data.append((gt_n1, gt_n2, gt_d1, gt_d2))
            continue

        precision, recall = _bipartite_match(
            marbles,
            holes,
            match_threshold,
            cos_angle_threshold,
        )
        results.append((precision, recall))
        seed_edge_data.append((gt_n1, gt_n2, gt_d1, gt_d2))

    if not results:
        return (0.0, 0.0, 0.0)

    results = _deduplicate_seeds(ground_truth, results, seed_edge_data, radius)

    avg_p = sum(p for p, _ in results) / len(results)
    avg_r = sum(r for _, r in results) / len(results)
    f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0

    return (avg_p, avg_r, f1)
