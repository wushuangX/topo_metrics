"""APLS precision and performance benchmark: Go reference vs Python implementation.

Outputs a JSON file with all measurements for report generation.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from topo_metrics import TopoGraph, apls

LAT_TOP_LEFT = 41.0
LON_TOP_LEFT = -71.0
GO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "external", "Sat2Graph", "metrics", "apls")
)
GO_BIN = os.path.join(GO_DIR, "apls_bin")
RESULTS = []


def xy_to_latlon(x, y):
    lat = LAT_TOP_LEFT - y / 111111.0
    lon = LON_TOP_LEFT + x / (111111.0 * math.cos(math.radians(LAT_TOP_LEFT)))
    return lat, lon


def make_go_json(nodes, edges):
    return [
        [[*xy_to_latlon(x, y)] for x, y in nodes],
        [[int(a), int(b)] for a, b in edges],
    ]


def run_go_apls(gt_json_path, prop_json_path, timeout=300):
    t0 = time.perf_counter()
    result = subprocess.run(
        [GO_BIN, gt_json_path, prop_json_path, "/dev/null"],
        capture_output=True, text=True, cwd=GO_DIR, timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        return None, elapsed, result.stderr
    for line in result.stdout.strip().split("\n"):
        if "apls:" in line:
            return float(line.split("apls:")[1].strip()), elapsed, None
    return None, elapsed, result.stdout


def run_py_apls(gt_nodes, gt_edges, prop_nodes, prop_edges, **kwargs):
    gt = TopoGraph(nodes=gt_nodes, edges=gt_edges, spacing=(1.0, 1.0))
    prop = TopoGraph(nodes=prop_nodes, edges=prop_edges, spacing=(1.0, 1.0))
    t0 = time.perf_counter()
    score = apls(prop, gt, **kwargs)
    elapsed = time.perf_counter() - t0
    return score, elapsed


def benchmark(name, gt_n, gt_e, pr_n, pr_e, tol=0.15, n_runs=3, **kwargs):
    defaults = dict(snapping_tol=10.0, min_path_length=100.0,
                    sample_interval=50.0, densify_interval=2.0)
    defaults.update(kwargs)

    # Write temp JSON files for Go
    gt_json = make_go_json(gt_n, gt_e)
    pr_json = make_go_json(pr_n, pr_e)

    # Go runs
    go_scores = []
    go_times = []
    go_error = None
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = os.path.join(tmpdir, "gt.json")
        pr_path = os.path.join(tmpdir, "pr.json")
        json.dump(gt_json, open(gt_path, "w"))
        json.dump(pr_json, open(pr_path, "w"))

        for _ in range(n_runs):
            s, t, err = run_go_apls(gt_path, pr_path)
            if err:
                go_error = err
                break
            go_scores.append(s)
            go_times.append(t)

    # Python runs
    py_scores = []
    py_times = []
    for _ in range(n_runs):
        s, t = run_py_apls(gt_n, gt_e, pr_n, pr_e, **defaults)
        py_scores.append(s)
        py_times.append(t)

    go_avg = float(np.mean(go_scores)) if go_scores else None
    py_avg = float(np.mean(py_scores))
    go_time_avg = float(np.mean(go_times)) if go_times else None
    py_time_avg = float(np.mean(py_times))

    diff = abs(go_avg - py_avg) if go_avg is not None else None
    gt_info = f"nodes={len(gt_n)} edges={len(gt_e)}"
    pr_info = f"nodes={len(pr_n)} edges={len(pr_e)}"

    result = {
        "name": name,
        "gt_info": gt_info,
        "pr_info": pr_info,
        "go_score": go_avg,
        "py_score": py_avg,
        "diff": diff,
        "precision_pass": diff is not None and diff < tol,
        "go_time_s": go_time_avg,
        "py_time_s": py_time_avg,
        "speedup": go_time_avg / py_time_avg if go_time_avg and py_time_avg else None,
        "go_error": go_error,
        "n_runs": n_runs,
    }
    RESULTS.append(result)

    status = "✅" if result["precision_pass"] else ("⚠️" if go_avg is None else "❌")
    diff_str = f"{diff:.6f}" if diff is not None else "N/A"
    go_str = f"{go_avg:.6f}" if go_avg is not None else "N/A (Go err)"
    py_str = f"{py_avg:.6f}"
    gt_str = f"{gt_n.shape[0]:>5}n {gt_e.shape[0]:>5}e"
    pr_str = f"{pr_n.shape[0]:>5}n {pr_e.shape[0]:>5}e"

    print(f"  {status} {name:45s}  GT={gt_str}  PR={pr_str}  "
          f"Go={go_str}  Py={py_str}  Δ={diff_str}")

    return result


# ============================================================
# Graph generators
# ============================================================

def _path(sx, sy, length, segs=1):
    sl = length / segs
    n = np.array([[sx + i * sl, sy] for i in range(segs + 1)])
    e = np.array([[i, i + 1] for i in range(segs)])
    return n, e


def _grid(cx, cy, sp, nx, ny):
    n = []
    for r in range(ny + 1):
        for c in range(nx + 1):
            n.append([cx - nx * sp / 2 + c * sp, cy - ny * sp / 2 + r * sp])
    e = []
    for r in range(ny + 1):
        for c in range(nx):
            e.append([r * (nx + 1) + c, r * (nx + 1) + c + 1])
    for r in range(ny):
        for c in range(nx + 1):
            e.append([r * (nx + 1) + c, (r + 1) * (nx + 1) + c])
    return np.array(n), np.array(e)


def _tree(depth, cx, cy, arm):
    """Binary tree graph."""
    nodes = [[cx, cy]]
    edges = []
    _tree_rec(0, cx, cy, arm, 0, depth, nodes, edges, -math.pi / 2)
    return np.array(nodes), np.array(edges)


def _tree_rec(parent, cx, cy, arm, d, max_d, nodes, edges, angle):
    if d >= max_d:
        return
    for sign in [-1, 1]:
        a = angle + sign * math.pi / 4
        nx = cx + arm * math.cos(a)
        ny = cy + arm * math.sin(a)
        idx = len(nodes)
        nodes.append([nx, ny])
        edges.append([parent, idx])
        _tree_rec(idx, nx, ny, arm * 0.7, d + 1, max_d, nodes, edges, a)


def _random_graph(n_nodes, region=1600, offset=200, seed=42):
    """Random connected graph via nearest-neighbor chain."""
    rng = np.random.RandomState(seed)
    nodes = rng.rand(n_nodes, 2) * region + offset
    from scipy.spatial import KDTree
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.sparse import csr_matrix
    tree = KDTree(nodes)
    D, _ = tree.query(nodes, k=min(n_nodes, 6))
    rows, cols, dists = [], [], []
    for i in range(n_nodes):
        for j_idx in range(1, min(n_nodes, 6)):
            j = int(_[i][j_idx]) if n_nodes > 6 else 0
            if n_nodes <= 6:
                j = int(tree.query(nodes[i], k=n_nodes)[1][j_idx])
            rows.append(i)
            cols.append(j)
            dists.append(float(np.linalg.norm(nodes[i] - nodes[j])))
    mat = csr_matrix((dists, (rows, cols)), shape=(n_nodes, n_nodes))
    mst = minimum_spanning_tree(mat)
    mst = mst.tocoo()
    edges = list(zip(mst.row.tolist(), mst.col.tolist()))
    # Add some extra edges for complexity
    for i in range(n_nodes):
        dists_i, idxs_i = tree.query(nodes[i], k=min(4, n_nodes))
        for k in range(1, min(4, n_nodes)):
            j = int(idxs_i[k])
            if [min(i, j), max(i, j)] not in [[min(a, b), max(a, b)] for a, b in edges]:
                edges.append([i, j])
    return nodes, np.array(edges)


# ============================================================
# Main benchmark
# ============================================================

def main():
    print("=" * 110)
    print("APLS Benchmark: Precision & Performance (Go Reference vs Python)")
    print("=" * 110)

    # ---- Section 1: Precision across diverse topologies ----
    print("\n" + "=" * 110)
    print("SECTION 1: PRECISION VERIFICATION")
    print("=" * 110)

    # 1a. Identical graphs
    print("\n--- 1.1 Identical Graphs ---")
    for segs in [1, 3, 5, 10, 20]:
        n, e = _path(300, 800, 1200, segs)
        benchmark(f"Identical: path {segs}-seg 1200m", n, e, n.copy(), e.copy())

    n, e = _grid(800, 800, 300, 2, 2)
    benchmark("Identical: 2x2 grid 300m", n, e, n.copy(), e.copy())
    n, e = _grid(800, 800, 200, 3, 3)
    benchmark("Identical: 3x3 grid 200m", n, e, n.copy(), e.copy())

    n = np.array([[300, 800], [700, 800], [1100, 800], [700, 400], [700, 1200]])
    e = np.array([[0, 1], [1, 2], [1, 3], [1, 4]])
    benchmark("Identical: cross 5-node", n, e, n.copy(), e.copy())

    n = np.array([[300, 800], [700, 800], [1100, 800], [700, 400], [700, 1200],
                   [1100, 400], [1100, 1200]])
    e = np.array([[0, 1], [1, 2], [1, 3], [1, 4], [2, 5], [2, 6]])
    benchmark("Identical: H-tree 7-node", n, e, n.copy(), e.copy())

    for depth in [2, 3, 4]:
        n, e = _tree(depth, 800, 400, 400)
        benchmark(f"Identical: binary tree depth={depth}", n, e, n.copy(), e.copy())

    # 1b. Partial overlap
    print("\n--- 1.2 Partial Overlap ---")
    gt_n = np.array([[300, 800], [800, 800], [1300, 800]])
    gt_e = np.array([[0, 1], [1, 2]])
    for frac, end_x in [(0.5, 800), (0.75, 1050)]:
        pr_n = np.array([[300, 800], [end_x, 800]])
        pr_e = np.array([[0, 1]])
        benchmark(f"Partial: pred covers {int(frac*100)}%", gt_n, gt_e, pr_n, pr_e)

    # Pred has extra branch
    gt_n = np.array([[300, 800], [800, 800], [1300, 800]])
    gt_e = np.array([[0, 1], [1, 2]])
    pr_n = np.array([[300, 800], [800, 800], [1300, 800], [800, 400], [800, 1200]])
    pr_e = np.array([[0, 1], [1, 2], [1, 3], [1, 4]])
    benchmark("Partial: pred has extra branches", gt_n, gt_e, pr_n, pr_e)

    # GT has extra branch
    gt_n = np.array([[300, 800], [800, 800], [1300, 800], [800, 400], [800, 1200]])
    gt_e = np.array([[0, 1], [1, 2], [1, 3], [1, 4]])
    pr_n = np.array([[300, 800], [800, 800], [1300, 800]])
    pr_e = np.array([[0, 1], [1, 2]])
    benchmark("Partial: GT has extra branches", gt_n, gt_e, pr_n, pr_e)

    # Different topologies
    print("\n--- 1.3 Different Topologies ---")
    gt_n = np.array([[300, 800], [1300, 800]])
    gt_e = np.array([[0, 1]])
    pr_n = np.array([[300, 800], [800, 400], [1300, 800]])
    pr_e = np.array([[0, 1], [1, 2]])
    benchmark("Topology: straight vs V-detour", gt_n, gt_e, pr_n, pr_e)

    gt_n, gt_e = _grid(800, 800, 300, 2, 2)
    pr_n = np.array([[500, 500], [1100, 500], [500, 800], [1100, 800],
                      [500, 1100], [1100, 1100]])
    pr_e = np.array([[0, 1], [2, 3], [4, 5]])
    benchmark("Topology: grid vs horizontal lines only", gt_n, gt_e, pr_n, pr_e)

    gt_n = np.array([[300, 400], [800, 400], [800, 1000]])
    gt_e = np.array([[0, 1], [1, 2]])
    pr_n = np.array([[300, 400], [800, 400], [800, 1000]])
    pr_e = np.array([[0, 1], [1, 2]])
    benchmark("Topology: L-shape identical", gt_n, gt_e, pr_n, pr_e)

    # Spatial offset
    print("\n--- 1.4 Spatial Offset ---")
    gt_n, gt_e = _path(300, 800, 1000, 3)
    for dx, dy in [(5, 0), (8, 0), (0, 5), (0, 8), (5, 5), (3, 7), (9, 2)]:
        pr_n = gt_n + np.array([dx, dy])
        benchmark(f"Offset: +({dx},{dy})m", gt_n, gt_e, pr_n, gt_e.copy())

    # Disconnected components
    print("\n--- 1.5 Disconnected Components ---")
    g1_n, g1_e = _path(300, 400, 400, 1)
    g2_n, g2_e = _path(900, 400, 400, 1)
    g3_n, g3_e = _path(600, 1200, 400, 1)

    all_n = np.concatenate([g1_n, g2_n])
    all_e = np.concatenate([g1_e, [[a + 2, b + 2] for a, b in g2_e]])
    benchmark("Disconnected: 2 comp identical", all_n, all_e, all_n.copy(), all_e.copy())

    all_n3 = np.concatenate([g1_n, g2_n, g3_n])
    all_e3 = np.concatenate([
        g1_e,
        [[a + 2, b + 2] for a, b in g2_e],
        [[a + 4, b + 4] for a, b in g3_e],
    ])
    pr_n3 = np.concatenate([g1_n, g3_n])
    pr_e3 = np.concatenate([g1_e, [[a + 2, b + 2] for a, b in g3_e]])
    benchmark("Disconnected: 3 comp pred matches 2", all_n3, all_e3, pr_n3, pr_e3)

    # ---- Section 2: Performance at scale ----
    print("\n" + "=" * 110)
    print("SECTION 2: PERFORMANCE BENCHMARK")
    print("=" * 110)

    print("\n--- 2.1 Scaling: Identical Path Graphs ---")
    for n_edges in [5, 10, 25, 50, 100, 200]:
        length = n_edges * 100
        n, e = _path(200, 800, length, n_edges)
        benchmark(f"Scale: path {n_edges} edges ({length}m)",
                  n, e, n.copy(), e.copy(), n_runs=3)

    print("\n--- 2.2 Scaling: Grid Graphs ---")
    for size in [(2, 2), (3, 3), (5, 5), (8, 8), (10, 10)]:
        nx, ny = size
        n, e = _grid(800, 800, 200, nx, ny)
        benchmark(f"Scale: {nx}x{ny} grid 200m", n, e, n.copy(), e.copy(), n_runs=3)

    print("\n--- 2.3 Scaling: Random Graphs ---")
    for n_nodes in [20, 50, 100, 200, 500]:
        n, e = _random_graph(n_nodes, seed=42)
        benchmark(f"Scale: random {n_nodes} nodes", n, e, n.copy(), e.copy(), n_runs=3)

    # ---- Section 3: Real example data (if available) ----
    gt_json_path = os.path.join(GO_DIR, "example", "gt.json")
    if os.path.exists(gt_json_path):
        print("\n--- 2.4 Real Data (Sat2Graph example) ---")
        with open(gt_json_path) as f:
            gt_data = json.load(f)
        gt_nodes_arr = np.array(gt_data[0])
        gt_edges_arr = np.array(gt_data[1], dtype=np.int64)
        # Convert latlon back to xy
        gt_xy = np.column_stack([
            (gt_nodes_arr[:, 1] - LON_TOP_LEFT) * 111111.0 * math.cos(math.radians(LAT_TOP_LEFT)),
            (LAT_TOP_LEFT - gt_nodes_arr[:, 0]) * 111111.0,
        ])
        benchmark("Real: Sat2Graph gt vs gt (identical)",
                  gt_xy, gt_edges_arr, gt_xy.copy(), gt_edges_arr.copy(), n_runs=1)

    # ---- Write results ----
    out_path = os.path.join(os.path.dirname(__file__), "..", "benchmark_results.json")
    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    # ---- Summary ----
    precision_tests = [r for r in RESULTS if r["go_score"] is not None]
    perf_tests = [r for r in RESULTS if r["go_time_s"] is not None]

    max_diff = max((r["diff"] for r in precision_tests), default=0)
    avg_diff = np.mean([r["diff"] for r in precision_tests])
    all_pass = all(r["precision_pass"] for r in precision_tests)

    print(f"\n{'=' * 110}")
    print(f"PRECISION SUMMARY: {len(precision_tests)} tests, "
          f"all within tolerance={all_pass}, max_diff={max_diff:.6f}, avg_diff={avg_diff:.6f}")
    print(f"PERFORMANCE SUMMARY: {len(perf_tests)} tests")

    # Speedup table
    print(f"\n{'Category':30s} {'Go (ms)':>10s} {'Python (ms)':>12s} {'Speedup':>10s}")
    print("-" * 65)
    for r in perf_tests:
        if r["go_time_s"] and r["py_time_s"]:
            go_ms = r["go_time_s"] * 1000
            py_ms = r["py_time_s"] * 1000
            su = r["go_time_s"] / r["py_time_s"]
            print(f"{r['name']:30s} {go_ms:10.1f} {py_ms:12.1f} {su:10.2f}x")

    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()
