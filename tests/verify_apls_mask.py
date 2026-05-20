"""Cross-verify Go vs Python APLS on real mask data (skeletonized via skan)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile

import numpy as np
from skimage.morphology import skeletonize
from skan import Skeleton, summarize

from topo_metrics import TopoGraph, apls

LAT_TOP_LEFT = 41.0
LON_TOP_LEFT = -71.0
GO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "external", "Sat2Graph", "metrics", "apls")
)


def mask_to_topograph(
    mask: np.ndarray, spacing: tuple[float, float] = (1.0, 1.0)
) -> TopoGraph:
    """Convert binary mask to TopoGraph via skeletonization + skan."""
    if mask.ndim == 3:
        mask = mask.squeeze()
    skeleton = skeletonize(mask > 0)
    if np.sum(skeleton) == 0:
        return TopoGraph(nodes=np.empty((0, 2)), edges=np.empty((0, 2)), spacing=spacing)

    skel = Skeleton(skeleton, spacing=spacing)
    branch_data = summarize(skel, separator="_")

    if len(branch_data) == 0:
        return TopoGraph(nodes=np.empty((0, 2)), edges=np.empty((0, 2)), spacing=spacing)

    src_coords = branch_data[["image_coord_src_0", "image_coord_src_1"]].values
    dst_coords = branch_data[["image_coord_dst_0", "image_coord_dst_1"]].values

    all_coords = np.vstack([src_coords, dst_coords])
    _, unique_idx, inverse = np.unique(
        np.round(all_coords, decimals=1), axis=0, return_index=True, return_inverse=True
    )
    nodes = all_coords[unique_idx]
    n_src = len(src_coords)
    edges = np.column_stack([inverse[:n_src], inverse[n_src:]])

    return TopoGraph(nodes=nodes, edges=edges, spacing=spacing)


def xy_to_latlon(x: float, y: float) -> tuple[float, float]:
    lat = LAT_TOP_LEFT - y / 111111.0
    lon = LON_TOP_LEFT + x / (111111.0 * math.cos(math.radians(LAT_TOP_LEFT)))
    return lat, lon


def make_go_json(nodes: np.ndarray, edges: np.ndarray) -> list:
    go_nodes = [[*xy_to_latlon(x, y)] for x, y in nodes]
    go_edges = [[int(e0), int(e1)] for e0, e1 in edges]
    return [go_nodes, go_edges]


def run_go_apls(gt_json_path: str, prop_json_path: str) -> float | None:
    result = subprocess.run(
        ["go", "run", "main.go", gt_json_path, prop_json_path, "/dev/null"],
        capture_output=True,
        text=True,
        cwd=GO_DIR,
        timeout=60,
    )
    for line in result.stdout.strip().split("\n"):
        if "apls:" in line:
            try:
                return float(line.split("apls:")[1].strip())
            except (IndexError, ValueError):
                pass
    return None


def main() -> None:
    test_dir = os.path.join(os.path.dirname(__file__), "test_data")
    files = sorted(f for f in os.listdir(test_dir) if f.endswith(".npy"))

    print(f"{'File':25s} {'Nodes':>6s} {'Edges':>6s} {'Road%':>7s} {'Py APLS':>8s} {'Go APLS':>8s} {'Diff':>8s}")
    print("-" * 80)

    for f in files:
        arr = np.load(os.path.join(test_dir, f))
        road_ratio = float(np.sum(arr > 0)) / arr.size

        g = mask_to_topograph(arr)

        if g.num_nodes == 0:
            print(f"{f:25s} {0:6d} {0:6d} {road_ratio:7.4f} {'EMPTY':>8s}")
            continue

        # Python APLS (identical graph → should be ~1.0)
        py_score = apls(g, g, sample_interval=50.0, densify_interval=2.0)

        # Go APLS
        go_data = make_go_json(g.nodes, g.edges)
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = os.path.join(tmpdir, "gt.json")
            prop_path = os.path.join(tmpdir, "prop.json")
            with open(gt_path, "w") as jf:
                json.dump(go_data, jf)
            with open(prop_path, "w") as jf:
                json.dump(go_data, jf)
            go_score = run_go_apls(gt_path, prop_path)

        diff = abs(py_score - go_score) if go_score is not None else None
        diff_str = f"{diff:.6f}" if diff is not None else "N/A"
        go_str = f"{go_score:.6f}" if go_score is not None else "N/A"

        print(f"{f:25s} {g.num_nodes:6d} {g.num_edges:6d} {road_ratio:7.4f} {py_score:8.6f} {go_str:>8s} {diff_str:>8s}")


if __name__ == "__main__":
    main()
