"""Random topological corruption for APLS cross-verification.

Generates random variants of each test mask by adding roads (line segments)
or cutting roads (erasing rectangular patches on existing road pixels), then
runs Go vs Python APLS comparison.

Usage:
    uv run python tests/apls_random_corrupt.py           # default: 5 variants per image
    uv run python tests/apls_random_corrupt.py --n 10    # 10 variants per image
    uv run python tests/apls_random_corrupt.py --seed 42 # fixed seed for reproducibility
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# Ensure src/ is on sys.path
_src = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

from topo_metrics import TopoGraph, apls, mask_to_graph

# ── Configuration ───────────────────────────────────────────────────────────

TEST_DIR = os.path.join(os.path.dirname(__file__), "test_data")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "apls_random_corrupt_report.txt")

GO_PIXEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "external", "Sat2Graph", "metrics", "apls_pixel")
)


# ── Go APLS interface ───────────────────────────────────────────────────────


def run_go_apls(gt_json_path: str, prop_json_path: str) -> float | None:
    """Run Go pixel-coordinate APLS binary and parse the result."""
    binary = os.path.join(GO_PIXEL_DIR, "apls_pixel")
    if not os.path.exists(binary):
        subprocess.run(["go", "build", "-o", "apls_pixel", "."], cwd=GO_PIXEL_DIR, check=True)
    result = subprocess.run(
        [binary, gt_json_path, prop_json_path, "/dev/null"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    for line in result.stdout.strip().split("\n"):
        if "apls:" in line:
            try:
                return float(line.split("apls:")[1].strip())
            except (IndexError, ValueError):
                pass
    return None


def topograph_to_go_json(tg: TopoGraph) -> list:
    """Convert TopoGraph to pixel-coordinate JSON format."""
    nodes = [[float(x), float(y)] for x, y in tg.nodes]
    edges = [[int(e0), int(e1)] for e0, e1 in tg.edges]
    return [nodes, edges]


# ── Random corruption primitives ────────────────────────────────────────────


def _bresenham_points(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Generate pixel coordinates along a line using Bresenham's algorithm."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((y0, x0))  # (row, col)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


def add_road(mask: np.ndarray, rng: np.random.Generator, width: int = 3) -> np.ndarray:
    """Add a random road segment (line with given width)."""
    h, w = mask.shape
    x0, y0 = rng.integers(0, w), rng.integers(0, h)
    x1 = x0 + rng.integers(-200, 200)
    y1 = y0 + rng.integers(-200, 200)
    x1, y1 = np.clip(x1, 0, w - 1), np.clip(y1, 0, h - 1)

    points = _bresenham_points(x0, y0, x1, y1)
    r = width // 2
    for row, col in points:
        r0, r1 = max(0, row - r), min(h, row + r + 1)
        c0, c1 = max(0, col - r), min(w, col + r + 1)
        mask[r0:r1, c0:c1] = 1
    return mask


def cut_road(mask: np.ndarray, rng: np.random.Generator, width: int = 5) -> np.ndarray:
    """Cut a random patch of existing road pixels."""
    road_pixels = np.argwhere(mask == 1)
    if len(road_pixels) == 0:
        return mask
    h, w = mask.shape

    # Pick a random road pixel as center
    idx = rng.integers(0, len(road_pixels))
    cy, cx = road_pixels[idx]

    # Cut a small rectangular area around it
    length = rng.integers(10, 40)
    angle = rng.uniform(0, 2 * np.pi)
    dx = int(length * np.cos(angle))
    dy = int(length * np.sin(angle))

    x0, x1 = cx, cx + dx
    y0, y1 = cy, cy + dy
    points = _bresenham_points(x0, y0, x1, y1)
    r = width // 2
    for row, col in points:
        r0, r1 = max(0, row - r), min(h, row + r + 1)
        c0, c1 = max(0, col - r), min(w, col + r + 1)
        mask[r0:r1, c0:c1] = 0
    return mask


def corrupt_mask(mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Apply random corruption: 1-3 operations of add/cut.

    Returns (corrupted_mask, description_string).
    """
    n_ops = rng.integers(1, 4)
    ops = []
    result = mask.copy()

    for _ in range(n_ops):
        op = rng.choice(["add", "cut"])
        if op == "add":
            width = rng.integers(2, 5)
            result = add_road(result, rng, width=width)
            ops.append(f"add(w={width})")
        else:
            width = rng.integers(3, 8)
            result = cut_road(result, rng, width=width)
            ops.append(f"cut(w={width})")

    desc = "+".join(ops)
    return result, desc


# ── Main pipeline ───────────────────────────────────────────────────────────


def generate_variants(
    n_variants: int, seed: int, data_dir: str = TEST_DIR
) -> dict[str, list[tuple[str, np.ndarray, str]]]:
    """Generate random corrupted variants for all test masks."""
    rng = np.random.default_rng(seed)
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npy"))
    all_edits: dict[str, list[tuple[str, np.ndarray, str]]] = {}

    print(f"Generating {n_variants} random variants per image (seed={seed})...")
    print()

    for f in files:
        arr = np.load(os.path.join(data_dir, f)).squeeze()
        mask = (arr > 0).astype(np.uint8)
        variants: list[tuple[str, np.ndarray, str]] = [("original", mask, "identity")]

        for v in range(n_variants):
            corrupted, desc = corrupt_mask(mask, rng)
            label = f"corrupt_{v+1}"
            variants.append((label, corrupted, desc))
            road_diff = int(np.sum(corrupted != mask))
            print(f"  {f} {label}: {desc} (changed {road_diff} px)")

        all_edits[f] = variants

    return all_edits


def compute_apls_comparison(
    all_edits: dict[str, list[tuple[str, np.ndarray, str]]],
) -> list[dict]:
    """Compute APLS (Go vs Python) for all corrupted masks."""
    rows = []

    for f, variants in all_edits.items():
        print(f"\nProcessing {f} ({len(variants)} variants)...")

        gt_label, gt_mask, gt_desc = variants[0]

        for pred_label, pred_mask, pred_desc in variants:
            gt_tg = mask_to_graph(gt_mask)
            pred_tg = mask_to_graph(pred_mask)

            if gt_tg.num_nodes == 0 or pred_tg.num_nodes == 0:
                rows.append({
                    "file": f, "desc": pred_desc,
                    "gt": gt_label, "pred": pred_label,
                    "gt_nodes": gt_tg.num_nodes, "pred_nodes": pred_tg.num_nodes,
                    "py_apls": 0.0 if gt_tg.num_nodes == 0 else None,
                    "go_apls": None, "diff": None,
                })
                continue

            py_score = apls(gt_tg, pred_tg, sample_interval=50.0, densify_interval=2.0)

            gt_json = topograph_to_go_json(gt_tg)
            pred_json = topograph_to_go_json(pred_tg)

            go_score = None
            with tempfile.TemporaryDirectory() as tmpdir:
                gt_path = os.path.join(tmpdir, "gt.json")
                prop_path = os.path.join(tmpdir, "prop.json")
                with open(gt_path, "w") as jf:
                    json.dump(gt_json, jf)
                with open(prop_path, "w") as jf:
                    json.dump(pred_json, jf)
                go_score = run_go_apls(gt_path, prop_path)

            diff = abs(py_score - go_score) if go_score is not None else None

            rows.append({
                "file": f, "desc": pred_desc,
                "gt": gt_label, "pred": pred_label,
                "gt_nodes": gt_tg.num_nodes, "pred_nodes": pred_tg.num_nodes,
                "py_apls": py_score,
                "go_apls": go_score,
                "diff": diff,
            })

            go_str = f"{go_score:.6f}" if go_score is not None else "N/A"
            diff_str = f"{diff:.6f}" if diff is not None else "N/A"
            print(f"  {pred_label} ({pred_desc}): py={py_score:.6f}, go={go_str}, diff={diff_str}")

    return rows


def print_report(rows: list[dict], report_path: str = REPORT_PATH) -> None:
    """Print and save APLS comparison report."""
    print("\n" + "=" * 115)
    print("APLS Random Corruption Report: Go vs Python (pixel-coordinate aligned)")
    print("=" * 115)
    header = (f"{'File':20s} {'Desc':20s} {'GT_N':>5s} {'Pred_N':>6s} "
              f"{'Py APLS':>8s} {'Go APLS':>8s} {'Diff':>8s}")
    print(header)
    print("-" * 115)

    lines = [header, "-" * 115]
    for r in rows:
        py_str = f"{r['py_apls']:.6f}" if r['py_apls'] is not None else "EMPTY"
        go_str = f"{r['go_apls']:.6f}" if r['go_apls'] is not None else "N/A"
        diff_str = f"{r['diff']:.6f}" if r['diff'] is not None else "N/A"
        line = (f"{r['file']:20s} {r['desc']:20s} "
                f"{r['gt_nodes']:5d} {r['pred_nodes']:6d} "
                f"{py_str:>8s} {go_str:>8s} {diff_str:>8s}")
        print(line)
        lines.append(line)

    # Summary
    diffs = [r['diff'] for r in rows if r['diff'] is not None]
    if diffs:
        summary_lines = [
            f"\nSummary ({len(diffs)} non-trivial comparisons):",
            f"  mean_diff  = {np.mean(diffs):.6f}",
            f"  median_diff= {np.median(diffs):.6f}",
            f"  max_diff   = {np.max(diffs):.6f}",
            f"  min_diff   = {np.min(diffs):.6f}",
            f"  p95_diff   = {np.percentile(diffs, 95):.6f}",
            f"  diff < 0.01: {sum(1 for d in diffs if d < 0.01)}/{len(diffs)}",
            f"  diff < 0.05: {sum(1 for d in diffs if d < 0.05)}/{len(diffs)}",
            f"  diff > 0.10: {sum(1 for d in diffs if d > 0.10)}/{len(diffs)}",
        ]
        for sl in summary_lines:
            print(sl)
            lines.append(sl)

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Random topological corruption APLS test")
    parser.add_argument("--dir", type=str, default=TEST_DIR, help="Data directory (default: test_data/)")
    parser.add_argument("--out", type=str, default=REPORT_PATH, help="Report output path")
    parser.add_argument("--n", type=int, default=5, help="Variants per image (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    data_dir = args.dir
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npy"))
    n_images = len(files)

    print("=" * 60)
    print("Random Topological Corruption — APLS Cross-Verification")
    print("=" * 60)
    print(f"Images:   {n_images} masks from {data_dir}")
    print(f"Variants: {args.n} per image = {n_images * args.n} comparisons (+ {n_images} identity)")
    print(f"Seed:     {args.seed}")
    print()

    all_edits = generate_variants(args.n, args.seed, data_dir)

    print("\n" + "=" * 60)
    print("Computing APLS (Go vs Python)")
    print("=" * 60)
    rows = compute_apls_comparison(all_edits)
    print_report(rows, args.out)


if __name__ == "__main__":
    main()
