"""Visualize the APLS computation pipeline step by step.

For a pair of (GT, prediction) masks, generates a multi-panel figure showing:
  Row 1: Input graphs (GT, Pred) on masks
  Row 2: Densified graphs + sampled control points
  Row 3: Snapped control points + shortest path pairs
  Row 4: Penalty heatmap + final score

Usage:
    uv run python tests/visualize_apls.py
    uv run python tests/visualize_apls.py --dir tests/test_data_100 --n 5 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


# Ensure src/ is on sys.path
_src = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

from topo_metrics import TopoGraph, apls, mask_to_graph
from topo_metrics.apls import (
    _sample_control_points,
    _snap_control_points,
)


# ── Drawing helpers ─────────────────────────────────────────────────────────


def _draw_graph(
    ax,
    graph: TopoGraph,
    mask: np.ndarray | None = None,
    title: str = "",
    node_color: str = "cyan",
    edge_color: str = "cyan",
    edge_alpha: float = 0.6,
    node_size: int = 8,
    show_degrees: bool = False,
) -> None:
    """Draw a TopoGraph on an axes, optionally with background mask."""
    if mask is not None:
        ax.imshow(mask, cmap="gray", origin="upper", alpha=0.4)

    if graph.num_nodes == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "empty graph", transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray")
        return

    # Edges as line collection for performance
    if graph.num_edges > 0:
        segments = []
        for e0, e1 in graph.edges:
            # nodes are (x, y), imshow is (row, col) so swap
            p0 = graph.nodes[e0]
            p1 = graph.nodes[e1]
            segments.append([(p0[0], p0[1]), (p1[0], p1[1])])
        lc = LineCollection(segments, colors=edge_color, linewidths=0.8, alpha=edge_alpha)
        ax.add_collection(lc)

    # Nodes colored by degree
    if show_degrees:
        adj = graph._build_adjacency()
        degree = np.array([len(adj.get(i, [])) for i in range(graph.num_nodes)])
        colors = []
        for d in degree:
            if d == 1:
                colors.append("lime")
            elif d >= 3:
                colors.append("red")
            else:
                colors.append("dodgerblue")
    else:
        colors = node_color

    # nodes[:,0]=x, nodes[:,1]=y; imshow uses (row=y, col=x)
    ax.scatter(graph.nodes[:, 0], graph.nodes[:, 1], c=colors, s=node_size,
               edgecolors="black", linewidths=0.3, zorder=5)

    ax.set_xlim(-5, 517)
    ax.set_ylim(517, -5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)


def _draw_paths(
    ax,
    graph: TopoGraph,
    source_cps: list[int],
    target_cps: list[int],
    paths_src: dict[int, dict[int, float]],
    paths_tgt: dict[int, dict[int, float]],
    n_show: int = 8,
) -> None:
    """Draw sampled shortest path pairs with color-coded penalty."""
    _draw_graph(ax, graph, title="Shortest Path Pairs (sample)")

    if len(source_cps) < 2:
        return

    cmap = plt.cm.RdYlGn_r  # green=0 penalty, red=1
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(source_cps)):
        for j in range(i + 1, len(source_cps)):
            cp1, cp2 = source_cps[i], source_cps[j]
            t1, t2 = target_cps[i], target_cps[j]
            if t1 < 0 or t2 < 0:
                pairs.append((i, j, 1.0))
                continue
            d_src = paths_src.get(cp1, {}).get(cp2, -1.0)
            if d_src < 0 or d_src <= 100.0:
                continue
            d_tgt = paths_tgt.get(t1, {}).get(t2, -1.0)
            if d_tgt < 0:
                d_tgt = 0.0
            penalty = min(abs(d_src - d_tgt) / d_src, 1.0)
            pairs.append((i, j, penalty))

    # Show the n_show pairs with highest penalty (most interesting)
    pairs.sort(key=lambda x: x[2], reverse=True)
    for idx, (i, j, penalty) in enumerate(pairs[:n_show]):
        cp1, cp2 = source_cps[i], source_cps[j]
        color = cmap(penalty)
        p1 = graph.nodes[cp1]
        p2 = graph.nodes[cp2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color=color, linewidth=2.0,
                alpha=0.8, zorder=4)

    # Draw control points as stars
    for i, cp in enumerate(source_cps):
        t = target_cps[i]
        p = graph.nodes[cp]
        marker_color = "yellow" if t >= 0 else "red"
        ax.plot(p[0], p[1], "*", color=marker_color, markersize=10,
                markeredgecolor="black", markeredgewidth=0.5, zorder=10)


# ── Main visualization ──────────────────────────────────────────────────────


def visualize_apls(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    title: str = "",
    sample_interval: float = 50.0,
    densify_interval: float = 2.0,
    snapping_tol: float = 10.0,
    min_path_length: float = 100.0,
) -> plt.Figure:
    """Create a 3×3 panel figure showing the APLS pipeline."""
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.squeeze()
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze()

    # Build graphs
    gt_tg = mask_to_graph(gt_mask)
    pred_tg = mask_to_graph(pred_mask)

    empty_gt = gt_tg.num_nodes == 0
    empty_pred = pred_tg.num_nodes == 0

    # Compute APLS score
    if empty_gt or empty_pred:
        score = 0.0 if (empty_gt != empty_pred) else 1.0
    else:
        score = apls(pred_tg, gt_tg, sample_interval=sample_interval,
                     densify_interval=densify_interval)

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    if title:
        fig.suptitle(f"{title}  —  APLS = {score:.4f}", fontsize=14, fontweight="bold")

    # ── Row 0: Input graphs on masks ─────────────────────────────────────
    _draw_graph(axes[0, 0], gt_tg, gt_mask, title=f"GT Graph ({gt_tg.num_nodes}N, {gt_tg.num_edges}E)",
                node_color="lime", edge_color="lime", show_degrees=True, node_size=15)
    _draw_graph(axes[0, 1], pred_tg, pred_mask,
                title=f"Pred Graph ({pred_tg.num_nodes}N, {pred_tg.num_edges}E)",
                node_color="orange", edge_color="orange", show_degrees=True, node_size=15)

    # Overlay both
    ax = axes[0, 2]
    _draw_graph(ax, gt_tg, gt_mask, title="Overlay (GT=green, Pred=orange)",
                node_color="lime", edge_color="lime", edge_alpha=0.4, node_size=8)
    if not empty_pred:
        if pred_tg.num_edges > 0:
            segs = [[(pred_tg.nodes[e0, 0], pred_tg.nodes[e0, 1]),
                      (pred_tg.nodes[e1, 0], pred_tg.nodes[e1, 1])]
                     for e0, e1 in pred_tg.edges]
            lc = LineCollection(segs, colors="orange", linewidths=0.8, alpha=0.4)
            ax.add_collection(lc)
        ax.scatter(pred_tg.nodes[:, 0], pred_tg.nodes[:, 1], c="orange", s=8,
                   edgecolors="black", linewidths=0.2, zorder=5)

    if empty_gt or empty_pred:
        for i in range(1, 3):
            for j in range(3):
                axes[i, j].text(0.5, 0.5, "skipped\n(empty graph)", transform=axes[i, j].transAxes,
                                ha="center", va="center", fontsize=11, color="gray")
        fig.tight_layout()
        return fig

    # ── Row 1: Densification + control points ────────────────────────────
    gt_dense = gt_tg.densify(densify_interval)
    pred_dense = pred_tg.densify(densify_interval)

    _draw_graph(axes[1, 0], gt_dense, gt_mask,
                title=f"GT Densified ({gt_dense.num_nodes}N)",
                node_color="lime", edge_color="lime", node_size=3)
    _draw_graph(axes[1, 1], pred_dense, pred_mask,
                title=f"Pred Densified ({pred_dense.num_nodes}N)",
                node_color="orange", edge_color="orange", node_size=3)

    # Control point sampling (GT → Pred direction)
    gt_cp_indices = _sample_control_points(gt_dense, sample_interval)
    ax = axes[1, 2]
    _draw_graph(ax, gt_dense, gt_mask, title=f"GT Control Points ({len(gt_cp_indices)} CP)",
                node_color="gray", edge_color="gray", node_size=2, edge_alpha=0.3)
    if gt_cp_indices:
        cp_coords = gt_dense.nodes[gt_cp_indices]
        ax.scatter(cp_coords[:, 0], cp_coords[:, 1], c="red", s=50,
                   marker="*", edgecolors="black", linewidths=0.5, zorder=10)

    # ── Row 2: Snapping + path comparison ────────────────────────────────

    # Snap GT control points to Pred graph
    snapped = _snap_control_points(gt_cp_indices, gt_dense, pred_dense, snapping_tol)

    # Show snapped mapping
    ax = axes[2, 0]
    _draw_graph(ax, gt_dense, gt_mask, title="Snapping (GT→Pred)",
                node_color="gray", edge_color="gray", node_size=2, edge_alpha=0.2)
    if not empty_pred:
        if pred_dense.num_edges > 0:
            segs = [[(pred_dense.nodes[e0, 0], pred_dense.nodes[e0, 1]),
                      (pred_dense.nodes[e1, 0], pred_dense.nodes[e1, 1])]
                     for e0, e1 in pred_dense.edges]
            lc = LineCollection(segs, colors="orange", linewidths=0.5, alpha=0.2)
            ax.add_collection(lc)

    matched = 0
    unmatched = 0
    for i, cp_idx in enumerate(gt_cp_indices):
        src = gt_dense.nodes[cp_idx]
        if snapped[i] >= 0:
            tgt = pred_dense.nodes[snapped[i]]
            ax.plot([src[0], tgt[0]], [src[1], tgt[1]], "-", color="yellow",
                    linewidth=0.8, alpha=0.6, zorder=6)
            ax.plot(src[0], src[1], "*", color="lime", markersize=8,
                    markeredgecolor="black", linewidth=0.3, zorder=10)
            ax.plot(tgt[0], tgt[1], "o", color="orange", markersize=5,
                    markeredgecolor="black", linewidth=0.3, zorder=10)
            matched += 1
        else:
            ax.plot(src[0], src[1], "*", color="red", markersize=8,
                    markeredgecolor="black", linewidth=0.3, zorder=10)
            unmatched += 1

    axes[2, 0].set_title(f"Snapping: {matched} matched, {unmatched} unmatched")

    # Show shortest path pairs (GT side)
    ax = axes[2, 1]
    _draw_graph(ax, gt_dense, gt_mask, title="GT Path Pairs (sample)",
                node_color="gray", edge_color="gray", node_size=2, edge_alpha=0.2)

    # Show shortest path pairs (Pred side)
    ax = axes[2, 2]
    _draw_graph(ax, pred_dense, pred_mask, title="Pred Path Pairs (sample)",
                node_color="gray", edge_color="gray", node_size=2, edge_alpha=0.2)

    # Compute paths for visualization
    matched_src = []
    matched_tgt = []
    for i, cp_idx in enumerate(gt_cp_indices):
        if snapped[i] >= 0:
            matched_src.append(cp_idx)
            matched_tgt.append(int(snapped[i]))

    if len(matched_src) >= 2:
        # Pick up to 6 pairs to visualize
        gt_sp = {}
        for cp in matched_src[:30]:
            gt_sp[cp] = gt_dense.shortest_paths(cp, matched_src[:30])
        pred_sp = {}
        tgt_unique = list(dict.fromkeys(matched_tgt[:30]))
        for cp in tgt_unique:
            pred_sp[cp] = pred_dense.shortest_paths(cp, tgt_unique)

        # Draw path pairs
        pairs_drawn = 0
        cmap_penalty = plt.cm.RdYlGn_r
        for i in range(min(len(matched_src), 30)):
            for j in range(i + 1, min(len(matched_src), 30)):
                cp1, cp2 = matched_src[i], matched_src[j]
                t1, t2 = matched_tgt[i], matched_tgt[j]
                d_src = gt_sp.get(cp1, {}).get(cp2, -1.0)
                if d_src < 0 or d_src <= min_path_length:
                    continue
                d_tgt = pred_sp.get(t1, {}).get(t2, -1.0)
                if d_tgt < 0:
                    d_tgt = 0.0
                penalty = min(abs(d_src - d_tgt) / d_src, 1.0)
                color = cmap_penalty(penalty)

                # Draw on GT axes
                p1 = gt_dense.nodes[cp1]
                p2 = gt_dense.nodes[cp2]
                axes[2, 1].plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color=color,
                                linewidth=2.0, alpha=0.7, zorder=4)

                # Draw on Pred axes
                p1t = pred_dense.nodes[t1]
                p2t = pred_dense.nodes[t2]
                axes[2, 2].plot([p1t[0], p2t[0]], [p1t[1], p2t[1]], "-", color=color,
                                linewidth=2.0, alpha=0.7, zorder=4)

                pairs_drawn += 1
                if pairs_drawn >= 8:
                    break
            if pairs_drawn >= 8:
                break

        # Draw control points on path axes
        for cp in matched_src[:30]:
            p = gt_dense.nodes[cp]
            axes[2, 1].plot(p[0], p[1], "*", color="lime", markersize=6,
                            markeredgecolor="black", linewidth=0.3, zorder=10)
        for t in matched_tgt[:30]:
            p = pred_dense.nodes[t]
            axes[2, 2].plot(p[0], p[1], "*", color="orange", markersize=6,
                            markeredgecolor="black", linewidth=0.3, zorder=10)

    # Add legend for penalty colors
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[2, 1], fraction=0.046, pad=0.04, label="Penalty")

    fig.tight_layout()
    return fig


# ── Random corruption (reused from apls_random_corrupt.py) ─────────────────


def _bresenham_points(x0, y0, x1, y1):
    points = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((y0, x0))
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


def add_road(mask, rng, width=3):
    h, w = mask.shape
    x0, y0 = rng.integers(0, w), rng.integers(0, h)
    x1 = np.clip(x0 + rng.integers(-200, 200), 0, w - 1)
    y1 = np.clip(y0 + rng.integers(-200, 200), 0, h - 1)
    r = width // 2
    for row, col in _bresenham_points(x0, y0, x1, y1):
        mask[max(0, row - r):min(h, row + r + 1), max(0, col - r):min(w, col + r + 1)] = 1
    return mask


def cut_road(mask, rng, width=5):
    road_px = np.argwhere(mask == 1)
    if len(road_px) == 0:
        return mask
    h, w = mask.shape
    cy, cx = road_px[rng.integers(0, len(road_px))]
    length = rng.integers(10, 40)
    angle = rng.uniform(0, 2 * np.pi)
    dx, dy = int(length * np.cos(angle)), int(length * np.sin(angle))
    r = width // 2
    for row, col in _bresenham_points(cx, cy, cx + dx, cy + dy):
        mask[max(0, row - r):min(h, row + r + 1), max(0, col - r):min(w, col + r + 1)] = 0
    return mask


def corrupt_mask(mask, rng):
    n_ops = rng.integers(1, 4)
    ops = []
    result = mask.copy()
    for _ in range(n_ops):
        op = rng.choice(["add", "cut"])
        if op == "add":
            w = rng.integers(2, 5)
            result = add_road(result, rng, width=w)
            ops.append(f"add(w={w})")
        else:
            w = rng.integers(3, 8)
            result = cut_road(result, rng, width=w)
            ops.append(f"cut(w={w})")
    return result, "+".join(ops)


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize APLS computation pipeline")
    parser.add_argument("--dir", type=str, default=os.path.join(os.path.dirname(__file__), "test_data"),
                        help="Data directory")
    parser.add_argument("--n", type=int, default=3, help="Number of corrupted variants per image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "vis_apls"),
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    files = sorted(f for f in os.listdir(args.dir) if f.endswith(".npy"))

    print(f"Visualizing APLS pipeline for {len(files)} images × {args.n} variants")
    print(f"Output: {args.out}/")
    print()

    count = 0
    for f in files:
        arr = np.load(os.path.join(args.dir, f)).squeeze()
        mask = (arr > 0).astype(np.uint8)

        if np.sum(mask) == 0:
            continue

        # 1. Identical graph
        fig = visualize_apls(mask, mask, title=f"{f} — identical")
        path = os.path.join(args.out, f.replace(".npy", "_identical.png"))
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        count += 1
        print(f"  {f} identical → {path}")

        # 2. Corrupted variants
        for v in range(args.n):
            corrupted, desc = corrupt_mask(mask, rng)
            fig = visualize_apls(mask, corrupted, title=f"{f} — {desc}")
            path = os.path.join(args.out, f.replace(".npy", f"_corrupt_{v + 1}.png"))
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            count += 1
            print(f"  {f} corrupt_{v + 1} ({desc}) → {path}")

    print(f"\nDone. {count} figures saved to {args.out}/")


if __name__ == "__main__":
    main()
