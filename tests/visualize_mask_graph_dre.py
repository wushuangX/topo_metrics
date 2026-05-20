"""Visualize mask_to_graph pipeline: mask → distance transform → keypoints → graph."""

from __future__ import annotations

import importlib.util
import os


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.feature import corner_harris, corner_peaks, peak_local_max
from skimage.measure import label
from skimage.morphology import skeletonize

# Import mask_to_graph directly from file to avoid __init__.py pulling in torch
_spec = importlib.util.spec_from_file_location(
    "graph",
    "external/DeepRoadExtraction/src/roaddeeplab/utils/graph.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
mask_to_graph = _mod.mask_to_graph
extract_edge_peaks = _mod._extract_edge_peaks
_sknw_mark = _mod._sknw_mark
_sknw_neighbors = _mod._sknw_neighbors
_sknw_parse_with_keypoints = _mod._sknw_parse_with_keypoints


def _transform_positions(node_positions: dict) -> dict:
    """Transform node_positions from (col, row) to (row, col) for imshow coords."""
    return {nid: (c, r) for nid, (r, c) in node_positions.items()}


def _call_mask_to_graph(mask: np.ndarray, method: str, fix_borders: bool) -> tuple[nx.Graph, dict]:
    """Call mask_to_graph with the right params for each method."""
    if method in ("keypoint", "hybrid"):
        return mask_to_graph(
            mask, min_distance=8, k=4, r=200, auto_relax=True,
            add_edge_nodes=True, edge_border_width=1,
            method=method, fix_borders=fix_borders,
        )
    return mask_to_graph(mask, method="sknw", fix_borders=fix_borders)


def _draw_graph_on_ax(
    ax, mask: np.ndarray, G: nx.Graph, node_positions: dict,
    num_mask_components: int, empty: bool, title_suffix: str = "",
) -> None:
    """Draw mask background + graph overlay + CC status text on a matplotlib Axes."""
    h, w = mask.shape
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)
    num_graph_components = nx.number_connected_components(G) if num_nodes > 0 else 0
    cc_match = num_mask_components == num_graph_components

    ax.imshow(mask, cmap="gray", origin="upper")

    if num_nodes > 0:
        pos = _transform_positions(node_positions)
        degrees = dict(G.degree())

        for u, v in G.edges:
            y1, x1 = pos[u]
            y2, x2 = pos[v]
            ax.plot([x1, x2], [y1, y2], "c-", linewidth=1.0, alpha=0.7)

        for nid, (row_p, col_p) in pos.items():
            d = degrees[nid]
            if d == 1:
                color = "lime"
            elif d >= 3:
                color = "red"
            else:
                color = "dodgerblue"
            ax.plot(col_p, row_p, "o", color=color, markersize=4,
                    markeredgecolor="black", markeredgewidth=0.5)
    else:
        ax.text(w / 2, h / 2, "No graph\n(empty mask)", ha="center",
                va="center", fontsize=14, color="gray")

    if title_suffix:
        ax.set_title(
            f"{title_suffix}\n"
            f"nodes={num_nodes}, edges={num_edges}\n"
            f"mask_cc={num_mask_components}, graph_cc={num_graph_components}, "
            f"match={cc_match}",
            fontsize=11,
        )

    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    status_text = (
        f"CC: mask={num_mask_components}, graph={num_graph_components}, "
        f"match={cc_match}"
    )
    ax.text(
        0.01, 0.01, status_text,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=8,
        color="white" if not empty else "black",
        bbox={"facecolor": "black" if not empty else "white",
              "alpha": 0.65, "pad": 2},
    )


def visualize_one(mask: np.ndarray, filename: str) -> plt.Figure:
    """Create a 2×3 figure showing mask_to_graph pipeline steps."""
    if mask.ndim == 3:
        mask = mask.squeeze()

    h, w = mask.shape
    mask_labels, num_mask_components = label(mask, connectivity=2, return_num=True)
    empty = num_mask_components == 0

    # --- Step 1: Skeleton ---
    skeleton = skeletonize(mask) if not empty else np.zeros_like(mask)

    # --- Step 2: Distance transform on skeleton ---
    dist_transform = distance_transform_edt(mask) * skeleton if not empty else np.zeros_like(mask)

    # --- Step 3: Keypoint extraction (same params as mask_to_graph) ---
    min_distance = 8
    if not empty:
        peak_coords = peak_local_max(
            dist_transform, min_distance=min_distance, exclude_border=False,
        )
        corner_coords = corner_peaks(
            corner_harris(skeleton, k=0.04),
            min_distance=5, threshold_rel=0.05, exclude_border=False,
        )
        if len(corner_coords) > 0:
            corner_keep = (
                (corner_coords[:, 0] >= 5)
                & (corner_coords[:, 0] < h - 5)
                & (corner_coords[:, 1] >= 5)
                & (corner_coords[:, 1] < w - 5)
            )
            corner_coords = corner_coords[corner_keep]
        edge_peak_coords = extract_edge_peaks(mask, border_width=1, min_distance=10)
    else:
        peak_coords = np.empty((0, 2), dtype=np.int64)
        corner_coords = np.empty((0, 2), dtype=np.int64)
        edge_peak_coords = np.empty((0, 2), dtype=np.int64)
    # --- Step 8+9: Build graph with k=4, auto_relax + edge peak nodes ---
    G, node_positions = mask_to_graph(
        mask, min_distance=min_distance, k=4, r=200, auto_relax=True,
        add_edge_nodes=True, edge_border_width=1,
    )
    # Transform: (col, row) → (row, col) for plotting
    pos = _transform_positions(node_positions)
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)
    num_graph_components = nx.number_connected_components(G) if num_nodes > 0 else 0

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle(
        f"{filename}   (nodes={num_nodes}, edges={num_edges}, "
        f"mask_cc={num_mask_components}, graph_cc={num_graph_components}, "
        f"match={num_mask_components == num_graph_components})",
        fontsize=14, fontweight="bold",
    )

    # (0,0) Original mask
    ax = axes[0, 0]
    ax.imshow(mask, cmap="gray", origin="upper")
    ax.set_title("1. Road Mask")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    # (0,1) Skeleton
    ax = axes[0, 1]
    ax.imshow(mask, cmap="gray", origin="upper", alpha=0.3)
    ax.imshow(skeleton, cmap="hot", origin="upper", alpha=0.9)
    ax.set_title(f"1+1. Skeleton (px={int(np.sum(skeleton))})")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    # (0,2) Distance transform on skeleton
    ax = axes[0, 2]
    ax.imshow(mask, cmap="gray", origin="upper", alpha=0.3)
    if not empty:
        dt_show = np.ma.masked_where(dist_transform == 0, dist_transform)
        ax.imshow(dt_show, cmap="inferno", origin="upper")
    ax.set_title(
        f"2. Distance Transform on Skeleton (max={dist_transform.max():.1f})"
    )
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    # (1,0) Keypoints overlay on mask
    ax = axes[1, 0]
    ax.imshow(mask, cmap="gray", origin="upper")
    if len(peak_coords) > 0:
        ax.plot(peak_coords[:, 1], peak_coords[:, 0], "o",
                color="red", markersize=4, markeredgecolor="white",
                markeredgewidth=0.5, label=f"peak ({len(peak_coords)})")
    if len(corner_coords) > 0:
        ax.plot(corner_coords[:, 1], corner_coords[:, 0], "s",
                color="cyan", markersize=3, markeredgecolor="white",
                markeredgewidth=0.5, label=f"corner ({len(corner_coords)})")
    if len(edge_peak_coords) > 0:
        ax.plot(edge_peak_coords[:, 1], edge_peak_coords[:, 0], "^",
                color="yellow", markersize=4, markeredgecolor="black",
                markeredgewidth=0.5, label=f"edge_peak ({len(edge_peak_coords)})")
    ax.set_title("3. Keypoints (peak=●, corner=■, edge_peak=▲, peak min_distance=8, corner_thr=0.05, drop_border_corner=5px)")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", fontsize=8)

    # (1,1) Graph structure
    ax = axes[1, 1]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title("8+9. Graph (k=4, r=200, auto_relax, min_distance=8, corner_thr=0.05, edge_nodes=on, drop_border_corner=5px)")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    if num_nodes > 0:
        degrees = dict(G.degree())

        for u, v in G.edges:
            y1, x1 = pos[u]
            y2, x2 = pos[v]
            ax.plot([x1, x2], [y1, y2], "c-", linewidth=1.0, alpha=0.7)

        for nid, (row, col) in pos.items():
            d = degrees[nid]
            if d == 1:
                color = "lime"
            elif d >= 3:
                color = "red"
            else:
                color = "dodgerblue"
            ax.plot(col, row, "o", color=color, markersize=4,
                    markeredgecolor="black", markeredgewidth=0.5)

        ax.legend(
            handles=[
                plt.Line2D([], [], marker="o", color="w",
                           markerfacecolor="lime", markeredgecolor="black",
                           markersize=8, label="endpoint (deg=1)"),
                plt.Line2D([], [], marker="o", color="w",
                           markerfacecolor="dodgerblue", markeredgecolor="black",
                           markersize=8, label="path (deg=2)"),
                plt.Line2D([], [], marker="o", color="w",
                           markerfacecolor="red", markeredgecolor="black",
                           markersize=8, label="junction (deg≥3)"),
            ],
            loc="lower right", fontsize=8,
        )
    else:
        ax.text(w / 2, h / 2, "No graph\n(empty mask)", ha="center",
                va="center", fontsize=14, color="gray")

    cc_match = num_mask_components == num_graph_components
    status_text = (
        f"CC check: mask={num_mask_components}, graph={num_graph_components}, "
        f"match={cc_match}"
    )
    # (1,2) Overlay: mask + keypoints + graph
    ax = axes[1, 2]
    ax.imshow(mask, cmap="gray", origin="upper", alpha=0.5)
    if not empty:
        ax.imshow(skeleton, cmap="autumn", origin="upper", alpha=0.4)
    if len(peak_coords) > 0:
        ax.plot(peak_coords[:, 1], peak_coords[:, 0], "o",
                color="red", markersize=3, markeredgecolor="white",
                markeredgewidth=0.3)
    if len(corner_coords) > 0:
        ax.plot(corner_coords[:, 1], corner_coords[:, 0], "s",
                color="cyan", markersize=2, markeredgecolor="white",
                markeredgewidth=0.3)
    if len(edge_peak_coords) > 0:
        ax.plot(edge_peak_coords[:, 1], edge_peak_coords[:, 0], "^",
                color="yellow", markersize=3, markeredgecolor="black",
                markeredgewidth=0.3)
    if num_nodes > 0:
        for u, v in G.edges:
            y1, x1 = pos[u]
            y2, x2 = pos[v]
            ax.plot([x1, x2], [y1, y2], "c-", linewidth=1.0, alpha=0.6)
        for nid, (row, col) in pos.items():
            d = degrees[nid]
            color = "lime" if d == 1 else ("red" if d >= 3 else "dodgerblue")
            ax.plot(col, row, "o", color=color, markersize=3,
                    markeredgecolor="black", markeredgewidth=0.3)
    ax.set_title("Overlay (mask + skeleton + keypoints + graph)")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.text(
        0.01,
        0.01,
        status_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="white" if not empty else "black",
        bbox={"facecolor": "black" if not empty else "white", "alpha": 0.65, "pad": 3},
    )

    for row in axes:
        for ax in row:
            ax.tick_params(labelsize=7)

    fig.tight_layout()
    return fig


def visualize_one_hybrid(mask: np.ndarray, filename: str) -> plt.Figure:
    """Create a 2x3 figure showing hybrid method pipeline steps."""
    if mask.ndim == 3:
        mask = mask.squeeze()

    h, w = mask.shape
    _, num_mask_components = label(mask, connectivity=2, return_num=True)
    empty = num_mask_components == 0

    min_distance = 8

    # --- Step 1: Skeleton ---
    skeleton = skeletonize(mask) if not empty else np.zeros_like(mask)

    # --- Step 2: Distance transform ---
    dist_transform = distance_transform_edt(mask) * skeleton if not empty else np.zeros_like(mask)

    # --- Step 3: Keypoint extraction (same params as mask_to_graph) ---
    if not empty:
        peak_coords = peak_local_max(
            dist_transform, min_distance=min_distance, exclude_border=False,
        )
        corner_coords = corner_peaks(
            corner_harris(skeleton, k=0.04),
            min_distance=5, threshold_rel=0.05, exclude_border=False,
        )
        if len(corner_coords) > 0:
            corner_keep = (
                (corner_coords[:, 0] >= 5)
                & (corner_coords[:, 0] < h - 5)
                & (corner_coords[:, 1] >= 5)
                & (corner_coords[:, 1] < w - 5)
            )
            corner_coords = corner_coords[corner_keep]
        edge_peak_coords = extract_edge_peaks(mask, border_width=1, min_distance=10)
    else:
        peak_coords = np.empty((0, 2), dtype=np.int64)
        corner_coords = np.empty((0, 2), dtype=np.int64)
        edge_peak_coords = np.empty((0, 2), dtype=np.int64)

    # --- Step 4: Build hybrid graph ---
    G, node_positions = mask_to_graph(
        mask, min_distance=min_distance, k=4, r=200, auto_relax=True,
        add_edge_nodes=True, edge_border_width=1,
        method="hybrid",
    )
    pos = _transform_positions(node_positions)
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)
    num_graph_components = nx.number_connected_components(G) if num_nodes > 0 else 0

    # --- Step 5: Skeleton pixel classification (mark) ---
    # Reproduce the mark step for visualization
    ske = skeleton.astype(np.uint8)
    buf = np.zeros((ske.shape[0] + 2, ske.shape[1] + 2), dtype=np.uint8)
    buf[1:-1, 1:-1] = ske
    mark_img = buf.astype(np.uint16).copy()
    nbs = _sknw_neighbors(mark_img.shape)
    _sknw_mark(mark_img, nbs)
    # mark_img values: 0=background, 1=edge, 2=node
    # Crop back to original size for display
    mark_display = mark_img[1:-1, 1:-1]

    # Classify which nodes are keypoint vs natural
    keypoint_nodes = set()
    natural_nodes = set()
    if num_nodes > 0:
        for nid in G.nodes:
            src = G.nodes[nid].get("source", "unknown")
            if src in ("keypoint", "both"):
                keypoint_nodes.add(nid)
            if src in ("natural", "both"):
                natural_nodes.add(nid)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle(
        f"{filename} [HYBRID]  (nodes={num_nodes}, edges={num_edges}, "
        f"mask_cc={num_mask_components}, graph_cc={num_graph_components}, "
        f"match={num_mask_components == num_graph_components})",
        fontsize=14, fontweight="bold",
    )

    # (0,0) Original mask
    ax = axes[0, 0]
    ax.imshow(mask, cmap="gray", origin="upper")
    ax.set_title("1. Road Mask")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    # (0,1) Skeleton pixel classification
    ax = axes[0, 1]
    ax.imshow(mask, cmap="gray", origin="upper", alpha=0.3)
    # Create RGB image for classification
    class_rgb = np.zeros((*mark_display.shape, 3), dtype=np.float64)
    # Edge pixels = blue
    class_rgb[mark_display == 1, 2] = 1.0  # blue channel
    # Node pixels = red
    class_rgb[mark_display == 2, 0] = 1.0  # red channel
    ax.imshow(class_rgb, origin="upper", alpha=0.8)
    n_edge = int(np.sum(mark_display == 1))
    n_node = int(np.sum(mark_display == 2))
    ax.set_title(f"2. Skeleton Classification (edge={n_edge}, node={n_node})")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="s", color="w", markerfacecolor="blue",
                       markersize=10, label=f"edge pixel ({n_edge})"),
            plt.Line2D([], [], marker="s", color="w", markerfacecolor="red",
                       markersize=10, label=f"node pixel ({n_node})"),
        ],
        loc="lower right", fontsize=8,
    )

    # (0,2) Keypoints overlay
    ax = axes[0, 2]
    ax.imshow(mask, cmap="gray", origin="upper")
    if len(peak_coords) > 0:
        ax.plot(peak_coords[:, 1], peak_coords[:, 0], "o",
                color="red", markersize=4, markeredgecolor="white",
                markeredgewidth=0.5, label=f"peak ({len(peak_coords)})")
    if len(corner_coords) > 0:
        ax.plot(corner_coords[:, 1], corner_coords[:, 0], "s",
                color="cyan", markersize=3, markeredgecolor="white",
                markeredgewidth=0.5, label=f"corner ({len(corner_coords)})")
    if len(edge_peak_coords) > 0:
        ax.plot(edge_peak_coords[:, 1], edge_peak_coords[:, 0], "^",
                color="yellow", markersize=4, markeredgecolor="black",
                markeredgewidth=0.5, label=f"edge_peak ({len(edge_peak_coords)})")
    # Also overlay natural skeleton nodes as small dots
    node_pixel_coords = np.argwhere(mark_display == 2)
    if len(node_pixel_coords) > 0:
        ax.plot(node_pixel_coords[:, 1], node_pixel_coords[:, 0], "x",
                color="magenta", markersize=2, alpha=0.5,
                label=f"skeleton node px ({len(node_pixel_coords)})")
    ax.set_title("3. Keypoints + Skeleton Nodes")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", fontsize=7)

    # (1,0) Graph nodes colored by source
    ax = axes[1, 0]
    ax.imshow(mask, cmap="gray", origin="upper", alpha=0.5)
    if num_nodes > 0:
        for u, v in G.edges:
            y1, x1 = pos[u]
            y2, x2 = pos[v]
            ax.plot([x1, x2], [y1, y2], "c-", linewidth=1.0, alpha=0.5)
        for nid, (row_p, col_p) in pos.items():
            src = G.nodes[nid].get("source", "unknown")
            if src in ("keypoint", "both"):
                color = "red"
            elif src == "natural":
                color = "magenta"
            else:
                color = "dodgerblue"
            ax.plot(col_p, row_p, "o", color=color, markersize=4,
                    markeredgecolor="black", markeredgewidth=0.5)
    ax.set_title(f"4. Graph Nodes by Source (keypoint=red, natural=magenta)")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", color="w", markerfacecolor="red",
                       markeredgecolor="black", markersize=8,
                       label=f"keypoint/both ({len(keypoint_nodes)})"),
            plt.Line2D([], [], marker="o", color="w", markerfacecolor="magenta",
                       markeredgecolor="black", markersize=8,
                       label=f"natural ({len(natural_nodes)})"),
        ],
        loc="lower right", fontsize=8,
    )

    # (1,1) Graph structure
    ax = axes[1, 1]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title("5. Graph (hybrid: keypoints + skeleton-traced edges)")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    if num_nodes > 0:
        degrees = dict(G.degree())

        for u, v in G.edges:
            y1, x1 = pos[u]
            y2, x2 = pos[v]
            ax.plot([x1, x2], [y1, y2], "c-", linewidth=1.0, alpha=0.7)

        for nid, (row, col) in pos.items():
            d = degrees[nid]
            if d == 1:
                color = "lime"
            elif d >= 3:
                color = "red"
            else:
                color = "dodgerblue"
            ax.plot(col, row, "o", color=color, markersize=4,
                    markeredgecolor="black", markeredgewidth=0.5)

        ax.legend(
            handles=[
                plt.Line2D([], [], marker="o", color="w",
                           markerfacecolor="lime", markeredgecolor="black",
                           markersize=8, label="endpoint (deg=1)"),
                plt.Line2D([], [], marker="o", color="w",
                           markerfacecolor="dodgerblue", markeredgecolor="black",
                           markersize=8, label="path (deg=2)"),
                plt.Line2D([], [], marker="o", color="w",
                           markerfacecolor="red", markeredgecolor="black",
                           markersize=8, label="junction (deg>=3)"),
            ],
            loc="lower right", fontsize=8,
        )
    else:
        ax.text(w / 2, h / 2, "No graph\n(empty mask)", ha="center",
                va="center", fontsize=14, color="gray")

    # (1,2) Overlay: mask + skeleton + graph
    ax = axes[1, 2]
    ax.imshow(mask, cmap="gray", origin="upper", alpha=0.5)
    if not empty:
        ax.imshow(skeleton, cmap="autumn", origin="upper", alpha=0.4)
    if num_nodes > 0:
        for u, v in G.edges:
            y1, x1 = pos[u]
            y2, x2 = pos[v]
            ax.plot([x1, x2], [y1, y2], "c-", linewidth=1.0, alpha=0.6)
        for nid, (row, col) in pos.items():
            d = degrees[nid]
            color = "lime" if d == 1 else ("red" if d >= 3 else "dodgerblue")
            ax.plot(col, row, "o", color=color, markersize=3,
                    markeredgecolor="black", markeredgewidth=0.3)
    cc_match = num_mask_components == num_graph_components
    status_text = (
        f"CC: mask={num_mask_components}, graph={num_graph_components}, "
        f"match={cc_match}"
    )
    ax.set_title("6. Overlay (mask + skeleton + graph)")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")
    ax.text(
        0.01, 0.01, status_text,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9,
        color="white" if not empty else "black",
        bbox={"facecolor": "black" if not empty else "white", "alpha": 0.65, "pad": 3},
    )

    for row in axes:
        for ax in row:
            ax.tick_params(labelsize=7)

    fig.tight_layout()
    return fig


def visualize_compare(
    mask: np.ndarray,
    filename: str,
    graphs: dict[tuple[str, bool], tuple[nx.Graph, dict]] | None = None,
) -> plt.Figure:
    """Create a 2x2 figure comparing keypoint/sknw x fix_borders off/on.

    Args:
        mask: 2D binary mask array.
        filename: Display name for the figure title.
        graphs: Optional pre-computed graphs keyed by (method, fix_borders).
            If provided, skips recomputation and uses these directly.
    """
    if mask.ndim == 3:
        mask = mask.squeeze()

    h, w = mask.shape
    _, num_mask_components = label(mask, connectivity=2, return_num=True)
    empty = num_mask_components == 0

    fig, axes = plt.subplots(3, 2, figsize=(20, 26))
    fig.suptitle(f"{filename} — Backend Comparison", fontsize=16, fontweight="bold")

    variants = [
        ("keypoint", False, 0, 0),
        ("keypoint", True,  0, 1),
        ("sknw",     False, 1, 0),
        ("sknw",     True,  1, 1),
        ("hybrid",   False, 2, 0),
        ("hybrid",   True,  2, 1),
    ]

    for method, fix_borders, r_idx, c_idx in variants:
        ax = axes[r_idx, c_idx]

        if graphs is not None and (method, fix_borders) in graphs:
            G, node_positions = graphs[(method, fix_borders)]
        else:
            G, node_positions = _call_mask_to_graph(mask, method, fix_borders)

        title = f"{method} | fix_borders={fix_borders}"
        _draw_graph_on_ax(ax, mask, G, node_positions, num_mask_components, empty, title)

    fig.legend(
        handles=[
            plt.Line2D([], [], marker="o", color="w",
                       markerfacecolor="lime", markeredgecolor="black",
                       markersize=8, label="endpoint (deg=1)"),
            plt.Line2D([], [], marker="o", color="w",
                       markerfacecolor="dodgerblue", markeredgecolor="black",
                       markersize=8, label="path (deg=2)"),
            plt.Line2D([], [], marker="o", color="w",
                       markerfacecolor="red", markeredgecolor="black",
                       markersize=8, label="junction (deg≥3)"),
        ],
        loc="lower center", ncol=3, fontsize=10,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06)
    return fig


def main() -> None:
    test_dir = "tests/test_data"
    out_dir = "tests/test_data_vis_dre"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(test_dir) if f.endswith(".npy"))

    summary = []
    compare_summary = []
    for f in files:
        arr = np.load(os.path.join(test_dir, f))
        arr2d = arr.squeeze()
        road_ratio = float(np.sum(arr2d > 0)) / arr2d.size
        _, num_mask_components = label(arr2d, connectivity=2, return_num=True)
        G, _ = mask_to_graph(
            arr2d, min_distance=8, k=4, r=200, auto_relax=True,
            add_edge_nodes=True, edge_border_width=1,
        )
        num_graph_components = nx.number_connected_components(G) if len(G.nodes) > 0 else 0
        summary.append((f, num_mask_components, num_graph_components))
        fig = visualize_one(arr, f"{f}  (road={road_ratio:.2%})")
        out_path = os.path.join(out_dir, f.replace(".npy", ".png"))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(
            f"Saved {out_path} | "
            f"mask_cc={num_mask_components}, graph_cc={num_graph_components}, "
            f"match={num_mask_components == num_graph_components}"
        )

        # --- Hybrid pipeline figure ---
        fig_hybrid = visualize_one_hybrid(arr, f"{f}  (road={road_ratio:.2%})")
        out_hybrid = os.path.join(out_dir, f.replace(".npy", "_hybrid.png"))
        fig_hybrid.savefig(out_hybrid, dpi=150, bbox_inches="tight")
        plt.close(fig_hybrid)
        print(f"Saved {out_hybrid}")

        # --- Compute all 6 variants once, reuse for both figure and summary ---
        graphs: dict[tuple[str, bool], tuple[nx.Graph, dict]] = {}
        for method in ["keypoint", "sknw", "hybrid"]:
            for fb in [False, True]:
                Gv, posv = _call_mask_to_graph(arr2d, method, fb)
                graphs[(method, fb)] = (Gv, posv)
                gcc = nx.number_connected_components(Gv) if len(Gv.nodes) > 0 else 0
                comp_match = num_mask_components == gcc
                compare_summary.append((f, method, fb, comp_match))
                print(
                    f"  {f}  {method:8s}  fix_borders={str(fb):5s}  "
                    f"mask_cc={num_mask_components}  graph_cc={gcc}  "
                    f"match={comp_match}"
                )

        fig_compare = visualize_compare(arr2d, f, graphs=graphs)
        out_compare = os.path.join(out_dir, f.replace(".npy", "_compare.png"))
        fig_compare.savefig(out_compare, dpi=150, bbox_inches="tight")
        plt.close(fig_compare)
        print(f"Saved {out_compare}")
        print()

    matched = sum(1 for _, mcc, gcc in summary if mcc == gcc)
    print(f"\nConnected-component check: {matched}/{len(summary)} files matched")
    # --- Final comparison summary ---
    print("\n--- Match Counts ---")
    for method in ["keypoint", "sknw", "hybrid"]:
        for fb in [False, True]:
            matched_v = sum(1 for _, m, fb2, ok in compare_summary
                           if m == method and fb2 == fb and ok)
            total = sum(1 for _, m, fb2, _ in compare_summary
                       if m == method and fb2 == fb)
            print(f"  {method:8s}  fix_borders={str(fb):5s}  {matched_v}/{total} matched")
    print()


if __name__ == "__main__":
    main()
