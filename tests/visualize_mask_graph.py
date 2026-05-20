"""Visualize road masks and their skeleton-derived graph structures side by side."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from skan import Skeleton, summarize


def mask_to_graph_info(mask: np.ndarray, spacing: tuple[float, float] = (1.0, 1.0)):
    """Skeletonize mask and extract graph nodes/edges via skan.

    Returns (skeleton_image, nodes, edges) or (None, None, None) for empty masks.
    """
    if mask.ndim == 3:
        mask = mask.squeeze()
    skeleton = skeletonize(mask > 0)
    if np.sum(skeleton) == 0:
        return None, None, None

    skel = Skeleton(skeleton, spacing=spacing)
    branch_data = summarize(skel, separator="_")

    if len(branch_data) == 0:
        return None, None, None

    src_coords = branch_data[["image_coord_src_0", "image_coord_src_1"]].values
    dst_coords = branch_data[["image_coord_dst_0", "image_coord_dst_1"]].values

    all_coords = np.vstack([src_coords, dst_coords])
    _, unique_idx, inverse = np.unique(
        np.round(all_coords, decimals=1), axis=0, return_index=True, return_inverse=True
    )
    nodes = all_coords[unique_idx]
    n_src = len(src_coords)
    edges = np.column_stack([inverse[:n_src], inverse[n_src:]])

    return skeleton, nodes, edges


def visualize_one(mask: np.ndarray, filename: str) -> None:
    """Create a 1×3 figure: mask | graph | overlay."""
    if mask.ndim == 3:
        mask = mask.squeeze()

    skeleton, nodes, edges = mask_to_graph_info(mask)
    h, w = mask.shape

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(filename, fontsize=14, fontweight="bold")

    # --- Subplot 1: Original mask ---
    ax = axes[0]
    ax.imshow(mask, cmap="gray", origin="upper")
    ax.set_title("Road Mask")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    # --- Subplot 2: Graph structure ---
    ax = axes[1]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title("Graph (skan)")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    if nodes is not None and len(nodes) > 0:
        # Draw edges
        for src_id, dst_id in edges:
            xs = [nodes[src_id, 1], nodes[dst_id, 1]]
            ys = [nodes[src_id, 0], nodes[dst_id, 0]]
            ax.plot(xs, ys, "c-", linewidth=1.5, alpha=0.8)

        # Draw nodes — color by degree
        degree = np.zeros(len(nodes), dtype=int)
        for src_id, dst_id in edges:
            degree[src_id] += 1
            degree[dst_id] += 1

        for i, (y, x) in enumerate(nodes):
            d = degree[i]
            if d == 1:
                color = "lime"
                label = "endpoint"
            elif d >= 3:
                color = "red"
                label = "junction"
            else:
                color = "dodgerblue"
                label = "path"
            ax.plot(x, y, "o", color=color, markersize=5, markeredgecolor="black",
                    markeredgewidth=0.5)

        ax.legend(
            handles=[
                plt.Line2D([], [], marker="o", color="w", markerfacecolor="lime",
                           markeredgecolor="black", markersize=8, label="endpoint (deg=1)"),
                plt.Line2D([], [], marker="o", color="w", markerfacecolor="dodgerblue",
                           markeredgecolor="black", markersize=8, label="path (deg=2)"),
                plt.Line2D([], [], marker="o", color="w", markerfacecolor="red",
                           markeredgecolor="black", markersize=8, label="junction (deg≥3)"),
            ],
            loc="lower right",
            fontsize=8,
        )
    else:
        ax.text(w / 2, h / 2, "No graph\n(empty skeleton)", ha="center", va="center",
                fontsize=14, color="gray")

    # --- Subplot 3: Overlay ---
    ax = axes[2]
    ax.imshow(mask, cmap="gray", origin="upper", alpha=0.7)
    if skeleton is not None:
        ax.imshow(skeleton, cmap="autumn", origin="upper", alpha=0.6)
    if nodes is not None and len(nodes) > 0:
        for src_id, dst_id in edges:
            xs = [nodes[src_id, 1], nodes[dst_id, 1]]
            ys = [nodes[src_id, 0], nodes[dst_id, 0]]
            ax.plot(xs, ys, "c-", linewidth=1.2, alpha=0.7)
        for i, (y, x) in enumerate(nodes):
            d = degree[i]  # noqa: PLW2901 — already defined above if nodes exist
            color = "lime" if d == 1 else ("red" if d >= 3 else "dodgerblue")
            ax.plot(x, y, "o", color=color, markersize=4, markeredgecolor="black",
                    markeredgewidth=0.5)
    ax.set_title("Overlay (mask + skeleton + graph)")
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    for ax in axes:
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    return fig


def main() -> None:
    test_dir = os.path.join(os.path.dirname(__file__), "test_data")
    out_dir = os.path.join(os.path.dirname(__file__), "test_data_vis")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(test_dir) if f.endswith(".npy"))

    for f in files:
        arr = np.load(os.path.join(test_dir, f))
        road_ratio = float(np.sum(arr > 0)) / arr.size
        fig = visualize_one(arr, f"{f}  (road={road_ratio:.2%})")
        out_path = os.path.join(out_dir, f.replace(".npy", ".png"))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    print(f"\nAll {len(files)} figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
