"""Interactive binary mask editor for APLS cross-verification.

Usage:
    uv run python tests/apls_editor.py

Workflow:
    1. For each of the 11 test masks, open an interactive editor.
    2. Left-click to ADD road pixels (brush).
    3. Right-click to REMOVE road pixels (eraser).
    4. Press 'y' to save current edit and move to next variant.
    5. Press 'n' to skip to the next image (no more variants for current image).
    6. Press 'q' to quit early.
    7. After editing, run Go vs Python APLS comparison on all saved results.
"""

from __future__ import annotations

import json

import os
import subprocess
import sys
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent

# Ensure src/ is on sys.path for direct execution without PYTHONPATH
_src = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

from topo_metrics import TopoGraph, apls, mask_to_graph

# ── Configuration ───────────────────────────────────────────────────────────

TEST_DIR = os.path.join(os.path.dirname(__file__), "test_data")
EDIT_DIR = os.path.join(os.path.dirname(__file__), "test_data_edited")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "apls_comparison_report.txt")

BRUSH_RADIUS = 3
VARIANTS_PER_IMAGE = 3



# Go pixel-coordinate APLS binary
GO_PIXEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "external", "Sat2Graph", "metrics", "apls_pixel")
)


def run_go_apls(gt_json_path: str, prop_json_path: str) -> float | None:
    """Run Go pixel-coordinate APLS binary and parse the result."""
    binary = os.path.join(GO_PIXEL_DIR, "apls_pixel")
    if not os.path.exists(binary):
        print(f"  Go pixel binary not found, building...")
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
    """Convert TopoGraph to pixel-coordinate JSON format [nodes, edges].

    Nodes are passed as raw [x, y] pixel coordinates (no lat/lon conversion).
    """
    nodes = [[float(x), float(y)] for x, y in tg.nodes]
    edges = [[int(e0), int(e1)] for e0, e1 in tg.edges]
    return [nodes, edges]


# ── Interactive editor ──────────────────────────────────────────────────────


class MaskEditor:
    """Matplotlib-based interactive binary mask editor.

    Left-click: add road (set to 1)
    Right-click: remove road (set to 0)
    Scroll: change brush radius
    's': save current edit
    'n': skip to next image
    'q': quit
    """

    def __init__(self, mask: np.ndarray, title: str) -> None:
        self.original = mask.copy()
        self.mask = mask.copy()
        self.title = title
        self.saved = False
        self.skipped = False
        self.brush_radius = BRUSH_RADIUS

    def run(self) -> np.ndarray | None:
        """Open editor window, return edited mask or None if skipped."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(self.title, fontsize=12, fontweight="bold")

        # Left: editable mask
        ax_edit = axes[0]
        ax_edit.set_title("Edit (Left=Add, Right=Erase, Scroll=Brush Size)")
        im = ax_edit.imshow(self.mask, cmap="gray", vmin=0, vmax=1)
        brush_circle = plt.Circle((0, 0), self.brush_radius, fill=False,
                                   color="lime", linewidth=1.5, visible=False)
        ax_edit.add_patch(brush_circle)
        ax_edit.set_xlim(-0.5, self.mask.shape[1] - 0.5)
        ax_edit.set_ylim(self.mask.shape[0] - 0.5, -0.5)

        # Right: diff from original
        ax_diff = axes[1]
        ax_diff.set_title("Diff from original (green=added, red=removed)")
        diff_display = np.zeros((*self.mask.shape, 3))
        im_diff = ax_diff.imshow(diff_display, origin="upper")
        ax_diff.set_xlim(-0.5, self.mask.shape[1] - 0.5)
        ax_diff.set_ylim(self.mask.shape[0] - 0.5, -0.5)

        h, w = self.mask.shape
        prev_pos: list[int | None] = [None, None]  # [prev_x, prev_y], mutable closure
        dirty = [False]  # whether diff panel needs refresh

        def _paint_circle(cx: int, cy: int, value: int) -> None:
            """Paint a single brush circle onto self.mask."""
            r = self.brush_radius
            r0, r1 = max(0, cy - r), min(h, cy + r + 1)
            c0, c1 = max(0, cx - r), min(w, cx + r + 1)
            yy, xx = np.mgrid[r0:r1, c0:c1]
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            self.mask[r0:r1, c0:c1][dist <= r] = value

        def _paint_line(x0: int, y0: int, x1: int, y1: int, value: int) -> None:
            """Paint circles along a line from (x0,y0) to (x1,y1) using Bresenham."""
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            steps = max(dx, dy, 1)
            for i in range(steps + 1):
                t = i / steps
                ix = int(round(x0 + t * (x1 - x0)))
                iy = int(round(y0 + t * (y1 - y0)))
                _paint_circle(ix, iy, value)

        def _get_value(event: MouseEvent) -> int | None:
            if event.button == MouseButton.LEFT:
                return 1
            elif event.button == MouseButton.RIGHT:
                return 0
            return None

        def _refresh_display() -> None:
            im.set_data(self.mask)
            if dirty[0]:
                self._update_diff(im_diff)
                dirty[0] = False
            fig.canvas.draw_idle()

        def _on_press(event: MouseEvent) -> None:
            if event.inaxes != ax_edit:
                return
            x, y = int(round(event.xdata)), int(round(event.ydata))
            if not (0 <= x < w and 0 <= y < h):
                return
            value = _get_value(event)
            if value is None:
                brush_circle.center = (x, y)
                brush_circle.set_visible(True)
                fig.canvas.draw_idle()
                return
            _paint_circle(x, y, value)
            prev_pos[0] = x
            prev_pos[1] = y  # type: ignore[call-overload]
            dirty[0] = True
            brush_circle.center = (x, y)
            brush_circle.set_visible(True)
            _refresh_display()

        def _on_motion(event: MouseEvent) -> None:
            if event.inaxes != ax_edit:
                brush_circle.set_visible(False)
                fig.canvas.draw_idle()
                prev_pos[0] = None
                return
            x, y = int(round(event.xdata)), int(round(event.ydata))
            brush_circle.center = (x, y)
            brush_circle.set_radius(self.brush_radius)
            brush_circle.set_visible(True)

            value = _get_value(event)
            if value is None:
                fig.canvas.draw_idle()
                return

            if 0 <= x < w and 0 <= y < h:
                if prev_pos[0] is not None:
                    _paint_line(prev_pos[0], prev_pos[1], x, y, value)  # type: ignore[arg-type]
                else:
                    _paint_circle(x, y, value)
                prev_pos[0] = x
                prev_pos[1] = y  # type: ignore[call-overload]
                dirty[0] = True
                _refresh_display()
            else:
                fig.canvas.draw_idle()

        def _on_release(event: MouseEvent) -> None:
            prev_pos[0] = None
            # Final full refresh including diff panel
            self._update_diff(im_diff)
            dirty[0] = False
            fig.canvas.draw_idle()

        def _on_scroll(event: MouseEvent) -> None:
            if event.button == "up":
                self.brush_radius = min(self.brush_radius + 1, 20)
            elif event.button == "down":
                self.brush_radius = max(self.brush_radius - 1, 1)
            brush_circle.set_radius(self.brush_radius)
            fig.canvas.draw_idle()

        def _on_key(event) -> None:
            if event.key == "y":
                self.saved = True
                plt.close(fig)
            elif event.key == "n":
                self.skipped = True
                plt.close(fig)
            elif event.key == "q":
                self.skipped = True
                plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", _on_press)
        fig.canvas.mpl_connect("motion_notify_event", _on_motion)
        fig.canvas.mpl_connect("button_release_event", _on_release)
        fig.canvas.mpl_connect("scroll_event", _on_scroll)
        fig.canvas.mpl_connect("key_press_event", _on_key)

        # Instructions
        fig.text(0.5, 0.01,
                 "[y] Save & next variant  |  [n] Skip to next image  |  "
                 "[q] Quit  |  Scroll=brush size",
                 ha="center", fontsize=10,
                 bbox={"facecolor": "lightyellow", "alpha": 0.8, "pad": 5})

        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        plt.show()

        if self.skipped:
            return None
        return self.mask

    def _update_diff(self, im_diff) -> None:
        diff = np.zeros((*self.mask.shape, 3))
        added = (self.mask == 1) & (self.original == 0)
        removed = (self.mask == 0) & (self.original == 1)
        diff[added] = [0, 1, 0]    # green = added
        diff[removed] = [1, 0, 0]  # red = removed
        im_diff.set_data(diff)


# ── Main workflow ───────────────────────────────────────────────────────────


def collect_edits() -> dict[str, list[tuple[str, np.ndarray]]]:
    """Run interactive editor on all test masks. Returns {filename: [(label, mask), ...]}."""
    matplotlib.use("macosx")  # macOS native backend (no tkinter needed)

    os.makedirs(EDIT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(TEST_DIR) if f.endswith(".npy"))
    results: dict[str, list[tuple[str, np.ndarray]]] = {}

    # Always include original (unmodified) as variant 0
    for f in files:
        arr = np.load(os.path.join(TEST_DIR, f)).squeeze()
        mask = (arr > 0).astype(np.uint8)
        results[f] = [("original", mask)]

    print("=" * 60)
    print("Interactive Mask Editor for APLS Cross-Verification")
    print("=" * 60)
    print(f"Test images: {len(files)}")
    print(f"Variants per image: {VARIANTS_PER_IMAGE} (including original)")
    print()
    print("Controls:")
    print("  Left-click drag  → Add road pixels")
    print("  Right-click drag → Remove road pixels")
    print("  Scroll           → Change brush size")
    print("  'y'              → Save current edit, next variant")
    print("  'n'              → Skip to next image")
    print("  'q'              → Quit editor")
    print()

    for f in files:
        arr = np.load(os.path.join(TEST_DIR, f)).squeeze()
        mask = (arr > 0).astype(np.uint8)
        results[f] = [("original", mask)]

        for v in range(1, VARIANTS_PER_IMAGE):
            label = f"edit_v{v}"
            editor = MaskEditor(mask.copy(), f"{f} — Edit Variant {v}/{VARIANTS_PER_IMAGE - 1}")
            edited = editor.run()

            if edited is None:
                print(f"  Skipped remaining variants for {f}")
                break

            # Save edited mask
            save_path = os.path.join(EDIT_DIR, f.replace(".npy", f"_{label}.npy"))
            np.save(save_path, edited)
            results[f].append((label, edited))
            print(f"  Saved {save_path}")

    return results


def compute_apls_comparison(
    all_edits: dict[str, list[tuple[str, np.ndarray]]],
) -> list[dict]:
    """Compute APLS (Go vs Python) for all edited masks."""
    rows = []

    for f, variants in all_edits.items():
        print(f"\nProcessing {f} ({len(variants)} variants)...")

        # The "original" is both GT and prediction for identical-test
        # For edits, original=GT, edit=prediction
        gt_label, gt_mask = variants[0]  # always "original"

        for pred_label, pred_mask in variants:
            # Build graphs using mask_to_graph (hybrid method)
            gt_tg = mask_to_graph(gt_mask)
            pred_tg = mask_to_graph(pred_mask)

            if gt_tg.num_nodes == 0 or pred_tg.num_nodes == 0:
                rows.append({
                    "file": f, "gt": gt_label, "pred": pred_label,
                    "gt_nodes": gt_tg.num_nodes, "pred_nodes": pred_tg.num_nodes,
                    "py_apls": 0.0 if gt_tg.num_nodes == 0 else None,
                    "go_apls": None,
                    "diff": None,
                })
                continue

            # Python APLS
            py_score = apls(gt_tg, pred_tg, sample_interval=50.0, densify_interval=2.0)

            # Go APLS
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
                "file": f, "gt": gt_label, "pred": pred_label,
                "gt_nodes": gt_tg.num_nodes, "pred_nodes": pred_tg.num_nodes,
                "py_apls": py_score,
                "go_apls": go_score,
                "diff": diff,
            })

            go_str = f"{go_score:.6f}" if go_score is not None else "N/A"
            diff_str = f"{diff:.6f}" if diff is not None else "N/A"
            print(f"  {gt_label}→{pred_label}: py={py_score:.6f}, go={go_str}, diff={diff_str}")

    return rows


def print_report(rows: list[dict]) -> None:
    """Print and save APLS comparison report."""
    print("\n" + "=" * 100)
    print("APLS Comparison Report: Go vs Python (mask_to_graph hybrid method)")
    print("=" * 100)
    header = f"{'File':20s} {'GT':10s} {'Pred':10s} {'GT_N':>5s} {'Pred_N':>6s} {'Py APLS':>8s} {'Go APLS':>8s} {'Diff':>8s}"
    print(header)
    print("-" * 100)

    lines = [header, "-" * 100]
    for r in rows:
        py_str = f"{r['py_apls']:.6f}" if r['py_apls'] is not None else "EMPTY"
        go_str = f"{r['go_apls']:.6f}" if r['go_apls'] is not None else "N/A"
        diff_str = f"{r['diff']:.6f}" if r['diff'] is not None else "N/A"
        line = (f"{r['file']:20s} {r['gt']:10s} {r['pred']:10s} "
                f"{r['gt_nodes']:5d} {r['pred_nodes']:6d} "
                f"{py_str:>8s} {go_str:>8s} {diff_str:>8s}")
        print(line)
        lines.append(line)

    # Summary
    diffs = [r['diff'] for r in rows if r['diff'] is not None]
    if diffs:
        summary = (f"\nSummary: {len(diffs)} comparisons, "
                    f"mean_diff={np.mean(diffs):.6f}, max_diff={np.max(diffs):.6f}")
        print(summary)
        lines.append(summary)

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {REPORT_PATH}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--compare-only":
        # Load previously saved edits and run comparison only
        print("--compare-only mode: loading saved edits...")
        files = sorted(f for f in os.listdir(TEST_DIR) if f.endswith(".npy"))
        all_edits: dict[str, list[tuple[str, np.ndarray]]] = {}
        for f in files:
            arr = np.load(os.path.join(TEST_DIR, f)).squeeze()
            mask = (arr > 0).astype(np.uint8)
            variants = [("original", mask)]
            # Load saved edits
            for v in range(1, VARIANTS_PER_IMAGE):
                label = f"edit_v{v}"
                edit_path = os.path.join(EDIT_DIR, f.replace(".npy", f"_{label}.npy"))
                if os.path.exists(edit_path):
                    edited = np.load(edit_path)
                    variants.append((label, edited))
            all_edits[f] = variants
        rows = compute_apls_comparison(all_edits)
        print_report(rows)
        return

    # Phase 1: Interactive editing
    all_edits = collect_edits()

    # Phase 2: APLS comparison
    print("\n" + "=" * 60)
    print("Phase 2: Computing APLS (Go vs Python)")
    print("=" * 60)
    rows = compute_apls_comparison(all_edits)
    print_report(rows)


if __name__ == "__main__":
    main()
