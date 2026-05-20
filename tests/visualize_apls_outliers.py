"""Visualize the 37 high-diff (>0.10) cases from the 100-image APLS benchmark.

Reproduces the exact corruptions using seed=42, then generates APLS pipeline
visualizations for each outlier.

Usage:
    uv run python tests/visualize_apls_outliers.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_src = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

# Reuse the corruption functions and APLS visualization from sibling scripts
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "visualize_apls", os.path.join(os.path.dirname(__file__), "visualize_apls.py")
)
_vis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vis)
corrupt_mask = _vis.corrupt_mask
visualize_apls = _vis.visualize_apls

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data_100")
OUT_DIR = os.path.join(os.path.dirname(__file__), "vis_apls_outliers")

# These are the exact 33 cases with diff > 0.10 from the seed=42 run.
# Format: (filename, 1-indexed variant number, diff)
OUTLIERS = [
    ("patch_000946.npy", 9, 0.499354),
    ("patch_001017.npy", 3, 0.422669),
    ("patch_000969.npy", 8, 0.342404),
    ("patch_001017.npy", 2, 0.260413),
    ("patch_001017.npy", 1, 0.250000),
    ("patch_001017.npy", 7, 0.250000),
    ("patch_001008.npy", 9, 0.224275),
    ("patch_000948.npy", 4, 0.215220),
    ("patch_000939.npy", 9, 0.200063),
    ("patch_001035.npy", 9, 0.187733),
    ("patch_000995.npy", 8, 0.180230),
    ("patch_001055.npy", 8, 0.171965),
    ("patch_001008.npy", 8, 0.166022),
    ("patch_000996.npy", 8, 0.158474),
    ("patch_001054.npy", 1, 0.153924),
    ("patch_000951.npy", 7, 0.153264),
    ("patch_001002.npy", 3, 0.152874),
    ("patch_000989.npy", 5, 0.144878),
    ("patch_001025.npy", 10, 0.136837),
    ("patch_001004.npy", 5, 0.136074),
    ("patch_000950.npy", 6, 0.133834),
    ("patch_001025.npy", 9, 0.129212),
    ("patch_001018.npy", 5, 0.120577),
    ("patch_001004.npy", 9, 0.119318),
    ("patch_001031.npy", 3, 0.114197),
    ("patch_000969.npy", 1, 0.109362),
    ("patch_000970.npy", 1, 0.107722),
    ("patch_001050.npy", 5, 0.107693),
    ("patch_001045.npy", 2, 0.107529),
    ("patch_001023.npy", 7, 0.106986),
    ("patch_000946.npy", 3, 0.106762),
    ("patch_001028.npy", 10, 0.105651),
    ("patch_001050.npy", 8, 0.102872),
]


def main() -> None:
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    # Pre-generate all corruptions in order to match the original run.
    # The original run processes files alphabetically, 10 variants each.
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".npy"))

    # Build a lookup: (filename, variant_idx) → (corrupted_mask, desc)
    corruption_cache: dict[tuple[str, int], tuple[np.ndarray, str]] = {}
    for f in files:
        arr = np.load(os.path.join(DATA_DIR, f)).squeeze()
        mask = (arr > 0).astype(np.uint8)
        for v in range(10):
            corrupted, desc = corrupt_mask(mask, rng)
            corruption_cache[(f, v + 1)] = (corrupted, desc)

    print(f"Visualizing {len(OUTLIERS)} outlier cases (diff > 0.10)")
    print(f"Output: {OUT_DIR}/")
    print()

    for filename, variant, diff in OUTLIERS:
        arr = np.load(os.path.join(DATA_DIR, filename)).squeeze()
        gt_mask = (arr > 0).astype(np.uint8)

        key = (filename, variant)
        if key not in corruption_cache:
            # Fallback: re-run with a fresh seed (shouldn't happen)
            print(f"  WARNING: {filename} variant {variant} not in cache, skipping")
            continue

        pred_mask, desc = corruption_cache[key]
        title = f"{filename} v{variant} ({desc}) — Go-Py diff={diff:.4f}"

        fig = visualize_apls(gt_mask, pred_mask, title=title)
        out_name = f"{filename.replace('.npy', '')}_v{variant:02d}_diff{diff:.3f}.png"
        out_path = os.path.join(OUT_DIR, out_name)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out_name}")

    print(f"\nDone. {len(OUTLIERS)} figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
