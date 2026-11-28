"""detection rate by barcode tilt angle

usage (from project root):
    python utils/evaluate.py
    python utils/angle_figure.py

when in development use SPLIT = "tune", switch to "test" for the final version
"""

import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UTILS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(UTILS_DIR))

from evaluate import IOU_THRESH, OUT_DIR

SPLIT = "tune"
BIN_EDGES = [0, 15, 30, 45, 60, 75, 90]
#comparison methods (with legends)
COMPARE = {
    "gxgy_shape": "Gx-Gy score",
    "tensor_shape": "structure tensor score"
}

def main():
    with open(OUT_DIR / "per_image_iou.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == SPLIT]

    tilts = np.array([float(r["gt_tilt"]) for r in rows])
    #bin index per image
    bins = np.digitize(tilts, BIN_EDGES[1:-1])
    n_bins = len(BIN_EDGES) - 1
    counts = np.bincount(bins, minlength=n_bins)

    #detection rate per bin per method
    rates = {}
    for m in COMPARE:
        hit = np.array([float(r[m]) >= IOU_THRESH for r in rows])
        rates[m] = [hit[bins == b].mean() if counts[b] else np.nan for b in range(n_bins)]

    #printing the table so we have numbers for the report
    labels = [f"{BIN_EDGES[b]}-{BIN_EDGES[b+1]}" for b in range(n_bins)]
    print(f"{SPLIT} split (n={len(rows)})")
    print(f"{'tilt':<8}{'n':>4}" + "".join(f"{m:>16}" for m in COMPARE))
    for b in range(n_bins):
        cells = "".join(f"{100 * rates[m][b]:>15.1f}%" if counts[b] else f"{'-':>16}"
                        for m in COMPARE)
        print(f"{labels[b]:<8}{counts[b]:>4}{cells}")

    # grouped bars, one group per tilt bin
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(n_bins)
    width = 0.8 / len(COMPARE)
    for j, (m, name) in enumerate(COMPARE.items()):
        ax.bar(x + (j - (len(COMPARE) - 1) / 2) * width, rates[m], width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}°\n(n={c})" for lab, c in zip(labels, counts)])
    ax.set_xlabel("barcode tilt from vertical bars")
    ax.set_ylabel(f"detection rate (IoU >= {IOU_THRESH})")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Detection rate by tilt angle ({SPLIT} split, n = {len(rows)} images)")
    ax.legend(loc="lower left")
    fig.tight_layout()

    out = OUT_DIR / f"angle_figure_{SPLIT}.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()