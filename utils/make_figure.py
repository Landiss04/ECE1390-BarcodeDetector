"""make the milestone figure from the results of evaluate.py

usage (from the project root):
    python utils/evaluate.py
    python utils/make_figure.py
"""
import sys
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# evaluate.py and detector.py live next to this file
from pathlib import Path
UTILS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(UTILS_DIR))

from detector import BarcodeImage, BarcodeDetector
from evaluate import METHODS, EXCLUDE, IOU_THRESH, DATASET, OUT_DIR

# nicer names for the bar chart
LABELS = {
    "baseline_dark": "baseline\n(dark = object)",
    "baseline_light": "baseline\n(light = object)",
    "gxgy": "Gx-Gy score",
    "tensor": "structure tensor\nscore",
    "gxgy_morph":      "Gx-Gy\n+ close/open",
    "tensor_morph":    "tensor\n+ close/open",
    "gxgy_no_close":   "Gx-Gy\nopen only",
    "tensor_no_close": "tensor\nopen only",
}


def pick_examples(rows, method):
    """pick three images by a fixed rule, so the examples are not hand picked"""
    ious = np.array([r[method] for r in rows])

    # typical hit: the hit closest to the median iou of all hits
    hits = np.where(ious >= IOU_THRESH)[0]
    typical = hits[np.argmin(np.abs(ious[hits] - np.median(ious[hits])))]

    # near miss: the miss closest to iou 0.4 (a blob that covers only part of the barcode)
    near = np.where((ious >= 0.3) & (ious < IOU_THRESH))[0]
    near_miss = near[np.argmin(np.abs(ious[near] - 0.4))]

    # wrong object: the middle one of all the misses with iou below 0.1
    wrong = np.where(ious < 0.1)[0]
    wrong_obj = wrong[len(wrong) // 2]

    return [("typical hit", rows[typical]["name"]),
            ("near miss (part of the barcode)", rows[near_miss]["name"]),
            ("wrong object", rows[wrong_obj]["name"])]


def overlay(name, method, detector):
    """image with the ground truth outline in green and our prediction in red"""
    img = BarcodeImage.from_name(name, DATASET)
    METHODS[method](detector, img)
    out = img.image.copy()
    for mask, color in ((img.gt_mask, (0, 255, 0)), (img.pred_mask, (0, 0, 255))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 3)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), img.iou


def main():
    # read the per image results and drop the excluded images
    with open(OUT_DIR / "per_image_iou.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["name"] not in EXCLUDE]
    for r in rows:
        for m in METHODS:
            r[m] = float(r[m])

    # detection rate and mean iou for every method
    rates = [np.mean([r[m] >= IOU_THRESH for r in rows]) for m in METHODS]
    means = [np.mean([r[m] for r in rows]) for m in METHODS]

    fig = plt.figure(figsize=(10, 7.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])

    # top: grouped bars, one pair per method
    ax = fig.add_subplot(grid[0, :])
    x = np.arange(len(METHODS))
    b1 = ax.bar(x - 0.2, rates, 0.4, label=f"detection rate (IoU >= {IOU_THRESH})")
    b2 = ax.bar(x + 0.2, means, 0.4, label="mean IoU")
    for bars, vals, pct in ((b1, rates, True), (b2, means, False)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{100 * v:.1f}%" if pct else f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHODS])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("detection rate / mean IoU")
    ax.set_title(f"Otsu + largest blob on each score vs baseline (n = {len(rows)} images, nothing tuned)")
    ax.legend(loc="upper left")

    # bottom: three example outputs from the structure tensor score
    detector = BarcodeDetector()
    for col, (title, name) in enumerate(pick_examples(rows, "tensor")):
        shown, iou = overlay(name, "tensor", detector)
        axi = fig.add_subplot(grid[1, col])
        axi.imshow(shown)
        axi.set_title(f"{title}\n{name}, IoU = {iou:.2f}", fontsize=9)
        axi.axis("off")
    fig.text(0.5, 0.005, "green = ground truth, red = our prediction (structure tensor score)",
             ha="center", fontsize=9)

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "milestone_figure.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()
