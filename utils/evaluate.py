"""run every method on every image and report detection rate and mean iou

usage (from the project root):  python utils/evaluate.py
"""
import sys
import csv
from pathlib import Path
import numpy as np

# detector.py lives next to this file, so add this folder to the import path
UTILS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(UTILS_DIR))

from detector import BarcodeImage, BarcodeDetector

PROJECT_ROOT = UTILS_DIR.parent
DATASET = PROJECT_ROOT / "barcodes" / "1d_barcode_extended_plain"
OUT_DIR = PROJECT_ROOT / "results"

# a prediction counts as a hit when iou is at least this
IOU_THRESH = 0.5

# images where another 1d barcode is visible but not labeled
# (candidates from looking through the dataset, confirm as a team)
EXCLUDE = {"IMAG0494", "IMAG0495", "IMAG0496", "IMAG0499", "PICT0027a", "PICT0028"}

# every method takes (detector, image) and leaves the iou on the image
METHODS = {
    "baseline_dark": lambda d, i: d.baseline(i, dark_is_object=True),
    "baseline_light": lambda d, i: d.baseline(i, dark_is_object=False),
    "gxgy": lambda d, i: d.detect(i, which="gxgy"),
    "tensor": lambda d, i: d.detect(i, which="tensor"),
}


def summarize(rows, names):
    """print detection rate and mean iou for each method over the given image names"""
    keep = [r for r in rows if r["name"] in names]
    print(f"{'method':<16}{'detection rate':>16}{'mean iou':>12}")
    for m in METHODS:
        ious = np.array([r[m] for r in keep])
        rate = 100 * np.mean(ious >= IOU_THRESH)
        print(f"{m:<16}{rate:>15.1f}%{ious.mean():>12.3f}")


def main():
    # list every image name in the dataset
    names = sorted(p.stem for p in (DATASET / "Original").glob("*.png"))
    detector = BarcodeDetector()

    # run every method on every image and keep the iou
    rows = []
    for name in names:
        img = BarcodeImage.from_name(name, DATASET)
        row = {"name": name, "gt_angle": img.gt_angle}
        for m, run in METHODS.items():
            run(detector, img)
            row[m] = img.iou
        rows.append(row)

    # save the per image results so the figure script can reuse them
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "per_image_iou.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "gt_angle", *METHODS])
        writer.writeheader()
        writer.writerows(rows)

    # report on every image, then without the excluded ones
    print(f"\nall images (n={len(names)})")
    summarize(rows, set(names))
    kept = set(names) - EXCLUDE
    print(f"\nwithout excluded images (n={len(kept)})")
    summarize(rows, kept)


if __name__ == "__main__":
    main()
