import sys
from pathlib import Path

# detector.py lives one folder up (utils/), so add it to the import path
UTILS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UTILS_DIR))

from detector import BarcodeImage

PROJECT_ROOT = UTILS_DIR.parent
DATASET = PROJECT_ROOT / "barcodes" / "1d_barcode_extended_plain"

image_paths = sorted((DATASET / "Original").glob("*.png"))[:20]

for img_path in image_paths:
    image = BarcodeImage.from_name(img_path.stem, DATASET)

    print(image.name)
    print("image:", image.image.shape, image.image.dtype)
    print("gray: ", image.gray.shape)
    print("mask: ", image.gt_mask.shape, image.gt_mask.dtype, "pixels set:", image.gt_mask.sum())
    print("angle: ", image.gt_angle)

    image.pred_mask = image.gt_mask.copy()
    print(f"iou (pred = gt): {image.compute_iou()}")
    image.reset()
    print()
