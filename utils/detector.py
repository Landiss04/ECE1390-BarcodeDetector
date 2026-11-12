import numpy as np
import cv2
from pathlib import Path

class BarcodeImage:
    """ Structure used to handle the individual image data

    Attributes:

    """
    def __init__(self, image_path, mask_path=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.name = Path(image_path).stem

        #inputs
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # ground truth (not cleared by reset)
        self.gt_mask = None
        self.gt_angle = None
        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {mask_path}")
            if mask.shape != self.gray.shape:
                raise ValueError(f"{self.name}: mask shape: {mask.shape} != image shape {self.gray.shape})")
            self.gt_mask = mask > 0
            self.compute_gt_angle()

        #pipeline values, values should be assigned via BarcodeDetector
        self.blurred = None
        self.gx = None
        self.gy = None

        self.jxx = None
        self.jyy = None
        self.jxy = None
        self.coherence = None
        self.energy = None
        self.score_gxgy = None
        self.score = None

        self.otsu_thresh = None
        self.binary = None
        self.closed = None
        self.opened = None

        self.labels = None
        self.stats = None
        self.candidates = None
        self.pred_rect = None
        self.pred_angle = None
        self.pred_mask = None

        self.iou = None

    @classmethod
    def from_name(cls, name, root):
        """Build a barcode image from a file stem I.E. "04102009081"

        """
        root = Path(root)
        image_path = root / "Original" / f"{name}.png"
        mask_path = root / "Detection" / f"{name}.png"

        return cls(image_path, mask_path)

    def compute_gt_angle(self):
        """Set gt_angle on gt_mask's nonzero pixles

        """
        if self.gt_mask is None:
            raise ValueError("Gt mask is None")

        pts = cv2.findNonZero(self.gt_mask.astype(np.uint8))
        if pts is None:
            raise ValueError(f"{self.name}: gt mask is empty")

        self.gt_angle = cv2.minAreaRect(pts)[2]
        return self.gt_angle

    def compute_iou(self):
        """setting iou, via gt and pred masks

        """
        if self.gt_mask is None or self.pred_mask is None:
            raise ValueError("gt mask and/or predicted mask is None")

        intersection = np.logical_and(self.gt_mask, self.pred_mask)
        union = np.logical_or(self.gt_mask, self.pred_mask)

        union_sum = np.sum(union)
        self.iou = float(np.sum(intersection) / union_sum) if union_sum > 0 else 0.0
        return self.iou

    def reset(self):
        """Reset all of the intermediate values

        """
        self.blurred = None
        self.gx = None
        self.gy = None

        self.jxx = None
        self.jyy = None
        self.jxy = None
        self.coherence = None
        self.energy = None
        self.score_gxgy = None
        self.score = None

        self.otsu_thresh = None
        self.binary = None
        self.closed = None
        self.opened = None

        self.labels = None
        self.stats = None
        self.candidates = None
        self.pred_rect = None
        self.pred_angle = None
        self.pred_mask = None

        self.iou = None

    def overlay(self):
        pass

    def save_stages(self, out_dir):
        pass

    def __repr__(self):
        h, w = self.gray.shape
        return (f"BarcodeImage(name={self.name!r}, size={w}x{h}, "
                f"has_mask={self.gt_mask is not None}, iou={self.iou})")



class BarcodeDetector:
    """ Detects the barcodes from the images, applys different image processing methods.

    Attributes:
    """
    def __init__(self):
        pass
