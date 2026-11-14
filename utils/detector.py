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

    Attributes:  blur_ksize:  odd Gaussian kernel size applied before gradients
        sobel_ksize: Sobel kernel size
        window:      side length of the box window used to integrate gradient
                     statistics (should be roughly the size of a few bars)
    """
    def __init__(self, blur_ksize=5, sobel_ksize=3, window=21):
        self.blur_ksize = blur_ksize    # how much to smooth the image first (must be odd)
        self.sobel_ksize = sobel_ksize  # size of the edge-detecting filter
        self.window = window            # size of the neighborhood we look at around each pixel

    def compute_score(self, img): #this is what Dallal was suggesting we do 
        k = self.blur_ksize
        w = (self.window, self.window)  # neighborhood as a (width, height) tuple
        eps = 1e-6                      # tiny number to avoid dividing by zero later

        img.blurred = cv2.GaussianBlur(img.gray, (k, k), 0) # Blur the grayscale image slightly so camera noise doesn't create fake edges
        # Convert to float so the edge math can go negative
        f = img.blurred.astype(np.float32)

        img.gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=self.sobel_ksize) #how fast brightness changes left-to-right (big at vertical bars)
        img.gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=self.sobel_ksize) #how fast brightness changes top-to-bottom (small at vertical bars)

        # Score 1: the simple version from our proposal 
        ax = cv2.boxFilter(np.abs(img.gx), -1, w)# Average the horizontal edge strength over each neighborhood
        ay = cv2.boxFilter(np.abs(img.gy), -1, w)# Average the vertical edge strength over each neighborhood
        # Barcode = strong horizontal change minus vertical change; negatives become 0
        img.score_gxgy = self._normalize(np.clip(ax - ay, 0, None))

        # Score 2: the tilt-proof version (structure tensor)
       
        img.jxx = cv2.boxFilter(img.gx * img.gx, -1, w) # Average gx*gx over each neighborhood: total horizontal edge energy
        img.jyy = cv2.boxFilter(img.gy * img.gy, -1, w)  # Average gy*gy: total vertical edge energy
        img.jxy = cv2.boxFilter(img.gx * img.gy, -1, w)# Average gx*gy: whether the edges lean diagonally (this is what captures tilt)

        # Energy = how strong the edges are overall in the neighborhood
        img.energy = img.jxx + img.jyy
        # Coherence = how much the edges agree on one direction
        # (near 1 = all parallel like barcode bars, near 0 = pointing everywhere)
        img.coherence = np.sqrt((img.jxx - img.jyy) ** 2 + 4 * img.jxy ** 2) \
                        / (img.energy + eps)

        # Multiply so a pixel needs both strong edges and a consistent direction.
        # (Coherence alone would light up on flat, noisy areas.)
        img.score = self._normalize(img.coherence * img.energy)
        return img.score

    @staticmethod
    def _normalize(m):
        m = m.astype(np.float32)
        lo, hi = float(m.min()), float(m.max())  # darkest and brightest values in the map
        # Stretch so the min becomes 0 and the max becomes 1; all zeros if the map is flat
        return (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)

    def segment(self, img, which="tensor"):
        """turn a score map into a binary image with otsu

        which: "tensor" uses the structure tensor score, "gxgy" uses the simple score
        """
        # pick which score map to threshold
        score_map = img.score if which == "tensor" else img.score_gxgy

        # otsu needs an 8-bit image, so stretch the 0..1 scores up to 0..255
        score8 = (score_map * 255).astype(np.uint8)

        # otsu picks the threshold from the histogram, nothing to tune
        img.otsu_thresh, img.binary = cv2.threshold(
            score8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img.binary

    def pick_largest(self, img):
        """keep only the biggest white blob in img.binary as the predicted mask"""
        # label every connected white blob (label 0 is the background)
        n, img.labels, img.stats, _ = cv2.connectedComponentsWithStats(
            img.binary, connectivity=8)

        # start with an empty prediction
        img.pred_mask = np.zeros(img.gray.shape, dtype=bool)

        # nothing white at all, so the prediction stays empty
        if n <= 1:
            return img.pred_mask

        # skip the background and take the blob with the largest area
        areas = img.stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        img.pred_mask = img.labels == best
        return img.pred_mask

    def detect(self, img, which="tensor"):
        """minimal pipeline: score -> otsu -> largest blob -> iou"""
        # clear anything left over from a previous run
        img.reset()

        # score every pixel, threshold it, keep the biggest blob
        self.compute_score(img)
        self.segment(img, which)
        self.pick_largest(img)

        # compare against the ground truth mask
        img.compute_iou()
        return img.pred_mask

    def baseline(self, img, dark_is_object=True):
        """baseline: otsu on the plain gray image, keep the largest blob

        dark_is_object: True treats the dark side of the otsu split as the object
        """
        # clear anything left over from a previous run
        img.reset()

        # otsu on the plain gray image, no blur and no gradients
        flag = cv2.THRESH_BINARY_INV if dark_is_object else cv2.THRESH_BINARY
        img.otsu_thresh, img.binary = cv2.threshold(
            img.gray, 0, 255, flag + cv2.THRESH_OTSU)

        # keep the biggest blob and score it
        self.pick_largest(img)
        img.compute_iou()
        return img.pred_mask
