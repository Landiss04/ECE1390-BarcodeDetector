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

        self.gx = None
        self.gy = None
        self.blurred = None
        self.score = None
        self.binary = None
        self.closed = None
        self.opened = None
        self.pred_mask = None

        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError("Invalid Filepath")
        self.name = Path(image_path).stem
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.gt_mask = (cv2.IMREAD_GRAYSCALE(mask_path) > 0 if mask_path is not None else None)
            


        
class BarcodeDetector:
    """ Detects the barcodes from the images, applys different image processing methods.
   
    Attributes:
    """
    def __init__(self):
        pass    



