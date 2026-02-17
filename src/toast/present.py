import cv2
import numpy as np

def toast_present(roi_bgr: np.ndarray,
                  green_ratio_thresh: float = 0.002,   # ~0.2% of ROI pixels
                  dark_ratio_thresh: float = 0.10):    # ~10% dark pixels
    """
    Returns (present: bool, meta: dict)
    Cheap gate: toast achievements have a green title on a dark panel.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return False, {"green_ratio": 0.0, "dark_ratio": 0.0}

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Minecraft "green" title: usually bright green; allow some tolerance
    # Hue for green ~ 35-90, Saturation high, Value medium-high
    lower_g = np.array([35, 80, 80], dtype=np.uint8)
    upper_g = np.array([90, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower_g, upper_g)

    green_ratio = float(np.count_nonzero(green_mask)) / green_mask.size

    # Dark panel heuristic (toast background is dark-ish)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    dark_mask = (gray < 70).astype(np.uint8)
    dark_ratio = float(np.count_nonzero(dark_mask)) / dark_mask.size

    present = (green_ratio >= green_ratio_thresh) and (dark_ratio >= dark_ratio_thresh)
    return present, {"green_ratio": green_ratio, "dark_ratio": dark_ratio}
