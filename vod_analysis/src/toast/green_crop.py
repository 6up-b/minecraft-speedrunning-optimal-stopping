import cv2
import numpy as np
from typing import Optional, Tuple, Dict

def green_text_crop(roi_bgr: np.ndarray,
                    pad: int = 6,
                    min_area: int = 150) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Returns (cropped_bgr_or_none, meta)
    - Crops ROI to bounding box of green text
    - Masks non-green pixels to black inside the crop
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None, {"reason": "empty"}

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    lower_g = np.array([35, 80, 80], dtype=np.uint8)
    upper_g = np.array([90, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_g, upper_g)

    # Clean up: fill small gaps in letters
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # Find connected components and bounding box union
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    h, w = mask.shape[:2]
    xs, ys, xe, ye = [], [], [], []
    total_area = 0

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < min_area:
            continue
        total_area += area
        xs.append(x); ys.append(y); xe.append(x + ww); ye.append(y + hh)

    if total_area == 0:
        return None, {"reason": "no_green_components"}

    x0 = max(0, min(xs) - pad)
    y0 = max(0, min(ys) - pad)
    x1 = min(w, max(xe) + pad)
    y1 = min(h, max(ye) + pad)

    crop = roi_bgr[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]

    # Mask out everything not green to remove white description lines
    crop[crop_mask == 0] = 0

    return crop, {
        "bbox": (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
        "green_area": int(total_area),
    }
