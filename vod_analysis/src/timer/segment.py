import cv2
import numpy as np
from typing import List, Tuple

def preprocess_for_cc(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    # Boost contrast a bit
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # HUD digits are typically bright invert so digits become white
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # if digits came out dark-on-light, invert
    if np.mean(bw) > 127:
        bw = 255 - bw

    # Cleanup
    bw = cv2.medianBlur(bw, 3)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), iterations=1)
    return bw

def segment_char_boxes(bw: np.ndarray) -> List[Tuple[int,int,int,int]]:
    # Connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    boxes = []
    h, w = bw.shape[:2]

    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        # filter tiny noise
        if area < 15:
            continue
        # Basic size constraints (tune per your ROI scale)
        if hh < 0.35*h or hh > 0.98*h:
            continue
        if ww < 2 or ww > 0.35*w:
            continue
        boxes.append((x, y, ww, hh))

    boxes.sort(key=lambda b: b[0])
    return boxes

def crop_and_normalize_chars(roi_bgr: np.ndarray,
                             target_size=32):
    bw = preprocess_for_cc(roi_bgr)
    boxes = segment_char_boxes(bw)

    chars = []
    for (x,y,w,h) in boxes:
        crop = bw[y:y+h, x:x+w]
        # Pad to square then resize
        pad = 4
        crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        hh, ww = crop.shape[:2]
        m = max(hh, ww)
        sq = np.zeros((m, m), dtype=np.uint8)
        y0 = (m - hh)//2
        x0 = (m - ww)//2
        sq[y0:y0+hh, x0:x0+ww] = crop
        sq = cv2.resize(sq, (target_size, target_size), interpolation=cv2.INTER_AREA)
        chars.append(sq)

    return chars, boxes
