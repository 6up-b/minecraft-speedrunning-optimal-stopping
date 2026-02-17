import numpy as np

def crop_toast_roi(frame_bgr: np.ndarray,
                   x0=0.48125, y0=0.7805555555555556, x1=0.675, y1=0.8638888888888889):
    h, w = frame_bgr.shape[:2]
    xa, ya = int(x0*w), int(y0*h)
    xb, yb = int(x1*w), int(y1*h)
    roi = frame_bgr[ya:yb, xa:xb].copy()
    return roi, (xa, ya, xb-xa, yb-ya)
