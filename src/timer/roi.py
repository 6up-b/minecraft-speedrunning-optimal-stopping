import numpy as np

def crop_timer_roi(frame_bgr: np.ndarray,
                   x0= 0.884375, y0=0.075, x1=0.984375, y1=0.10833333333333334):
    h, w = frame_bgr.shape[:2]
    xa, ya = int(x0*w), int(y0*h)
    xb, yb = int(x1*w), int(y1*h)
    roi = frame_bgr[ya:yb, xa:xb].copy()
    return roi, (xa, ya, xb-xa, yb-ya)
