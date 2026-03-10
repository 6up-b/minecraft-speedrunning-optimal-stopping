"""
Debug: test template matching around a known match point (18:42 in part538.mp4).
Try both green mask and grayscale approaches at multiple scales.
"""
import json
import cv2
import numpy as np
from pathlib import Path

Path("debug").mkdir(exist_ok=True)

# HSV config
cfg = json.loads(open("configs/hsv_config.json").read())
lower = np.array(cfg["lower_green"], dtype=np.uint8)
upper = np.array(cfg["upper_green"], dtype=np.uint8)

# Template
tmpl = cv2.imread("templates/we_need_to_go_template.png")
tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
tmpl_hsv = cv2.cvtColor(tmpl, cv2.COLOR_BGR2HSV)
tmpl_mask = cv2.inRange(tmpl_hsv, lower, upper)
print(f"Template: {tmpl.shape}")
print(f"Template green mask: {np.count_nonzero(tmpl_mask)}/{tmpl_mask.size} px")

# Save template debug
cv2.imwrite("debug/tmpl_bgr.png", tmpl)
cv2.imwrite("debug/tmpl_gray.png", tmpl_gray)
cv2.imwrite("debug/tmpl_green_mask.png", tmpl_mask)

# Toast ROI from roi.json
ROI_X0, ROI_Y0 = 0.2359375, 0.7064814814814815
ROI_X1, ROI_Y1 = 0.653125, 0.9194444444444444

cap = cv2.VideoCapture("part538.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {total} frames, {fps} fps, {total/fps:.0f}s")

# Known match at 18:42 = 1122s
target_s = 18 * 60 + 42
target_frame = int(target_s * fps)
print(f"Target: {target_s}s = frame {target_frame}")

# Sample frames around the target (+/- 10s at 1fps)
scales = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0)
step = max(1, int(round(fps)))

print(f"\n--- Testing frames around 18:42 ({target_s-10}s to {target_s+10}s) ---")
for offset in range(-10, 11):
    t = target_s + offset
    fi = int(t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    roi = frame[int(ROI_Y0*h):int(ROI_Y1*h), int(ROI_X0*w):int(ROI_X1*w)]

    # Green mask match
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_mask = cv2.inRange(roi_hsv, lower, upper)
    green_ratio = np.count_nonzero(roi_mask) / roi_mask.size

    best_green = (0, None, None)
    best_gray = (0, None, None)

    for s in scales:
        tw = int(round(tmpl.shape[1] * s))
        th = int(round(tmpl.shape[0] * s))
        if tw < 4 or th < 4:
            continue
        if th >= roi.shape[0] or tw >= roi.shape[1]:
            continue

        # Green mask
        ts_mask = cv2.resize(tmpl_mask, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(roi_mask, ts_mask, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if mx > best_green[0]:
            best_green = (mx, ml, s)

        # Grayscale
        ts_gray = cv2.resize(tmpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), ts_gray, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if mx > best_gray[0]:
            best_gray = (mx, ml, s)

    tag = " <<<" if best_green[0] > 0.4 or best_gray[0] > 0.4 else ""
    print(f"  t={t:>5d}s  green_ratio={green_ratio:.4f}  green_score={best_green[0]:.4f}@s={best_green[2]}  gray_score={best_gray[0]:.4f}@s={best_gray[2]}{tag}")

    # Save debug for frames near target
    if abs(offset) <= 2:
        cv2.imwrite(f"debug/target_{t}s_roi.png", roi)
        cv2.imwrite(f"debug/target_{t}s_mask.png", roi_mask)
        cv2.imwrite(f"debug/target_{t}s_full.png", frame)

cap.release()
print("\nDone. Check debug/ for saved images.")
