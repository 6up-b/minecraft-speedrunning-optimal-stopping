import argparse
import json
import cv2

def clamp01(x): 
    return max(0.0, min(1.0, x))

def rect_to_norm(rect, w, h):
    x, y, ww, hh = rect
    return {
        "x0": clamp01(x / w),
        "y0": clamp01(y / h),
        "x1": clamp01((x + ww) / w),
        "y1": clamp01((y + hh) / h),
    }

def draw_roi(frame, rect, label):
    if rect is None:
        return
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, max(0, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

def read_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return ok, frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--out", default="configs/roi.json", help="Output JSON path")
    ap.add_argument("--jump", type=int, default=0, help="Start at frame index (absolute)")
    ap.add_argument("--step", type=int, default=300, help="N for forward/back jumps (frames)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    frame_idx = max(0, min(args.jump, max(0, total-1)))
    step_n = max(1, args.step)

    timer_rect = None
    toast_rect = None

    window = "ROI Calibrator (t=timer, o=toast, s=save, f/b jump, q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def refresh():
        ok, frame = read_frame(cap, frame_idx)
        if not ok:
            return False, None
        overlay = frame.copy()
        draw_roi(overlay, timer_rect, "TIMER")
        draw_roi(overlay, toast_rect, "TOAST")
        text = f"frame {frame_idx}/{max(0,total-1)}  fps={fps:.2f}  step={step_n}"
        cv2.putText(overlay, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow(window, overlay)
        return True, frame

    try:
        while True:
            ok, frame = refresh()
            if not ok:
                print("Could not read frame (end of video or decode issue).")
                break

            key = cv2.waitKey(0) & 0xFF

            # Quit
            if key in (27, ord('q')):  # ESC or q
                break

            # Single-step
            if key in (ord('d'), 83):  # d or right arrow
                frame_idx = min(frame_idx + 1, max(0, total-1))
            elif key in (ord('a'), 81):  # a or left arrow
                frame_idx = max(frame_idx - 1, 0)

            # Jump +/- step_n
            elif key == ord('f'):
                frame_idx = min(frame_idx + step_n, max(0, total-1))
            elif key == ord('b'):
                frame_idx = max(frame_idx - step_n, 0)

            # Jump to absolute frame
            elif key == ord('g'):
                s = input(f"Go to frame [0..{max(0,total-1)}]: ").strip()
                if s.isdigit():
                    frame_idx = max(0, min(int(s), max(0, total-1)))

            # Select ROIs (uses OpenCV’s built-in selector)
            elif key == ord('t'):
                r = cv2.selectROI(window, frame, fromCenter=False, showCrosshair=True)
                if r and r[2] > 0 and r[3] > 0:
                    timer_rect = tuple(map(int, r))
            elif key == ord('o'):
                r = cv2.selectROI(window, frame, fromCenter=False, showCrosshair=True)
                if r and r[2] > 0 and r[3] > 0:
                    toast_rect = tuple(map(int, r))

            # Save
            elif key == ord('s'):
                if timer_rect is None or toast_rect is None:
                    print("Select both TIMER (t) and TOAST (o) before saving.")
                    continue

                h, w = frame.shape[:2]
                cfg = {
                    "timer": rect_to_norm(timer_rect, w, h),
                    "toast": rect_to_norm(toast_rect, w, h),
                    "meta": {
                        "frame_wh": [w, h],
                        "fps": fps,
                        "calibrated_at_frame": frame_idx,
                        "jump_step_frames": step_n
                    }
                }
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
                print(f"Saved ROI config to {args.out}")

            # If user types digits, treat as "set step"
            elif 48 <= key <= 57:  # '0'..'9'
                s = input(f"Set jump step (frames), current={step_n}: ").strip()
                if s.isdigit() and int(s) > 0:
                    step_n = int(s)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
