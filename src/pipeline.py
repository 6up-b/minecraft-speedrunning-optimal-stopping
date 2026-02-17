import json
import cv2
import torch

from timer.model import TinyCharCNN
from timer.infer import infer_timer

from toast.roi import crop_toast_roi
from toast.present import toast_present
from toast.green_crop import green_text_crop
from toast.debounce import Debouncer
from toast.doctr_ocr import ToastOCRDoctr


def analyze_video(
    video_path: str,
    out_jsonl: str,
    timer_model_path: str,
    device: str = "gpu",
    sample_every_n_frames: int = 2,

    # Toast gate thresholds
    green_ratio_thresh: float = 0.002,
    dark_ratio_thresh: float = 0.10,

    # Debounce
    toast_min_conf: float = 0.55,
    toast_stable_frames: int = 6,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Timer model
    timer_model = TinyCharCNN()
    timer_model.load_state_dict(torch.load(timer_model_path, map_location=device))
    timer_model.to(device).eval()

    # Toast OCR (doctr)
    toast_ocr = ToastOCRDoctr(device=device)
    toast_debouncer = Debouncer(stable_frames=toast_stable_frames, min_conf=toast_min_conf)

    frame_idx = 0

    with open(out_jsonl, "w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if sample_every_n_frames > 1 and (frame_idx % sample_every_n_frames != 0):
                frame_idx += 1
                continue

            t_video = frame_idx / fps

            # --- TIMER ---
            timer = infer_timer(frame, timer_model, device=device)

            # --- TOAST ROI ---
            toast_roi, toast_bbox = crop_toast_roi(frame)

            # --- TOAST PRESENT GATE ---
            present, present_meta = toast_present(
                toast_roi,
                green_ratio_thresh=green_ratio_thresh,
                dark_ratio_thresh=dark_ratio_thresh,
            )

            if present:
                # --- GREEN TITLE ONLY CROP ---
                green_crop, green_meta = green_text_crop(toast_roi)

                if green_crop is not None:
                    ocr_res = toast_ocr.infer(green_crop)
                    fired = toast_debouncer.update(ocr_res.text, ocr_res.conf)

                    if fired is not None:
                        event = {
                            "video_path": video_path,
                            "t_video": float(t_video),
                            "frame_idx": int(frame_idx),
                            "fps": float(fps),

                            "igt_raw": timer["text"],
                            "t_igt": timer["seconds"],
                            "igt_conf": float(timer["conf"]),
                            "timer_bbox": timer["bbox"],

                            "event_type": "toast_achievement_title",
                            "event_text": fired["text"],
                            "event_conf": float(fired["conf"]),
                            "toast_bbox": toast_bbox,

                            # Debug/meta to tune thresholds later
                            "toast_present_meta": present_meta,
                            "green_crop_meta": green_meta,
                        }
                        f.write(json.dumps(event, ensure_ascii=False) + "\n")

            frame_idx += 1

    cap.release()
    print(f"Wrote events to {out_jsonl}")
