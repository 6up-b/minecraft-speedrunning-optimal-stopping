"""
Save the toast ROI and green-filtered crop for a chosen video frame.

Examples:
    python scripts/dump_toast_debug.py --video part538.mp4 --timestamp 32:00
    python scripts/dump_toast_debug.py --video part538.mp4 --seconds 1920
    python scripts/dump_toast_debug.py --video part538.mp4 --frame 115200
"""
import argparse
from pathlib import Path
from typing import Optional

import cv2

from src.toast.green_crop import green_text_crop, load_hsv_config
from src.toast.template_match import crop_toast_roi


def parse_timestamp(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 2:
        mm, ss = parts
        return int(mm) * 60 + float(ss)
    if len(parts) == 3:
        hh, mm, ss = parts
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    raise ValueError(f"Unsupported timestamp format: {value}")


def resolve_frame_index(
    fps: float,
    frame: Optional[int],
    seconds: Optional[float],
    timestamp: Optional[str],
) -> int:
    provided = [frame is not None, seconds is not None, timestamp is not None]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of --frame, --seconds, or --timestamp")

    if frame is not None:
        return int(frame)
    if seconds is not None:
        return int(round(seconds * fps))
    return int(round(parse_timestamp(timestamp) * fps))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump toast ROI and green crop for a video frame")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--frame", type=int, default=None, help="Absolute frame index")
    parser.add_argument("--seconds", type=float, default=None, help="Video time in seconds")
    parser.add_argument("--timestamp", default=None, help="Video time as MM:SS or HH:MM:SS(.sss)")
    parser.add_argument("--hsv_config", default="configs/hsv_config.json",
                        help="Path to HSV config for green filtering")
    parser.add_argument("--min_green_area", type=int, default=150,
                        help="Minimum connected green area for green crop extraction")
    parser.add_argument("--out_dir", default="debug",
                        help="Directory to write debug images into")
    parser.add_argument("--prefix", default="toast_debug",
                        help="Filename prefix for outputs")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0.0 or total_frames <= 0:
        cap.release()
        raise RuntimeError("Could not read video metadata.")

    frame_idx = resolve_frame_index(
        fps=fps,
        frame=args.frame,
        seconds=args.seconds,
        timestamp=args.timestamp,
    )
    frame_idx = max(0, min(total_frames - 1, frame_idx))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {args.video}")

    roi_bgr, roi_xywh = crop_toast_roi(frame)
    lower_green, upper_green = load_hsv_config(args.hsv_config)
    green_crop, meta = green_text_crop(
        roi_bgr,
        min_area=args.min_green_area,
        lower_green=lower_green,
        upper_green=upper_green,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_path = out_dir / f"{args.prefix}_roi.png"
    cv2.imwrite(str(roi_path), roi_bgr)

    green_path = None
    if green_crop is not None:
        green_path = out_dir / f"{args.prefix}_green_crop.png"
        cv2.imwrite(str(green_path), green_crop)

    t_seconds = frame_idx / fps
    print(f"video={args.video}")
    print(f"frame={frame_idx}/{total_frames - 1}")
    print(f"fps={fps:.3f}")
    print(f"timestamp_s={t_seconds:.3f}")
    print(f"toast_roi_xywh={roi_xywh}")
    print(f"saved_roi={roi_path}")
    if green_path is not None:
        print(f"saved_green_crop={green_path}")
        print(f"green_meta={meta}")
    else:
        print("saved_green_crop=None")
        print(f"green_meta={meta}")


if __name__ == "__main__":
    main()
