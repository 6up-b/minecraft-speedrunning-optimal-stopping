"""
Scan a VOD at 1 fps for toast template matches and output timestamps.
(Toast-only, no timer inference. For the full pipeline use src/pipeline.py)

Usage (from vod_analysis/):
    python scripts/find_toast_timestamps.py --video part538.mp4

Writes matched timestamps to a txt file next to the video.
"""
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from src.toast.template_match import crop_toast_roi, load_toast_templates, match_toast_templates


def fmt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def main():
    parser = argparse.ArgumentParser(description="Find toast template matches in a VOD")
    parser.add_argument("--video", required=True, help="Path to the VOD mp4")
    parser.add_argument("--toast_templates", default="templates/toast_templates",
                        help="Directory containing toast template PNGs")
    parser.add_argument("--min_score", type=float, default=0.55,
                        help="Minimum template match score")
    parser.add_argument("--out", default=None,
                        help="Output txt path (default: <video>_toast_matches.txt)")
    args = parser.parse_args()

    templates = load_toast_templates(args.toast_templates)
    print(f"Loaded {len(templates)} templates: {[t.name for t in templates]}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(round(fps)))  # ~1 fps sampling

    out_path = args.out or str(Path(args.video).stem + "_toast_matches.txt")

    matches = []
    frame_idx = 0

    pbar = tqdm(total=total_frames // frame_step, desc="Scanning", unit="frame")
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        roi, _ = crop_toast_roi(frame)
        results = match_toast_templates(roi, templates, min_score=args.min_score)

        for r in results:
            t = frame_idx / fps
            matches.append((t, frame_idx, r.name, r.score))
            pbar.set_postfix(hits=len(matches), last=fmt_timestamp(t))

        frame_idx += frame_step
        pbar.update(1)

    pbar.close()
    cap.release()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Toast template matches for {args.video}\n")
        f.write(f"# templates={args.toast_templates}  min_score={args.min_score}\n")
        f.write(f"# fps={fps}  total_frames={total_frames}\n")
        f.write(f"# {len(matches)} matches found\n\n")
        f.write(f"{'timestamp':<16} {'frame':>8} {'template':<30} {'score':>8}\n")
        f.write(f"{'-'*16} {'-'*8} {'-'*30} {'-'*8}\n")
        for t, fi, name, sc in matches:
            f.write(f"{fmt_timestamp(t):<16} {fi:>8d} {name:<30} {sc:>8.4f}\n")

    print(f"\n{len(matches)} matches written to {out_path}")


if __name__ == "__main__":
    main()
