import cv2
import torch
import random
import argparse
from src.timer.model import TinyCharCNN
from src.timer.infer import load_timer_layout, infer_timer

device = "cuda" if torch.cuda.is_available() else "cpu"

layout = load_timer_layout("configs/timer_layout_from_anchor.json")
ap = argparse.ArgumentParser()
ap.add_argument("--video", default="input_lowres.mp4", help="Input video path")
ap.add_argument("--frame", type=int, default=None, help="Optional absolute frame index (overrides random)")
args = ap.parse_args()


model = TinyCharCNN(num_classes=10, in_channels=3)
model.load_state_dict_compat(torch.load("timer_model.pth", map_location=device))
model.eval()
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {args.video}")

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
# Choose frame
frame_idx = args.frame if args.frame is not None else random.randrange(0, total)
frame_idx = max(0, min(total - 1, frame_idx))

# Seek + read
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ok, frame = cap.read()
cap.release()
if not ok or frame is None:
    raise RuntimeError(f"Failed to read frame {frame_idx} from {args.video}")
out = infer_timer(frame, model, layout, device=device, pad_px=0)
print(out["text"], out["conf"], out["seconds"])
