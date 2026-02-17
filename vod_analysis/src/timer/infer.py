import numpy as np
import torch
import torch.nn.functional as F
from .model import TinyCharCNN, LABELS
from .roi import crop_timer_roi
from .segment import crop_and_normalize_chars

def parse_time_str(s: str):
    # Accept formats like "1:23.45" or "12:34.56"
    try:
        if ":" in s:
            mm, rest = s.split(":")
            if "." in rest:
                ss, cs = rest.split(".")
                return int(mm)*60 + int(ss) + int(cs)/100.0
            return int(mm)*60 + float(rest)
        return float(s)
    except Exception:
        return None

@torch.no_grad()
def infer_timer(frame_bgr, model: TinyCharCNN, device="cpu"):
    roi, bbox = crop_timer_roi(frame_bgr)
    char_imgs, boxes = crop_and_normalize_chars(roi, target_size=32)

    if len(char_imgs) == 0:
        return {"text": None, "seconds": None, "conf": 0.0, "bbox": bbox}

    x = np.stack(char_imgs).astype(np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(1).to(device)  # (N,1,32,32)

    logits = model(x)
    probs = F.softmax(logits, dim=1)
    confs, idxs = probs.max(dim=1)

    text = "".join(LABELS[i] for i in idxs.cpu().numpy())
    conf = float(confs.mean().cpu().item())
    seconds = parse_time_str(text)

    return {"text": text, "seconds": seconds, "conf": conf, "bbox": bbox}
