import cv2
import numpy as np
import torch
from torchvision import datasets


def yellow_score_mask(bgr: np.ndarray, thr: int = 40) -> np.ndarray:
    """
    score = min(R,G) - B
    mask = score >= thr
    returns uint8 mask {0,255}
    """
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    score = np.minimum(r, g) - b
    score = np.clip(score, 0, 255).astype(np.uint8)
    return (score >= thr).astype(np.uint8) * 255


def sobel_mag(gray_u8: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag, 0, 255)
    return mag.astype(np.uint8)


class TimerEdgeDataset(datasets.ImageFolder):
    """
    Expects folder structure:
      root/{0..9}/*.png   (digits only)

    Returns:
      x: torch.FloatTensor [3,32,32] in [0,1]
         x[0] = raw grayscale
         x[1] = edge magnitude computed from yellow-masked grayscale
         x[2] = yellow mask (binary-ish) as float
      y: int label
    """
    def __init__(self, root, yellow_thr: int = 40):
        super().__init__(root=root, transform=None)
        self.yellow_thr = int(yellow_thr)
        self.in_channels = 3  # handy for training script

        # Optional: enforce that this is digits-only (0..9)
        expected = {str(i) for i in range(10)}
        found = set(self.class_to_idx.keys())
        if found != expected:
            raise RuntimeError(
                f"TimerEdgeDataset expects digit folders 0..9 only.\n"
                f"Found: {sorted(found)}"
            )

    def __getitem__(self, idx):
        path, y = self.samples[idx]

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)

        # ensure size
        if bgr.shape[0] != 32 or bgr.shape[1] != 32:
            bgr = cv2.resize(bgr, (32, 32), interpolation=cv2.INTER_AREA)

        # Channel 0: raw grayscale (no masking)
        gray_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Yellow mask
        mask_u8 = yellow_score_mask(bgr, thr=self.yellow_thr)  # 0 or 255

        # Channel 1: edges computed from yellow-masked grayscale
        gray_masked = cv2.bitwise_and(gray_raw, gray_raw, mask=mask_u8)
        edge = sobel_mag(gray_masked)

        # Channel 2: yellow mask itself
        ymask = mask_u8  # already u8 0/255

        x0 = gray_raw.astype(np.float32) / 255.0
        x1 = edge.astype(np.float32) / 255.0
        x2 = ymask.astype(np.float32) / 255.0

        x = np.stack([x0, x1, x2], axis=0)  # [3,H,W]
        x = torch.from_numpy(x)

        return x, y
