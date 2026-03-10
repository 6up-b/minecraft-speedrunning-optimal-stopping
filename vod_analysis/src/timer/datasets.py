import random

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


def _bgr_to_3ch(bgr: np.ndarray, yellow_thr: int = 40) -> np.ndarray:
    """Convert 32x32 BGR to [3,32,32] float32 (gray, edge, yellow_mask)."""
    if bgr.shape[0] != 32 or bgr.shape[1] != 32:
        bgr = cv2.resize(bgr, (32, 32), interpolation=cv2.INTER_AREA)

    gray_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask_u8 = yellow_score_mask(bgr, thr=yellow_thr)
    gray_masked = cv2.bitwise_and(gray_raw, gray_raw, mask=mask_u8)
    edge = sobel_mag(gray_masked)

    x0 = gray_raw.astype(np.float32) / 255.0
    x1 = edge.astype(np.float32) / 255.0
    x2 = mask_u8.astype(np.float32) / 255.0
    return np.stack([x0, x1, x2], axis=0)


def _augment_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Apply augmentations to a 32x32 BGR crop BEFORE 3-channel preprocessing.
    All augmentations operate on the raw BGR image.
    """
    h, w = bgr.shape[:2]

    # 1) Scale jitter: resize [0.6, 1.4] then re-letterbox to 32x32
    scale = random.uniform(0.6, 1.4)
    new_w = max(4, int(round(w * scale)))
    new_h = max(4, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=interp)
    # Re-letterbox to 32x32
    out = np.zeros((32, 32, 3), dtype=np.uint8)
    fit_scale = min(32 / new_h, 32 / new_w)
    fw = int(round(new_w * fit_scale))
    fh = int(round(new_h * fit_scale))
    fw = max(1, min(fw, 32))
    fh = max(1, min(fh, 32))
    resized = cv2.resize(resized, (fw, fh), interpolation=cv2.INTER_AREA)
    y0 = (32 - fh) // 2
    x0 = (32 - fw) // 2
    out[y0:y0+fh, x0:x0+fw] = resized
    bgr = out

    # 2) Translation: random shift ±3px
    dx = random.randint(-3, 3)
    dy = random.randint(-3, 3)
    if dx != 0 or dy != 0:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        bgr = cv2.warpAffine(bgr, M, (32, 32), borderValue=(0, 0, 0))

    # 3) Brightness/contrast: brightness ±30, contrast [0.7, 1.3]
    alpha = random.uniform(0.7, 1.3)
    beta = random.uniform(-30, 30)
    bgr = np.clip(bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 4) Gaussian noise: sigma 0-5
    sigma = random.uniform(0, 5)
    if sigma > 0.5:
        noise = np.random.normal(0, sigma, bgr.shape).astype(np.float32)
        bgr = np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 5) Rotation: ±3 degrees (30% probability)
    if random.random() < 0.3:
        angle = random.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((16, 16), angle, 1.0)
        bgr = cv2.warpAffine(bgr, M, (32, 32), borderValue=(0, 0, 0))

    # 6) JPEG artifacts: encode at quality [50, 95] (50% probability)
    if random.random() < 0.5:
        quality = random.randint(50, 95)
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return bgr


class TimerEdgeAugDataset(datasets.ImageFolder):
    """
    Same as TimerEdgeDataset but with augmentation on the BGR crop
    BEFORE 3-channel preprocessing. For training only.
    """
    def __init__(self, root, yellow_thr: int = 40):
        super().__init__(root=root, transform=None)
        self.yellow_thr = int(yellow_thr)
        self.in_channels = 3

        expected = {str(i) for i in range(10)}
        found = set(self.class_to_idx.keys())
        if found != expected:
            raise RuntimeError(
                f"TimerEdgeAugDataset expects digit folders 0..9 only.\n"
                f"Found: {sorted(found)}"
            )

    def __getitem__(self, idx):
        path, y = self.samples[idx]

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)

        if bgr.shape[0] != 32 or bgr.shape[1] != 32:
            bgr = cv2.resize(bgr, (32, 32), interpolation=cv2.INTER_AREA)

        bgr = _augment_bgr(bgr)
        x = _bgr_to_3ch(bgr, self.yellow_thr)
        x = torch.from_numpy(x)

        return x, y
