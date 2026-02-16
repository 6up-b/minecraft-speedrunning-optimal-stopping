import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.timer.model import TinyCharCNN
from src.timer.datasets import TimerEdgeDataset  # 3-channel dataset


CLASS_NAMES = [str(i) for i in range(10)]


def build_class_weights(train_ds):
    # Handle any residual imbalance
    counts = np.zeros(len(train_ds.classes), dtype=np.int64)
    for _, y in train_ds.samples:
        counts[y] += 1
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/timer_digits_raw", help="Root with train/ and val/ (RGB crops)")
    ap.add_argument("--out", default="timer_model.pth", help="Where to save best weights")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=0)  # Windows-safe by default

    # edge dataset options
    ap.add_argument("--yellow_thr", type=int, default=40, help="Threshold for yellow_score_mask")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Expected train/ and val/ under data_root")

    train_ds = TimerEdgeDataset(str(train_dir), yellow_thr=args.yellow_thr)
    val_ds   = TimerEdgeDataset(str(val_dir),   yellow_thr=args.yellow_thr)

    print("Train class_to_idx:", train_ds.class_to_idx)
    print("Val   class_to_idx:", val_ds.class_to_idx)

    # Sanity: ensure consistent mapping between train and val
    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise RuntimeError("Train/val class folders mismatch. Ensure both have same class names.")

    # Sanity: ensure digits-only
    expected = {c: i for i, c in enumerate(CLASS_NAMES)}
    if train_ds.class_to_idx != expected:
        raise RuntimeError(f"Expected digit-only folders 0..9 with mapping {expected}, got {train_ds.class_to_idx}")

    n_classes = len(train_ds.classes)  # should be 10

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # NEW: 3-channel model
    model = TinyCharCNN(num_classes=n_classes, in_channels=3).to(args.device)

    class_weights = build_class_weights(train_ds).to(args.device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"train {epoch}/{args.epochs}"):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            train_loss += float(loss.item()) * y.numel()
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.numel()

        scheduler.step()

        train_acc = train_correct / max(1, train_total)
        train_loss = train_loss / max(1, train_total)

        # ---- Val ----
        model.eval()
        val_correct = 0
        val_total = 0
        all_t = []
        all_p = []

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"val {epoch}/{args.epochs}"):
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)
                logits = model(x)
                pred = logits.argmax(1)

                val_correct += (pred == y).sum().item()
                val_total += y.numel()

                all_t.extend(y.cpu().numpy().tolist())
                all_p.extend(pred.cpu().numpy().tolist())

        val_acc = val_correct / max(1, val_total)
        cm = confusion_matrix(all_t, all_p, n_classes)

        print(f"\nEpoch {epoch}")
        print(f"  train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
        print(f"  val_acc={val_acc:.4f}  lr={opt.param_groups[0]['lr']:.6f}")

        # Print top confusions
        cm_off = cm.copy()
        np.fill_diagonal(cm_off, 0)
        flat = cm_off.flatten()
        top_idx = flat.argsort()[-8:][::-1]
        print("  Top confusions:")
        for idx in top_idx:
            c = flat[idx]
            if c == 0:
                break
            i = idx // n_classes
            j = idx % n_classes
            print(f"    {train_ds.classes[i]} -> {train_ds.classes[j]} : {c}")

        # Save best
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"✅ Saved best model to {args.out} (val_acc={best_val:.4f})")

    print(f"\nDone. Best val_acc={best_val:.4f}. Model at: {args.out}")


if __name__ == "__main__":
    main()
