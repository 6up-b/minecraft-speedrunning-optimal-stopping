import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.timer.model import TinyCharCNN
from src.timer.datasets import TimerEdgeDataset, TimerEdgeAugDataset


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


def save_misclassified(val_ds, model, device, out_dir: Path, max_per_class=20):
    """Save misclassified validation images for inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    counts = {}

    with torch.no_grad():
        for i in range(len(val_ds)):
            x, y_true = val_ds[i]
            x_batch = x.unsqueeze(0).to(device)
            logits = model(x_batch)
            y_pred = logits.argmax(1).item()

            if y_pred != y_true:
                key = f"true{y_true}_pred{y_pred}"
                counts[key] = counts.get(key, 0) + 1
                if counts[key] <= max_per_class:
                    path = val_ds.samples[i][0]
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    if img is not None:
                        fname = f"{key}_{counts[key]}.png"
                        cv2.imwrite(str(out_dir / fname), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/timer_digits_v2", help="Root with train/ and val/ (RGB crops)")
    ap.add_argument("--out", default="timer_model.pth", help="Where to save best weights")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=0)  # Windows-safe by default

    # edge dataset options
    ap.add_argument("--yellow_thr", type=int, default=40, help="Threshold for yellow_score_mask")

    # new options
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                    help="Use augmented dataset for training (default: True)")
    ap.add_argument("--dropout", type=float, default=0.1, help="Dropout2d probability after 2nd maxpool")
    ap.add_argument("--save_misclassified", action=argparse.BooleanOptionalAction, default=True,
                    help="Save misclassified val images to debug/misclassified/")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Expected train/ and val/ under data_root")

    # Training: augmented or plain
    if args.augment:
        print("Using TimerEdgeAugDataset (with augmentation) for training")
        train_ds = TimerEdgeAugDataset(str(train_dir), yellow_thr=args.yellow_thr)
    else:
        print("Using TimerEdgeDataset (no augmentation) for training")
        train_ds = TimerEdgeDataset(str(train_dir), yellow_thr=args.yellow_thr)

    # Validation: always plain (no augmentation)
    val_ds = TimerEdgeDataset(str(val_dir), yellow_thr=args.yellow_thr)

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

    # Print class distribution
    train_counts = np.zeros(n_classes, dtype=np.int64)
    for _, y in train_ds.samples:
        train_counts[y] += 1
    val_counts = np.zeros(n_classes, dtype=np.int64)
    for _, y in val_ds.samples:
        val_counts[y] += 1
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Train per class: {dict(zip(CLASS_NAMES, train_counts.tolist()))}")
    print(f"Val   per class: {dict(zip(CLASS_NAMES, val_counts.tolist()))}")

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

    model = TinyCharCNN(num_classes=n_classes, in_channels=3, dropout=args.dropout).to(args.device)
    print(f"Model: TinyCharCNN(dropout={args.dropout})")

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
            print(f"  Saved best model to {args.out} (val_acc={best_val:.4f})")

    # Save misclassified images from best model
    if args.save_misclassified and best_val < 1.0:
        print("\nSaving misclassified validation images...")
        model.load_state_dict(torch.load(args.out, map_location=args.device))
        save_misclassified(val_ds, model, args.device, Path("debug/misclassified"))
        print("Saved to debug/misclassified/")

    print(f"\nDone. Best val_acc={best_val:.4f}. Model at: {args.out}")


if __name__ == "__main__":
    main()
