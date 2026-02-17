import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

from model import TinyCharCNN

def main():
    ds = datasets.ImageFolder(
        "../../data/timer_chars/train",
        transform=transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])
    )

    # take a small fixed subset (e.g., 512 samples)
    idx = np.random.RandomState(0).choice(len(ds), size=min(512, len(ds)), replace=False)
    sub = Subset(ds, idx)

    dl = DataLoader(sub, batch_size=128, shuffle=True, num_workers=0)

    model = TinyCharCNN(num_classes=len(ds.classes)).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(1, 501):
        x, y = next(iter(dl))
        x, y = x.cuda(), y.cuda()

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"step={step} loss={loss.item():.3f} acc={acc:.3f}")

if __name__ == "__main__":
    main()


