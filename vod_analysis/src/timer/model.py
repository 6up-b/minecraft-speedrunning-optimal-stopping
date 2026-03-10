import torch
import torch.nn as nn

LABELS = [str(i) for i in range(10)]  # digits only

class TinyCharCNN(nn.Module):
    def __init__(self, num_classes=len(LABELS), in_channels=3, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout),  # always present; p=0 is a no-op
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)

    def load_state_dict_compat(self, state_dict, **kwargs):
        """Load state dict, remapping old checkpoints (pre-dropout) if needed."""
        if "net.6.weight" in state_dict and "net.7.weight" not in state_dict:
            # Old format: Conv2d was at net.6, now at net.7 due to Dropout2d at net.6
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith("net.6."):
                    remapped[k.replace("net.6.", "net.7.")] = v
                else:
                    remapped[k] = v
            state_dict = remapped
        return self.load_state_dict(state_dict, **kwargs)
