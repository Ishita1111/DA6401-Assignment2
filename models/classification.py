from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout
import torch
import torch.nn as nn


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super(VGG11Classifier, self).__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        # IMPORTANT: This head must match the provided checkpoint keys.
        # (Linear layers at indices 1, 4, 7 inside `self.classifier`.)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.classifier(x)
        return x