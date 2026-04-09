"""Localization modules
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3):
        super(VGG11Localizer, self).__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)      # [B, 512, 7, 7]
        x = self.regressor(x)    # [B, 4]
        return x
