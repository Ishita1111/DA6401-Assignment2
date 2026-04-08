"""Segmentation model
"""

import torch
import torch.nn as nn

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super(VGG11UNet, self).__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Upsampling layers
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # 14 → 28
        self.dec5 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 28 → 56
        self.dec4 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 56 → 112
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 112 → 224
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.up1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, feats = self.encoder(x, return_features=True)

        # Decoder
        # 7 → 14
        x = self.up5(bottleneck)
        x = torch.cat([x, feats["block5"]], dim=1)
        x = self.dec5(x)

        # 14 → 28
        x = self.up4(x)
        x = torch.cat([x, feats["block4"]], dim=1)
        x = self.dec4(x)

        # 28 → 56
        x = self.up3(x)
        x = torch.cat([x, feats["block3"]], dim=1)
        x = self.dec3(x)

        # 56 → 112
        x = self.up2(x)
        x = torch.cat([x, feats["block2"]], dim=1)
        x = self.dec2(x)

        # 112 → 224
        x = self.up1(x)
        x = torch.cat([x, feats["block1"]], dim=1)
        x = self.dec1(x)

        # Final segmentation map
        x = self.final_conv(x)

        return x
