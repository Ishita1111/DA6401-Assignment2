"""Unified multi-task model
"""

import torch
import torch.nn as nn
import os
import gdown


from models.vgg11 import VGG11Encoder


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3):
        super(MultiTaskPerceptionModel, self).__init__()
        
        # checkpoint paths
        classifier_path = "classifier.pth"
        localizer_path = "localizer.pth"
        unet_path = "unet.pth"
        
        if not os.path.exists(classifier_path):
            gdown.download(id="18tzPDAapn1Lx7qW0tA6LXBJQPu6TCudm", output=classifier_path, quiet=False)

        if not os.path.exists(localizer_path):
            gdown.download(id="1PjiHWfVUwMvpRz3MidyIHM0I01TJaWGU", output=localizer_path, quiet=False)

        if not os.path.exists(unet_path):
            gdown.download(id="17u9sG04y0a7_wxedLdE_JJ3m45wTFHnb", output=unet_path, quiet=False)

        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_breeds)
        )

        # Localization head
        self.localizer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)
        )

        # Segmentation decoder (same idea as UNet, but inline)
                # 7 → 14
        self.up5 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 14 → 28
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256 + 512, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 28 → 56
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 56 → 112
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 112 → 224
        self.up1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # load checkpoints
        classifier_ckpt = torch.load(classifier_path, map_location="cpu")
        localizer_ckpt = torch.load(localizer_path, map_location="cpu")
        unet_ckpt = torch.load(unet_path, map_location="cpu")

        # handle both formats
        classifier_state = classifier_ckpt.get("state_dict", classifier_ckpt)
        localizer_state = localizer_ckpt.get("state_dict", localizer_ckpt)
        unet_state = unet_ckpt.get("state_dict", unet_ckpt)

        # load encoder (from classifier)
        self.encoder.load_state_dict(
            {k.replace("encoder.", ""): v for k, v in classifier_state.items() if k.startswith("encoder.")},
            strict=False
        )

        # load classifier head
        self.classifier.load_state_dict(
            {k.replace("classifier.", ""): v for k, v in classifier_state.items() if k.startswith("classifier.")},
            strict=False
        )

        # load localizer head
        self.localizer.load_state_dict(
            {k.replace("regressor.", ""): v for k, v in localizer_state.items() if k.startswith("regressor.")},
            strict=False
        )

        # load segmentation decoder
        self.load_state_dict(unet_state, strict=False)

        self.final_conv = nn.Conv2d(64, seg_classes, 1)

    def forward(self, x: torch.Tensor):

        bottleneck, feats = self.encoder(x, return_features=True)

        # Classification
        cls_out = self.classifier(bottleneck)

        # Localization
        bbox_out = self.localizer(bottleneck)

        # Segmentation decoder
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

        seg_out = self.final_conv(x)

        seg_out = self.final_conv(x)

        return {
            "classification": cls_out,
            "localization": bbox_out,
            "segmentation": seg_out,
        }
