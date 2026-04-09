"""Unified multi-task model
"""

import torch
import torch.nn as nn
import os
import gdown

from models.vgg11 import VGG11Encoder
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3):
        super(MultiTaskPerceptionModel, self).__init__()

        # ---------------- DOWNLOAD CHECKPOINTS ----------------
        classifier_path = "classifier.pth"
        localizer_path = "localizer.pth"
        unet_path = "unet.pth"

        if not os.path.exists(classifier_path):
            gdown.download(id="18tzPDAapn1Lx7qW0tA6LXBJQPu6TCudm", output=classifier_path, quiet=False)

        if not os.path.exists(localizer_path):
            gdown.download(id="1PjiHWfVUwMvpRz3MidyIHM0I01TJaWGU", output=localizer_path, quiet=False)

        if not os.path.exists(unet_path):
            gdown.download(id="17u9sG04y0a7_wxedLdE_JJ3m45wTFHnb", output=unet_path, quiet=False)

        # ---------------- SHARED ENCODER ----------------
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ---------------- CLASSIFICATION HEAD ----------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_breeds)
        )

        # ---------------- LOCALIZATION HEAD ----------------
        self.localizer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)
        )

        # ---------------- SEGMENTATION DECODER ----------------
        self.up5 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256 + 512, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(64, seg_classes, 1)

        # ---------------- LOAD CHECKPOINTS CORRECTLY ----------------
        classifier_ckpt = torch.load(classifier_path, map_location="cpu")
        localizer_ckpt = torch.load(localizer_path, map_location="cpu")
        unet_ckpt = torch.load(unet_path, map_location="cpu")

        classifier_state = classifier_ckpt.get("state_dict", classifier_ckpt)
        localizer_state = localizer_ckpt.get("state_dict", localizer_ckpt)
        unet_state = unet_ckpt.get("state_dict", unet_ckpt)

        # ---- CLASSIFIER ----
        temp_classifier = VGG11Classifier(num_breeds, in_channels)
        temp_classifier.load_state_dict(classifier_state)

        self.encoder.load_state_dict(temp_classifier.encoder.state_dict())
        self.classifier.load_state_dict(temp_classifier.classifier.state_dict())

        # ---- LOCALIZER ----
        temp_localizer = VGG11Localizer(in_channels)
        temp_localizer.load_state_dict(localizer_state)

        self.localizer.load_state_dict(temp_localizer.regressor.state_dict())

        # ---- SEGMENTATION ----
        temp_unet = VGG11UNet(seg_classes, in_channels)
        temp_unet.load_state_dict(unet_state)

        self.up5.load_state_dict(temp_unet.up5.state_dict())
        self.dec5.load_state_dict(temp_unet.dec5.state_dict())

        self.up4.load_state_dict(temp_unet.up4.state_dict())
        self.dec4.load_state_dict(temp_unet.dec4.state_dict())

        self.up3.load_state_dict(temp_unet.up3.state_dict())
        self.dec3.load_state_dict(temp_unet.dec3.state_dict())

        self.up2.load_state_dict(temp_unet.up2.state_dict())
        self.dec2.load_state_dict(temp_unet.dec2.state_dict())

        self.up1.load_state_dict(temp_unet.up1.state_dict())
        self.dec1.load_state_dict(temp_unet.dec1.state_dict())

        self.final_conv.load_state_dict(temp_unet.final_conv.state_dict())

    def forward(self, x: torch.Tensor):

        bottleneck, feats = self.encoder(x, return_features=True)

        # Classification
        cls_out = self.classifier(bottleneck)

        # Localization
        bbox_out = self.localizer(bottleneck)

        # Segmentation
        x = self.up5(bottleneck)
        x = torch.cat([x, feats["block5"]], dim=1)
        x = self.dec5(x)

        x = self.up4(x)
        x = torch.cat([x, feats["block4"]], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, feats["block3"]], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, feats["block2"]], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, feats["block1"]], dim=1)
        x = self.dec1(x)

        seg_out = self.final_conv(x)

        return {
            "classification": cls_out,
            "localization": bbox_out,
            "segmentation": seg_out,
        }