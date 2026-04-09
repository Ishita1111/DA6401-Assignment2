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
    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3):
        super().__init__()

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

        # ---------------- LOAD CHECKPOINTS ----------------
        classifier_ckpt = torch.load(classifier_path, map_location="cpu")
        localizer_ckpt = torch.load(localizer_path, map_location="cpu")
        unet_ckpt = torch.load(unet_path, map_location="cpu")

        classifier_state = classifier_ckpt.get("state_dict", classifier_ckpt)
        localizer_state = localizer_ckpt.get("state_dict", localizer_ckpt)
        unet_state = unet_ckpt.get("state_dict", unet_ckpt)

        # ---------------- LOAD CLASSIFIER ----------------
        temp_cls = VGG11Classifier(num_breeds, in_channels)
        temp_cls.load_state_dict(classifier_state)

        self.encoder.load_state_dict(temp_cls.encoder.state_dict())
        self.classifier_model = temp_cls

        # ---------------- LOAD LOCALIZER ----------------
        temp_loc = VGG11Localizer(in_channels)
        temp_loc.load_state_dict(localizer_state)

        self.localizer_model = temp_loc

        # ---------------- LOAD SEGMENTATION ----------------
        self.segmenter = VGG11UNet(seg_classes, in_channels)
        self.segmenter.load_state_dict(unet_state)

    def forward(self, x):
        
        cls_out = self.classifier_model(x)
        bbox_out = self.localizer_model(x)

        _, _, H, W = x.shape
        scale = torch.tensor([W, H, W, H], device=bbox_out.device)
        bbox_out = bbox_out * scale

        seg_out = self.segmenter(x)

        return {
            "classification": cls_out,
            "localization": bbox_out,
            "segmentation": seg_out,
        }