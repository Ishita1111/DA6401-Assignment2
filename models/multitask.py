"""Unified multi-task model
"""

import torch
import torch.nn as nn
import os
import gdown

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_breeds=37, seg_classes=3, in_channels=3):
        super().__init__()

        # ---------------- DOWNLOAD CHECKPOINTS ----------------
        classifier_path = "classifier.pth"
        localizer_path = "localizer.pth"
        unet_path = "unet.pth"

        if not os.path.exists(classifier_path):
            gdown.download(id="1ojjWWJ-XFozoCS1Tmxk8S873-Vk6YJRP", output=classifier_path, quiet=False)

        if not os.path.exists(localizer_path):
            gdown.download(id="1REslGGryB4ph8jFy4p9m0Je5OJ1JUhgC", output=localizer_path, quiet=False)

        if not os.path.exists(unet_path):
            gdown.download(id="1J1gCdlC8tWJiwE7gBcCBQg5bWT_CFip-", output=unet_path, quiet=False)

        # ---------------- LOAD MODELS ----------------
        self.classifier_model = VGG11Classifier(num_breeds, in_channels)
        self.localizer_model = VGG11Localizer(in_channels)
        self.segmenter = VGG11UNet(seg_classes, in_channels)

        # ---------------- LOAD WEIGHTS ----------------
        cls_ckpt = torch.load(classifier_path, map_location="cpu")
        loc_ckpt = torch.load(localizer_path, map_location="cpu")
        seg_ckpt = torch.load(unet_path, map_location="cpu")

        self.classifier_model.load_state_dict(cls_ckpt.get("state_dict", cls_ckpt))
        self.localizer_model.load_state_dict(loc_ckpt.get("state_dict", loc_ckpt))
        self.segmenter.load_state_dict(seg_ckpt.get("state_dict", seg_ckpt))

        # ---------------- SET TO EVAL MODE ----------------
        self.classifier_model.eval()
        self.localizer_model.eval()
        self.segmenter.eval()

    def forward(self, x):

        # ---------------- CLASSIFICATION ----------------
        cls_out = self.classifier_model(x)

        # ---------------- LOCALIZATION ----------------
        bbox_out = self.localizer_model(x)
        _, _, H, W = x.shape
        scale = torch.tensor([W, H, W, H], device=bbox_out.device, dtype=torch.float32)
        bbox_out = bbox_out * scale

        # ---------------- SEGMENTATION ----------------
        seg_out = self.segmenter(x)

        return {
            "classification": cls_out,
            "localization": bbox_out,
            "segmentation": seg_out,
        }