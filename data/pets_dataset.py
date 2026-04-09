"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class OxfordIIITPetDataset(Dataset):
    def __init__(self, root_dir, image_size=224, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.ann_dir = os.path.join(root_dir, "annotations")
        self.trimap_dir = os.path.join(self.ann_dir, "trimaps")
        self.xml_dir = os.path.join(self.ann_dir, "xmls")

        self.image_size = image_size
        self.transform = transform

        self.image_files = sorted(os.listdir(self.image_dir))
        
        self.image_files = sorted(os.listdir(self.image_dir))

        self.image_files = [
            f for f in self.image_files
            if os.path.exists(os.path.join(self.xml_dir, f.replace(".jpg", ".xml")))
        ]

        # Build label map
        breeds = sorted(list(set(["_".join(f.replace(".jpg", "").split("_")[:-1]) for f in self.image_files])))
        self.breed_to_idx = {b: i for i, b in enumerate(breeds)}

    def __len__(self):
        return len(self.image_files)

    def _load_bbox(self, xml_path, orig_w, orig_h):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        bbox = root.find("object").find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # convert to normalized (x_center, y_center, w, h)
        x_center = ((xmin + xmax) / 2) / orig_w
        y_center = ((ymin + ymax) / 2) / orig_h
        width = (xmax - xmin) / orig_w
        height = (ymax - ymin) / orig_h

        return torch.tensor([x_center, y_center, width, height], dtype=torch.float32)

    def _load_mask(self, mask_path):
        mask = Image.open(mask_path)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        mask = np.array(mask)

        mask = mask - 1

        return torch.tensor(mask, dtype=torch.long)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # paths
        img_path = os.path.join(self.image_dir, img_name)
        xml_path = os.path.join(self.xml_dir, img_name.replace(".jpg", ".xml"))
        mask_path = os.path.join(self.trimap_dir, img_name.replace(".jpg", ".png"))

        # image
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.image_size, self.image_size))

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

        # label
        breed = "_".join(img_name.replace(".jpg", "").split("_")[:-1])
        label = self.breed_to_idx[breed]

        # bbox
        bbox = self._load_bbox(xml_path, orig_w, orig_h)

        # mask
        mask = self._load_mask(mask_path)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "bbox": bbox,
            "mask": mask,
        }