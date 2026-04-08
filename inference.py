"""Inference and evaluation
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse

from models.multitask import MultiTaskPerceptionModel


def load_image(path, size=224):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size))

    img = np.array(img)
    img = torch.tensor(img).permute(2, 0, 1).float() / 255.0

    return img.unsqueeze(0)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskPerceptionModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    image = load_image(args.image_path).to(device)

    with torch.no_grad():
        outputs = model(image)

    # Classification
    cls_logits = outputs["classification"]
    pred_class = torch.argmax(cls_logits, dim=1).item()

    # Bounding box
    bbox = outputs["localization"].squeeze().cpu().numpy()

    # Segmentation
    seg_logits = outputs["segmentation"]
    seg_mask = torch.argmax(seg_logits, dim=1).squeeze().cpu().numpy()

    print(f"Predicted class: {pred_class}")
    print(f"Bounding box (xc, yc, w, h): {bbox}")
    print(f"Segmentation mask shape: {seg_mask.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)

    args = parser.parse_args()
    main(args)
    