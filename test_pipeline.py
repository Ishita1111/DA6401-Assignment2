from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
import torch

ds = OxfordIIITPetDataset("dataset")

sample = ds[0]

print("Image:", sample["image"].shape)
print("Label:", sample["label"])
print("BBox:", sample["bbox"])
print("Mask:", sample["mask"].shape)

model = MultiTaskPerceptionModel()

x = sample["image"].unsqueeze(0)
out = model(x)

print("Classification:", out["classification"].shape)
print("BBox:", out["localization"].shape)
print("Seg:", out["segmentation"].shape)
