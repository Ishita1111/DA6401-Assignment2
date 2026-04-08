"""Training entrypoint
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import argparse

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    ce_loss = nn.CrossEntropyLoss()
    iou_loss = IoULoss()
    seg_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        bboxes = batch["bbox"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)

        cls_out = outputs["classification"]
        bbox_out = outputs["localization"]
        seg_out = outputs["segmentation"]

        loss_cls = ce_loss(cls_out, labels)
        loss_bbox = iou_loss(bbox_out, bboxes)
        loss_seg = seg_loss_fn(seg_out, masks)

        loss = loss_cls + loss_bbox + loss_seg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OxfordIIITPetDataset(args.data_dir)
    dataset = torch.utils.data.Subset(dataset, range(10))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = MultiTaskPerceptionModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), args.save_path)
    print("Model saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default="model.pth")

    args = parser.parse_args()
    main(args)