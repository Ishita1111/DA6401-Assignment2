"""Training entrypoint
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import argparse
from tqdm import tqdm

from data.pets_dataset import OxfordIIITPetDataset
from losses.iou_loss import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


def train_one_epoch(model, loader, optimizer, device, task):
    model.train()

    ce_loss = nn.CrossEntropyLoss()
    iou_loss = IoULoss()
    seg_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)

    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        bboxes = batch["bbox"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)

        if task == "classification":
            loss = ce_loss(outputs, labels)

        elif task == "localization":
            loss = iou_loss(outputs, bboxes)

        elif task == "segmentation":
            loss = seg_loss_fn(outputs, masks)

        else:
            raise ValueError("Invalid task")

        pbar.set_postfix({"loss": loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OxfordIIITPetDataset(args.data_dir)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Select model
    if args.task == "classification":
        model = VGG11Classifier().to(device)
    elif args.task == "localization":
        model = VGG11Localizer().to(device)
    elif args.task == "segmentation":
        model = VGG11UNet().to(device)
    else:
        raise ValueError(f"Invalid task: {args.task}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Auto filename (IMPORTANT)
    save_name = {
        "classification": "classifier.pth",
        "localization": "localizer.pth",
        "segmentation": "unet.pth"
    }[args.task]

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.task)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}")

    # Save checkpoint (autograder-safe)
    torch.save({
        "state_dict": model.state_dict(),
        "epoch": args.epochs
    }, save_name)

    print(f"Model saved as {save_name}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "localization", "segmentation"]
    )

    args = parser.parse_args()
    main(args)