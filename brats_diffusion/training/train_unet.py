"""U-Net segmentation training loop with validation Dice and checkpointing."""
import torch
from torch.utils.data import DataLoader

from brats_diffusion.data import BratsFlairSliceDataset
from brats_diffusion.metrics.dice import multiclass_dice
from brats_diffusion.models import UNetSeg


def train_unet(
    data_root: str,
    output_path: str,
    epochs: int = 19,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: torch.device = None,
    start_epoch: int = 1,
    resume_checkpoint: str = None,
):
    """
    Train segmentation U-Net; save best checkpoint to output_path.
    If resume_checkpoint is set, load it and train from start_epoch (e.g. 20) to epochs (e.g. 45).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = BratsFlairSliceDataset(data_root, "train")
    val_ds = BratsFlairSliceDataset(data_root, "val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = UNetSeg(in_channels=1, num_classes=5).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_dice = 0.0
    if resume_checkpoint:
        state = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(state)
        best_dice = 0.6894  # or pass as arg

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch} Train Loss: {train_loss / len(train_loader):.4f}")

        model.eval()
        val_d = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_d += multiclass_dice(preds, masks).item()
        val_d /= len(val_loader)
        print(f"Epoch {epoch} Val Dice: {val_d:.4f}")

        if val_d > best_dice:
            best_dice = val_d
            torch.save(model.state_dict(), output_path)
            print("Saved best model!")
    return best_dice
