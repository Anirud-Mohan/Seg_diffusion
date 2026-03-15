#!/usr/bin/env python3
"""Train segmentation U-Net. Entry point from Segmentation_U_Net_training notebook."""
import argparse
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brats_diffusion.config import set_seed, DATA_ROOT
from brats_diffusion.training.train_unet import train_unet


def main():
    p = argparse.ArgumentParser(description="Train segmentation U-Net (5-class BraTS)")
    p.add_argument("--data_root", type=str, default=None, help="Path to brats_final_split (default: config DATA_ROOT)")
    p.add_argument("--output", type=str, default="unet_best_5class.pth", help="Output checkpoint path")
    p.add_argument("--epochs", type=int, default=19, help="Number of epochs")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint (e.g. after epoch 19, train to 45)")
    p.add_argument("--start_epoch", type=int, default=1, help="Start epoch when resuming")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()
    data_root = args.data_root or str(DATA_ROOT)
    set_seed(args.seed)
    train_unet(
        data_root=data_root,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_checkpoint=args.resume,
        start_epoch=args.start_epoch,
    )


if __name__ == "__main__":
    main()
