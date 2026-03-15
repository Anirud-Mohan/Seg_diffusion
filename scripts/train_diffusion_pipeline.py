#!/usr/bin/env python3
"""Train segmentation-guided diffusion. Entry point from DL_project_training_pipeline_no_MAT notebook."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brats_diffusion.config import TrainingConfig, set_seed
from brats_diffusion.training.train_diffusion import run_diffusion_training


def main():
    p = argparse.ArgumentParser(description="Train segmentation-guided diffusion")
    p.add_argument("--train_img_dir", type=str, default=None)
    p.add_argument("--train_mask_dir", type=str, default=None)
    p.add_argument("--val_img_dir", type=str, default=None)
    p.add_argument("--val_mask_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    set_seed(args.seed)
    config = TrainingConfig()
    if args.train_img_dir:
        config.TRAIN_IMG_DIR = args.train_img_dir
    if args.train_mask_dir:
        config.TRAIN_MASK_DIR = args.train_mask_dir
    if args.val_img_dir:
        config.VAL_IMG_DIR = args.val_img_dir
    if args.val_mask_dir:
        config.VAL_MASK_DIR = args.val_mask_dir
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    config.num_epochs = args.num_epochs
    config.train_batch_size = args.train_batch_size
    config.learning_rate = args.learning_rate
    config.seed = args.seed
    run_diffusion_training(config=config)


if __name__ == "__main__":
    main()
