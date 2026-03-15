#!/usr/bin/env python3
"""Run full evaluation (FID/KID, SSIM, Dice/IoU, tumor residual, diff-map IoU)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seg_diffusion.config import DATA_ROOT, M1_DIR, M2_DIR, M3_DIR, set_seed
from seg_diffusion.evaluation.eval_metrics import run_full_evaluation


def main():
    p = argparse.ArgumentParser(description="Run evaluation for M1/M2/M3 and segmentation U-Net")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--m1_dir", type=str, default=None)
    p.add_argument("--m2_dir", type=str, default=None)
    p.add_argument("--m3_dir", type=str, default=None)
    p.add_argument("--seg_ckpt", type=str, default=None, help="Path to unet_best_5class.pth")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    data_root = Path(args.data_root) if args.data_root else DATA_ROOT
    m1_dir = Path(args.m1_dir) if args.m1_dir else M1_DIR
    m2_dir = Path(args.m2_dir) if args.m2_dir else M2_DIR
    m3_dir = Path(args.m3_dir) if args.m3_dir else M3_DIR
    seg_ckpt = Path(args.seg_ckpt) if args.seg_ckpt else (data_root.parent / "checkpoints" / "unet_best_5class.pth")
    run_full_evaluation(
        data_root=data_root,
        m1_dir=m1_dir,
        m2_dir=m2_dir,
        m3_dir=m3_dir,
        seg_ckpt=seg_ckpt,
    )


if __name__ == "__main__":
    main()
