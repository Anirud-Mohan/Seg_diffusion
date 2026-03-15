#!/usr/bin/env python3
"""Setup data for Model1 (copy flair to local). Optionally run external main.py for DDIM training."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brats_diffusion.config import DATA_ROOT, LOCAL_IMG_ROOT, BASE
from brats_diffusion.data import copy_flair_split, count_pngs


def main():
    p = argparse.ArgumentParser(description="Model1 setup: copy flair splits to local dir")
    p.add_argument("--data_root", type=str, default=None, help="Path to brats_final_split")
    p.add_argument("--local_root", type=str, default=None, help="Local destination for flair copies")
    p.add_argument("--run_main_py", type=str, default=None, help="If set, run this path as main.py with train args")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train_batch_size", type=int, default=24)
    args = p.parse_args()
    data_root = Path(args.data_root or DATA_ROOT)
    local_root = Path(args.local_root or LOCAL_IMG_ROOT)
    local_root.mkdir(parents=True, exist_ok=True)
    copy_flair_split(data_root, local_root)
    for split in ("train", "val", "test"):
        sp = local_root / split
        print(f"{split} pngs: {count_pngs(sp)}")
    if args.run_main_py:
        main_py = Path(args.run_main_py)
        if not main_py.exists():
            print(f"main.py not found at {main_py}; skipping run.")
            return
        cmd = [
            sys.executable,
            str(main_py),
            "--mode", "train",
            "--model_type", "DDIM",
            "--img_size", str(args.img_size),
            "--num_img_channels", "1",
            "--img_dir", str(local_root),
            "--train_batch_size", str(args.train_batch_size),
            "--num_epochs", str(args.epochs),
        ]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
