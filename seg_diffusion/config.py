"""
Centralized configuration: paths, constants, training and evaluation config.
Use pathlib.Path and env vars or CLI overrides so the package runs outside Colab.
"""
import os
import random
from pathlib import Path

import numpy as np
import torch

# Default paths (override via env or CLI)
BASE = Path(os.environ.get("SEG_DIFFUSION_BASE", "."))
DATA_ROOT = BASE / os.environ.get("SEG_DIFFUSION_DATA", "brats_final_split")
RUNS = BASE / "runs"
LOCAL = Path(os.environ.get("SEG_DIFFUSION_LOCAL", "/tmp/seg_diffusion_work"))
LOCAL_IMG_ROOT = LOCAL / "brats_final_split_flair"

# Model and artifact identifiers (evaluation)
ENTITY = "challenger"
PROJECT = "brats-counterfactual-diffusion"
M1_ARTIFACT = f"{ENTITY}/{PROJECT}/model1_uncond_ddim:v0"
M2_ARTIFACT = f"{ENTITY}/{PROJECT}/checkpoint-epoch-60:v2"
M3_ARTIFACT = f"{ENTITY}/{PROJECT}/checkpoint-epoch-60:v0"
M1_DIR = Path(os.environ.get("SEG_DIFFUSION_M1_DIR", "/tmp/model_checkpoint/m1_checkpoint"))
M2_DIR = Path(os.environ.get("SEG_DIFFUSION_M2_DIR", "/tmp/model_checkpoint/m2_checkpoint"))
M3_DIR = Path(os.environ.get("SEG_DIFFUSION_M3_DIR", "/tmp/model_checkpoint/m3_checkpoint"))
SEG_UNET_CKPT = BASE / "checkpoints" / "unet_best_5class.pth"

# Dataset layout
VAL_IMG_DIR = DATA_ROOT / "val" / "flair"
VAL_MASK_DIR = DATA_ROOT / "val" / "mask"
TEST_IMG_DIR = DATA_ROOT / "test" / "flair"
TEST_MASK_DIR = DATA_ROOT / "test" / "mask"

IMG_SIZE = 256
NUM_CLASSES = 5

# RGB mask -> class index (Segmentation U-Net and Evaluation)
COLOR_TO_LABEL = {
    (0, 0, 0): 0,        # Background
    (0, 0, 255): 1,      # CSF
    (0, 255, 0): 2,      # Gray Matter
    (255, 255, 0): 3,    # White Matter
    (255, 0, 0): 4,      # Tumor
}

# Diffusion mask grayscale value -> class (Evaluation)
DIFF_MASK_VALUE_TO_CLASS = {
    0: 0,    # Background
    76: 4,   # Tumor
    29: 1,   # CSF
    150: 2,  # Gray Matter
    226: 3,  # White Matter
}

# Global evaluation configuration
EVAL_CFG = {
    "fid_kid": {
        "N_GEN": 1024,
        "BATCH": 16,
        "STEPS": 20,
    },
    "ssim": {
        "N_PAIRS": 256,
        "STEPS": 50,
    },
    "gen_seg": {
        "N_PAIRS": 256,
        "BATCH": 8,
        "STEPS": 50,
    },
}

SEED = 42


class TrainingConfig:
    """Training configuration for segmentation-guided diffusion (DL_project)."""
    TRAIN_IMG_DIR = "/content/dataset/train/flair"
    TRAIN_MASK_DIR = "/content/dataset/train/mask"
    VAL_IMG_DIR = "/content/dataset/val/flair"
    VAL_MASK_DIR = "/content/dataset/val/mask"
    OUTPUT_DIR = "/content/drive/MyDrive/SegGuidedDiff/output_full_mask_no_mat"
    model_type = "DDPM"
    image_size = 256
    train_batch_size = 64
    eval_batch_size = 64
    num_epochs = 200
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = "fp16"
    segmentation_guided = True
    segmentation_channel_mode = "single"
    num_segmentation_classes = 5
    use_ablated_segmentations = False
    ablation_prob = 0.0
    gradient_accumulation_steps = 1
    push_to_hub = False
    hub_private_repo = False
    overwrite_output_dir = True
    seed = 0
    class_conditional = False
    resume_epoch = None


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
