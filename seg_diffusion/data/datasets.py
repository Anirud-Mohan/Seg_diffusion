"""Dataset classes: BratsFlairSliceDataset (segmentation), BraTSDataset (diffusion)."""
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from seg_diffusion.config import IMG_SIZE, NUM_CLASSES
from seg_diffusion.data.utils import rgb_to_label


class BratsFlairSliceDataset(Dataset):
    """
    Paired FLAIR PNG + color-coded mask PNG.
    Expects root/split/flair and root/split/mask with matching .png filenames.
    """
    def __init__(self, root: Union[str, Path], split: str = "train"):
        self.root = Path(root)
        self.split = split
        self.items = []
        flair_dir = self.root / split / "flair"
        mask_dir = self.root / split / "mask"
        if not flair_dir.exists():
            flair_dir = Path(os.path.join(self.root, split, "flair"))
            mask_dir = Path(os.path.join(self.root, split, "mask"))
        flair_files = sorted(
            f for f in os.listdir(flair_dir) if f.lower().endswith(".png")
        )
        for fname in flair_files:
            img_path = flair_dir / fname if isinstance(flair_dir, Path) else Path(os.path.join(flair_dir, fname))
            mask_path = mask_dir / fname if isinstance(mask_dir, Path) else Path(os.path.join(mask_dir, fname))
            if (Path(img_path) if not isinstance(img_path, Path) else img_path).exists() and (
                Path(mask_path) if not isinstance(mask_path, Path) else mask_path
            ).exists():
                self.items.append((img_path, mask_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]
        img_path = Path(img_path) if not isinstance(img_path, Path) else img_path
        mask_path = Path(mask_path) if not isinstance(mask_path, Path) else mask_path

        img = Image.open(img_path).convert("L")
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)

        mask_rgb = np.array(Image.open(mask_path))
        if mask_rgb.ndim == 2:
            from seg_diffusion.config import DIFF_MASK_VALUE_TO_CLASS
            label = np.zeros_like(mask_rgb, dtype=np.int64)
            for src_val, cls in DIFF_MASK_VALUE_TO_CLASS.items():
                label[mask_rgb == src_val] = cls
            mask = torch.from_numpy(label).long()
        else:
            mask = rgb_to_label(mask_rgb)
            mask = torch.from_numpy(mask).long()
        return img, mask


class BraTSDataset(Dataset):
    """
    BraTS FLAIR + mask for segmentation-guided diffusion.
    Mask pixels mapped to classes 0-4, encoded as class/255.
    """
    def __init__(
        self,
        img_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        image_size: int = IMG_SIZE,
        use_ablated_segmentations: bool = False,
        ablation_prob: float = 0.0,
    ):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.use_ablated_segmentations = use_ablated_segmentations
        self.ablation_prob = ablation_prob
        self.filenames = sorted(
            f for f in os.listdir(self.img_dir) if f.lower().endswith(".png")
        )
        self.mapping = {
            0: 0,
            76: 4,
            29: 1,
            150: 2,
            226: 3,
        }
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = self.img_dir / fname
        mask_path = self.mask_dir / fname
        image = Image.open(img_path).convert("L")
        image = self.img_transform(image)
        mask = Image.open(mask_path).convert("L")
        mask = self.mask_transform(mask)
        mask_arr = np.array(mask)
        new_mask = np.zeros_like(mask_arr, dtype=np.float32)
        for src_val, target_cls in self.mapping.items():
            new_mask[mask_arr == src_val] = target_cls
        if self.use_ablated_segmentations and torch.rand(1).item() < self.ablation_prob:
            new_mask[new_mask == 4] = 0
        new_mask = new_mask / 255.0
        mask_tensor = torch.from_numpy(new_mask).unsqueeze(0)
        return {"images": image, "seg_mask": mask_tensor}
