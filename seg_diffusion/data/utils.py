"""Data utilities: rgb_to_label, copy_flair_split, count_pngs, collect_paired_paths, load_image, etc."""
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from seg_diffusion.config import COLOR_TO_LABEL, DATA_ROOT, DIFF_MASK_VALUE_TO_CLASS, IMG_SIZE


def rgb_to_label(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB mask (H,W,3) to integer label image (H,W) in [0..NUM_CLASSES-1]."""
    h, w = mask_rgb.shape[0], mask_rgb.shape[1]
    if mask_rgb.ndim == 3:
        label = np.zeros((h, w), dtype=np.int64)
        for rgb, cls in COLOR_TO_LABEL.items():
            label[(mask_rgb == np.array(rgb)).all(axis=-1)] = cls
    else:
        label = np.zeros((h, w), dtype=np.int64)
        for src_val, target_cls in DIFF_MASK_VALUE_TO_CLASS.items():
            label[mask_rgb == src_val] = target_cls
    return label


def copy_flair_split(
    data_root: Union[str, Path],
    local_img_root: Union[str, Path],
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> None:
    """Copy flair slices from data_root/split/flair to local_img_root/split for faster I/O."""
    data_root = Path(data_root)
    local_img_root = Path(local_img_root)
    for split in splits:
        src = data_root / split / "flair"
        dst = local_img_root / split
        if dst.exists() and list(dst.iterdir()):
            print(f"{split} already copied to local: {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying {split} flair slices from {src} -> {dst} ...")
        shutil.copytree(str(src), str(dst))
        print("Done:", split)


def count_pngs(folder: Union[str, Path]) -> int:
    """Count total .png files under folder."""
    folder = Path(folder)
    total = 0
    for root, _, files in os.walk(folder):
        total += sum(1 for f in files if f.lower().endswith(".png"))
    return total


def collect_paired_paths(
    img_dir: Path,
    mask_dir: Path,
    limit: int = None,
) -> Tuple[List[Path], List[Path]]:
    """
    Return filename-matched pairs of (flair, mask) as lists of Paths.
    Only keep files where both flair and mask exist.
    """
    all_imgs = sorted(img_dir.glob("*.png"))
    all_masks = sorted(mask_dir.glob("*.png"))
    mask_lookup = {m.name: m for m in all_masks}
    paired_imgs = []
    paired_masks = []
    for img_path in all_imgs:
        fname = img_path.name
        if fname in mask_lookup:
            paired_imgs.append(img_path)
            paired_masks.append(mask_lookup[fname])
    if limit is not None:
        paired_imgs = paired_imgs[:limit]
        paired_masks = paired_masks[:limit]
    return paired_imgs, paired_masks


# Default transforms for evaluation (can be overridden)
_img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_image(path: Path, img_size: int = IMG_SIZE) -> torch.Tensor:
    """Load a FLAIR slice as tensor (1,H,W), normalized to [0,1]."""
    tr = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("L")
    return tr(img)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert tensor (1,1,H,W) or (1,H,W) in [0,1] to grayscale PIL image."""
    if t.dim() == 4:
        t = t[0, 0]
    elif t.dim() == 3:
        t = t[0]
    arr = t.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def load_diffusion_mask(mask_path: Path, device: torch.device, img_size: int = IMG_SIZE) -> torch.Tensor:
    """
    Load diffusion mask: map BraTS grayscale values to class indices, encode as cls/255.
    Returns (1,1,H,W) on device.
    """
    mask_img = Image.open(mask_path).convert("L")
    mask_img = mask_img.resize((img_size, img_size), Image.NEAREST)
    mask_arr = np.array(mask_img, dtype=np.int64)
    cls_arr = np.zeros_like(mask_arr, dtype=np.float32)
    for src_val, target_cls in DIFF_MASK_VALUE_TO_CLASS.items():
        cls_arr[mask_arr == src_val] = float(target_cls)
    norm = cls_arr / 255.0
    return torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)


# --- From DL_project: segmentation-guided diffusion ---

def ablate_masks(segs: torch.Tensor, num_segmentation_classes: int = 5) -> torch.Tensor:
    """Randomly remove class label(s) from segs with 0.5 probability per non-BG class."""
    segs = segs.clone()
    class_removals = (torch.rand(num_segmentation_classes - 1, device=segs.device) < 0.5)
    for class_idx, remove_class in enumerate(class_removals.tolist()):
        if remove_class:
            segs[(255 * segs).int() == class_idx + 1] = 0
    return segs


def convert_segbatch_to_multiclass(
    shape: Tuple[int, ...],
    segmentations_batch,
    device: torch.device,
    use_ablated_segmentations: bool = False,
    num_segmentation_classes: int = 5,
    config=None,
) -> torch.Tensor:
    """Combine segmentation maps into one single-channel map; optionally ablate."""
    if config is not None:
        use_ablated_segmentations = getattr(config, "use_ablated_segmentations", False)
        num_segmentation_classes = getattr(config, "num_segmentation_classes", 5)
    segs = torch.zeros(shape, device=device)
    if isinstance(segmentations_batch, dict):
        for k, seg in segmentations_batch.items():
            if k.startswith("seg_"):
                seg = seg.to(device)
                segs[segs == 0] = seg[segs == 0]
    else:
        segs = segmentations_batch.to(device)
    if use_ablated_segmentations:
        segs = ablate_masks(segs, num_segmentation_classes)
    return segs


def add_segmentations_to_noise(
    noisy_images: torch.Tensor,
    batch: dict,
    device: torch.device,
    segmentation_channel_mode: str = "single",
    use_ablated_segmentations: bool = False,
    num_segmentation_classes: int = 5,
    config=None,
) -> torch.Tensor:
    """Concatenate single-channel mask to the noisy image. Only 'single' mode supported."""
    if config is not None:
        segmentation_channel_mode = getattr(config, "segmentation_channel_mode", "single")
        use_ablated_segmentations = getattr(config, "use_ablated_segmentations", False)
        num_segmentation_classes = getattr(config, "num_segmentation_classes", 5)
    if segmentation_channel_mode != "single":
        raise NotImplementedError("Only 'single' channel mode is supported.")
    multiclass_masks_shape = (
        noisy_images.shape[0], 1, noisy_images.shape[2], noisy_images.shape[3]
    )
    segs = convert_segbatch_to_multiclass(
        multiclass_masks_shape,
        batch,
        device,
        use_ablated_segmentations=use_ablated_segmentations,
        num_segmentation_classes=num_segmentation_classes,
        config=config,
    )
    return torch.cat((noisy_images, segs), dim=1)
