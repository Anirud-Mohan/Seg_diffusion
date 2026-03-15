"""
Evaluation: FID/KID, SSIM, Dice/IoU on generated images, tumor residual, difference-map IoU.
All functions take device, data_root, seg_model, etc. as arguments where needed.
"""
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline, UNet2DModel, DDPMScheduler

from seg_diffusion.config import (
    DATA_ROOT,
    EVAL_CFG,
    IMG_SIZE,
    NUM_CLASSES,
    COLOR_TO_LABEL,
    DIFF_MASK_VALUE_TO_CLASS,
)
from seg_diffusion.data import (
    BratsFlairSliceDataset,
    load_image,
    load_diffusion_mask,
    tensor_to_pil,
    rgb_to_label,
    collect_paired_paths,
)
from seg_diffusion.models import UNetSeg
from seg_diffusion.metrics.dice import multiclass_dice, multiclass_iou
from seg_diffusion.metrics.fid_ssim import (
    extract_seg_encoder_features,
    compute_mean_cov,
    compute_fid,
    compute_kid,
)


def load_uncond_from_dir(checkpoint_dir: Path, device: torch.device):
    """Load unconditional DiffusionPipeline from checkpoint dir."""
    pipe = DiffusionPipeline.from_pretrained(str(checkpoint_dir))
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_seg_guided_components(ckpt_dir: Path, device: torch.device):
    """Load UNet2D and DDPMScheduler for seg-guided model (M2/M3)."""
    unet = UNet2DModel.from_pretrained(str(ckpt_dir / "unet"))
    scheduler = DDPMScheduler.from_pretrained(str(ckpt_dir / "scheduler"))
    unet.to(device)
    unet.eval()
    return unet, scheduler


def sample_seg_guided(
    unet,
    train_scheduler,
    real_mask: torch.Tensor,
    init_image: torch.Tensor,
    device: torch.device,
    num_steps: int = 50,
    seed: int = 42,
    mode: str = "healthy",
    strength: float = 0.35,
) -> torch.Tensor:
    """Seg-guided sampling (M2/M3). mode='healthy' or 'reconstruct'."""
    strength = float(max(0.0, min(1.0, strength)))
    real_mask = real_mask.to(device)
    cond_mask = real_mask.clone()
    image_0 = init_image.to(device) * 2.0 - 1.0
    tumor_thresh = (4.0 / 255.0) - 1e-4
    tumor_pixels = real_mask > tumor_thresh
    if mode == "healthy":
        cond_mask[tumor_pixels] = 0.0
    elif mode != "reconstruct":
        raise ValueError(f"Unknown mode={mode}")
    unet_to_use = unet.module if hasattr(unet, "module") else unet
    if hasattr(unet_to_use, "_orig_mod"):
        unet_to_use = unet_to_use._orig_mod
    scheduler = DDPMScheduler.from_config(train_scheduler.config)
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps
    if strength <= 0.0:
        image = image_0
        t_iter = timesteps
    else:
        start_idx = int((1.0 - strength) * (len(timesteps) - 1))
        t_start = timesteps[start_idx]
        torch.manual_seed(seed)
        noise = torch.randn_like(image_0, device=device)
        image = scheduler.add_noise(image_0, noise, t_start)
        t_iter = timesteps[start_idx:]
    for t in t_iter:
        model_input = torch.cat([image, cond_mask], dim=1)
        with torch.no_grad():
            noise_pred = unet_to_use(model_input, t).sample
        image = scheduler.step(noise_pred, t, image).prev_sample
    return (image / 2.0 + 0.5).clamp(0.0, 1.0)


def sample_seg_guided_from_noise(
    unet,
    train_scheduler,
    real_mask: torch.Tensor,
    device: torch.device,
    num_steps: int = 50,
    seed: int = 42,
) -> torch.Tensor:
    """Pure seg-guided generation from noise conditioned on real_mask."""
    real_mask = real_mask.to(device)
    cond_mask = real_mask.clone()
    unet_to_use = unet.module if hasattr(unet, "module") else unet
    if hasattr(unet_to_use, "_orig_mod"):
        unet_to_use = unet_to_use._orig_mod
    scheduler = DDPMScheduler.from_config(train_scheduler.config)
    scheduler.set_timesteps(num_steps)
    _, _, H, W = cond_mask.shape
    torch.manual_seed(seed)
    image = torch.randn(1, 1, H, W, device=device)
    for t in scheduler.timesteps:
        model_input = torch.cat([image, cond_mask], dim=1)
        with torch.no_grad():
            noise_pred = unet_to_use(model_input, t).sample
        image = scheduler.step(noise_pred, t, image).prev_sample
    return (image / 2.0 + 0.5).clamp(0.0, 1.0)


def generate_aligned_samples_from_split(
    pipe,
    out_dir: Path,
    data_root: Path,
    device: torch.device,
    img_size: int = IMG_SIZE,
    split: str = "test",
    limit: int = None,
    steps: int = 50,
) -> None:
    """Generate unconditional samples aligned to dataset order and save to out_dir."""
    ds = BratsFlairSliceDataset(data_root, split=split)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(ds) if limit is None else min(limit, len(ds))
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    for i in range(n):
        img_path, _ = ds.items[i]
        fname = Path(img_path).name
        out = pipe(batch_size=1, num_inference_steps=steps)
        pil = out.images[0].convert("L")
        pil = pil.resize((img_size, img_size), Image.BILINEAR)
        pil.save(out_dir / fname)


def evaluate_unet_on_split(
    seg_model: UNetSeg,
    data_root: Path,
    device: torch.device,
    split: str = "val",
    batch_size: int = 8,
    num_classes: int = NUM_CLASSES,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """Compute mean Dice and IoU for seg model on a split."""
    ds = BratsFlairSliceDataset(data_root, split=split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    seg_model.eval()
    inter = torch.zeros(num_classes, device=device, dtype=torch.float64)
    union_dice = torch.zeros(num_classes, device=device, dtype=torch.float64)
    union_iou = torch.zeros(num_classes, device=device, dtype=torch.float64)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = seg_model(imgs)
            preds = logits.argmax(dim=1)
            for cls in range(num_classes):
                pred_c = preds == cls
                tgt_c = masks == cls
                inter[cls] += (pred_c & tgt_c).sum()
                union_dice[cls] += pred_c.sum() + tgt_c.sum()
                union_iou[cls] += (pred_c | tgt_c).sum()
    valid = union_dice > 0
    dice_per_class = (2 * inter[valid] + eps) / (union_dice[valid] + eps)
    iou_per_class = (inter[valid] + eps) / (union_iou[valid] + eps)
    return dice_per_class.mean().item(), iou_per_class.mean().item()


def compute_fid_kid_for_pipe(
    name: str,
    pipe,
    seg_model: UNetSeg,
    real_feats: np.ndarray,
    mu_real: np.ndarray,
    sigma_real: np.ndarray,
    img_transform: transforms.Compose,
    device: torch.device,
    n_gen: int = None,
    batch_size: int = 8,
    num_steps: int = 50,
) -> Tuple[Optional[float], Optional[float]]:
    """Generate images from pipe, extract seg-UNet features, compute FID/KID vs real_feats."""
    if pipe is None:
        return None, None
    n_gen = n_gen or real_feats.shape[0]
    gen_feats = []
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    with torch.no_grad():
        num_done = 0
        while num_done < n_gen:
            cur_bs = min(batch_size, n_gen - num_done)
            out = pipe(batch_size=cur_bs, num_inference_steps=num_steps)
            for img_pil in out.images:
                img_t = img_transform(img_pil.convert("L")).unsqueeze(0).to(device)
                feats = extract_seg_encoder_features(seg_model, img_t)
                gen_feats.append(feats.cpu().numpy())
                num_done += 1
                if num_done >= n_gen:
                    break
    gen_feats = np.concatenate(gen_feats, axis=0)
    mu_g, sigma_g = compute_mean_cov(gen_feats)
    fid = compute_fid(mu_real, sigma_real, mu_g, sigma_g)
    kid = compute_kid(real_feats, gen_feats)
    return fid, kid


def compute_fid_kid_for_seg_guided(
    name: str,
    unet,
    scheduler,
    seg_model: UNetSeg,
    real_feats: np.ndarray,
    mu_real: np.ndarray,
    sigma_real: np.ndarray,
    data_root: Path,
    device: torch.device,
    img_transform: transforms.Compose,
    n_gen: int = 1024,
    num_steps: int = 50,
    split: str = "val",
) -> Tuple[float, float]:
    """FID/KID for seg-guided model in seg-UNet feature space."""
    ds = BratsFlairSliceDataset(data_root, split=split)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    n = min(n_gen, len(indices))
    unet.eval()
    gen_feats = []
    used = 0
    with torch.no_grad():
        for idx in indices:
            img_path, mask_path = ds.items[idx]
            real_mask = load_diffusion_mask(Path(mask_path), device)
            gen = sample_seg_guided_from_noise(
                unet=unet,
                train_scheduler=scheduler,
                real_mask=real_mask,
                device=device,
                num_steps=num_steps,
                seed=42 + idx,
            )
            pil = tensor_to_pil(gen)
            img_t = img_transform(pil).unsqueeze(0).to(device)
            feats = extract_seg_encoder_features(seg_model, img_t)
            gen_feats.append(feats.cpu().numpy())
            used += 1
            if used >= n:
                break
    gen_feats = np.concatenate(gen_feats, axis=0)
    mu_g, sigma_g = compute_mean_cov(gen_feats)
    fid = compute_fid(mu_real, sigma_real, mu_g, sigma_g)
    kid = compute_kid(real_feats, gen_feats)
    return fid, kid


def ssim_from_tensors(
    real_img: torch.Tensor,
    gen_img: torch.Tensor,
    ssim_metric,
    valid_mask: Optional[torch.Tensor] = None,
) -> float:
    """SSIM between real and gen tensors; optional valid_mask. Uses histogram matching."""
    from skimage.exposure import match_histograms
    real_01 = real_img.clamp(0.0, 1.0)
    gen_01 = gen_img.clamp(0.0, 1.0)
    if valid_mask is not None:
        real_01 = real_01 * valid_mask
        gen_01 = gen_01 * valid_mask
    real_np = real_01[0, 0].detach().cpu().numpy()
    gen_np = gen_01[0, 0].detach().cpu().numpy()
    gen_matched = match_histograms(gen_np, real_np, channel_axis=None)
    gen_01_m = torch.from_numpy(gen_matched).unsqueeze(0).unsqueeze(0).to(real_img.device)
    with torch.no_grad():
        score = ssim_metric(gen_01_m, real_01)
    return float(score.item())


def compute_ssim_for_m1_uncond(
    name: str,
    pipe,
    paired_val_imgs: List[Path],
    load_image_fn,
    img_transform: transforms.Compose,
    ssim_metric,
    device: torch.device,
    n_pairs: int = 64,
    num_steps: int = 50,
) -> Optional[float]:
    """SSIM for M1 unconditional pipeline vs validation images."""
    if pipe is None or len(paired_val_imgs) == 0:
        return None
    n_pairs = min(n_pairs, len(paired_val_imgs))
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    ssim_scores = []
    with torch.no_grad():
        for i in range(n_pairs):
            img_path = paired_val_imgs[i]
            real = load_image_fn(img_path).unsqueeze(0).to(device)
            out = pipe(batch_size=1, num_inference_steps=num_steps)
            gen_pil = out.images[0].convert("L")
            gen = img_transform(gen_pil).unsqueeze(0).to(device)
            ssim_val = ssim_from_tensors(real, gen, ssim_metric)
            ssim_scores.append(ssim_val)
    return float(np.mean(ssim_scores))


def compute_ssim_for_seg_guided(
    name: str,
    unet,
    train_scheduler,
    seg_model: UNetSeg,
    data_root: Path,
    device: torch.device,
    load_image_fn,
    load_diffusion_mask_fn,
    img_transform: transforms.Compose,
    ssim_metric,
    n_pairs: int = 64,
    num_steps: int = 50,
    split: str = "val",
    mode: str = "healthy",
    strength: float = 0.35,
) -> float:
    """SSIM for seg-guided model (M2/M3)."""
    ds = BratsFlairSliceDataset(data_root, split=split)
    n_pairs = min(n_pairs, len(ds))
    ssim_scores = []
    unet.eval()
    with torch.no_grad():
        for i in range(n_pairs):
            img_path, mask_path = ds.items[i]
            real = load_image_fn(img_path).unsqueeze(0).to(device)
            real_mask = load_diffusion_mask_fn(Path(mask_path), device)
            gen = sample_seg_guided(
                unet=unet,
                train_scheduler=train_scheduler,
                real_mask=real_mask,
                init_image=real,
                device=device,
                num_steps=num_steps,
                seed=42 + i,
                mode=mode,
                strength=strength,
            )
            if mode == "healthy":
                tumor_thresh = (4.0 / 255.0) - 1e-4
                non_tumor_mask = (real_mask <= tumor_thresh).float()
                ssim_val = ssim_from_tensors(real, gen, ssim_metric, valid_mask=non_tumor_mask)
            else:
                ssim_val = ssim_from_tensors(real, gen, ssim_metric)
            ssim_scores.append(ssim_val)
    return float(np.mean(ssim_scores))


def load_generated_pairs(
    gen_root: Path,
    data_root: Path,
    split: str = "test",
    limit: int = None,
) -> List[Tuple[Path, Path]]:
    """Return list of (gen_path, gt_mask_path) for generated images that have GT."""
    gen_dir = Path(gen_root)
    mask_dir = data_root / split / "mask"
    pairs = []
    for p in sorted(gen_dir.glob("*.png")):
        gt = mask_dir / p.name
        if gt.exists():
            pairs.append((p, gt))
            if limit and len(pairs) >= limit:
                break
    return pairs


def evaluate_generated_with_unet(
    model_name: str,
    seg_model: UNetSeg,
    gen_root: Path,
    data_root: Path,
    device: torch.device,
    split: str = "test",
    limit: int = None,
    batch_size: int = 8,
    num_classes: int = NUM_CLASSES,
) -> Tuple[float, float]:
    """Dice/IoU on generated images by running seg-UNet and comparing to GT masks."""
    pairs = load_generated_pairs(gen_root, data_root, split=split, limit=limit)
    if not pairs:
        return 0.0, 0.0
    seg_model.eval()
    dice_scores, iou_scores = [], []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            gen_imgs = []
            gt_masks = []
            for gen_path, gt_mask_path in batch:
                img = Image.open(gen_path).convert("L")
                arr = np.array(img, dtype=np.float32) / 255.0
                gen_imgs.append(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0))
                mask_rgb = np.array(Image.open(gt_mask_path).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
                if mask_rgb.ndim == 3:
                    lbl = rgb_to_label(mask_rgb)
                else:
                    lbl = np.zeros_like(mask_rgb, dtype=np.int64)
                    for src_val, cls in DIFF_MASK_VALUE_TO_CLASS.items():
                        lbl[mask_rgb == src_val] = cls
                gt_masks.append(torch.from_numpy(lbl).long().unsqueeze(0))
            gen_imgs = torch.cat(gen_imgs, dim=0).to(device)
            gt_masks = torch.cat(gt_masks, dim=0).to(device)
            logits = seg_model(gen_imgs)
            for b in range(gen_imgs.size(0)):
                d = multiclass_dice(
                    logits[b : b + 1], gt_masks[b : b + 1], num_classes=num_classes
                )
                iou = multiclass_iou(
                    logits[b : b + 1], gt_masks[b : b + 1], num_classes=num_classes
                )
                dice_scores.append(d)
                iou_scores.append(iou)
    return float(np.mean(dice_scores)), float(np.mean(iou_scores))


def compute_tumor_residual_ratio(
    model_name: str,
    seg_model: UNetSeg,
    gen_root: Path,
    data_root: Path,
    device: torch.device,
    split: str = "test",
    limit: int = None,
    batch_size: int = 8,
    void_thresh: float = 0.05,
    void_weight: float = 2.0,
) -> float:
    """Residual abnormality ratio in generated healthy counterfactuals."""
    pairs = load_generated_pairs(gen_root, data_root, split=split, limit=limit)
    if not pairs:
        return 0.0
    ds = BratsFlairSliceDataset(data_root, split=split)
    orig_lookup = {Path(img).name: (img, mask) for img, mask in ds.items}
    total_bad = 0.0
    total_real_tumor = 0.0
    seg_model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            gen_imgs, real_masks = [], []
            for gen_path, _ in batch:
                if gen_path.name not in orig_lookup:
                    continue
                real_img_path, real_mask_path = orig_lookup[gen_path.name]
                gen_img = Image.open(gen_path).convert("L")
                gen_arr = np.array(gen_img, dtype=np.float32) / 255.0
                gen_imgs.append(torch.from_numpy(gen_arr).unsqueeze(0).unsqueeze(0))
                mask_rgb = np.array(
                    Image.open(real_mask_path).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
                )
                if mask_rgb.ndim == 3:
                    mask_lbl = rgb_to_label(mask_rgb)
                else:
                    mask_lbl = np.zeros_like(mask_rgb, dtype=np.int64)
                    for src_val, cls in DIFF_MASK_VALUE_TO_CLASS.items():
                        mask_lbl[mask_rgb == src_val] = cls
                real_masks.append(torch.from_numpy(mask_lbl).long().unsqueeze(0))
            if not gen_imgs:
                continue
            gen_imgs = torch.cat(gen_imgs, dim=0).to(device)
            real_masks = torch.cat(real_masks, dim=0).to(device)
            logits = seg_model(gen_imgs)
            preds = logits.argmax(dim=1)
            real_tumor_mask = real_masks == 4
            for b in range(gen_imgs.size(0)):
                tumor_mask_b = real_tumor_mask[b]
                tumor_pix = tumor_mask_b.float().sum().item()
                if tumor_pix <= 0:
                    continue
                pred_tumor_mask_b = preds[b] == 4
                gen_b = gen_imgs[b, 0]
                void_mask_b = (gen_b < void_thresh) & tumor_mask_b
                bad_pix = pred_tumor_mask_b.float().sum().item() + void_weight * void_mask_b.float().sum().item()
                total_bad += bad_pix
                total_real_tumor += tumor_pix
    return total_bad / (total_real_tumor + 1e-6)


def compute_difference_map_iou(
    model_name: str,
    gen_root: Path,
    data_root: Path,
    load_image_fn,
    img_size: int = IMG_SIZE,
    split: str = "test",
    diff_thresh: float = 0.10,
    limit: int = None,
    focus_class: int = 4,
    eps: float = 1e-6,
) -> Optional[float]:
    """Difference-map IoU: IoU between |real - gen| > thresh and GT tumor mask."""
    pairs = load_generated_pairs(gen_root, data_root, split=split, limit=limit)
    if not pairs:
        return None
    flair_dir = data_root / split / "flair"
    ious = []
    for gen_path, gt_mask_path in pairs:
        real_img_path = flair_dir / gen_path.name
        if not real_img_path.exists():
            continue
        real = load_image_fn(real_img_path).squeeze(0)
        gen_img = Image.open(gen_path).convert("L")
        gen_arr = np.array(gen_img, dtype=np.float32) / 255.0
        gen = torch.from_numpy(gen_arr)
        diff = (real - gen).abs()
        change_mask = (diff > diff_thresh).float()
        mask_rgb = np.array(
            Image.open(gt_mask_path).resize((img_size, img_size), Image.NEAREST)
        )
        if mask_rgb.ndim == 3:
            mask_lbl = rgb_to_label(mask_rgb)
        else:
            mask_lbl = np.zeros_like(mask_rgb, dtype=np.int64)
            for src_val, cls in DIFF_MASK_VALUE_TO_CLASS.items():
                mask_lbl[mask_rgb == src_val] = cls
        tumor_mask = torch.from_numpy((mask_lbl == focus_class).astype(np.float32))
        inter = (change_mask * tumor_mask).sum()
        union = change_mask.sum() + tumor_mask.sum() - inter
        if union <= 0:
            continue
        ious.append((inter + eps) / (union + eps))
    return float(np.mean(ious)) if ious else None


def run_full_evaluation(
    data_root: Path = None,
    m1_dir: Path = None,
    m2_dir: Path = None,
    m3_dir: Path = None,
    seg_ckpt: Path = None,
    device: torch.device = None,
    eval_cfg: dict = None,
) -> dict:
    """
    Load M1/M2/M3 and seg model, compute real features, then FID/KID, SSIM, Dice/IoU,
    tumor residual, difference-map IoU. Return results dict and print DataFrame.
    """
    import pandas as pd
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    if data_root is None:
        data_root = DATA_ROOT
    if m1_dir is None:
        m1_dir = data_root.parent / "model_checkpoint" / "m1_checkpoint"
    if m2_dir is None:
        m2_dir = data_root.parent / "model_checkpoint" / "m2_checkpoint"
    if m3_dir is None:
        m3_dir = data_root.parent / "model_checkpoint" / "m3_checkpoint"
    m1_dir = Path(m1_dir)
    m2_dir = Path(m2_dir)
    m3_dir = Path(m3_dir)
    if seg_ckpt is None:
        seg_ckpt = data_root.parent / "checkpoints" / "unet_best_5class.pth"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if eval_cfg is None:
        eval_cfg = EVAL_CFG

    img_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    val_img_dir = data_root / "val" / "flair"
    val_mask_dir = data_root / "val" / "mask"
    paired_val_imgs, _ = collect_paired_paths(val_img_dir, val_mask_dir)

    seg_model = UNetSeg(in_channels=1, num_classes=NUM_CLASSES).to(device)
    state = torch.load(seg_ckpt, map_location=device)
    seg_model.load_state_dict(state)
    seg_model.eval()
    for p in seg_model.parameters():
        p.requires_grad = False

    real_feats = []
    with torch.no_grad():
        for img_path in paired_val_imgs:
            img = load_image(img_path).unsqueeze(0).to(device)
            feats = extract_seg_encoder_features(seg_model, img)
            real_feats.append(feats.cpu().numpy())
    real_feats = np.concatenate(real_feats, axis=0)
    mu_real, sigma_real = compute_mean_cov(real_feats)

    pipe_m1 = load_uncond_from_dir(m1_dir, device) if m1_dir.exists() else None
    unet_m2, sched_m2 = load_seg_guided_components(m2_dir, device) if m2_dir.exists() else (None, None)
    unet_m3, sched_m3 = load_seg_guided_components(m3_dir, device) if m3_dir.exists() else (None, None)

    fk_cfg = eval_cfg["fid_kid"]
    fid1, kid1 = compute_fid_kid_for_pipe(
        "M1", pipe_m1, seg_model, real_feats, mu_real, sigma_real,
        img_transform, device,
        n_gen=min(fk_cfg["N_GEN"], real_feats.shape[0]),
        batch_size=fk_cfg["BATCH"],
        num_steps=fk_cfg["STEPS"],
    ) if pipe_m1 else (None, None)
    fid2, kid2 = (compute_fid_kid_for_seg_guided(
        "M2", unet_m2, sched_m2, seg_model, real_feats, mu_real, sigma_real,
        data_root, device, img_transform,
        n_gen=min(fk_cfg["N_GEN"], real_feats.shape[0]),
        num_steps=fk_cfg["STEPS"],
    ) if unet_m2 else (None, None))
    fid3, kid3 = (compute_fid_kid_for_seg_guided(
        "M3", unet_m3, sched_m3, seg_model, real_feats, mu_real, sigma_real,
        data_root, device, img_transform,
        n_gen=min(fk_cfg["N_GEN"], real_feats.shape[0]),
        num_steps=fk_cfg["STEPS"],
    ) if unet_m3 else (None, None))

    ss_cfg = eval_cfg["ssim"]
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_m1 = compute_ssim_for_m1_uncond(
        "M1", pipe_m1, paired_val_imgs, load_image, img_transform, ssim_metric, device,
        n_pairs=ss_cfg["N_PAIRS"], num_steps=ss_cfg["STEPS"],
    ) if pipe_m1 else None
    ssim_m2 = compute_ssim_for_seg_guided(
        "M2", unet_m2, sched_m2, seg_model, data_root, device,
        load_image, load_diffusion_mask, img_transform, ssim_metric,
        n_pairs=ss_cfg["N_PAIRS"], num_steps=ss_cfg["STEPS"], mode="healthy", strength=0.3,
    ) if unet_m2 else None
    ssim_m3 = compute_ssim_for_seg_guided(
        "M3", unet_m3, sched_m3, seg_model, data_root, device,
        load_image, load_diffusion_mask, img_transform, ssim_metric,
        n_pairs=ss_cfg["N_PAIRS"], num_steps=ss_cfg["STEPS"], mode="healthy", strength=0.3,
    ) if unet_m3 else None

    gs_cfg = eval_cfg["gen_seg"]
    gen_m1_root = m1_dir / "test" if (m1_dir / "test").exists() else m1_dir
    gen_m2_root = m2_dir / "test" / "recon" if (m2_dir / "test" / "recon").exists() else m2_dir
    gen_m3_root = m3_dir / "test" / "recon" if (m3_dir / "test" / "recon").exists() else m3_dir
    dice_m1, iou_m1 = evaluate_generated_with_unet(
        "M1", seg_model, gen_m1_root, data_root, device,
        split="test", limit=gs_cfg["N_PAIRS"], batch_size=gs_cfg["BATCH"],
    )
    dice_m2, iou_m2 = evaluate_generated_with_unet(
        "M2", seg_model, gen_m2_root, data_root, device,
        split="test", limit=gs_cfg["N_PAIRS"], batch_size=gs_cfg["BATCH"],
    )
    dice_m3, iou_m3 = evaluate_generated_with_unet(
        "M3", seg_model, gen_m3_root, data_root, device,
        split="test", limit=gs_cfg["N_PAIRS"], batch_size=gs_cfg["BATCH"],
    )
    tumor_m1 = compute_tumor_residual_ratio(
        "M1", seg_model, gen_m1_root, data_root, device,
        split="test", limit=gs_cfg["N_PAIRS"], batch_size=gs_cfg["BATCH"],
    )
    tumor_m2 = compute_tumor_residual_ratio(
        "M2", seg_model, m2_dir / "test" / "healthy", data_root, device,
        split="test", limit=gs_cfg["N_PAIRS"], batch_size=gs_cfg["BATCH"],
    )
    tumor_m3 = compute_tumor_residual_ratio(
        "M3", seg_model, m3_dir / "test" / "healthy", data_root, device,
        split="test", limit=gs_cfg["N_PAIRS"], batch_size=gs_cfg["BATCH"],
    )
    diff_m1 = compute_difference_map_iou("M1", gen_m1_root, data_root, load_image, limit=gs_cfg["N_PAIRS"])
    diff_m2 = compute_difference_map_iou("M2", m2_dir / "test" / "healthy", data_root, load_image, limit=gs_cfg["N_PAIRS"])
    diff_m3 = compute_difference_map_iou("M3", m3_dir / "test" / "healthy", data_root, load_image, limit=gs_cfg["N_PAIRS"])

    results = {
        "M1": {"FID": fid1, "KID": kid1, "SSIM": ssim_m1, "GenDice": dice_m1, "GenIoU": iou_m1, "TumorResidual": tumor_m1, "DiffMapIoU": diff_m1},
        "M2": {"FID": fid2, "KID": kid2, "SSIM": ssim_m2, "GenDice": dice_m2, "GenIoU": iou_m2, "TumorResidual": tumor_m2, "DiffMapIoU": diff_m2},
        "M3": {"FID": fid3, "KID": kid3, "SSIM": ssim_m3, "GenDice": dice_m3, "GenIoU": iou_m3, "TumorResidual": tumor_m3, "DiffMapIoU": diff_m3},
    }
    metric_order = ["FID", "KID", "SSIM", "GenDice", "GenIoU", "TumorResidual", "DiffMapIoU"]
    df = pd.DataFrame(results).T[metric_order]
    print("\n=== Summary metrics table ===")
    print(df.round(4))
    return results
