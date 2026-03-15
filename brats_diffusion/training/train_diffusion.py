"""Segmentation-guided diffusion training loop with WandB and checkpointing."""
import os
import gc
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
from safetensors.torch import load_file as load_safetensors

import diffusers
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from brats_diffusion.config import TrainingConfig
from brats_diffusion.data import BraTSDataset, add_segmentations_to_noise


def get_model_and_scheduler(config: TrainingConfig, device: torch.device):
    """Build UNet2D and DDPMScheduler from config."""
    in_channels = 1
    if getattr(config, "segmentation_guided", True):
        if getattr(config, "segmentation_channel_mode", "single") == "single":
            in_channels += 1
        else:
            raise NotImplementedError("Only single channel mode supported.")
    model = UNet2DModel(
        sample_size=config.image_size,
        in_channels=in_channels,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D", "UpBlock2D",
        ),
    )
    model.to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    return model, noise_scheduler


def find_latest_wandb_run(config, entity="challenger", project_name="brats-counterfactual-diffusion"):
    """Search WandB for a previous run with the same config."""
    try:
        import wandb
        api = wandb.Api()
    except Exception:
        return None
    target_name = f"run-noMAT-epochs-{config.num_epochs}-bs-{config.train_batch_size}"
    try:
        runs = api.runs(path=f"{entity}/{project_name}")
    except ValueError:
        return None
    for run in runs:
        if run.name == target_name:
            return run
    return None


def download_checkpoint_from_wandb(run, model, optimizer, noise_scheduler, config):
    """Download latest checkpoint from WandB and load into model."""
    try:
        import wandb
    except ImportError:
        return 0
    artifacts = run.logged_artifacts()
    model_artifacts = [a for a in artifacts if a.type == "model"]
    if not model_artifacts:
        return 0
    latest_artifact = sorted(model_artifacts, key=lambda x: x.updated_at)[-1]
    artifact_dir = latest_artifact.download()
    unet_dir = os.path.join(artifact_dir, "unet")
    safe_path = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")
    state_dict = load_safetensors(safe_path) if os.path.exists(safe_path) else None
    if state_dict is None:
        return 0
    model_to_load = model
    if hasattr(model_to_load, "_orig_mod"):
        model_to_load = model_to_load._orig_mod
    if hasattr(model_to_load, "module"):
        model_to_load = model_to_load.module
    model_to_load.load_state_dict(state_dict)
    try:
        epoch_str = latest_artifact.name.split(":")[0].split("-")[-1]
        start_epoch = int(epoch_str)
    except Exception:
        start_epoch = 0
    return start_epoch


def log_validation_images(model, scheduler, val_ds, config, device, epoch):
    """Generate validation sample and log SSIM to WandB."""
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        from skimage.exposure import match_histograms
        import numpy as np
        import wandb
    except ImportError:
        return
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    sample_idx = 0
    for i in range(len(val_ds)):
        if val_ds[i]["seg_mask"][0].max() > 0.015:
            sample_idx = i
            break
    batch = val_ds[sample_idx]
    real_img = batch["images"].unsqueeze(0).to(device)
    real_mask = batch["seg_mask"].unsqueeze(0).to(device)
    input_mask = real_mask.clone()
    input_mask[real_mask > 0.014] = 0
    unet_to_use = model.module if hasattr(model, "module") else model
    if hasattr(unet_to_use, "_orig_mod"):
        unet_to_use = unet_to_use._orig_mod
    inference_scheduler = DDIMScheduler.from_config(scheduler.config)
    inference_scheduler.set_timesteps(50)
    generator = torch.Generator(device="cpu").manual_seed(42)
    image = torch.randn(1, 1, config.image_size, config.image_size, generator=generator).to(device)
    for t in inference_scheduler.timesteps:
        model_input = torch.cat([image, input_mask], dim=1)
        with torch.no_grad():
            noise_pred = unet_to_use(model_input, t).sample
        image = inference_scheduler.step(noise_pred, t, image).prev_sample
    gen_np = (image[0, 0].cpu().numpy() / 2 + 0.5).clip(0, 1)
    real_np = (real_img[0, 0].cpu().numpy() / 2 + 0.5).clip(0, 1)
    gen_matched = match_histograms(gen_np, real_np, channel_axis=None)
    gen_norm = torch.from_numpy(gen_matched).unsqueeze(0).unsqueeze(0).to(device)
    real_norm = torch.from_numpy(real_np).unsqueeze(0).unsqueeze(0).to(device)
    roi_mask = (real_mask[:, 0:1, :, :] < 0.015).float()
    gen_masked = gen_norm * roi_mask
    real_masked = real_norm * roi_mask
    score = ssim(gen_masked, real_masked)
    wandb.log({
        "val_ssim": score.item(),
        "epoch": epoch,
        "Validation_Samples": [
            wandb.Image(real_np, caption="Real Image"),
            wandb.Image(gen_matched, caption=f"Generated (SSIM: {score.item():.4f})"),
        ],
    })


def run_diffusion_training(
    config: TrainingConfig = None,
    device: torch.device = None,
    entity: str = "challenger",
    project: str = "brats-counterfactual-diffusion",
):
    """Run full diffusion training with WandB resume and artifact logging."""
    if config is None:
        config = TrainingConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    train_ds = BraTSDataset(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        image_size=config.image_size,
        use_ablated_segmentations=config.use_ablated_segmentations,
        ablation_prob=config.ablation_prob,
    )
    val_ds = BraTSDataset(
        config.VAL_IMG_DIR,
        config.VAL_MASK_DIR,
        image_size=config.image_size,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.train_batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_ds, batch_size=config.eval_batch_size, shuffle=False)

    model, noise_scheduler = get_model_and_scheduler(config, device)
    try:
        model = torch.compile(model)
    except Exception:
        pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs,
    )

    import wandb
    existing_run = find_latest_wandb_run(config, entity=entity, project_name=project)
    resume_id = existing_run.id if existing_run else None
    run = wandb.init(
        project=project,
        entity=entity,
        config=vars(config),
        name=f"run-noMAT-epochs-{config.num_epochs}-bs-{config.train_batch_size}",
        id=resume_id,
        resume="allow",
    )
    wandb.define_metric("epoch")
    wandb.define_metric("val_loss", step_metric="epoch")
    wandb.define_metric("val_ssim", step_metric="epoch")
    wandb.define_metric("avg_train_loss", step_metric="epoch")

    start_epoch = 0
    if existing_run:
        start_epoch = download_checkpoint_from_wandb(
            existing_run, model, optimizer, noise_scheduler, config
        )
    scaler = GradScaler("cuda", enabled=(config.mixed_precision == "fp16"))

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        train_loss_total = 0.0
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}")
        for batch in train_loader:
            clean_images = batch["images"].to(device)
            noise = torch.randn_like(clean_images).to(device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=device
            ).long()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(config.mixed_precision == "fp16")):
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                if config.segmentation_guided:
                    model_input = add_segmentations_to_noise(
                        noisy_images, batch, device, config=config
                    )
                else:
                    model_input = noisy_images
                noise_pred = model(model_input, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_loss_total += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            wandb.log({"train_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})

        avg_train_loss = train_loss_total / len(train_loader)
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                clean_images = batch["images"].to(device)
                noise = torch.randn_like(clean_images).to(device)
                bs = clean_images.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bs,), device=device
                ).long()
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(config.mixed_precision == "fp16")):
                    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                    if config.segmentation_guided:
                        model_input = add_segmentations_to_noise(
                            noisy_images, batch, device, config=config
                        )
                    else:
                        model_input = noisy_images
                    noise_pred = model(model_input, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, noise)
                val_loss_total += loss.item()
        avg_val_loss = val_loss_total / len(val_loader)
        wandb.log({
            "epoch": epoch + 1,
            "val_loss": avg_val_loss,
            "avg_train_loss": avg_train_loss,
        })
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if (epoch + 1) % config.save_model_epochs == 0:
            save_path = os.path.join(config.OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
            model_to_save = model.module if hasattr(model, "module") else model
            if hasattr(model_to_save, "_orig_mod"):
                model_to_save = model_to_save._orig_mod
            pipeline = diffusers.DDPMPipeline(unet=model_to_save, scheduler=noise_scheduler)
            pipeline.save_pretrained(save_path)
            artifact = wandb.Artifact(
                name=f"checkpoint-epoch-{epoch+1}",
                type="model",
                description=f"Model at epoch {epoch+1}",
            )
            artifact.add_dir(save_path)
            run.log_artifact(artifact)
            log_validation_images(model, noise_scheduler, val_ds, config, device, epoch + 1)
    wandb.finish()


def run_training(config=None, model=None, noise_scheduler=None, optimizer=None, device=None):
    """Entry point that builds model/scheduler if not provided and calls run_diffusion_training."""
    if config is None:
        config = TrainingConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None or noise_scheduler is None:
        model, noise_scheduler = get_model_and_scheduler(config, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    run_diffusion_training(config=config, device=device)
