"""FID/KID and SSIM helpers using segmentation U-Net feature space."""
import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel
from skimage.exposure import match_histograms
import torch
import torch.nn.functional as F

from seg_diffusion.models.unet import UNetSeg


def extract_seg_encoder_features(seg_model: UNetSeg, x: torch.Tensor) -> torch.Tensor:
    """
    x: (N,1,H,W) tensor in [0,1]
    Returns: (N,C) feature vectors from deepest encoder level.
    """
    with torch.no_grad():
        x1 = seg_model.inc(x)
        x2 = seg_model.down1(x1)
        x3 = seg_model.down2(x2)
        x4 = seg_model.down3(x3)
        x5 = seg_model.down4(x4)
        feats = F.adaptive_avg_pool2d(x5, (1, 1))
        feats = feats.view(feats.size(0), -1)
        return feats


def compute_mean_cov(feats: np.ndarray):
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    offset = np.eye(sigma1.shape[0]) * eps
    sigma1 = sigma1 + offset
    sigma2 = sigma2 + offset
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * (10 * eps)
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)


def compute_kid(
    feats_real: np.ndarray,
    feats_gen: np.ndarray,
    degree=3,
    gamma=None,
    coef0=1.0,
):
    """KID via polynomial kernel MMD^2 (unbiased)."""
    k_rr = polynomial_kernel(feats_real, feats_real, degree=degree, gamma=gamma, coef0=coef0)
    k_gg = polynomial_kernel(feats_gen, feats_gen, degree=degree, gamma=gamma, coef0=coef0)
    k_rg = polynomial_kernel(feats_real, feats_gen, degree=degree, gamma=gamma, coef0=coef0)
    n_r = feats_real.shape[0]
    n_g = feats_gen.shape[0]
    np.fill_diagonal(k_rr, 0.0)
    np.fill_diagonal(k_gg, 0.0)
    mmd_rr = k_rr.sum() / (n_r * (n_r - 1))
    mmd_gg = k_gg.sum() / (n_g * (n_g - 1))
    mmd_rg = k_rg.mean()
    kid = mmd_rr + mmd_gg - 2 * mmd_rg
    return float(kid)
