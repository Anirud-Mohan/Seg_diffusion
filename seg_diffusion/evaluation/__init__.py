"""Evaluation metrics and aggregation for diffusion and segmentation."""
from seg_diffusion.evaluation.eval_metrics import (
    evaluate_unet_on_split,
    load_uncond_from_dir,
    load_seg_guided_components,
    generate_aligned_samples_from_split,
    compute_fid_kid_for_pipe,
    compute_fid_kid_for_seg_guided,
    evaluate_generated_with_unet,
    compute_tumor_residual_ratio,
    compute_difference_map_iou,
    run_full_evaluation,
)

__all__ = [
    "evaluate_unet_on_split",
    "load_uncond_from_dir",
    "load_seg_guided_components",
    "generate_aligned_samples_from_split",
    "compute_fid_kid_for_pipe",
    "compute_fid_kid_for_seg_guided",
    "evaluate_generated_with_unet",
    "compute_tumor_residual_ratio",
    "compute_difference_map_iou",
    "run_full_evaluation",
]
