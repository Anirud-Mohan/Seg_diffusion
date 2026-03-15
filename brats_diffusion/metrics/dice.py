"""Multi-class Dice (and IoU) for segmentation."""
import torch


def multiclass_dice(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 5,
    eps: float = 1e-6,
) -> float:
    """
    Hard macro Dice.
    pred_logits: (B, C, H, W); target: (B, H, W) with class indices in [0..C-1].
    Returns mean Dice over all classes.
    """
    pred_labels = pred_logits.argmax(dim=1)
    per_class_scores = []
    for cls in range(num_classes):
        pred_c = (pred_labels == cls).float()
        tgt_c = (target == cls).float()
        inter = (pred_c * tgt_c).sum()
        union = pred_c.sum() + tgt_c.sum()
        if union <= 0:
            continue
        dice = (2 * inter + eps) / (union + eps)
        per_class_scores.append(dice)
    if not per_class_scores:
        return 0.0
    return torch.stack(per_class_scores).mean().item()


def multiclass_iou(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 5,
    eps: float = 1e-6,
) -> float:
    """
    Hard macro IoU (Jaccard).
    pred_logits: (B, C, H, W); target: (B, H, W).
    Returns mean IoU over all classes.
    """
    pred_labels = pred_logits.argmax(dim=1)
    per_class_scores = []
    for cls in range(num_classes):
        pred_c = (pred_labels == cls).float()
        tgt_c = (target == cls).float()
        inter = (pred_c * tgt_c).sum()
        union = pred_c.sum() + tgt_c.sum() - inter
        if union <= 0:
            continue
        iou = (inter + eps) / (union + eps)
        per_class_scores.append(iou)
    if not per_class_scores:
        return 0.0
    return torch.stack(per_class_scores).mean().item()
