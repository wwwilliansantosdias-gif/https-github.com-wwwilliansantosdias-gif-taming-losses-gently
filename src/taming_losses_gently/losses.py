"""Robust and smooth loss functions for ML training.

Each function is a pure, differentiable operation on `torch.Tensor` inputs and
returns a reduced scalar tensor (unless `reduction="none"` is requested).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unknown reduction: {reduction!r}")


def charbonnier_loss(pred: Tensor, target: Tensor, eps: float = 1e-3, reduction: str = "mean") -> Tensor:
    """Smooth, differentiable approximation of the L1 loss.

    Behaves like L2 near zero and like L1 for large residuals, avoiding the
    vanishing gradient of L2 at large errors and the non-differentiability of
    L1 at zero. Commonly used in image restoration / super-resolution.
    """
    diff = pred - target
    loss = torch.sqrt(diff * diff + eps * eps)
    return _reduce(loss, reduction)


def focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> Tensor:
    """Binary focal loss (Lin et al., 2017) for class-imbalanced classification.

    `logits` are raw (pre-sigmoid) scores and `targets` are binary labels in
    {0, 1}, both of shape `(N, ...)`. Down-weights well-classified examples so
    training focuses on hard, misclassified ones.
    """
    targets = targets.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * bce
    return _reduce(loss, reduction)


def label_smoothing_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    smoothing: float = 0.1,
    reduction: str = "mean",
) -> Tensor:
    """Cross-entropy loss with label smoothing for multi-class classification.

    `logits` has shape `(N, C)` and `targets` are class indices of shape
    `(N,)`. Softens hard one-hot targets to discourage over-confident
    predictions and improve calibration.
    """
    num_classes = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    smooth = -log_probs.mean(dim=-1)
    loss = (1 - smoothing) * nll + smoothing * smooth
    return _reduce(loss, reduction)


def dice_loss(logits: Tensor, targets: Tensor, smooth: float = 1.0) -> Tensor:
    """Soft Dice loss for binary segmentation.

    `logits` are raw (pre-sigmoid) scores and `targets` are binary masks in
    {0, 1}, both of shape `(N, ...)`. Directly optimizes overlap between
    prediction and target, which is robust to class imbalance between
    foreground and background pixels.
    """
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1).to(probs.dtype)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return (1 - dice).mean()
