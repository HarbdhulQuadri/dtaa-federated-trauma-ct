"""
Loss functions for DTAA: SoftDiceLoss, FocalLoss, and CompoundFocalDiceLoss with Manifold Regularization.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Multi-class Soft Dice Loss supporting voxel ignore masks (-1)."""
    def __init__(self, num_classes: int = 6, smooth: float = 1e-4, include_background: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid = (targets != -1)
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        clamped_targets = torch.where(valid, targets, torch.zeros_like(targets))
        target_one_hot = F.one_hot(clamped_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid_exp = valid.unsqueeze(1).float()

        probs = probs * valid_exp
        target_one_hot = target_one_hot * valid_exp

        start_c = 0 if self.include_background else 1
        d_losses = []
        for c in range(start_c, self.num_classes):
            p_c = probs[:, c, :, :]
            t_c = target_one_hot[:, c, :, :]
            intersection = torch.sum(p_c * t_c)
            cardinality = torch.sum(p_c + t_c)
            dice_c = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            d_losses.append(1.0 - dice_c)

        return torch.mean(torch.stack(d_losses)) if len(d_losses) > 0 else torch.tensor(0.0, device=logits.device)


class FocalLoss(nn.Module):
    """Class-weighted Focal Loss to address extreme active extravasation sparsity (<0.7% volume)."""
    def __init__(self, gamma: float = 2.0, alpha: Optional[List[float]] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = (targets != -1)
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        valid_logits = logits.permute(0, 2, 3, 1)[valid]
        valid_targets = targets[valid]

        ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            weights = torch.tensor(self.alpha, device=logits.device, dtype=torch.float32)
            focal_loss = focal_loss * weights[valid_targets]

        return torch.mean(focal_loss)


class CompoundFocalDiceLoss(nn.Module):
    """
    Compound Focal + Dice objective with optional manifold KL consistency regularization.
    """
    def __init__(
        self,
        num_classes: int = 6,
        dice_weight: float = 0.60,
        focal_weight: float = 0.40,
        alpha_weights: Optional[List[float]] = None,
        gamma: float = 2.0
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = SoftDiceLoss(num_classes=num_classes)
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha_weights)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        p_consensus: Optional[torch.Tensor] = None,
        manifold_weight: float = 0.10
    ) -> torch.Tensor:
        l_dice = self.dice_loss(logits, targets)
        l_focal = self.focal_loss(logits, targets)
        total_loss = self.dice_weight * l_dice + self.focal_weight * l_focal

        if p_consensus is not None and manifold_weight > 0.0:
            p_student_log = F.log_softmax(logits, dim=1)
            # Symmetrized KL divergence between student and consensus distribution
            kl_div = F.kl_div(p_student_log, p_consensus, reduction='batchmean')
            total_loss += manifold_weight * kl_div

        return total_loss
