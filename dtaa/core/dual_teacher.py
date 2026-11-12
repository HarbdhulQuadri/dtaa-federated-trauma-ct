"""
Dual-Teacher Triad Manager: Global Anatomy Teacher, Local Scanner EMA Teacher, and Active Student.
"""

from typing import Dict, Tuple, Any
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from dtaa.core.agreement import (
    compute_inter_teacher_agreement,
    compute_mutual_confidence
)


class DualTeacherManager:
    """
    Manages the Triad of models on each hospital edge client:
      1. Student (theta_S): Gradient-updated on labeled and pseudo-labeled slices.
      2. Global Anchor Teacher (theta_G): Frozen at the latest global server weights.
      3. Local Scanner Teacher (theta_L): Exponential Moving Average (EMA) of student.
    """
    def __init__(
        self,
        base_model: nn.Module,
        device: torch.device,
        ema_alpha: float = 0.90
    ):
        self.device = device
        self.ema_alpha = ema_alpha

        self.student = base_model.to(device)
        self.global_teacher = copy.deepcopy(base_model).to(device)
        self.local_teacher = copy.deepcopy(base_model).to(device)

        # Freeze both teachers from direct PyTorch autograd
        for p in self.global_teacher.parameters():
            p.requires_grad = False
        for p in self.local_teacher.parameters():
            p.requires_grad = False

        self.global_teacher.eval()
        self.local_teacher.eval()

    def update_global_anchor(self, new_global_state_dict: Dict[str, torch.Tensor]) -> None:
        """Loads synchronized central server weights into the Global Anchor Teacher."""
        self.global_teacher.load_state_dict(new_global_state_dict)
        self.global_teacher.eval()

    def update_local_ema(self) -> None:
        """Updates Local Scanner Teacher via EMA: theta_L <- alpha * theta_L + (1 - alpha) * theta_S."""
        with torch.no_grad():
            for p_local, p_student in zip(self.local_teacher.parameters(), self.student.parameters()):
                p_local.data.mul_(self.ema_alpha).add_(p_student.data, alpha=(1.0 - self.ema_alpha))

    def generate_agreement_pseudo_labels(
        self,
        x_unlabeled: torch.Tensor,
        tau_t: float
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Executes Dual-Teacher inference and agreement gating on unannotated slices.
        
        Returns:
            pseudo_labels: Target class tensor [B, H, W] with filtered unconfident voxels set to -1 (ignore)
            agreement_mask: Boolean tensor [B, H, W] indicating accepted voxels
            telemetry: Dictionary containing mean agreement, confidence, and voxel acceptance rate
        """
        self.global_teacher.eval()
        self.local_teacher.eval()

        with torch.no_grad():
            logits_g = self.global_teacher(x_unlabeled)
            logits_l = self.local_teacher(x_unlabeled)

            p_g = F.softmax(logits_g, dim=1)
            p_l = F.softmax(logits_l, dim=1)

            agreement = compute_inter_teacher_agreement(p_g, p_l)     # [B, 1, H, W]
            confidence = compute_mutual_confidence(p_g, p_l)           # [B, 1, H, W]

            score = agreement * confidence                              # [B, 1, H, W]
            valid_mask = (score >= tau_t).squeeze(1)                   # [B, H, W] bool

            # Consensus pseudo-label is the argmax of the averaged teacher probabilities
            p_consensus = 0.5 * (p_g + p_l)
            pseudo_class = torch.argmax(p_consensus, dim=1)            # [B, H, W]

            # Unaccepted voxels receive -1 (ignored in loss computation)
            filtered_targets = torch.where(valid_mask, pseudo_class, torch.tensor(-1, device=self.device, dtype=torch.long))

            accepted_ratio = float(valid_mask.float().mean().item())
            mean_agreement = float(agreement.mean().item())
            mean_confidence = float(confidence.mean().item())

        telemetry = {
            "accepted_ratio": accepted_ratio,
            "mean_agreement": mean_agreement,
            "mean_confidence": mean_confidence,
            "p_consensus": p_consensus,
            "p_g": p_g,
            "p_l": p_l
        }
        return filtered_targets, valid_mask, telemetry
