"""Core algorithms, loss functions, metrics, and teacher management for DTAA."""
from dtaa.core.dual_teacher import DualTeacherManager
from dtaa.core.agreement import (
    compute_inter_teacher_agreement,
    compute_mutual_confidence,
    compute_dynamic_threshold,
    compute_pseudo_weight
)
from dtaa.core.losses import SoftDiceLoss, FocalLoss, CompoundFocalDiceLoss
from dtaa.core.metrics import compute_multiorgan_dice, compute_parameters_sha256

__all__ = [
    "DualTeacherManager",
    "compute_inter_teacher_agreement",
    "compute_mutual_confidence",
    "compute_dynamic_threshold",
    "compute_pseudo_weight",
    "SoftDiceLoss",
    "FocalLoss",
    "CompoundFocalDiceLoss",
    "compute_multiorgan_dice",
    "compute_parameters_sha256",
]
