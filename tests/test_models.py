"""
Unit tests for TraumaUNet and Agreement gating logic.
"""

import unittest
import torch
from dtaa.models.trauma_unet import TraumaUNet, _get_num_groups
from dtaa.core.agreement import (
    compute_inter_teacher_agreement,
    compute_mutual_confidence,
    compute_dynamic_threshold,
    compute_pseudo_weight
)
from dtaa.core.losses import CompoundFocalDiceLoss


def test_trauma_unet_shape_and_params():
    model = TraumaUNet(in_channels=1, num_classes=6, feature_channels=[8, 16, 32, 64, 128])
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 6, 64, 64), f"Expected shape (2, 6, 64, 64), got {out.shape}"

    num_params = sum(p.numel() for p in model.parameters())
    assert 400_000 < num_params < 600_000, f"Expected ~509k parameters, got {num_params}"


def test_group_norm_divisors():
    assert _get_num_groups(8) == 8
    assert _get_num_groups(16) == 8
    assert _get_num_groups(32) == 8
    assert _get_num_groups(12) == 4
    assert _get_num_groups(7) == 1


def test_agreement_and_confidence():
    p_g = torch.zeros(2, 6, 32, 32)
    p_g[:, 1, :, :] = 1.0  # Global teacher predicts Liver with prob 1.0

    p_l = torch.zeros(2, 6, 32, 32)
    p_l[:, 1, :, :] = 1.0  # Local teacher also predicts Liver with prob 1.0

    agreement = compute_inter_teacher_agreement(p_g, p_l)
    confidence = compute_mutual_confidence(p_g, p_l)

    # Identical predictions must yield perfect agreement (1.0) and confidence (1.0)
    assert torch.allclose(agreement, torch.ones_like(agreement))
    assert torch.allclose(confidence, torch.ones_like(confidence))


def test_cosine_threshold_schedule():
    t_start = compute_dynamic_threshold(1, 20, tau_min=0.70, tau_max=0.95)
    t_end = compute_dynamic_threshold(20, 20, tau_min=0.70, tau_max=0.95)
    assert round(t_start, 2) == 0.95
    assert round(t_end, 2) == 0.70


def test_compound_loss():
    loss_fn = CompoundFocalDiceLoss(num_classes=6)
    logits = torch.randn(2, 6, 32, 32)
    targets = torch.randint(0, 6, (2, 32, 32))
    # Inject some ignore index voxels
    targets[0, 0:5, 0:5] = -1

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


if __name__ == "__main__":
    test_trauma_unet_shape_and_params()
    test_group_norm_divisors()
    test_agreement_and_confidence()
    test_cosine_threshold_schedule()
    test_compound_loss()
    print("All unit tests passed successfully!")
