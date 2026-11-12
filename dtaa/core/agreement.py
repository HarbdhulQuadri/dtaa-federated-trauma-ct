"""
Mathematical agreement, confidence, and annealing curriculum functions for DTAA.
"""

import math
import torch


def compute_inter_teacher_agreement(p_global: torch.Tensor, p_local: torch.Tensor) -> torch.Tensor:
    """
    Computes inter-teacher consensus A(v) as the complement of Total Variation Distance:
    
        A(v) = 1 - 0.5 * sum_{c=1}^C | p_g^(c)(v) - p_l^(c)(v) |  in [0.0, 1.0]
        
    Args:
        p_global: Softmax probabilities from Global Anatomy Teacher [B, C, H, W]
        p_local: Softmax probabilities from Local Scanner Teacher [B, C, H, W]
        
    Returns:
        Agreement map tensor [B, 1, H, W] in range [0.0, 1.0]
    """
    tv_distance = 0.5 * torch.sum(torch.abs(p_global - p_local), dim=1, keepdim=True)
    agreement = torch.clamp(1.0 - tv_distance, min=0.0, max=1.0)
    return agreement


def compute_mutual_confidence(p_global: torch.Tensor, p_local: torch.Tensor) -> torch.Tensor:
    """
    Computes mutual geometric confidence C(v):
    
        C(v) = sqrt( max_c p_g^(c)(v) * max_c p_l^(c)(v) )  in [0.0, 1.0]
        
    Args:
        p_global: Softmax probabilities from Global Anatomy Teacher [B, C, H, W]
        p_local: Softmax probabilities from Local Scanner Teacher [B, C, H, W]
        
    Returns:
        Confidence map tensor [B, 1, H, W] in range [0.0, 1.0]
    """
    conf_g, _ = torch.max(p_global, dim=1, keepdim=True)
    conf_l, _ = torch.max(p_local, dim=1, keepdim=True)
    mutual_conf = torch.sqrt(torch.clamp(conf_g * conf_l, min=1e-8, max=1.0))
    return mutual_conf


def compute_dynamic_threshold(
    current_round: int,
    total_rounds: int,
    tau_min: float = 0.70,
    tau_max: float = 0.95
) -> float:
    """
    Curriculum Cosine Annealing Threshold:
    
        tau(t) = tau_min + 0.5 * (tau_max - tau_min) * (1 + cos(pi * t / T))
        
    Starts strict (tau_max = 0.95) to prevent early confirmation bias poisoning,
    then relaxes toward tau_min = 0.70 to supervise fine organ boundaries.
    """
    if total_rounds <= 1:
        return tau_max
    progress = float(current_round - 1) / float(max(total_rounds - 1, 1))
    cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return tau_min + (tau_max - tau_min) * cos_decay


def compute_pseudo_weight(
    current_round: int,
    total_rounds: int,
    lambda_max: float = 1.0
) -> float:
    """
    Gaussian ramp-up for unsupervised loss weight lambda_u(t):
    
        lambda_u(t) = lambda_max * exp(-5 * (1 - t / T)^2)
    """
    if total_rounds <= 1:
        return lambda_max
    progress = float(current_round) / float(total_rounds)
    return lambda_max * math.exp(-5.0 * ((1.0 - progress) ** 2))
