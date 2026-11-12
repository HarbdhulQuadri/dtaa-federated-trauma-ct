"""
DTAA: Dual-Teacher Adaptive Agreement for Semi-Supervised Federated Segmentation
Abdominal Trauma CT Multi-Organ Segmentation under Non-IID Scanner Shifts
Author: Abdulquadri Olanrewaju Adegbiji
Institution: University of Lagos, Department of Computer Science
Supervisor: Dr. F.O. Alamu
"""

__version__ = "1.0.0"
__author__ = "Abdulquadri Olanrewaju Adegbiji"

from dtaa.models.trauma_unet import TraumaUNet
from dtaa.core.dual_teacher import DualTeacherManager
from dtaa.core.agreement import (
    compute_inter_teacher_agreement,
    compute_mutual_confidence,
    compute_dynamic_threshold,
    compute_pseudo_weight
)
from dtaa.core.losses import SoftDiceLoss, FocalLoss, CompoundFocalDiceLoss
from dtaa.federated.client import DTAAFederatedClient
from dtaa.federated.server import DTAAFederatedServerStrategy
