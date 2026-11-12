"""
PyTorch Dataset for RSNA Abdominal Trauma CT cohort.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset


class RSNATraumaDataset(Dataset):
    """
    PyTorch Dataset yielding (image_tensor, mask_tensor, metadata_dict).
    
    When masks=None (unlabeled client splits), mask_tensor is filled with -1 (ignored).
    Images are shaped [1, H, W] for single-channel input or [C_in, H, W] for multi-channel input.
    """
    def __init__(
        self,
        images: List[np.ndarray],
        masks: Optional[List[Optional[np.ndarray]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        self.images = images
        self.masks = masks
        self.metadata = metadata or [{} for _ in range(len(images))]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_np = self.images[idx]
        if img_np.ndim == 2:
            img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)
        else:
            img_tensor = torch.tensor(img_np, dtype=torch.float32)

        if self.masks is not None and self.masks[idx] is not None:
            mask_tensor = torch.tensor(self.masks[idx], dtype=torch.long)
        else:
            mask_tensor = torch.full((img_np.shape[-2], img_np.shape[-1]), -1, dtype=torch.long)

        return img_tensor, mask_tensor, self.metadata[idx]
