"""
Non-IID federated dataset partitioner across heterogeneous hospital scanner silos.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader

from dtaa.data.dataset import RSNATraumaDataset


@dataclass
class ClientDatasetSplit:
    """Encapsulates dataset partitions for a single hospital federated client."""
    client_id: int
    name: str
    labeled_ratio: float
    train_labeled_loader: DataLoader
    train_unlabeled_loader: DataLoader
    val_loader: DataLoader
    num_labeled: int
    num_unlabeled: int
    num_val: int


class NonIIDPartitioner:
    """
    Partitions cohort into:
      - 15% Held-Out Unknown Global Test Set (evaluated across all rounds)
      - 4 Heterogeneous Hospital Clients with varying scanner profiles and label ratios:
        * Node 0: Teaching Hospital (Siemens - 80% labeled)
        * Node 1: Community Hospital A (GE - 25% labeled)
        * Node 2: Trauma Center B (Philips - 15% labeled)
        * Node 3: Regional Emergency C (Canon Onboarding - 8% labeled)
    """
    def __init__(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        num_clients: int = 4,
        labeled_ratios: Optional[List[float]] = None,
        batch_size: int = 8,
        seed: int = 42
    ):
        self.images = images
        self.masks = masks
        self.metadata = metadata
        self.num_clients = num_clients
        self.labeled_ratios = labeled_ratios or [0.80, 0.25, 0.15, 0.08]
        self.batch_size = batch_size
        self.seed = seed

    def create_partitions(self) -> Tuple[List[ClientDatasetSplit], DataLoader]:
        rng = np.random.RandomState(self.seed)
        indices = np.arange(len(self.images))
        rng.shuffle(indices)

        # 15% completely unseen Global Test Set
        num_test = max(int(0.15 * len(self.images)), 1)
        test_indices = indices[:num_test]
        client_pool_indices = indices[num_test:]

        test_ds = RSNATraumaDataset(
            [self.images[i] for i in test_indices],
            [self.masks[i] for i in test_indices],
            [self.metadata[i] for i in test_indices]
        )
        global_test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        client_chunks = np.array_split(client_pool_indices, self.num_clients)
        hospital_names = [
            "Node 0: Teaching Hospital (Siemens - 80% Labeled)",
            "Node 1: Community Hospital A (GE - 25% Labeled)",
            "Node 2: Trauma Center B (Philips - 15% Labeled)",
            "Node 3: Regional Emergency C (Canon - 8% Labeled)"
        ]

        client_splits = []
        for k in range(self.num_clients):
            k_indices = client_chunks[k]
            rng.shuffle(k_indices)

            num_k_val = max(int(0.20 * len(k_indices)), 1)
            k_val_idx = k_indices[:num_k_val]
            k_train_idx = k_indices[num_k_val:]

            k_ratio = self.labeled_ratios[k % len(self.labeled_ratios)]
            num_k_labeled = max(int(k_ratio * len(k_train_idx)), 1)

            k_labeled_idx = k_train_idx[:num_k_labeled]
            k_unlabeled_idx = k_train_idx[num_k_labeled:]

            labeled_ds = RSNATraumaDataset(
                [self.images[i] for i in k_labeled_idx],
                [self.masks[i] for i in k_labeled_idx],
                [self.metadata[i] for i in k_labeled_idx]
            )
            unlabeled_ds = RSNATraumaDataset(
                [self.images[i] for i in k_unlabeled_idx],
                [None for _ in k_unlabeled_idx],
                [self.metadata[i] for i in k_unlabeled_idx]
            )
            val_ds = RSNATraumaDataset(
                [self.images[i] for i in k_val_idx],
                [self.masks[i] for i in k_val_idx],
                [self.metadata[i] for i in k_val_idx]
            )

            labeled_loader = DataLoader(labeled_ds, batch_size=self.batch_size, shuffle=True)
            unlabeled_loader = DataLoader(unlabeled_ds, batch_size=self.batch_size, shuffle=True) if len(k_unlabeled_idx) > 0 else labeled_loader
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

            client_splits.append(ClientDatasetSplit(
                client_id=k,
                name=hospital_names[k],
                labeled_ratio=k_ratio,
                train_labeled_loader=labeled_loader,
                train_unlabeled_loader=unlabeled_loader,
                val_loader=val_loader,
                num_labeled=len(k_labeled_idx),
                num_unlabeled=len(k_unlabeled_idx),
                num_val=len(k_val_idx)
            ))

        return client_splits, global_test_loader
