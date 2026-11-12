"""
Hospital edge federated client executing DTAA local optimization and pseudo-labeling.
"""

from typing import List, Dict, Tuple, Optional, Any
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dtaa.core.dual_teacher import DualTeacherManager
from dtaa.core.agreement import compute_dynamic_threshold, compute_pseudo_weight
from dtaa.core.losses import CompoundFocalDiceLoss
from dtaa.core.metrics import evaluate_segmentation_batch
from dtaa.data.partitioner import ClientDatasetSplit


class DTAAFederatedClient:
    """
    Federated client running on local hospital hardware.
    
    Supports 3 modes:
      - 'dtaa': Dual-Teacher Adaptive Agreement with Dynamic Cosine Thresholding
      - 'vanilla_fssl': Single-Teacher FixMatch pseudo-labeling
      - 'supervised_only': Standard FedAvg baseline (ignores unlabeled scans)
    """
    def __init__(
        self,
        client_id: int,
        split: ClientDatasetSplit,
        base_model: nn.Module,
        device: torch.device,
        learning_rate: float = 6e-3,
        ema_alpha: float = 0.90,
        mode: str = "dtaa"
    ):
        self.client_id = client_id
        self.split = split
        self.device = device
        self.mode = mode
        self.teacher_manager = DualTeacherManager(base_model, device, ema_alpha=ema_alpha)
        self.optimizer = torch.optim.AdamW(
            self.teacher_manager.student.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        self.criterion = CompoundFocalDiceLoss(
            num_classes=6,
            dice_weight=0.60,
            focal_weight=0.40,
            alpha_weights=[0.2, 1.0, 1.5, 2.0, 1.4, 5.0]
        )

    def get_parameters(self) -> List[np.ndarray]:
        return [val.detach().cpu().numpy().copy() for _, val in self.teacher_manager.student.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        state_dict = {
            k: torch.tensor(p_np, dtype=v.dtype, device=self.device)
            for (k, v), p_np in zip(self.teacher_manager.student.state_dict().items(), parameters)
        }
        self.teacher_manager.student.load_state_dict(state_dict)
        self.teacher_manager.update_global_anchor(state_dict)

    def train_epoch(
        self,
        tau_t: float,
        lambda_pseudo: float,
        gt_lookup: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        self.teacher_manager.student.train()
        tot_loss, tot_sup, tot_unsup = 0.0, 0.0, 0.0
        acc_vox_count, tot_vox_count = 0, 0
        agree_accum = []
        conf_bias_fp = 0

        unlab_iter = iter(self.split.train_unlabeled_loader)

        for x_lab, y_lab, _ in self.split.train_labeled_loader:
            x_lab, y_lab = x_lab.to(self.device), y_lab.to(self.device)
            self.optimizer.zero_grad()

            logits_lab = self.teacher_manager.student(x_lab)
            l_sup = self.criterion(logits_lab, y_lab)

            l_unsup = torch.tensor(0.0, device=self.device)
            if self.mode != "supervised_only":
                try:
                    x_unlab, _, meta_unlab = next(unlab_iter)
                except StopIteration:
                    unlab_iter = iter(self.split.train_unlabeled_loader)
                    x_unlab, _, meta_unlab = next(unlab_iter)

                x_unlab = x_unlab.to(self.device)

                if self.mode == "dtaa":
                    pseudo_y, v_mask, telem = self.teacher_manager.generate_agreement_pseudo_labels(x_unlab, tau_t)
                    logits_unlab = self.teacher_manager.student(x_unlab)
                    l_unsup = self.criterion(
                        logits_unlab,
                        pseudo_y,
                        p_consensus=telem["p_consensus"],
                        manifold_weight=0.10
                    )
                    agree_accum.append(telem["mean_agreement"])
                    acc_vox_count += int(v_mask.sum().item())
                    tot_vox_count += v_mask.numel()
                else:  # vanilla_fssl
                    with torch.no_grad():
                        logits_vanilla = self.teacher_manager.student(x_unlab)
                        p_vanilla = F.softmax(logits_vanilla, dim=1)
                        max_p, pseudo_y = torch.max(p_vanilla, dim=1)
                        v_mask = (max_p >= 0.85)
                        pseudo_y = torch.where(v_mask, pseudo_y, torch.tensor(-1, device=self.device))

                    logits_unlab = self.teacher_manager.student(x_unlab)
                    l_unsup = self.criterion(logits_unlab, pseudo_y)
                    acc_vox_count += int(v_mask.sum().item())
                    tot_vox_count += v_mask.numel()

            loss = l_sup + lambda_pseudo * l_unsup
            loss.backward()
            self.optimizer.step()

            if self.mode == "dtaa":
                self.teacher_manager.update_local_ema()

            tot_loss += float(loss.item())
            tot_sup += float(l_sup.item())
            tot_unsup += float(l_unsup.item())

        n_batches = max(len(self.split.train_labeled_loader), 1)
        accepted_ratio = (float(acc_vox_count) / float(tot_vox_count)) if tot_vox_count > 0 else 0.0

        return {
            "loss": tot_loss / n_batches,
            "loss_sup": tot_sup / n_batches,
            "loss_unsup": tot_unsup / n_batches,
            "accepted_voxels_pct": round(100.0 * accepted_ratio, 2),
            "mean_agreement": float(np.mean(agree_accum)) if agree_accum else 1.0,
            "confirmation_bias_fp": conf_bias_fp
        }

    def evaluate_local(self) -> Dict[str, float]:
        self.teacher_manager.student.eval()
        val_metrics = []
        with torch.no_grad():
            for x_val, y_val, _ in self.split.val_loader:
                logits = self.teacher_manager.student(x_val.to(self.device))
                val_metrics.append(evaluate_segmentation_batch(logits, y_val.to(self.device)))
        return {k: float(np.mean([m[k] for m in val_metrics])) for k in val_metrics[0].keys()}

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
        gt_lookup: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        self.set_parameters(parameters)
        rnd = int(config.get("current_round", 1))
        tot = int(config.get("total_rounds", 10))
        epochs = int(config.get("local_epochs", 1))
        tau_t = compute_dynamic_threshold(rnd, tot)
        lam_p = compute_pseudo_weight(rnd, tot)

        last_m = {}
        for _ in range(epochs):
            last_m = self.train_epoch(tau_t=tau_t, lambda_pseudo=lam_p, gt_lookup=gt_lookup)

        val_ev = self.evaluate_local()
        n_samp = self.split.num_labeled if self.mode == "supervised_only" else (self.split.num_labeled + self.split.num_unlabeled)
        return self.get_parameters(), n_samp, {
            "client_id": self.client_id,
            "client_name": self.split.name,
            **last_m,
            "val_mean_foreground_dice": val_ev["mean_foreground_dice"],
            "val_hd95": val_ev["hd95"]
        }
