"""
Central federated coordinator executing aggregation, cryptographic audit, and global evaluation.
"""

from typing import List, Dict, Tuple, Any
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dtaa.core.metrics import (
    compute_parameters_sha256,
    compute_l2_norm,
    compute_delta_l2_norm,
    evaluate_segmentation_batch
)


class DTAAFederatedServerStrategy:
    """
    Central Federated Coordinator:
      - Aggregates client parameter matrices using sample-weighted averaging.
      - Maintains cryptographic SHA-256 tamper-evident telemetry logs.
      - Tracks global convergence L2 norms and parameter deltas.
      - Evaluates current global model on the 15% Unknown Global Test Set.
    """
    def __init__(self, global_model: nn.Module, global_test_loader: DataLoader, device: torch.device):
        self.global_model = copy.deepcopy(global_model).to(device)
        self.test_loader = global_test_loader
        self.device = device
        self.round_telemetry_log: List[Dict[str, Any]] = []

    def get_parameters(self) -> List[np.ndarray]:
        """Extracts current global model weights as NumPy arrays."""
        return [val.detach().cpu().numpy().copy() for _, val in self.global_model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Loads updated parameter arrays back into global model state_dict."""
        state_dict = {
            k: torch.tensor(p_np, dtype=v.dtype, device=self.device)
            for (k, v), p_np in zip(self.global_model.state_dict().items(), parameters)
        }
        self.global_model.load_state_dict(state_dict)

    def aggregate(self, client_results: List[Tuple[List[np.ndarray], int, Dict[str, Any]]]) -> List[np.ndarray]:
        """
        Federated sample-weighted parameter aggregation:
        
            theta_{t+1} = sum_{k=1}^K (n_k / N) * theta_{t, k}
        """
        prev_params = self.get_parameters()
        total_samples = sum(n for _, n, _ in client_results)
        agg_params = [np.zeros_like(layer, dtype=np.float64) for layer in client_results[0][0]]

        for c_params, n_samp, _ in client_results:
            w_k = float(n_samp) / float(total_samples)
            for l in range(len(agg_params)):
                agg_params[l] += c_params[l].astype(np.float64) * w_k

        final_params = [layer.astype(client_results[0][0][l].dtype) for l, layer in enumerate(agg_params)]
        self.round_telemetry_log.append({
            "prev_global_hash": compute_parameters_sha256(prev_params),
            "new_global_hash": compute_parameters_sha256(final_params),
            "new_global_l2_norm": round(compute_l2_norm(final_params), 4),
            "global_step_delta": round(compute_delta_l2_norm(final_params, prev_params), 4),
        })
        return final_params

    def ingest_onboarded_client_update(
        self,
        client_state_dict: Dict[str, torch.Tensor],
        client_samples: int,
        existing_federation_samples: int
    ) -> Dict[str, Any]:
        """
        Dynamically merges an onboarded hospital's locally adapted weights into the federation:
        
            theta_v2 = (N_exist / N_total) * theta_v1 + (N_onboard / N_total) * theta_onboard
        """
        prev_params = self.get_parameters()
        new_client_params = [client_state_dict[k].detach().cpu().numpy().copy() for k in self.global_model.state_dict().keys()]
        total_n = float(existing_federation_samples + client_samples)
        w_exist = float(existing_federation_samples) / total_n
        w_onb = float(client_samples) / total_n

        blended = [
            (w_exist * po.astype(np.float64) + w_onb * pn.astype(np.float64)).astype(po.dtype)
            for po, pn in zip(prev_params, new_client_params)
        ]
        self.set_parameters(blended)
        return {
            "w_existing": round(w_exist, 4),
            "w_onboarded": round(w_onb, 4),
            "prev_hash": compute_parameters_sha256(prev_params),
            "onboard_hash": compute_parameters_sha256(new_client_params),
            "merged_v2_hash": compute_parameters_sha256(blended),
            "merged_v2_l2": round(compute_l2_norm(blended), 4)
        }

    def evaluate_global(self) -> Dict[str, float]:
        """Evaluates current global model against the 15% Unknown Global Test Set."""
        self.global_model.eval()
        evals = []
        with torch.no_grad():
            for b in self.test_loader:
                evals.append(evaluate_segmentation_batch(self.global_model(b[0].to(self.device)), b[1].to(self.device)))
        return {k: float(np.mean([m[k] for m in evals])) for k in evals[0].keys()}
