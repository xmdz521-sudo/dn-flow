from typing import Dict

import torch
import torch.nn.functional as functional


def _safe_corr(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    centered = x - x.mean(dim=0, keepdim=True)
    covariance = centered.T.mm(centered) / max(x.shape[0] - 1, 1)
    scale = torch.sqrt(torch.clamp(torch.diag(covariance), min=eps))
    return covariance / torch.clamp(scale[:, None] * scale[None, :], min=eps)


def compute_losses(
    output: Dict[str, torch.Tensor],
    x_1: torch.Tensor,
    target_v: torch.Tensor,
    num_dim: int,
    weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    flow = functional.mse_loss(output["v_final"], target_v)
    driver = functional.mse_loss(output["v_driver"], target_v)
    selector = output["selector_num"]
    zero = flow.new_zeros(())
    reconstruction = zero
    correlation = zero
    sparsity = zero
    binary = zero
    if num_dim:
        squared = (output["x_num_hat"] - x_1[:, :num_dim]).pow(2)
        denominator = squared.shape[0] * selector.sum() + 1e-6
        reconstruction = (squared * selector).sum() / denominator
        if num_dim >= 2 and weights["correlation"] > 0:
            corr_hat = _safe_corr(output["x_num_hat"])
            corr_true = _safe_corr(x_1[:, :num_dim])
            column_weights = selector.squeeze(0)
            pair_weights = column_weights[:, None] * column_weights[None, :]
            correlation = (
                (corr_hat - corr_true).pow(2) * pair_weights
            ).sum() / (pair_weights.sum() + 1e-6)
        sparsity = selector.mean()
        binary = (selector * (1.0 - selector)).mean()
    total = (
        weights["flow"] * flow
        + weights["driver"] * driver
        + weights["reconstruction"] * reconstruction
        + weights["correlation"] * correlation
        + weights["sparsity"] * sparsity
        + weights["binary"] * binary
    )
    return {
        "total": total,
        "flow": flow,
        "driver": driver,
        "reconstruction": reconstruction,
        "correlation": correlation,
        "sparsity": sparsity,
        "binary": binary,
    }
