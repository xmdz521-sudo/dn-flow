from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .backbone import TimeConditionedTransformer


class DNFlow(nn.Module):
    """Formal shared DN-Flow model from the paper implementation."""

    def __init__(
        self,
        in_dim: int,
        num_dim: int,
        hidden_dim: int = 512,
        layers: int = 6,
        heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.05,
        alpha: float = 0.05,
        selector_init_bias: float = 0.0,
    ):
        super().__init__()
        if not 0 <= num_dim <= in_dim:
            raise ValueError("num_dim must be between zero and in_dim")
        self.in_dim = in_dim
        self.num_dim = num_dim
        self.cat_dim = in_dim - num_dim
        self.alpha = float(alpha)
        self.backbone = TimeConditionedTransformer(
            in_dim, hidden_dim, layers, heads, feedforward_dim, dropout
        )
        self.head_driver_v = nn.Linear(hidden_dim, in_dim)
        self.head_nav_num = (
            nn.Linear(hidden_dim, num_dim) if num_dim > 0 else None
        )
        self.head_gate = nn.Sequential(
            nn.Linear(hidden_dim + in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
            nn.Sigmoid(),
        )
        if num_dim > 0:
            self.selector_logits = nn.Parameter(
                torch.full((1, num_dim), float(selector_init_bias))
            )
        else:
            self.register_parameter("selector_logits", None)

    def get_soft_selector(self) -> torch.Tensor:
        if self.selector_logits is None:
            return torch.zeros(1, 0, device=next(self.parameters()).device)
        return torch.sigmoid(self.selector_logits)

    def get_selector_scores(self) -> np.ndarray:
        return self.get_soft_selector().detach().cpu().numpy().reshape(-1)

    def _full_selector(self, x_t: torch.Tensor) -> torch.Tensor:
        selector = self.get_soft_selector().to(x_t.device)
        if self.cat_dim:
            zeros = torch.zeros(
                1, self.cat_dim, dtype=x_t.dtype, device=x_t.device
            )
            selector = torch.cat((selector, zeros), dim=1)
        return selector

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, self_cond: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        h = self.backbone(x_t, t, self_cond)
        v_driver = self.head_driver_v(h)
        selector_num = self.get_soft_selector().to(x_t.device)
        if self.num_dim:
            raw_x_num = self.head_nav_num(h)
            x_num_hat = selector_num * raw_x_num + (1.0 - selector_num) * x_t[:, : self.num_dim]
        else:
            raw_x_num = x_t[:, :0]
            x_num_hat = raw_x_num
        # Categorical one-hot dimensions remain on the Driver path.
        x1_hat = torch.cat((x_num_hat, x_t[:, self.num_dim :]), dim=1)
        gate = self.head_gate(torch.cat((h, x_t), dim=-1))
        route_v = (x1_hat - x_t) / torch.clamp(
            1.0 - t.view(-1, 1), min=0.05
        )
        selector_full = self._full_selector(x_t)
        v_final = v_driver + self.alpha * selector_full * gate * (
            route_v - v_driver
        )
        return {
            "v_final": v_final,
            "v_driver": v_driver,
            "x1_hat": x1_hat,
            "raw_x_num": raw_x_num,
            "x_num_hat": x_num_hat,
            "selector_num": selector_num,
            "selector_full": selector_full,
            "gate": gate,
            "route_v": route_v,
        }
