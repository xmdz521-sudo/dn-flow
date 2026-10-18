import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(
            torch.arange(half_dim, device=time.device, dtype=time.dtype) * -scale
        )
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class TimeConditionedTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.selfcond_proj = nn.Linear(in_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, self_cond: torch.Tensor = None
    ) -> torch.Tensor:
        h = self.input_proj(x_t) + self.time_mlp(t)
        if self_cond is not None:
            h = h + self.selfcond_proj(self_cond)
        return self.transformer(h.unsqueeze(1)).squeeze(1)
