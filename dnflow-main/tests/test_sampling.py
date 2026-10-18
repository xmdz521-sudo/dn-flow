import numpy as np
import torch

from dnflow.model import DNFlow
from dnflow.sampler import sample_heun


def test_small_heun_sample():
    model = DNFlow(
        in_dim=6,
        num_dim=2,
        hidden_dim=16,
        layers=1,
        heads=2,
        feedforward_dim=32,
        dropout=0.0,
    )
    result = sample_heun(model, 5, 6, 3, torch.device("cpu"), batch_size=2)
    assert result.shape == (5, 6)
    assert np.isfinite(result).all()
