import torch

from dnflow.losses import compute_losses
from dnflow.model import DNFlow


def test_model_forward_and_losses_are_finite():
    model = DNFlow(
        in_dim=9,
        num_dim=3,
        hidden_dim=16,
        layers=1,
        heads=2,
        feedforward_dim=32,
        dropout=0.0,
    )
    x_1 = torch.randn(8, 9)
    x_0 = torch.randn_like(x_1)
    t = torch.rand(8)
    x_t = t[:, None] * x_1 + (1 - t[:, None]) * x_0
    output = model(x_t, t)
    assert output["v_final"].shape == x_1.shape
    assert torch.count_nonzero(output["selector_full"][:, 3:]) == 0
    losses = compute_losses(
        output,
        x_1,
        x_1 - x_0,
        3,
        {
            "flow": 1.0,
            "driver": 0.1,
            "reconstruction": 0.1,
            "correlation": 0.05,
            "sparsity": 0.001,
            "binary": 0.001,
        },
    )
    assert all(torch.isfinite(value) for value in losses.values())


def test_reconstruction_loss_uses_selector_interpolated_endpoint():
    x_1 = torch.tensor([[1.0], [2.0]])
    output = {
        "v_final": torch.zeros(2, 1),
        "v_driver": torch.zeros(2, 1),
        "selector_num": torch.ones(1, 1),
        "raw_x_num": torch.tensor([[100.0], [100.0]]),
        "x_num_hat": x_1.clone(),
    }
    losses = compute_losses(
        output,
        x_1,
        torch.zeros_like(x_1),
        1,
        {
            "flow": 0.0,
            "driver": 0.0,
            "reconstruction": 1.0,
            "correlation": 0.0,
            "sparsity": 0.0,
            "binary": 0.0,
        },
    )
    assert torch.isclose(losses["reconstruction"], torch.tensor(0.0))
