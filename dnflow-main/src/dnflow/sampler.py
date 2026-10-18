from typing import Optional

import numpy as np
import torch


def sample_heun(
    model: torch.nn.Module,
    n_samples: int,
    in_dim: int,
    steps: int,
    device: torch.device,
    batch_size: int = 512,
    self_conditioning: bool = False,
) -> np.ndarray:
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            size = min(batch_size, n_samples - start)
            x_t = torch.randn(size, in_dim, device=device)
            self_cond: Optional[torch.Tensor] = None
            dt = 1.0 / steps
            for step in range(steps):
                t_value = step / steps
                t = torch.full((size,), t_value, device=device)
                first = model(x_t, t, self_cond)
                euler = x_t + first["v_final"] * dt
                if step < steps - 1:
                    next_t = torch.full((size,), (step + 1) / steps, device=device)
                    second_cond = first["x1_hat"] if self_conditioning else None
                    second = model(euler, next_t, second_cond)
                    x_t = x_t + 0.5 * (
                        first["v_final"] + second["v_final"]
                    ) * dt
                else:
                    x_t = euler
                self_cond = first["x1_hat"] if self_conditioning else None
            batches.append(torch.clamp(x_t, -5.0, 5.0).cpu().numpy())
    return np.concatenate(batches, axis=0)[:n_samples]
