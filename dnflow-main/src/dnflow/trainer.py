import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from .losses import compute_losses


def train_model(
    model: torch.nn.Module,
    train_data: np.ndarray,
    validation_data: np.ndarray,
    config: Dict,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, float]:
    training = config["training"]
    loss_weights = config["loss"]
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data).float()),
        batch_size=training["batch_size"],
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation_data).float()),
        batch_size=training["batch_size"],
        shuffle=False,
    )
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training["scheduler_factor"],
        patience=training["scheduler_patience"],
        min_lr=training["minimum_learning_rate"],
    )
    best = float("inf")
    last = {}
    for epoch in range(1, training["epochs"] + 1):
        model.train()
        train_sum = 0.0
        for (x_1,) in train_loader:
            x_1 = x_1.to(device)
            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.shape[0], device=device)
            x_t = t[:, None] * x_1 + (1.0 - t[:, None]) * x_0
            target = x_1 - x_0
            self_cond = None
            probability = training["self_conditioning_probability"]
            if probability > 0 and random.random() < probability:
                with torch.no_grad():
                    self_cond = model(x_t, t)["x1_hat"].detach()
            losses = compute_losses(
                model(x_t, t, self_cond), x_1, target, model.num_dim, loss_weights
            )
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training["gradient_clip"]
            )
            optimizer.step()
            train_sum += losses["total"].item() * len(x_1)
        model.eval()
        validation_sum = 0.0
        with torch.no_grad():
            for (x_1,) in validation_loader:
                x_1 = x_1.to(device)
                x_0 = torch.randn_like(x_1)
                t = torch.rand(x_1.shape[0], device=device)
                x_t = t[:, None] * x_1 + (1.0 - t[:, None]) * x_0
                target = x_1 - x_0
                losses = compute_losses(
                    model(x_t, t), x_1, target, model.num_dim, loss_weights
                )
                validation_sum += losses["total"].item() * len(x_1)
        last = {
            "train_loss": train_sum / len(train_data),
            "validation_loss": validation_sum / len(validation_data),
        }
        scheduler.step(last["validation_loss"])
        if last["validation_loss"] < best:
            best = last["validation_loss"]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": config}, checkpoint_path)
        print(
            "epoch={}/{} train={:.6f} validation={:.6f}".format(
                epoch, training["epochs"], last["train_loss"], last["validation_loss"]
            )
        )
    last["best_validation_loss"] = best
    return last
