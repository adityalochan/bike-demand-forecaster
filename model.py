"""The network: a stack of Linear layers with ReLU, ending in one number."""

import torch
from torch import nn


class BikeNet(nn.Module):
    """Predicts the number of bikes rented in one station-hour.

    The last layer has no activation function, because the target is a count
    that can be any size, not a probability squeezed between 0 and 1.
    """

    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width_in = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(width_in, hidden), nn.ReLU()]
            width_in = hidden
        layers.append(nn.Linear(width_in, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # squeeze(1) turns shape (batch, 1) into (batch,), matching the targets
        return self.layers(x).squeeze(1)
