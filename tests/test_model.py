"""Tests for the network."""

import pytest
import torch
from torch import nn

from model import BikeNet


@pytest.mark.parametrize("hidden", [4, 64, 256])
def test_network_returns_one_value_per_row(hidden):
    model = BikeNet(n_features=8, hidden=hidden)
    assert model(torch.randn(10, 8)).shape == (10,)


def test_one_optimizer_step_changes_every_layer(splits):
    torch.manual_seed(0)
    model = BikeNet(splits.n_features)
    before = {name: p.clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x, y = splits.train.x[:32], splits.train.y[:32]
    optimizer.zero_grad()
    nn.MSELoss()(model(x), y).backward()
    optimizer.step()
    for name, p in model.named_parameters():
        assert not torch.equal(before[name], p), f"{name} did not change"
