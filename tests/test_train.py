"""Tests for the epoch function and the training function."""

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from model import BikeNet
from train import run_epoch, train_model


def test_network_can_memorize_one_small_batch(splits, small_loader):
    # Fails if zero_grad() or step() is missing from run_epoch.
    model = BikeNet(splits.n_features)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    start, _ = run_epoch(model, small_loader, nn.MSELoss())
    for _ in range(300):
        run_epoch(model, small_loader, nn.MSELoss(), optimizer)
    end, _ = run_epoch(model, small_loader, nn.MSELoss())
    assert end < 0.05 * start


def test_validation_epoch_leaves_weights_unchanged(splits, small_loader):
    model = BikeNet(splits.n_features)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    run_epoch(model, small_loader, nn.MSELoss())  # no optimizer: measure only
    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value)


def test_train_model_logs_every_epoch_and_restores_best_weights(splits):
    torch.manual_seed(0)
    train_loader = DataLoader(Subset(splits.train, range(256)), batch_size=64)
    val_loader = DataLoader(Subset(splits.val, range(256)), batch_size=256)
    model = BikeNet(splits.n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # deliberately jumpy
    history, best_epoch = train_model(
        model, train_loader, val_loader, nn.MSELoss(), optimizer, epochs=8
    )
    assert len(history) == 8
    assert set(history[0]) == {"epoch", "train_loss", "val_loss", "val_mae"}
    best_val = min(row["val_loss"] for row in history)
    assert history[best_epoch - 1]["val_loss"] == best_val
    restored_val, _ = run_epoch(model, val_loader, nn.MSELoss())
    assert abs(restored_val - best_val) < 1e-4
