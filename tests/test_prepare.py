"""Tests for the split, the preparation, the Dataset and the loaders."""

import torch

from bikes_data import make_rentals
from prepare import NUMERIC, make_loaders, split_frame


def test_dataset_length_and_tensor_types(splits):
    x, y = splits.train[0]
    assert len(splits.train) == 3600
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert x.shape == (splits.n_features,) and y.shape == ()


def test_splits_have_expected_sizes_and_do_not_overlap():
    frame = make_rentals()
    train, val, test = split_frame(frame)
    assert (len(train), len(val), len(test)) == (3600, 1200, 1200)
    assert len(train) + len(val) + len(test) == len(frame)
    ids = [set(part.index) for part in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_scaler_was_fitted_on_training_rows_only(splits):
    n = len(NUMERIC)  # the scaled columns come first
    train_means = splits.train.x[:, :n].mean(dim=0)
    val_means = splits.val.x[:, :n].mean(dim=0)
    assert torch.allclose(train_means, torch.zeros(n), atol=1e-5)
    assert not torch.allclose(val_means, torch.zeros(n), atol=1e-3)


def test_training_loader_shuffles_but_is_repeatable(splits):
    first, _, _ = make_loaders(splits, seed=1)
    second, _, _ = make_loaders(splits, seed=1)
    batch_a, _ = next(iter(first))
    batch_b, _ = next(iter(second))
    assert torch.equal(batch_a, batch_b)  # same seed, same batch
    assert not torch.equal(batch_a, splits.train.x[:64])  # not in table order
