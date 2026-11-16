"""Shared fixtures for the tests."""

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from prepare import make_splits


@pytest.fixture(scope="session")
def splits():
    """The real prepared splits, built once for the whole test run."""
    return make_splits()


@pytest.fixture
def small_loader(splits):
    """One small batch of 16 training rows, for quick training checks."""
    torch.manual_seed(0)
    return DataLoader(Subset(splits.train, range(16)), batch_size=16)
