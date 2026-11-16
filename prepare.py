"""Split the rentals table, prepare the features, and feed them to PyTorch.

The order matters: the table is split first, and the scaler and the encoder
learn only from the training rows. Validation and test rows are transformed
with what the training rows taught them, so nothing leaks backwards.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET, make_rentals

SEED = 42
NUMERIC = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORICAL = ["station_type"]


def split_frame(
    frame: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the rows 60% / 20% / 20% into training, validation and test."""
    train, rest = train_test_split(frame, test_size=0.4, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, random_state=seed)
    return train, val, test


def make_preprocessor() -> ColumnTransformer:
    """Scale the numeric columns and one-hot encode the station type."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )


class RentalsDataset(Dataset):
    """Prepared features and rental counts, handed out one row at a time."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


@dataclass
class Splits:
    """The three prepared datasets, plus what was fitted to make them."""

    train: RentalsDataset
    val: RentalsDataset
    test: RentalsDataset
    preprocessor: ColumnTransformer

    @property
    def n_features(self) -> int:
        return self.train.x.shape[1]


def make_splits(seed: int = SEED) -> Splits:
    """Split the table, fit the preparation on training rows, build datasets."""
    train, val, test = split_frame(make_rentals(), seed)
    preprocessor = make_preprocessor()
    preprocessor.fit(train[NUMERIC + CATEGORICAL])  # training rows only

    def to_dataset(rows: pd.DataFrame) -> RentalsDataset:
        features = preprocessor.transform(rows[NUMERIC + CATEGORICAL])
        return RentalsDataset(features, rows[TARGET].to_numpy())

    return Splits(to_dataset(train), to_dataset(val), to_dataset(test), preprocessor)


def make_loaders(
    splits: Splits, batch_size: int = 64, seed: int = SEED
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Shuffle the training batches only, with a seeded generator."""
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        splits.train, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(splits.val, batch_size=256)
    test_loader = DataLoader(splits.test, batch_size=256)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    splits = make_splits()
    print(
        "rows train / val / test:", len(splits.train), len(splits.val), len(splits.test)
    )
    print("features per row:", splits.n_features)
    train_loader, _, _ = make_loaders(splits)
    xb, yb = next(iter(train_loader))
    print("one batch:", tuple(xb.shape), xb.dtype, tuple(yb.shape), yb.dtype)
    print("batches per epoch:", len(train_loader))
