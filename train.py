"""Train the network with a hand-written loop, compare it with two baselines,
run the experiments, and write everything to reports/.

Run it with:  python train.py
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader

from model import BikeNet
from prepare import SEED, Splits, make_loaders, make_splits

REPORTS = Path(__file__).parent / "reports"

# The main run is the network I believe in. Each experiment changes one thing.
# Settings were chosen on the validation rows only (lr 1e-3, 3e-3 and 1e-2 tried).
MAIN = {"name": "main", "lr": 3e-3, "hidden": 64, "n_layers": 2, "batch_size": 64}
EXPERIMENTS = [
    {**MAIN, "name": "lr_too_small", "lr": 1e-5},
    {**MAIN, "name": "lr_too_large", "lr": 1e-1},
    {**MAIN, "name": "too_wide", "hidden": 512},
]
EPOCHS = 300


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """One pass over the loader. Trains if given an optimizer, else only measures.

    Returns the mean loss and the mean absolute error in rentals.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, total_abs_error, count = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:  # 1. a batch
            prediction = model(x)  # 2. forward
            loss = loss_fn(prediction, y)  # 3. loss
            if training:
                optimizer.zero_grad()  # clear last step's gradients
                loss.backward()  # 4. backward
                optimizer.step()  # 5. step
            total_loss += loss.item() * len(y)  # weight by batch size
            total_abs_error += (prediction - y).abs().sum().item()
            count += len(y)
    return total_loss / count, total_abs_error / count


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> tuple[list[dict], int]:
    """Train for many epochs, log every epoch, and restore the best weights.

    Returns the history (one row per epoch) and the best epoch number.
    """
    history = []
    best_loss, best_epoch = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)  # go back to the best epoch
    return history, best_epoch


def baselines(splits: Splits) -> dict[str, dict[str, float]]:
    """Mean-of-training and linear regression, scored on the same rows."""
    x_train, y_train = splits.train.x.numpy(), splits.train.y.numpy()
    linear = LinearRegression().fit(x_train, y_train)
    mean = float(y_train.mean())
    results = {}
    for name, rows in (("val", splits.val), ("test", splits.test)):
        x, y = rows.x.numpy(), rows.y.numpy()
        results[name] = {
            "mean_baseline_mae": mean_absolute_error(y, np.full_like(y, mean)),
            "linear_regression_mae": mean_absolute_error(y, linear.predict(x)),
        }
    return results


def run(config: dict, splits: Splits) -> tuple[BikeNet, list[dict], int]:
    """Train one network from a fresh seed with the settings in config."""
    torch.manual_seed(SEED)
    train_loader, val_loader, _ = make_loaders(splits, config["batch_size"], SEED)
    model = BikeNet(splits.n_features, config["hidden"], config["n_layers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    history, best_epoch = train_model(
        model, train_loader, val_loader, nn.MSELoss(), optimizer, EPOCHS
    )
    print(
        f"{config['name']:14s} best epoch {best_epoch:3d}  "
        f"val MAE {history[best_epoch - 1]['val_mae']:.3f}"
    )
    return model, history, best_epoch


def plot_curves(history: list[dict], best_epoch: int, title: str, path: Path) -> None:
    """Save the training and validation loss for every epoch, marking the best."""
    frame = pd.DataFrame(history)
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    ax.plot(frame["epoch"], frame["train_loss"], label="training loss")
    ax.plot(frame["epoch"], frame["val_loss"], label="validation loss")
    ax.axvline(
        best_epoch, color="grey", linestyle="--", label=f"best epoch {best_epoch}"
    )
    ax.set(xlabel="epoch", ylabel="MSE loss (rentals squared)", title=title)
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    splits = make_splits(SEED)

    rows = []
    main_model = None
    for config in [MAIN, *EXPERIMENTS]:
        model, history, best_epoch = run(config, splits)
        name = config["name"]
        plot_curves(history, best_epoch, f"Run: {name}", REPORTS / f"curves_{name}.png")
        rows.append(
            {
                **config,
                "epochs": EPOCHS,
                "best_epoch": best_epoch,
                "best_val_mae": history[best_epoch - 1]["val_mae"],
                "final_train_loss": history[-1]["train_loss"],
                "final_val_loss": history[-1]["val_loss"],
            }
        )
        if name == "main":
            main_model = model
            pd.DataFrame(history).to_csv(REPORTS / "history.csv", index=False)
            plot_curves(
                history, best_epoch, "Main run", REPORTS / "learning_curves.png"
            )
    pd.DataFrame(rows).to_csv(REPORTS / "experiments.csv", index=False)

    # The test set is used once, for the main network, after everything is chosen.
    _, val_loader, test_loader = make_loaders(splits, MAIN["batch_size"], SEED)
    _, val_mae = run_epoch(main_model, val_loader, nn.MSELoss())
    _, test_mae = run_epoch(main_model, test_loader, nn.MSELoss())
    results = baselines(splits)
    results["val"]["network_mae"] = val_mae
    results["test"]["network_mae"] = test_mae
    metrics = {
        "seed": SEED,
        "torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
        "main_run": rows[0],
        "mae_in_rentals": results,
    }
    (REPORTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
