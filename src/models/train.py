"""
Training infrastructure for ReturnPredictor.

Key design decisions (see LEARNING.md §4.2):

ChronologicalBatchSampler
  Batches are sampled within each calendar month, never across months.
  Months are iterated in chronological order. This enforces the temporal
  ordering constraint: the model never sees future data during training.
  Stocks within a month are shuffled to reduce gradient correlation.

Validation split
  The last ~20% of the training period by time (val_start in DataConfig,
  default 1992-01) is held out for early stopping. It is NOT used for
  gradient updates. This is distinct from the calibration set.

Checkpointing
  Saves every N epochs AND whenever a new validation-loss minimum is found.
  On CPU, a full training run may take many hours; checkpointing lets you
  resume after interruption.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

from src.utils.config import DataConfig, ModelConfig


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StockReturnDataset(Dataset):
    """
    Panel dataset of (features, return) pairs.

    NaN values in features are imputed to 0, which equals the cross-sectional
    median after rank normalisation to [-1, 1].  This matches Gu et al. (2020).
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list[str]):
        X = df[feature_cols].fillna(0.0).values.astype(np.float32)
        y = df["ret"].values.astype(np.float32)

        # Drop rows where the return itself is NaN (stock delisted mid-month)
        valid = ~np.isnan(y)
        self.X = torch.from_numpy(X[valid])
        self.y = torch.from_numpy(y[valid])

        # Keep date/permno arrays (numpy) for the sampler and prediction output
        self.dates   = df["date"].values[valid]
        self.permnos = df["permno"].values[valid]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Chronological batch sampler
# ---------------------------------------------------------------------------

class ChronologicalBatchSampler(Sampler):
    """
    Yield batches that:
      1. Never span more than one calendar month.
      2. Are emitted in chronological month order.
      3. Shuffle stocks within each month (controlled by seed for reproducibility).

    Used as the `batch_sampler` argument to DataLoader (not `batch_size`).
    """

    def __init__(
        self,
        dates: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dates = dates
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        # Pre-compute month → index list mapping (sorted by date)
        unique_dates = sorted(set(dates))
        self._month_indices: List[np.ndarray] = []
        for d in unique_dates:
            idx = np.where(dates == d)[0]
            self._month_indices.append(idx)

    def set_epoch(self, epoch: int) -> None:
        """Call before each epoch to get different within-month shuffles."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        for month_idx in self._month_indices:
            idx = month_idx.copy()
            if self.shuffle:
                rng.shuffle(idx)
            for start in range(0, len(idx), self.batch_size):
                yield idx[start : start + self.batch_size].tolist()

    def __len__(self) -> int:
        return sum(
            math.ceil(len(m) / self.batch_size) for m in self._month_indices
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def find_best_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the 'best_model.pt' checkpoint if it exists, else None."""
    best = checkpoint_dir / "best_model.pt"
    return best if best.exists() else None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    model_cfg: ModelConfig,
    checkpoint_dir: Path,
    resume: bool = True,
) -> dict:
    """
    Train ReturnPredictor with early stopping and periodic checkpointing.

    Parameters
    ----------
    model           : uninitialised ReturnPredictor (on CPU)
    train_df        : training set DataFrame (dates < val_start)
    val_df          : validation set DataFrame (dates >= val_start, within training period)
    feature_cols    : list of feature column names
    model_cfg       : ModelConfig hyperparameters
    checkpoint_dir  : directory to write checkpoints
    resume          : if True and a best_model.pt exists, resume from it

    Returns
    -------
    dict with keys: best_val_loss, best_epoch, total_epochs, training_time_s
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(model_cfg.seed)

    # ---- datasets & loaders ------------------------------------------------
    train_ds = StockReturnDataset(train_df, feature_cols)
    val_ds   = StockReturnDataset(val_df,   feature_cols)

    train_sampler = ChronologicalBatchSampler(
        train_ds.dates, model_cfg.batch_size, shuffle=True, seed=model_cfg.seed
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=model_cfg.num_workers,
        pin_memory=model_cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=model_cfg.batch_size * 4,  # larger batches OK for inference
        shuffle=False,
        num_workers=model_cfg.num_workers,
        pin_memory=model_cfg.pin_memory,
    )

    # ---- optimiser & loss --------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_cfg.lr,
        weight_decay=model_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=model_cfg.lr_scheduler_patience,
        factor=model_cfg.lr_scheduler_factor,
        min_lr=model_cfg.lr_min,
    )
    loss_fn = nn.HuberLoss(delta=model_cfg.huber_delta)

    # ---- resume from checkpoint --------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    if resume:
        best_ckpt = find_best_checkpoint(checkpoint_dir)
        if best_ckpt is not None:
            ckpt = load_checkpoint(best_ckpt, model, optimizer)
            start_epoch   = ckpt["epoch"] + 1
            best_val_loss = ckpt["val_loss"]
            print(f"  Resumed from epoch {ckpt['epoch']}  val_loss={best_val_loss:.6f}")

    # ---- training loop -----------------------------------------------------
    best_epoch = start_epoch
    t_start    = time.time()

    for epoch in range(start_epoch, model_cfg.max_epochs):
        # -- train ---
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss_sum = 0.0
        train_batches  = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches  += 1

        train_loss = train_loss_sum / max(train_batches, 1)

        # -- validate ---
        val_loss = _eval_loss(model, val_loader, loss_fn)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  epoch {epoch:03d}  "
            f"train={train_loss:.5f}  val={val_loss:.5f}  "
            f"lr={lr_now:.2e}"
        )

        # -- best checkpoint -------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_epoch      = epoch
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_loss,
                            checkpoint_dir / "best_model.pt")
        else:
            epochs_no_improve += 1

        # -- periodic checkpoint ---------------------------------------------
        if (epoch + 1) % model_cfg.checkpoint_every_n_epochs == 0:
            save_checkpoint(model, optimizer, epoch, val_loss,
                            checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt")

        # -- early stopping --------------------------------------------------
        if epochs_no_improve >= model_cfg.early_stopping_patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(no improvement for {model_cfg.early_stopping_patience} epochs)")
            break

    elapsed = time.time() - t_start
    print(f"\n  Training complete: best epoch={best_epoch}, "
          f"best_val_loss={best_val_loss:.6f}, "
          f"time={elapsed/60:.1f} min")

    # Reload the best weights before returning
    load_checkpoint(checkpoint_dir / "best_model.pt", model)

    return {
        "best_val_loss": best_val_loss,
        "best_epoch":    best_epoch,
        "total_epochs":  epoch + 1,
        "training_time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model: nn.Module,
    df: pd.DataFrame,
    feature_cols: list[str],
    batch_size: int = 4096,
) -> pd.DataFrame:
    """
    Run inference on a DataFrame and return a DataFrame with columns:
      permno, date, y_true, y_pred, residual

    NaN returns are retained with NaN y_true (useful for tracking
    which observations were dropped during training).
    """
    model.eval()

    ds = StockReturnDataset(df, feature_cols)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    preds = []
    for X_batch, _ in loader:
        preds.append(model(X_batch).numpy())

    y_pred = np.concatenate(preds)
    y_true = ds.y.numpy()

    return pd.DataFrame({
        "permno":   ds.permnos,
        "date":     ds.dates,
        "y_true":   y_true,
        "y_pred":   y_pred,
        "residual": y_true - y_pred,
    })


# ---------------------------------------------------------------------------
# Train/val split within the training period
# ---------------------------------------------------------------------------

def split_train_val(
    train_df: pd.DataFrame,
    val_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the training DataFrame at val_start.

    Returns (pure_train, val) where pure_train is used for gradient updates
    and val is used only for early stopping.
    """
    val_start_period = pd.Period(val_start, freq="M")
    mask = train_df["date"] >= val_start_period
    return train_df[~mask].copy(), train_df[mask].copy()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> float:
    model.eval()
    total, n_batches = 0.0, 0
    for X_batch, y_batch in loader:
        pred = model(X_batch)
        total += loss_fn(pred, y_batch).item()
        n_batches += 1
    model.train()
    return total / max(n_batches, 1)
