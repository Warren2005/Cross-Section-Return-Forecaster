"""
Phase 2 — Train the ReturnPredictor neural network.

Usage:
    python scripts/02_train_model.py [--fast-dev] [--config PATH] [--no-resume]

Prerequisites:
    Phase 1 must have been run first (data/splits/ must contain
    train.parquet, cal.parquet, test.parquet, column_meta.json).

What this script does:
  1. Load train / cal / test splits from data/splits/
  2. Split the training set into pure-train (1957–1991) and val (1992–1999)
  3. Train ReturnPredictor with ChronologicalBatchSampler + early stopping
  4. Load the best checkpoint and run inference on cal and test sets
  5. Save predictions to data/splits/cal_predictions.parquet
                          data/splits/test_predictions.parquet
  6. Print OOS R², L-S Sharpe, and IC for the calibration set

Verification gate (from implementation plan):
  Cal-set OOS R² should be in [0.20%, 0.65%]
  Cal-set L-S Sharpe should be > 0.8 annualised
  If either check fails, see LEARNING.md §4.3 for debugging guidance.

CPU training note:
  Full training (200 stocks/month × 500 months) takes 4–12 hours on CPU.
  Use --fast-dev for a quick end-to-end smoke test (~5 minutes).
  The script is resumable: re-running it picks up the best checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.data.splits import fast_dev_subsample
from src.models.evaluate import print_metrics
from src.models.network import ReturnPredictor, count_parameters
from src.models.train import predict, split_train_val, train
from src.utils.config import load_config
from src.utils.io import load_parquet, save_parquet


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSRF Phase 2: train ReturnPredictor")
    p.add_argument("--config",    default="config.yaml")
    p.add_argument("--fast-dev",  action="store_true")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore existing checkpoint and retrain from scratch.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    if args.fast_dev:
        cfg.fast_dev = True

    print("=" * 60)
    print("CSRF Phase 2 — Train ReturnPredictor")
    print(f"  fast_dev : {cfg.fast_dev}")
    print(f"  resume   : {not args.no_resume}")
    print("=" * 60)

    splits_dir     = Path(cfg.data.splits_dir)
    checkpoint_dir = Path(cfg.model.checkpoint_dir)

    # -- 1. Load splits -------------------------------------------------------
    print("\n[1/6] Loading splits...")
    with open(splits_dir / "column_meta.json") as f:
        meta = json.load(f)
    feature_cols  = meta["feature_cols"]
    char_cols     = meta["char_cols"]

    train_df = load_parquet(splits_dir / "train.parquet")
    cal_df   = load_parquet(splits_dir / "cal.parquet")
    test_df  = load_parquet(splits_dir / "test.parquet")

    if cfg.fast_dev:
        print("  [fast_dev] Subsampling all splits...")
        train_df = fast_dev_subsample(train_df, cfg.data)
        cal_df   = fast_dev_subsample(cal_df,   cfg.data)
        test_df  = fast_dev_subsample(test_df,  cfg.data)

    print(f"  train: {len(train_df):,} rows  |  "
          f"cal: {len(cal_df):,} rows  |  "
          f"test: {len(test_df):,} rows")
    print(f"  features: {len(feature_cols)} columns")

    # -- 2. Split train → pure-train + val ------------------------------------
    print(f"\n[2/6] Splitting training set at val_start={cfg.data.val_start}...")
    pure_train_df, val_df = split_train_val(train_df, cfg.data.val_start)
    print(f"  pure_train: {len(pure_train_df):,} rows  |  val: {len(val_df):,} rows")

    # Training mean return (used as OOS R² benchmark)
    train_mean_return = float(train_df["ret"].mean())
    print(f"  train mean return: {train_mean_return*100:.4f}%/month")

    # -- 3. Instantiate model -------------------------------------------------
    print(f"\n[3/6] Instantiating ReturnPredictor...")
    model = ReturnPredictor(
        n_chars=len(feature_cols),
        dropout=cfg.model.dropout,
    )
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Architecture: {len(feature_cols)} → 32 → 16 → 8 → 4 → 1")

    # -- 4. Train -------------------------------------------------------------
    print(f"\n[4/6] Training (max_epochs={cfg.model.max_epochs}, "
          f"patience={cfg.model.early_stopping_patience})...")
    print(f"  Checkpoints → {checkpoint_dir}/")

    stats = train(
        model         = model,
        train_df      = pure_train_df,
        val_df        = val_df,
        feature_cols  = feature_cols,
        model_cfg     = cfg.model,
        checkpoint_dir= checkpoint_dir,
        resume        = not args.no_resume,
    )

    print(f"\n  Summary: best_epoch={stats['best_epoch']}, "
          f"best_val_loss={stats['best_val_loss']:.6f}, "
          f"time={stats['training_time_s']/60:.1f}min")

    # -- 5. Inference on cal and test -----------------------------------------
    print(f"\n[5/6] Running inference on calibration and test sets...")
    cal_preds  = predict(model, cal_df,  feature_cols, batch_size=cfg.model.batch_size * 4)
    test_preds = predict(model, test_df, feature_cols, batch_size=cfg.model.batch_size * 4)

    save_parquet(cal_preds,  splits_dir / "cal_predictions.parquet")
    save_parquet(test_preds, splits_dir / "test_predictions.parquet")
    print(f"  Saved: cal_predictions.parquet  ({len(cal_preds):,} rows)")
    print(f"  Saved: test_predictions.parquet ({len(test_preds):,} rows)")

    # Also save training-set predictions (needed for SPCI residual model in Phase 3)
    print("  Generating training-set predictions for Phase 3 residuals...")
    train_preds = predict(model, train_df, feature_cols, batch_size=cfg.model.batch_size * 4)
    save_parquet(train_preds, splits_dir / "train_predictions.parquet")
    print(f"  Saved: train_predictions.parquet ({len(train_preds):,} rows)")

    # -- 6. Evaluation report -------------------------------------------------
    print(f"\n[6/6] Evaluation (calibration set — NOT the test set):")
    print_metrics(
        label             = "Calibration set (2000-2007)",
        pred_df           = cal_preds,
        train_mean_return = train_mean_return,
        n_deciles         = cfg.portfolio.n_deciles,
    )

    print("\nPhase 2 complete.")
    print("Gate check: OOS R² should be in [0.20%, 0.65%], L-S Sharpe > 0.8")
    print("Next step: python scripts/03_conformal_calibrate.py")


if __name__ == "__main__":
    main()
