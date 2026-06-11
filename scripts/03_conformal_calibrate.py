"""
Phase 3 — Conformal calibration (SPCI).

Usage:
    python scripts/03_conformal_calibrate.py [--fast-dev] [--config PATH]

Prerequisites:
    Phase 2 must have been run (data/splits/ must contain
    train_predictions.parquet, cal_predictions.parquet, test_predictions.parquet).

What this script does:
  1. Load residuals from all three prediction files
  2. Build the combined residual history panel (train + cal + test)
  3. Compute lagged features for every (permno, month) row
  4. Fit CrossSectionalSPCI on calibration-set rows
  5. Compute the fallback quantile from calibration absolute residuals
  6. Predict asymmetric intervals for the test set
  7. Save intervals to data/splits/test_intervals.parquet
  8. Print the full coverage validation table

Verification gate (from implementation plan):
  Full test-set coverage must be ≥ 90.0%
  2008-09 crisis coverage must be ≥ 85.0%
  If either fails, do NOT proceed to Phase 4 — see LEARNING.md §5.4.

Outputs (all in data/splits/):
  spci_model.pkl           — serialised CrossSectionalSPCI (for inspection)
  test_intervals.parquet   — intervals: permno, date, y_pred, y_true,
                             lower, upper, width, used_spci
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.conformal.coverage import print_coverage_report
from src.conformal.fallback import calibration_quantile
from src.conformal.spci import CrossSectionalSPCI, build_feature_panel
from src.data.splits import fast_dev_subsample
from src.utils.config import load_config
from src.utils.io import load_parquet, save_parquet


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSRF Phase 3: conformal calibration")
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--fast-dev", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    if args.fast_dev:
        cfg.fast_dev = True

    print("=" * 65)
    print("CSRF Phase 3 — Conformal Calibration (SPCI)")
    print(f"  fast_dev : {cfg.fast_dev}")
    print(f"  alpha    : {cfg.spci.alpha}  ({(1-cfg.spci.alpha)*100:.0f}% intervals)")
    print(f"  L        : {cfg.spci.L} lagged residuals")
    print("=" * 65)

    splits_dir = Path(cfg.data.splits_dir)

    # -- 1. Load prediction files -------------------------------------------
    print("\n[1/7] Loading prediction files...")
    train_preds = load_parquet(splits_dir / "train_predictions.parquet")
    cal_preds   = load_parquet(splits_dir / "cal_predictions.parquet")
    test_preds  = load_parquet(splits_dir / "test_predictions.parquet")

    if cfg.fast_dev:
        print("  [fast_dev] Subsampling prediction files...")
        train_preds = fast_dev_subsample(train_preds.rename(columns={"date": "date"}), cfg.data)
        cal_preds   = fast_dev_subsample(cal_preds,   cfg.data)
        test_preds  = fast_dev_subsample(test_preds,  cfg.data)

    # Also load train split to get return history for vol-decile coverage
    train_df = load_parquet(splits_dir / "train.parquet",
                            columns=["permno", "date", "ret"])
    cal_df   = load_parquet(splits_dir / "cal.parquet",
                            columns=["permno", "date", "ret"])
    test_df  = load_parquet(splits_dir / "test.parquet",
                            columns=["permno", "date", "ret"])
    return_history = pd.concat([train_df, cal_df, test_df], ignore_index=True)

    print(f"  train preds: {len(train_preds):,}  |  "
          f"cal preds: {len(cal_preds):,}  |  "
          f"test preds: {len(test_preds):,}")

    # -- 2. Build combined residual panel ------------------------------------
    print("\n[2/7] Building combined residual history panel...")

    def _make_residual_panel(df: pd.DataFrame, source: str) -> pd.DataFrame:
        return pd.DataFrame({
            "permno":   df["permno"],
            "date":     df["date"],
            "residual": df["residual"],
            "source":   source,
        })

    combined = pd.concat([
        _make_residual_panel(train_preds, "train"),
        _make_residual_panel(cal_preds,   "cal"),
        _make_residual_panel(test_preds,  "test"),
    ], ignore_index=True)

    print(f"  Combined panel: {len(combined):,} rows "
          f"({combined['permno'].nunique():,} stocks)")

    # -- 3. Build lagged feature panel ---------------------------------------
    print(f"\n[3/7] Building lagged feature panel (L={cfg.spci.L})...")
    t0 = time.time()
    feature_panel = build_feature_panel(combined, L=cfg.spci.L)
    print(f"  Done ({time.time()-t0:.1f}s)  "
          f"shape: {feature_panel.shape}")

    # -- 4. Fit SPCI quantile models ----------------------------------------
    print("\n[4/7] Fitting SPCI quantile models on calibration set...")
    t0 = time.time()

    spci = CrossSectionalSPCI(
        alpha         = cfg.spci.alpha,
        L             = cfg.spci.L,
        min_history   = cfg.spci.min_history,
        n_estimators  = cfg.spci.n_estimators,
        max_depth     = cfg.spci.max_depth,
        learning_rate = cfg.spci.learning_rate,
        subsample     = cfg.spci.subsample,
    )
    spci.fit(feature_panel, source_filter="cal")
    print(f"  Fitting complete ({(time.time()-t0)/60:.1f} min)")

    # Save model
    model_path = splits_dir / "spci_model.pkl"
    spci.save(model_path)
    print(f"  Saved SPCI model → {model_path}")

    # -- 5. Compute fallback quantile ----------------------------------------
    print("\n[5/7] Computing fallback quantile from calibration residuals...")
    cal_residuals  = cal_preds["residual"].dropna().values
    fallback_q     = calibration_quantile(cal_residuals, cfg.spci.alpha)
    print(f"  Fallback half-width: {fallback_q*10_000:.1f} bps  "
          f"(symmetric: [{-fallback_q*10_000:.1f}, +{fallback_q*10_000:.1f}] bps)")

    # -- 6. Predict test-set intervals ---------------------------------------
    print("\n[6/7] Predicting intervals for test set...")
    t0 = time.time()

    # pred_df for test: needs y_pred and y_true
    test_pred_df = test_preds[["permno", "date", "y_pred", "y_true"]].copy()

    test_intervals = spci.predict_intervals(
        pred_df            = test_pred_df,
        feature_panel      = feature_panel,
        fallback_half_width= fallback_q,
    )

    print(f"  Done ({time.time()-t0:.1f}s)  "
          f"intervals: {len(test_intervals):,} rows")

    n_spci = test_intervals["used_spci"].sum()
    n_fb   = (~test_intervals["used_spci"]).sum()
    print(f"  SPCI: {n_spci:,} ({n_spci/len(test_intervals)*100:.1f}%)  |  "
          f"Fallback: {n_fb:,} ({n_fb/len(test_intervals)*100:.1f}%)")

    # -- 7. Save and validate ------------------------------------------------
    print("\n[7/7] Saving intervals and validating coverage...")
    save_parquet(test_intervals, splits_dir / "test_intervals.parquet")
    print(f"  Saved: test_intervals.parquet")

    print_coverage_report(
        interval_df      = test_intervals,
        return_history   = return_history,
        market_cap_panel = None,   # CRSP-DEPENDENT — swap in CRSPLoader output here
        alpha            = cfg.spci.alpha,
    )

    print("\nPhase 3 complete.")
    print("Next step: python scripts/04_width_analysis.py")


if __name__ == "__main__":
    main()
