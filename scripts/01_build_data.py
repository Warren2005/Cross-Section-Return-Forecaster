"""
Phase 1 — Data pipeline.

Usage:
    python scripts/01_build_data.py [--fast-dev] [--config PATH] [--gkx-file PATH]

What this script does:
  1. Load the raw GKX characteristics panel from data/raw/
  2. Apply filing-delay lags to each characteristic (look-ahead bias prevention)
  3. Apply cross-sectional rank normalization to [-1, 1]
  4. Split into train / calibration / test sets
  5. Save each split to data/splits/ as parquet
  6. Print validation statistics

Before running:
  - Download the GKX data file from https://dachxiu.chicagobooth.edu/#rp
    and place it in data/raw/ (any filename; pass with --gkx-file if non-default)
  - Optionally download Fama-French 5 factors and VIX (see LEARNING.md §10)

Outputs (all in data/splits/):
  train.parquet
  cal.parquet
  test.parquet
  column_meta.json  — char_cols, industry_cols, feature_cols lists
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from the project root without pip install
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.data.loader import GKXLoader, detect_char_cols, detect_industry_cols
from src.data.preprocess import preprocess, summarize_characteristics, count_stocks_per_month
from src.data.splits import build_splits, fast_dev_subsample
from src.utils.config import load_config
from src.utils.io import save_parquet


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSRF Phase 1: build data splits")
    p.add_argument(
        "--gkx-file",
        default=None,
        help="Path to the raw GKX data file. "
             "Auto-detected inside data/raw/ if not provided.",
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml in project root)",
    )
    p.add_argument(
        "--fast-dev",
        action="store_true",
        help="Subsample to a tiny slice for rapid end-to-end testing.",
    )
    return p.parse_args()


def find_gkx_file(raw_dir: Path) -> Path:
    """Auto-detect the GKX data file inside data/raw/."""
    candidates = list(raw_dir.glob("*.zip")) + \
                 list(raw_dir.glob("*.csv")) + \
                 list(raw_dir.glob("*.csv.gz")) + \
                 list(raw_dir.glob("*.sas7bdat")) + \
                 list(raw_dir.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No data file found in {raw_dir}. "
            "Download the GKX dataset from https://dachxiu.chicagobooth.edu/#rp "
            "and place it in data/raw/."
        )
    if len(candidates) > 1:
        print(f"  Multiple files in {raw_dir}: {[c.name for c in candidates]}")
        print(f"  Using: {candidates[0].name}")
    return candidates[0]


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def print_validation_report(splits, char_cols: list[str]) -> None:
    """
    Print statistics that can be cross-checked against Gu et al. (2020) Table 1.

    After rank normalization every characteristic should have:
      mean ≈ 0,   std ≈ 0.577 (uniform on [-1,1]),   min = -1,   max = +1
    """
    print("\n" + "=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    print("\nSplit sizes:")
    print(splits.describe())

    print("\nStocks per month (train set, sample years):")
    counts = count_stocks_per_month(splits.train)
    for yr in [1960, 1970, 1980, 1990, 1999]:
        yr_counts = counts[counts.index.year == yr]
        if len(yr_counts):
            print(f"  {yr}: avg {yr_counts.mean():.0f} stocks/month")

    print("\nCharacteristic summary after rank normalization")
    print("  (expected: mean≈0, std≈0.577, min=-1, max=+1)")
    stats = summarize_characteristics(splits.train, char_cols, n_sample=8)
    print(stats.to_string())

    print("\nReturn statistics (train set):")
    ret = splits.train["ret"].dropna()
    print(f"  N obs:  {len(ret):,}")
    print(f"  Mean:   {ret.mean():.4f}  ({ret.mean()*100:.2f}%/month)")
    print(f"  Std:    {ret.std():.4f}")
    print(f"  Sharpe: {ret.mean() / ret.std() * (12**0.5):.2f} annualized")

    missing_pct = splits.train[char_cols].isna().mean().mean() * 100
    print(f"\nMissing values in characteristics (train): {missing_pct:.1f}%")
    print("  (expected: 10–30% — characteristics are missing for small/new firms)")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.fast_dev:
        cfg.fast_dev = True

    print("=" * 70)
    print("CSRF Phase 1 — Data Pipeline")
    print(f"  fast_dev: {cfg.fast_dev}")
    print("=" * 70)

    # -- 1. Load raw data ----------------------------------------------------
    raw_dir = Path(cfg.data.raw_dir)
    gkx_path = Path(args.gkx_file) if args.gkx_file else find_gkx_file(raw_dir)
    print(f"\n[1/5] Loading GKX data from: {gkx_path}")

    t0 = time.time()
    loader = GKXLoader(gkx_path)
    df = loader.load()
    print(f"      Loaded {len(df):,} rows × {len(df.columns)} columns  ({time.time()-t0:.1f}s)")

    # Detect column types
    industry_cols = detect_industry_cols(df)
    char_cols = detect_char_cols(df, industry_cols)
    print(f"      Detected: {len(char_cols)} characteristic cols, {len(industry_cols)} industry cols")

    # -- 1b. fast_dev subsample ---------------------------------------------
    if cfg.fast_dev:
        print(f"\n  [fast_dev] Subsampling to "
              f"{cfg.data.fast_dev_n_stocks} stocks × {cfg.data.fast_dev_n_months} months")
        df = fast_dev_subsample(df, cfg.data)
        print(f"  [fast_dev] Subsampled to {len(df):,} rows")

    # -- 2. Enforce lags -----------------------------------------------------
    print(f"\n[2/5] Enforcing characteristic lags...")
    t0 = time.time()
    from src.data.preprocess import enforce_lags
    df = enforce_lags(df, char_cols, cfg.data.lag_rules)
    print(f"      Done ({time.time()-t0:.1f}s)")

    # -- 3. Rank normalization -----------------------------------------------
    print(f"\n[3/5] Rank-normalizing features...")
    t0 = time.time()
    from src.data.preprocess import rank_normalize
    all_feature_cols = char_cols + industry_cols
    df = rank_normalize(df, all_feature_cols, show_progress=True)
    print(f"      Done ({time.time()-t0:.1f}s)")

    # -- 4. Build splits -----------------------------------------------------
    print(f"\n[4/5] Building train / calibration / test splits...")
    splits = build_splits(df, cfg.data, char_cols, industry_cols)
    print(splits.describe())

    # -- 5. Save to parquet --------------------------------------------------
    splits_dir = Path(cfg.data.splits_dir)
    print(f"\n[5/5] Saving splits to {splits_dir}/")

    save_parquet(splits.train, splits_dir / "train.parquet")
    save_parquet(splits.cal,   splits_dir / "cal.parquet")
    save_parquet(splits.test,  splits_dir / "test.parquet")

    # Save column metadata so downstream scripts don't need to re-detect
    meta = {
        "char_cols": char_cols,
        "industry_cols": industry_cols,
        "feature_cols": all_feature_cols,
    }
    with open(splits_dir / "column_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved: train.parquet, cal.parquet, test.parquet, column_meta.json")

    # -- 6. Validation report ------------------------------------------------
    print_validation_report(splits, char_cols)

    print("\nPhase 1 complete. Next step: python scripts/02_train_model.py")


if __name__ == "__main__":
    main()
