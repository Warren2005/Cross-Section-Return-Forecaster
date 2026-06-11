"""
Train / calibration / test split construction.

The three sets are strictly non-overlapping by date:

  Train       1957-01 to 1999-12  (504 months)
  Calibration 2000-01 to 2007-12  ( 96 months)  — conformal calibration + λ tuning only
  Test        2008-01 to 2021-12  (168 months)  — all evaluation here

The calibration set is quarantined: it is never used for NN training and never
touched for evaluation. Its sole purposes are:
  1. Computing SPCI quantile models (Phase 3)
  2. Tuning the portfolio weight λ (Phase 5)

See LEARNING.md §3.4 for the rationale behind these date boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from src.utils.config import DataConfig


@dataclass
class DataSplits:
    train: pd.DataFrame
    cal: pd.DataFrame
    test: pd.DataFrame
    char_cols: list[str]
    industry_cols: list[str]

    @property
    def feature_cols(self) -> list[str]:
        return self.char_cols + self.industry_cols

    def describe(self) -> str:
        lines = []
        for name, df in [("train", self.train), ("cal", self.cal), ("test", self.test)]:
            n_months = df["date"].nunique()
            n_stocks = df["permno"].nunique()
            n_obs = len(df)
            date_range = f"{df['date'].min()} – {df['date'].max()}"
            lines.append(
                f"  {name:<5}: {n_obs:>8,} obs  |  {n_months:>4} months  |  "
                f"~{n_stocks:>5,} stocks  |  {date_range}"
            )
        return "\n".join(lines)


def build_splits(
    df: pd.DataFrame,
    cfg: DataConfig,
    char_cols: list[str],
    industry_cols: list[str],
) -> DataSplits:
    """
    Partition the preprocessed panel into train / calibration / test sets.

    Parameters
    ----------
    df            : preprocessed panel (permno, date as Period M, ret, features)
    cfg           : DataConfig (provides date boundary strings)
    char_cols     : the 94 continuous characteristic column names
    industry_cols : the 74 industry dummy column names

    Returns DataSplits with validation assertions already checked.
    """
    train_start = pd.Period(cfg.train_start, freq="M")
    train_end   = pd.Period(cfg.train_end,   freq="M")
    cal_start   = pd.Period(cfg.cal_start,   freq="M")
    cal_end     = pd.Period(cfg.cal_end,     freq="M")
    test_start  = pd.Period(cfg.test_start,  freq="M")
    test_end    = pd.Period(cfg.test_end,    freq="M")

    date = df["date"]

    train = df[(date >= train_start) & (date <= train_end)].copy()
    cal   = df[(date >= cal_start)   & (date <= cal_end)].copy()
    test  = df[(date >= test_start)  & (date <= test_end)].copy()

    _validate_splits(train, cal, test)

    return DataSplits(
        train=train.reset_index(drop=True),
        cal=cal.reset_index(drop=True),
        test=test.reset_index(drop=True),
        char_cols=char_cols,
        industry_cols=industry_cols,
    )


def _validate_splits(
    train: pd.DataFrame,
    cal: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Hard assertions that the three sets are temporally disjoint."""
    train_dates = set(train["date"].unique())
    cal_dates   = set(cal["date"].unique())
    test_dates  = set(test["date"].unique())

    overlap_tc = train_dates & cal_dates
    overlap_tt = train_dates & test_dates
    overlap_ct = cal_dates   & test_dates

    if overlap_tc:
        raise ValueError(
            f"Train and calibration sets share {len(overlap_tc)} dates: "
            f"{sorted(overlap_tc)[:5]}..."
        )
    if overlap_tt:
        raise ValueError(
            f"Train and test sets share {len(overlap_tt)} dates: "
            f"{sorted(overlap_tt)[:5]}..."
        )
    if overlap_ct:
        raise ValueError(
            f"Calibration and test sets share {len(overlap_ct)} dates: "
            f"{sorted(overlap_ct)[:5]}..."
        )

    assert len(train) > 0, "Training set is empty"
    assert len(cal)   > 0, "Calibration set is empty"
    assert len(test)  > 0, "Test set is empty"


def fast_dev_subsample(
    df: pd.DataFrame,
    cfg: DataConfig,
) -> pd.DataFrame:
    """
    Subsample to a small slice for rapid end-to-end testing on CPU.

    Keeps the last fast_dev_n_months months of each split's date range
    and a random sample of fast_dev_n_stocks stocks within that window.
    This preserves temporal structure while shrinking the dataset ~1000x.
    """
    import numpy as np

    rng = np.random.default_rng(42)

    # Take last N months of each split to keep temporal ordering intact
    all_dates = sorted(df["date"].unique())
    keep_dates = set(all_dates[-cfg.fast_dev_n_months:])
    df = df[df["date"].isin(keep_dates)].copy()

    # Sample stocks
    all_permnos = df["permno"].unique()
    n = min(cfg.fast_dev_n_stocks, len(all_permnos))
    keep_permnos = set(rng.choice(all_permnos, size=n, replace=False))
    df = df[df["permno"].isin(keep_permnos)].copy()

    return df.reset_index(drop=True)
