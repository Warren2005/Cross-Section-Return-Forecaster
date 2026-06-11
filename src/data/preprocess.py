"""
Characteristic preprocessing for the CSRF pipeline.

Two operations, applied in order:
  1. enforce_lags    — shift each characteristic back by its filing delay
  2. rank_normalize  — map each characteristic to [-1, 1] cross-sectionally

These exactly replicate the Gu, Kelly & Xiu (2020) preprocessing protocol
(Section II.B). See LEARNING.md §3.2 and §3.3 for the design rationale.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import (
    DEFAULT_LAG_RULES,
    MONTHLY_CHARS,
    QUARTERLY_CHARS,
)


# ---------------------------------------------------------------------------
# Step 1 — Lag enforcement (look-ahead bias prevention)
# ---------------------------------------------------------------------------

def classify_columns(
    char_cols: Sequence[str],
    monthly_chars: Sequence[str] = MONTHLY_CHARS,
    quarterly_chars: Sequence[str] = QUARTERLY_CHARS,
) -> dict[str, list[str]]:
    """
    Assign each characteristic column to its update frequency.

    Returns {'monthly': [...], 'quarterly': [...], 'annual': [...]}

    Columns not in monthly_chars or quarterly_chars are treated as annual
    (6-month filing delay). This is conservative — if unsure, use the longer lag.
    """
    monthly = [c for c in char_cols if c in set(monthly_chars)]
    quarterly = [c for c in char_cols if c in set(quarterly_chars)]
    annual = [c for c in char_cols
              if c not in set(monthly_chars) and c not in set(quarterly_chars)]

    return {"monthly": monthly, "quarterly": quarterly, "annual": annual}


def enforce_lags(
    df: pd.DataFrame,
    char_cols: Sequence[str],
    lag_rules: dict[str, int] = DEFAULT_LAG_RULES,
    monthly_chars: Sequence[str] = MONTHLY_CHARS,
    quarterly_chars: Sequence[str] = QUARTERLY_CHARS,
) -> pd.DataFrame:
    """
    Shift each characteristic by its filing-delay lag, per stock.

    The shift is row-based within each permno group (after sorting by date).
    This is valid under the assumption that each stock appears at most once
    per month in the panel — which holds for the GKX dataset.

    Industry dummy columns are passed through unchanged (no lag needed:
    SIC codes are contemporaneously observable).

    Parameters
    ----------
    df          : panel DataFrame with columns [permno, date, ret, *chars, *ind_dummies]
    char_cols   : the 94 continuous characteristic column names
    lag_rules   : {'monthly': N, 'quarterly': N, 'annual': N}
    """
    df = df.sort_values(["permno", "date"]).copy()

    freq_map = classify_columns(char_cols, monthly_chars, quarterly_chars)

    for freq, cols in freq_map.items():
        lag = lag_rules[freq]
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        # groupby().shift() is vectorised — avoids Python-level loops
        df[present] = df.groupby("permno")[present].shift(lag)

    return df


# ---------------------------------------------------------------------------
# Step 2 — Cross-sectional rank normalization
# ---------------------------------------------------------------------------

def _rank_normalize_series(x: pd.Series) -> pd.Series:
    """
    Map a cross-sectional slice to [-1, 1] by rank.

    Ties receive the average rank. Missing values are left as NaN (they will be
    imputed to 0 by the NN DataLoader, which is the cross-sectional median
    after normalization to [-1, 1]).
    """
    ranks = x.rank(method="average", na_option="keep")
    n_valid = ranks.notna().sum()
    if n_valid < 2:
        return x
    return 2.0 * (ranks - 1.0) / (n_valid - 1.0) - 1.0


def rank_normalize(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Apply cross-sectional rank normalization to all feature columns.

    For each (date, column) pair, rank stocks and map to [-1, 1].
    Applied to BOTH continuous characteristics AND industry dummies.
    For binary dummies this simply separates 0s from 1s in cross-section.

    Parameters
    ----------
    df            : panel DataFrame; must contain a 'date' column
    feature_cols  : columns to normalize (chars + industry dummies)
    show_progress : print a tqdm progress bar (useful for the full panel)
    """
    df = df.copy()
    cols = [c for c in feature_cols if c in df.columns]

    dates = df["date"].unique()
    iter_dates = tqdm(dates, desc="Rank normalizing", unit="month") if show_progress else dates

    for date in iter_dates:
        mask = df["date"] == date
        df.loc[mask, cols] = df.loc[mask, cols].apply(_rank_normalize_series)

    return df


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    df: pd.DataFrame,
    char_cols: Sequence[str],
    industry_cols: Sequence[str],
    lag_rules: dict[str, int] = DEFAULT_LAG_RULES,
    monthly_chars: Sequence[str] = MONTHLY_CHARS,
    quarterly_chars: Sequence[str] = QUARTERLY_CHARS,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline: lag enforcement → rank normalization.

    Steps (order matters):
      1. Lag characteristics by filing delay (modifies the raw values in place)
      2. Rank-normalize characteristics AND industry dummies cross-sectionally

    Industry dummies are NOT lagged (SIC codes are contemporaneously known)
    but ARE rank-normalized along with characteristics.

    Returns the preprocessed DataFrame. Does not modify the input.
    """
    print("Step 1/2: Enforcing characteristic lags...")
    df = enforce_lags(df, char_cols, lag_rules, monthly_chars, quarterly_chars)

    print("Step 2/2: Rank-normalizing features...")
    all_feature_cols = list(char_cols) + list(industry_cols)
    df = rank_normalize(df, all_feature_cols, show_progress=show_progress)

    return df


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def summarize_characteristics(
    df: pd.DataFrame,
    char_cols: Sequence[str],
    n_sample: int = 10,
) -> pd.DataFrame:
    """
    Return summary statistics for a sample of characteristics.
    Used in 01_build_data.py to verify preprocessing matches Gu et al. Table 1.

    After rank normalization, each characteristic should have:
      mean ≈ 0,  std ≈ 0.577  (uniform on [-1,1] has std = 1/√3 ≈ 0.577)
      min = -1,  max = +1
    """
    sample_cols = list(char_cols)[:n_sample]
    stats = df[sample_cols].describe().T[["mean", "std", "min", "max"]]
    return stats


def count_stocks_per_month(df: pd.DataFrame) -> pd.Series:
    """Number of unique stocks in each month — useful for coverage checks."""
    return df.groupby("date")["permno"].nunique()
