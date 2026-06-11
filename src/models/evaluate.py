"""
Evaluation metrics for the base return-prediction model.

All metrics here operate on prediction DataFrames produced by train.predict(),
which have columns: permno, date, y_true, y_pred, residual.

OOS R²
  Follows the Gu et al. (2020) formula exactly: the benchmark is the
  historical mean return from the training set (not the test-set mean).
  This is more conservative and more realistic — in live trading you don't
  know the out-of-sample mean in advance.

Decile L-S Sharpe
  Each month, rank stocks by y_pred into 10 deciles. The long-short (L-S)
  portfolio holds the top decile (D10) and shorts the bottom decile (D1).
  Sharpe is annualised: (mean monthly L-S return / std) × √12.

Information Coefficient (IC)
  Spearman rank correlation of y_pred with y_true, computed monthly then
  averaged. IR = mean(IC) / std(IC).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# OOS R²
# ---------------------------------------------------------------------------

def oos_r2(
    pred_df: pd.DataFrame,
    train_mean_return: float,
) -> float:
    """
    Out-of-sample R² following Gu et al. (2020).

    R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ_train)²

    where ȳ_train is the mean return over the training period.
    A positive R² means the model beats the historical-mean naive forecast.

    Parameters
    ----------
    pred_df           : DataFrame with y_true, y_pred columns
    train_mean_return : mean return computed on the training set
    """
    df = pred_df.dropna(subset=["y_true", "y_pred"])
    y      = df["y_true"].values
    y_hat  = df["y_pred"].values

    ss_res   = np.sum((y - y_hat) ** 2)
    ss_total = np.sum((y - train_mean_return) ** 2)

    return float(1.0 - ss_res / ss_total)


# ---------------------------------------------------------------------------
# Decile long-short Sharpe
# ---------------------------------------------------------------------------

def decile_ls_sharpe(
    pred_df: pd.DataFrame,
    n_deciles: int = 10,
) -> dict:
    """
    Each month, sort stocks into n_deciles by y_pred.
    Compute equal-weighted returns of each decile and the D_top - D_bottom spread.

    Returns
    -------
    dict with keys:
      decile_mean_returns : array of shape (n_deciles,) — avg monthly return per decile
      ls_mean             : mean monthly L-S return
      ls_std              : std of monthly L-S return
      ls_sharpe           : annualised Sharpe of L-S
      ls_tstat            : t-statistic (mean / (std / √T))
    """
    df = pred_df.dropna(subset=["y_true", "y_pred"]).copy()

    # Assign decile label within each month (0 = bottom, n_deciles-1 = top)
    df["decile"] = df.groupby("date")["y_pred"].transform(
        lambda x: pd.qcut(x, n_deciles, labels=False, duplicates="drop")
    )
    df = df.dropna(subset=["decile"])
    df["decile"] = df["decile"].astype(int)

    # Monthly L-S return: top decile minus bottom decile
    monthly_ls = (
        df.groupby(["date", "decile"])["y_true"]
        .mean()
        .unstack("decile")
    )

    top    = n_deciles - 1
    bottom = 0
    ls_series = monthly_ls[top] - monthly_ls[bottom]
    ls_series = ls_series.dropna()

    T = len(ls_series)
    ls_mean   = float(ls_series.mean())
    ls_std    = float(ls_series.std(ddof=1))
    ls_sharpe = float(ls_mean / ls_std * np.sqrt(12)) if ls_std > 0 else np.nan
    ls_tstat  = float(ls_mean / (ls_std / np.sqrt(T))) if ls_std > 0 else np.nan

    # Per-decile mean return (averaged across months)
    decile_means = (
        df.groupby(["date", "decile"])["y_true"].mean()
        .groupby("decile").mean()
        .values
    )

    return {
        "decile_mean_returns": decile_means,
        "ls_mean":    ls_mean,
        "ls_std":     ls_std,
        "ls_sharpe":  ls_sharpe,
        "ls_tstat":   ls_tstat,
        "n_months":   T,
    }


# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------

def information_coefficient(pred_df: pd.DataFrame) -> dict:
    """
    Monthly Spearman IC between y_pred and y_true.

    Returns
    -------
    dict with: mean_ic, std_ic, ir (information ratio), n_months
    """
    df = pred_df.dropna(subset=["y_true", "y_pred"])

    monthly_ic = (
        df.groupby("date")
        .apply(lambda g: spearmanr(g["y_pred"], g["y_true"]).statistic,
               include_groups=False)
    )

    mean_ic = float(monthly_ic.mean())
    std_ic  = float(monthly_ic.std(ddof=1))
    ir      = float(mean_ic / std_ic) if std_ic > 0 else np.nan

    return {"mean_ic": mean_ic, "std_ic": std_ic, "ir": ir, "n_months": len(monthly_ic)}


# ---------------------------------------------------------------------------
# Formatted metrics report
# ---------------------------------------------------------------------------

def print_metrics(
    label: str,
    pred_df: pd.DataFrame,
    train_mean_return: float,
    n_deciles: int = 10,
) -> None:
    r2     = oos_r2(pred_df, train_mean_return)
    ls     = decile_ls_sharpe(pred_df, n_deciles)
    ic     = information_coefficient(pred_df)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  OOS R²:          {r2*100:+.4f}%")
    print(f"  L-S Sharpe (EW): {ls['ls_sharpe']:+.3f}  "
          f"(mean={ls['ls_mean']*100:.3f}%/mo, "
          f"t={ls['ls_tstat']:.2f}, n={ls['n_months']}mo)")
    print(f"  Mean IC:         {ic['mean_ic']:.4f}  "
          f"(IR={ic['ir']:.2f})")
    print(f"\n  Decile mean returns (D1=short -> D{n_deciles}=long):")
    for i, r in enumerate(ls["decile_mean_returns"]):
        bar = "#" * int(abs(r * 1000))
        sign = "+" if r >= 0 else "-"
        print(f"    D{i+1:2d}: {sign}{abs(r)*100:.3f}%  {bar}")
    print(f"{'='*55}")

    # Gu et al. benchmark check
    if r2 < 0.002:
        print(f"  [!]  OOS R2 ({r2*100:.4f}%) is below 0.20% -- check preprocessing.")
    elif r2 > 0.0065:
        print(f"  [!]  OOS R2 ({r2*100:.4f}%) is above 0.65% -- may indicate data leakage.")
    else:
        print(f"  [OK]  OOS R2 in expected range [0.20%, 0.65%] -- implementation looks correct.")
