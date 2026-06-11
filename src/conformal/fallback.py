"""
Fallback conformal quantile for stocks with insufficient residual history.

Used by CrossSectionalSPCI when a stock has fewer than min_history months of
residuals — typically IPOs, new listings, or stocks that entered the dataset
recently.

The fallback is standard split conformal prediction: the empirical (1-α)
quantile of absolute calibration residuals, with the finite-sample correction
factor (n+1)/n. This gives a symmetric interval [ŷ - q, ŷ + q] with the
guaranteed coverage property.

See LEARNING.md §5.3 for the design rationale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_quantile(
    cal_residuals: np.ndarray | pd.Series,
    alpha: float,
) -> float:
    """
    Standard split conformal empirical quantile with finite-sample correction.

    The correction ⌈(n+1)(1-α)⌉/n ensures exact (not just approximate) coverage
    for finite calibration sets. As n → ∞ this converges to the ordinary
    (1-α) quantile.

    Parameters
    ----------
    cal_residuals : 1-D array of calibration-set residuals (signed or absolute)
    alpha         : miscoverage level (e.g. 0.10 for 90% intervals)

    Returns
    -------
    q : non-negative half-width for the symmetric fallback interval
        Interval = [ŷ - q, ŷ + q]
    """
    abs_resids = np.abs(np.asarray(cal_residuals, dtype=float))
    abs_resids = abs_resids[~np.isnan(abs_resids)]
    n = len(abs_resids)
    if n == 0:
        raise ValueError("cal_residuals contains no valid (non-NaN) values.")

    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(abs_resids, level))


def apply_fallback_intervals(
    pred_df: pd.DataFrame,
    q: float,
) -> pd.DataFrame:
    """
    Apply the symmetric fallback interval to all rows in pred_df.

    Returns a DataFrame matching the interval output format of
    CrossSectionalSPCI.predict_intervals(), with used_spci=False everywhere.
    """
    result = pred_df[["permno", "date", "y_pred", "y_true"]].copy()
    result["lower"]     = pred_df["y_pred"] - q
    result["upper"]     = pred_df["y_pred"] + q
    result["width"]     = 2.0 * q
    result["used_spci"] = False
    return result
