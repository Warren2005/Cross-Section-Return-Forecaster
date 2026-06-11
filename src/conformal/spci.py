"""
CrossSectionalSPCI — Sequential Predictive Conformal Inference adapted for
the cross-sectional return prediction panel.

Reference: Xu & Xie (2023), "Sequential Predictive Conformal Inference for
Time Series", ICML 2023.

See LEARNING.md §5.1 and §5.2 for full design rationale.

Core idea
---------
Instead of using a fixed global calibration quantile (standard split conformal),
SPCI fits a quantile regression model that predicts the CONDITIONAL quantile of
the next residual given its recent history. This produces narrower intervals for
predictable stocks and wider intervals for volatile/uncertain ones — exactly
the variation we want for the width signal to be meaningful.

Interval type
-------------
We predict ASYMMETRIC intervals using two quantile models:
  - qgbm_lo : α/2  quantile of the signed residual ε (negative tail)
  - qgbm_hi : 1-α/2 quantile of the signed residual ε (positive tail)
  Interval = [ŷ + lo_pred, ŷ + hi_pred]
  Width    = hi_pred - lo_pred

This is more general than the symmetric version in the spec and is the
formulation used in the original SPCI paper.

History convention
------------------
Features at test time are built from the most recent L residuals available
BEFORE the test date. Because this is a retrospective backtest, we use the
full combined history (train + calibration residuals). The quantile model
itself is fit exclusively on calibration-set observations.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_panel(
    residual_panel: pd.DataFrame,
    L: int,
) -> pd.DataFrame:
    """
    For every (permno, date) row in residual_panel, compute lagged features
    using that stock's residual history up to (but not including) that date.

    Features (total = 2L + 2 columns):
      f_abs_1  … f_abs_L  : |ε_{t-1}|, …, |ε_{t-L}|
      f_sign_1 … f_sign_L : sign(ε_{t-1}), …, sign(ε_{t-L})
      f_rolling_std        : std of |ε| over trailing 12 months
      f_rolling_mae        : mean of |ε| over trailing 12 months

    NaN lag values arise when a stock has fewer than L months of history.
    They are handled downstream: SPCI uses only rows with sufficient history,
    the fallback handles the rest.

    Parameters
    ----------
    residual_panel : DataFrame with columns [permno, date, residual, source]
    L              : number of lags
    """
    df = residual_panel.sort_values(["permno", "date"]).copy()

    def _add_lags(g: pd.DataFrame) -> pd.DataFrame:
        r = g["residual"]
        for lag in range(1, L + 1):
            g[f"f_abs_{lag}"]  = r.abs().shift(lag)
            g[f"f_sign_{lag}"] = np.sign(r).shift(lag)
        abs_shifted = r.abs().shift(1)
        g["f_rolling_std"] = abs_shifted.rolling(12, min_periods=3).std()
        g["f_rolling_mae"] = abs_shifted.rolling(12, min_periods=3).mean()
        return g

    df = df.groupby("permno", group_keys=False).apply(_add_lags)
    return df


def feature_columns(L: int) -> list[str]:
    return (
        [f"f_abs_{i}"  for i in range(1, L + 1)] +
        [f"f_sign_{i}" for i in range(1, L + 1)] +
        ["f_rolling_std", "f_rolling_mae"]
    )


# ---------------------------------------------------------------------------
# CrossSectionalSPCI
# ---------------------------------------------------------------------------

class CrossSectionalSPCI:
    """
    Parameters
    ----------
    alpha       : miscoverage level (0.10 → 90% prediction intervals)
    L           : number of lagged residuals to use as features
    min_history : minimum number of non-null lag values required to use SPCI;
                  stocks below this threshold fall back to the global quantile
    n_estimators: GBM trees for each quantile model
    max_depth   : tree depth (kept shallow to avoid overfitting noisy residuals)
    learning_rate: GBM shrinkage
    subsample   : fraction of data used per tree (stochastic GBM)
    """

    def __init__(
        self,
        alpha: float = 0.10,
        L: int = 24,
        min_history: int = 12,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
    ):
        self.alpha        = alpha
        self.L            = L
        self.min_history  = min_history
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.learning_rate = learning_rate
        self.subsample    = subsample

        self.qgbm_lo: GradientBoostingRegressor | None = None
        self.qgbm_hi: GradientBoostingRegressor | None = None
        self._feat_cols: list[str] = feature_columns(L)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, feature_panel: pd.DataFrame, source_filter: str = "cal") -> None:
        """
        Fit two quantile GBM models on calibration-set observations.

        Parameters
        ----------
        feature_panel : output of build_feature_panel(), with a 'source' column
        source_filter : which rows to use as the fitting set (default 'cal')
        """
        cal_rows = feature_panel[feature_panel["source"] == source_filter].copy()

        # Require at least min_history non-null abs lags
        abs_lag_cols = [f"f_abs_{i}" for i in range(1, self.min_history + 1)]
        cal_rows = cal_rows.dropna(subset=abs_lag_cols)

        X = cal_rows[self._feat_cols].fillna(0.0).values
        y = cal_rows["residual"].values   # signed residual — asymmetric intervals

        print(f"  SPCI fitting on {len(X):,} cal observations, "
              f"{len(self._feat_cols)} features...")

        self.qgbm_lo = GradientBoostingRegressor(
            loss="quantile",
            alpha=self.alpha / 2,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=42,
        )
        self.qgbm_hi = GradientBoostingRegressor(
            loss="quantile",
            alpha=1.0 - self.alpha / 2,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=42,
        )

        print("  Fitting lower quantile model...")
        self.qgbm_lo.fit(X, y)
        print("  Fitting upper quantile model...")
        self.qgbm_hi.fit(X, y)
        print("  SPCI fitting complete.")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_intervals(
        self,
        pred_df: pd.DataFrame,
        feature_panel: pd.DataFrame,
        fallback_half_width: float,
    ) -> pd.DataFrame:
        """
        Produce prediction intervals for every row in pred_df.

        Uses SPCI (quantile GBM) where sufficient residual history exists;
        falls back to the global symmetric fallback interval otherwise.

        Parameters
        ----------
        pred_df           : DataFrame with [permno, date, y_pred, y_true]
        feature_panel     : output of build_feature_panel() (contains test rows)
        fallback_half_width: symmetric fallback interval half-width (from fallback.py)

        Returns
        -------
        DataFrame with columns:
          permno, date, y_pred, y_true, lower, upper, width, used_spci
        """
        assert self.qgbm_lo is not None, "Call fit() before predict_intervals()"

        # Merge features into pred_df
        feat_panel_test = feature_panel[feature_panel["source"] == "test"][
            ["permno", "date"] + self._feat_cols
        ].copy()

        merged = pred_df.merge(feat_panel_test, on=["permno", "date"], how="left")

        # Determine which rows have enough history for SPCI
        abs_lag_cols = [f"f_abs_{i}" for i in range(1, self.min_history + 1)]
        has_history  = merged[abs_lag_cols].notna().all(axis=1)

        X_all = merged[self._feat_cols].fillna(0.0).values

        lo_pred = np.full(len(merged), np.nan)
        hi_pred = np.full(len(merged), np.nan)

        # SPCI rows
        spci_mask = has_history.values
        if spci_mask.any():
            lo_pred[spci_mask] = self.qgbm_lo.predict(X_all[spci_mask])
            hi_pred[spci_mask] = self.qgbm_hi.predict(X_all[spci_mask])

        # Fallback rows (symmetric)
        fb_mask = ~spci_mask
        lo_pred[fb_mask] = -fallback_half_width
        hi_pred[fb_mask] =  fallback_half_width

        y_pred = merged["y_pred"].values

        result = merged[["permno", "date", "y_pred", "y_true"]].copy()
        result["lower"]     = y_pred + lo_pred
        result["upper"]     = y_pred + hi_pred
        result["width"]     = hi_pred - lo_pred
        result["used_spci"] = spci_mask

        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "CrossSectionalSPCI":
        with open(path, "rb") as f:
            return pickle.load(f)
