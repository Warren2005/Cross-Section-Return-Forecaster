"""
Portfolio construction and evaluation for the CSRF pipeline.

Three sorting strategies compared across the full test period (2008-2021):

  Strategy          lambda   Description
  --------          ------   -----------
  Gu baseline         0      Sort on standardised point estimate only
  Width-only          1      Point estimate minus full width-percentile penalty
  Uncertainty-adj    lam*    Calibration-set optimal blend (see LEARNING.md 7.1)

Score formula (cross-sectional within each month):
  score_{i,t} = pe_std_{i,t} - lambda * width_pct_{i,t}

  pe_std    : z-score of y_pred per date  (mean 0, std 1)
  width_pct : percentile rank of width per date in [0, 1]

Lambda tuning:
  Grid search over PortfolioConfig.lambda_grid on the cal_intervals ONLY.
  Lambda is frozen before any test-set data is viewed.
  See LEARNING.md 7.1 for why this constraint is non-negotiable.

Transaction costs:
  net = gross - 2 * (half_spread_bps / 10_000) * monthly_turnover
  Default half_spread_bps = 20 (conservative for full universe, incl. small caps).

FF5 alpha:
  Time-series OLS of monthly L-S returns on Fama-French 5 factors.
  Newey-West HAC standard errors (6 lags). Factor file from load_ff5_factors().
  Column names expected: mkt_rf, smb, hml, rmw, cma.
  Pass ff5_df=None to skip (e.g. if factors file is unavailable).

Value-weighting:
  # CRSP-DEPENDENT: true VW requires raw market cap from CRSP.
  # The GKX `me` column is rank-normalised in [-1, 1] and cannot be used
  # directly as portfolio weights. If CRSP is unavailable, all results are EW.
  # Pass market_cap_col=None (default) to use EW only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Score construction
# ---------------------------------------------------------------------------

def uncertainty_adjusted_score(
    df: pd.DataFrame,
    lambda_: float,
) -> pd.DataFrame:
    """
    Add 'pe_std', 'width_pct', and 'score' columns to df.

    All transformations are applied cross-sectionally within each date so
    that the score is directly comparable across months.

    Parameters
    ----------
    df      : DataFrame with columns [date, y_pred, width]
    lambda_ : penalty weight on interval width (0 = pure PE, 1 = full penalty)

    Returns
    -------
    df copy with three new columns added
    """
    df = df.copy()

    df["pe_std"] = df.groupby("date")["y_pred"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std(ddof=1) > 0 else 0.0
    )
    df["width_pct"] = df.groupby("date")["width"].transform(
        lambda x: x.rank(pct=True)
    )
    df["score"] = df["pe_std"] - lambda_ * df["width_pct"]
    return df


# ---------------------------------------------------------------------------
# Lambda tuning
# ---------------------------------------------------------------------------

def lambda_tune(
    cal_df: pd.DataFrame,
    lambda_grid: list[float],
    n_deciles: int = 10,
) -> tuple[float, pd.DataFrame]:
    """
    Grid search over lambda_grid using CALIBRATION set intervals.

    For each candidate lambda:
      1. Compute uncertainty-adjusted scores
      2. Construct monthly L-S portfolio (EW)
      3. Record annualised Sharpe

    Returns (lambda_star, tuning_table) where lambda_star maximises Sharpe.
    Lambda is never adjusted after this point -- see LEARNING.md 7.1.

    Parameters
    ----------
    cal_df      : calibration-set intervals (permno, date, y_pred, y_true, width)
    lambda_grid : candidate lambda values to search
    n_deciles   : portfolio decile count (must match test evaluation)
    """
    rows = []
    for lam in lambda_grid:
        scored = uncertainty_adjusted_score(cal_df, lam)
        port   = construct_ls_portfolio(scored, score_col="score", n_deciles=n_deciles)
        stats  = portfolio_stats(port["ls_ew"])
        rows.append({
            "lambda":       lam,
            "sharpe":       stats["sharpe"],
            "mean_monthly": stats["mean_monthly"],
            "tstat":        stats["tstat"],
            "n_months":     stats["n_months"],
        })

    tune_df     = pd.DataFrame(rows)
    best_idx    = tune_df["sharpe"].idxmax()
    lambda_star = float(tune_df.loc[best_idx, "lambda"])
    return lambda_star, tune_df


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def construct_ls_portfolio(
    df: pd.DataFrame,
    score_col: str = "score",
    n_deciles: int = 10,
    market_cap_col: str | None = None,
) -> dict:
    """
    Monthly long-short portfolio sorted on score.

    Long:  top decile of score  (high PE, low uncertainty if lambda > 0)
    Short: bottom decile        (low PE, high uncertainty if lambda > 0)

    EW returns are always computed.
    VW returns are computed when market_cap_col is provided and non-null.
    Turnover tracking starts from the second month (no prior holdings in month 1).

    Parameters
    ----------
    df             : DataFrame with [permno, date, y_true, score_col]
    score_col      : column to rank stocks on
    n_deciles      : number of equal groups
    market_cap_col : column containing positive market-cap weights; None = EW only

    Returns
    -------
    dict with:
      ls_ew    : pd.Series  monthly EW L-S returns (indexed by date)
      ls_vw    : pd.Series  monthly VW L-S returns (NaN if no market cap)
      turnover : pd.Series  monthly one-way turnover fraction (from month 2)
    """
    required = {"permno", "date", "y_true", score_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"construct_ls_portfolio: missing columns {missing}")

    dates = sorted(df["date"].unique())

    ls_ew_rows   = []
    ls_vw_rows   = []
    turnover_rows= []
    prev_long    = None
    prev_short   = None

    for date in dates:
        mdf = df[df["date"] == date].dropna(subset=[score_col, "y_true"]).copy()
        if len(mdf) < n_deciles * 2:
            continue

        mdf["_dec"] = pd.qcut(mdf[score_col], n_deciles, labels=False, duplicates="drop")
        mdf = mdf.dropna(subset=["_dec"])
        mdf["_dec"] = mdf["_dec"].astype(int)

        max_dec    = mdf["_dec"].max()
        long_mask  = mdf["_dec"] == max_dec
        short_mask = mdf["_dec"] == 0

        long_df  = mdf[long_mask]
        short_df = mdf[short_mask]

        # EW L-S
        ls_ew = float(long_df["y_true"].mean() - short_df["y_true"].mean())
        ls_ew_rows.append({"date": date, "ls_ew": ls_ew})

        # VW L-S  # CRSP-DEPENDENT: raw market cap needed for true VW
        ls_vw = np.nan
        if market_cap_col and market_cap_col in mdf.columns:
            lw = long_df[market_cap_col].clip(lower=0)
            sw = short_df[market_cap_col].clip(lower=0)
            if lw.sum() > 0 and sw.sum() > 0:
                lr = np.average(long_df["y_true"].values, weights=lw)
                sr = np.average(short_df["y_true"].values, weights=sw)
                ls_vw = float(lr - sr)
        ls_vw_rows.append({"date": date, "ls_vw": ls_vw})

        # Turnover (one-way, averaged over both legs)
        cur_long  = set(long_df["permno"])
        cur_short = set(short_df["permno"])

        if prev_long is not None:
            n_l = len(cur_long) + len(prev_long)
            n_s = len(cur_short) + len(prev_short)
            ch_l = len(cur_long.symmetric_difference(prev_long))
            ch_s = len(cur_short.symmetric_difference(prev_short))
            to   = (ch_l / n_l + ch_s / n_s) / 2 if (n_l > 0 and n_s > 0) else np.nan
            turnover_rows.append({"date": date, "turnover": to})

        prev_long  = cur_long
        prev_short = cur_short

    def _to_series(rows, val_key, name):
        return pd.Series(
            {r["date"]: r[val_key] for r in rows},
            name=name,
            dtype=float,
        )

    return {
        "ls_ew":    _to_series(ls_ew_rows,    "ls_ew",    "ls_ew"),
        "ls_vw":    _to_series(ls_vw_rows,    "ls_vw",    "ls_vw"),
        "turnover": _to_series(turnover_rows, "turnover", "turnover"),
    }


# ---------------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------------

def portfolio_stats(ls_series: pd.Series) -> dict:
    """
    Full performance summary for a monthly L-S return series.

    Returns
    -------
    dict with: mean_monthly, std_monthly, ann_return, sharpe (ann), tstat,
               max_drawdown, skewness, n_months
    """
    ls = ls_series.dropna()
    T  = len(ls)

    if T < 2:
        nan = {k: np.nan for k in [
            "mean_monthly", "std_monthly", "ann_return",
            "sharpe", "tstat", "max_drawdown", "skewness",
        ]}
        nan["n_months"] = T
        return nan

    mu  = float(ls.mean())
    sd  = float(ls.std(ddof=1))

    # Maximum drawdown on cumulative L-S NAV
    cum = (1 + ls).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())

    return {
        "mean_monthly": mu,
        "std_monthly":  sd,
        "ann_return":   mu * 12,
        "sharpe":       mu / sd * np.sqrt(12) if sd > 0 else np.nan,
        "tstat":        mu / (sd / np.sqrt(T)) if sd > 0 else np.nan,
        "max_drawdown": max_dd,
        "skewness":     float(ls.skew()),
        "n_months":     T,
    }


# ---------------------------------------------------------------------------
# Transaction cost adjustment
# ---------------------------------------------------------------------------

def apply_transaction_costs(
    ls_series: pd.Series,
    turnover_series: pd.Series,
    half_spread_bps: float = 20.0,
) -> pd.Series:
    """
    Deduct round-trip transaction costs from gross L-S returns.

    net_return_t = gross_return_t - 2 * (half_spread_bps / 10_000) * turnover_t

    The first month always has turnover = 1.0 (building positions from scratch).
    Any remaining months without tracked turnover use the sample mean.

    Parameters
    ----------
    ls_series       : gross monthly L-S returns (indexed by date)
    turnover_series : fraction of portfolio turned over each month (from month 2)
    half_spread_bps : one-way transaction cost in basis points (default 20)
    """
    cost_rate = 2.0 * (half_spread_bps / 10_000.0)

    to = turnover_series.reindex(ls_series.index)

    # First month: full portfolio construction cost
    first_date = ls_series.index[0]
    if pd.isna(to.get(first_date, np.nan)):
        to[first_date] = 1.0

    # Fill remaining NaNs with the mean tracked turnover
    mean_to = to.dropna().mean()
    to = to.fillna(mean_to if not np.isnan(mean_to) else 0.5)

    net = ls_series - cost_rate * to
    net.name = "ls_net"
    return net


# ---------------------------------------------------------------------------
# Newey-West HAC variance helper
# ---------------------------------------------------------------------------

def _nw_vcov(X: np.ndarray, u: np.ndarray, n_lags: int) -> np.ndarray:
    """
    Newey-West HAC variance-covariance matrix for OLS.

    V(beta_hat) = (X'X)^{-1} S (X'X)^{-1}
    S = sum_{j=-(L)}^{L} w(j) * Gamma_j
    Gamma_j = Xu[j:].T @ Xu[:-j]   (for j >= 0)
    w(j) = 1 - |j|/(L+1)           Bartlett kernel
    """
    Xu    = X * u[:, np.newaxis]          # (T, k)
    XtXinv = np.linalg.inv(X.T @ X)

    S = Xu.T @ Xu                         # Gamma_0 (no lag)

    for j in range(1, n_lags + 1):
        w  = 1.0 - j / (n_lags + 1)
        Gj = Xu[j:].T @ Xu[:-j]          # Gamma_j
        S += w * (Gj + Gj.T)              # symmetrise

    return XtXinv @ S @ XtXinv


# ---------------------------------------------------------------------------
# FF5 alpha
# ---------------------------------------------------------------------------

FF5_FACTORS = ["mkt_rf", "smb", "hml", "rmw", "cma"]


def ff5_alpha(
    ls_series: pd.Series,
    ff5_df: pd.DataFrame | None,
    n_lags: int = 6,
) -> dict | None:
    """
    Fama-French 5-factor alpha for a monthly L-S return series.

    Model: R_{ls,t} = alpha + beta_MKT*MKT_t + beta_SMB*SMB_t
                    + beta_HML*HML_t + beta_RMW*RMW_t + beta_CMA*CMA_t + e_t

    Standard errors use the Newey-West (1987) HAC estimator with n_lags
    Bartlett lags to correct for autocorrelation in residuals.

    Parameters
    ----------
    ls_series : monthly gross L-S return series (indexed by date, pd.Period M)
    ff5_df    : DataFrame from load_ff5_factors(); must have column 'date' and
                factor columns listed in FF5_FACTORS. Pass None to skip.
    n_lags    : Newey-West lag order (default 6, matching FM bandwidth)

    Returns
    -------
    dict with: alpha_monthly, alpha_annual, t_stat, r2, betas, n_months
    None if ff5_df is None or insufficient overlap.
    """
    if ff5_df is None:
        return None

    missing_cols = set(FF5_FACTORS) - set(ff5_df.columns)
    if missing_cols:
        raise ValueError(f"ff5_df is missing factor columns: {missing_cols}")

    # Align on date index
    ls_df  = pd.DataFrame({"ret": ls_series})
    ls_df.index.name = "date"
    ff5_idx = ff5_df.set_index("date")[FF5_FACTORS]
    merged  = ls_df.join(ff5_idx, how="inner").dropna()

    if len(merged) < 12:
        return None

    Y = merged["ret"].values
    X = np.column_stack([np.ones(len(merged)), merged[FF5_FACTORS].values])

    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    u = Y - X @ beta

    Vcov     = _nw_vcov(X, u, n_lags)
    se_alpha = float(np.sqrt(max(Vcov[0, 0], 0.0)))
    alpha    = float(beta[0])

    ss_res = float(np.sum(u ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))

    return {
        "alpha_monthly": alpha,
        "alpha_annual":  alpha * 12,
        "t_stat":        alpha / se_alpha if se_alpha > 0 else np.nan,
        "r2":            1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "betas":         {f: float(b) for f, b in zip(FF5_FACTORS, beta[1:])},
        "n_months":      len(merged),
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_results_table(
    strategies: dict[str, dict],
    ff5_results: dict[str, dict | None],
    cost_stats:  dict[str, dict],
    turnover_means: dict[str, float],
) -> pd.DataFrame:
    """
    Assemble a single DataFrame comparing all strategies.

    Parameters
    ----------
    strategies      : {label: portfolio_stats dict}
    ff5_results     : {label: ff5_alpha dict or None}
    cost_stats      : {label: portfolio_stats dict for net returns}
    turnover_means  : {label: mean monthly turnover fraction}

    Returns
    -------
    DataFrame indexed by metric, columns = strategy labels
    """
    metrics = {
        "Mean ret (%/mo)":    lambda s, _f, _c: s["mean_monthly"] * 100,
        "Ann. return (%)":    lambda s, _f, _c: s["ann_return"] * 100,
        "Ann. Sharpe":        lambda s, _f, _c: s["sharpe"],
        "t-stat":             lambda s, _f, _c: s["tstat"],
        "Max drawdown (%)":   lambda s, _f, _c: s["max_drawdown"] * 100,
        "Skewness":           lambda s, _f, _c: s["skewness"],
        "N months":           lambda s, _f, _c: s["n_months"],
        "Mean turnover (%)":  None,
        "Net Sharpe":         lambda s, _f, c: c["sharpe"],
        "FF5 alpha (%/mo)":   lambda s, f, _c: f["alpha_monthly"] * 100 if f else np.nan,
        "FF5 alpha t-stat":   lambda s, f, _c: f["t_stat"]               if f else np.nan,
        "FF5 R2":             lambda s, f, _c: f["r2"]                   if f else np.nan,
    }

    data = {}
    for label in strategies:
        s  = strategies[label]
        f  = ff5_results.get(label)
        c  = cost_stats.get(label, {})
        to = turnover_means.get(label, np.nan)

        col = {}
        for met, fn in metrics.items():
            if met == "Mean turnover (%)":
                col[met] = to * 100 if not np.isnan(to) else np.nan
            else:
                col[met] = fn(s, f, c)
        data[label] = col

    return pd.DataFrame(data)


def print_results_table(tbl: pd.DataFrame, half_spread_bps: float = 20.0) -> None:
    w_label = 24
    w_col   = 14

    header = f"{'Metric':<{w_label}}" + "".join(f"{c:>{w_col}}" for c in tbl.columns)
    sep    = "=" * (w_label + w_col * len(tbl.columns))

    print(f"\n{sep}")
    print("  PORTFOLIO RESULTS (test period)")
    print(sep)
    print(header)
    print("-" * (w_label + w_col * len(tbl.columns)))

    fmt_map = {
        "N months":  lambda v: f"{int(v)}",
        "default":   lambda v: f"{v:.3f}" if not np.isnan(v) else "N/A",
    }

    for metric, row in tbl.iterrows():
        if metric == "Mean turnover (%)":
            print("-" * (w_label + w_col * len(tbl.columns)))
            print(f"  Transaction costs ({half_spread_bps:.0f} bps half-spread)")
            print("-" * (w_label + w_col * len(tbl.columns)))
        elif metric == "FF5 alpha (%/mo)":
            print("-" * (w_label + w_col * len(tbl.columns)))
            print("  Fama-French 5-factor alpha")
            print("-" * (w_label + w_col * len(tbl.columns)))

        fmt = fmt_map.get(metric, fmt_map["default"])
        row_str = f"  {metric:<{w_label-2}}" + "".join(
            f"{fmt(v):>{w_col}}" for v in row.values
        )
        print(row_str)

    print(sep)
