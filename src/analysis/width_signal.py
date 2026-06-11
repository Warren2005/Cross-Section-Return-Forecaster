"""
Width-as-signal analysis.

Three tests of the hypothesis that conformal interval width is a standalone
cross-sectional return predictor (see LEARNING.md §6):

  1. Univariate decile sort on width
     Each month, sort stocks into deciles by interval width. Long D1 (narrowest,
     model most confident) and short D10 (widest, model least confident).
     Expected: D1 outperforms D10 — uncertainty predicts lower returns.

  2. Fama-MacBeth regression
     Monthly cross-sectional OLS of forward returns on width + controls.
     Time-series average of monthly coefficients with Newey-West t-stats.
     Key test: is the width coefficient significantly NEGATIVE?

  3. Double sort: width tercile × point-estimate decile
     Does the point-estimate sort have higher predictive validity when the
     model is confident (narrow width) than when uncertain (wide width)?
     Expected: L-S Sharpe from PE sort is highest in the narrow-width tercile.

  4. Width IC
     Spearman correlation of width with subsequent absolute prediction error.
     Positive IC confirms width is a meaningful uncertainty proxy.

All functions operate on the interval DataFrame produced by Phase 3:
  columns: permno, date, y_pred, y_true, lower, upper, width, used_spci
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ls_stats(ls_series: pd.Series) -> dict:
    """Annualised stats for a monthly long-short return series."""
    ls = ls_series.dropna()
    T  = len(ls)
    mu = float(ls.mean())
    sd = float(ls.std(ddof=1))
    return {
        "mean_monthly": mu,
        "std_monthly":  sd,
        "sharpe":       mu / sd * np.sqrt(12) if sd > 0 else np.nan,
        "tstat":        mu / (sd / np.sqrt(T)) if sd > 0 else np.nan,
        "n_months":     T,
        "ann_return":   mu * 12,
    }


def _cross_rank(s: pd.Series) -> pd.Series:
    """Cross-sectionally rank-normalise a series to [0, 1] within each date group."""
    return s.rank(pct=True)


# ---------------------------------------------------------------------------
# Test 1 — Univariate width decile sort
# ---------------------------------------------------------------------------

def width_decile_sort(
    interval_df: pd.DataFrame,
    n_deciles: int = 10,
) -> dict:
    """
    Each month sort stocks into deciles by interval width. Compute per-decile
    equal-weighted returns. Report D1 (narrowest) − D10 (widest) L-S portfolio.

    Uses SAME-PERIOD returns: the width for month t predicts the return IN
    month t (the return the interval was constructed to cover). This is the
    most direct test of whether high-uncertainty stocks have different realised
    returns.

    Returns
    -------
    dict with:
      decile_stats : DataFrame — per-decile mean/std/Sharpe
      ls_series    : monthly L-S return series (narrow − wide)
      ls_stats     : L-S performance summary
    """
    df = interval_df.dropna(subset=["y_true", "width"]).copy()

    df["width_decile"] = df.groupby("date")["width"].transform(
        lambda x: pd.qcut(x, n_deciles, labels=False, duplicates="drop")
    )
    df = df.dropna(subset=["width_decile"])
    df["width_decile"] = df["width_decile"].astype(int)

    # Monthly per-decile returns
    monthly = (
        df.groupby(["date", "width_decile"])["y_true"]
        .mean()
        .unstack("width_decile")
    )

    # L-S: D0 (narrowest = 0) minus D_{n-1} (widest)
    ls_series = monthly[0] - monthly[n_deciles - 1]

    # Per-decile summary across all months
    decile_stats = []
    for d in range(n_deciles):
        if d not in monthly.columns:
            continue
        col = monthly[d].dropna()
        decile_stats.append({
            "Decile":       f"D{d+1}",
            "Label":        "narrow" if d == 0 else ("wide" if d == n_deciles - 1 else ""),
            "Mean ret (%/mo)": f"{col.mean()*100:.3f}",
            "Std (%/mo)":      f"{col.std()*100:.3f}",
            "Sharpe (ann)":    f"{col.mean()/col.std()*np.sqrt(12):.2f}" if col.std() > 0 else "N/A",
        })

    return {
        "decile_stats": pd.DataFrame(decile_stats),
        "ls_series":    ls_series,
        "ls_stats":     _ls_stats(ls_series),
    }


# ---------------------------------------------------------------------------
# Test 2 — Fama-MacBeth regression
# ---------------------------------------------------------------------------

# Controls sourced from the GKX panel (already rank-normalised to [-1, 1]).
# These proxy for the canonical Fama-MacBeth controls:
#   me      → log(size)          (rank-normalised; monotone with log-size)
#   bm      → book-to-market     (rank-normalised)
#   mom12m  → momentum 12-1      (12-month cumulative return, skipping 1m)
#   mom1m   → short-term reversal (last-month return)
# Using rank-normalised proxies is standard practice when raw CRSP data is
# unavailable. The t-statistics are unaffected by the monotone transformation.
# See LEARNING.md §6.1 for full discussion.
CONTROL_COLS = ["me", "bm", "mom12m", "mom1m"]


def fama_macbeth(
    interval_df: pd.DataFrame,
    gkx_panel: pd.DataFrame,
    forward_periods: int = 1,
    bandwidth: int = 6,
) -> object:
    """
    Fama-MacBeth regression of forward returns on width + controls.

    Specification:
      ret_{i,t+forward} = a_t + b_t*ŷ_{i,t} + c_t*width_pct_{i,t}
                        + d_t*me_{i,t} + e_t*bm_{i,t}
                        + f_t*mom12m_{i,t} + g_t*mom1m_{i,t} + ε

    width_pct is the cross-sectional percentile rank of width (in [0,1]),
    making the coefficient comparable across months with different mean widths.

    Parameters
    ----------
    interval_df     : Phase 3 output (permno, date, y_pred, y_true, width)
    gkx_panel       : test split with GKX characteristics (controls)
    forward_periods : months ahead for the dependent variable (default 1)
    bandwidth       : Newey-West lag order for kernel standard errors

    Returns
    -------
    linearmodels FamaMacBeth result object (has .summary, .params, .tstats)
    """
    from linearmodels import FamaMacBeth

    # Merge intervals with controls
    controls_available = [c for c in CONTROL_COLS if c in gkx_panel.columns]
    merge_cols = ["permno", "date"] + controls_available

    df = interval_df[["permno", "date", "y_pred", "y_true", "width"]].merge(
        gkx_panel[merge_cols], on=["permno", "date"], how="left"
    ).dropna(subset=["y_true", "width"])

    # Cross-sectional percentile rank of width (same convention as characteristics)
    df["width_pct"] = df.groupby("date")["width"].transform(_cross_rank)

    # Forward return: shift y_true forward by forward_periods within each stock
    df = df.sort_values(["permno", "date"])
    df["ret_fwd"] = df.groupby("permno")["y_true"].shift(-forward_periods)
    df = df.dropna(subset=["ret_fwd"])

    # Build regressor list
    regressors = ["y_pred", "width_pct"] + controls_available

    # Add constant manually (linearmodels FamaMacBeth doesn't add one by default)
    df["const"] = 1.0
    regressors  = ["const"] + regressors

    # Set MultiIndex (entity, time)
    df_idx = df.set_index(["permno", "date"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = FamaMacBeth(
            dependent = df_idx[["ret_fwd"]],
            exog      = df_idx[regressors],
        )
        result = model.fit(cov_type="kernel", bandwidth=bandwidth)

    return result


def print_fm_result(result) -> None:
    """Print a concise Fama-MacBeth summary focused on the width coefficient."""
    params  = result.params
    tstats  = result.tstats
    pvalues = result.pvalues

    print(f"\n{'='*60}")
    print("  FAMA-MACBETH REGRESSION RESULTS")
    print(f"  Dep. var: forward 1-month return")
    print(f"  SE: Newey-West kernel (bandwidth=6)")
    print(f"{'='*60}")
    print(f"  {'Variable':<15} {'Coeff':>10} {'t-stat':>10} {'p-val':>8}")
    print(f"  {'-'*48}")
    for var in params.index:
        coeff = params[var]
        tstat = tstats[var]
        pval  = pvalues[var]
        flag  = " ***" if pval < 0.01 else (" **" if pval < 0.05 else (" *" if pval < 0.10 else ""))
        print(f"  {var:<15} {coeff:>10.4f} {tstat:>10.2f} {pval:>8.3f}{flag}")

    width_t = float(tstats.get("width_pct", np.nan))
    width_c = float(params.get("width_pct", np.nan))
    print(f"\n  KEY: width_pct coeff = {width_c:.4f}, t = {width_t:.2f}")
    if width_t < -2.0:
        print("  [PASS]  Width is significantly NEGATIVE -- wider intervals -> lower returns")
    elif width_t < 0:
        print("  [~]  Width coefficient is negative but not significant (t < 2)")
    else:
        print("  [FAIL]  Width coefficient is NOT negative -- hypothesis not supported")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Test 3 — Double sort: width tercile × point-estimate decile
# ---------------------------------------------------------------------------

def double_sort(
    interval_df: pd.DataFrame,
    n_width_groups: int = 3,
    n_pe_deciles: int = 10,
) -> dict:
    """
    For each width group, sort stocks on y_pred and compute the L-S Sharpe.

    Expected: L-S Sharpe from PE sort is HIGHEST in the narrow-width tercile
    (T1) and LOWEST in the wide-width tercile (T3). This confirms that model
    confidence improves the signal-to-noise of the point estimate.

    Returns
    -------
    dict: {width_group_label: ls_stats_dict}
    """
    df = interval_df.dropna(subset=["y_true", "y_pred", "width"]).copy()

    df["width_group"] = df.groupby("date")["width"].transform(
        lambda x: pd.qcut(x, n_width_groups, labels=False, duplicates="drop")
    )
    df = df.dropna(subset=["width_group"])
    df["width_group"] = df["width_group"].astype(int)

    results = {}
    labels  = {0: "T1 (narrow)", 1: "T2 (medium)", 2: "T3 (wide)"}

    for g in range(n_width_groups):
        sub = df[df["width_group"] == g].copy()

        sub["pe_decile"] = sub.groupby("date")["y_pred"].transform(
            lambda x: pd.qcut(x, n_pe_deciles, labels=False, duplicates="drop")
        )
        sub = sub.dropna(subset=["pe_decile"])
        sub["pe_decile"] = sub["pe_decile"].astype(int)

        monthly = (
            sub.groupby(["date", "pe_decile"])["y_true"]
            .mean()
            .unstack("pe_decile")
        )

        if n_pe_deciles - 1 not in monthly.columns or 0 not in monthly.columns:
            results[labels.get(g, f"T{g+1}")] = {"sharpe": np.nan, "note": "insufficient data"}
            continue

        ls = monthly[n_pe_deciles - 1] - monthly[0]  # top − bottom PE decile
        stats = _ls_stats(ls)
        results[labels.get(g, f"T{g+1}")] = stats

    return results


def print_double_sort(results: dict) -> None:
    print(f"\n{'='*60}")
    print("  DOUBLE SORT: width tercile x point-estimate decile")
    print("  (within each width group, L-S = top PE decile - bottom)")
    print(f"{'='*60}")
    print(f"  {'Group':<15} {'L-S Sharpe':>12} {'Mean (%/mo)':>12} {'t-stat':>8}")
    print(f"  {'-'*50}")
    for label, stats in results.items():
        sh  = stats.get("sharpe", np.nan)
        mn  = stats.get("mean_monthly", np.nan)
        t   = stats.get("tstat", np.nan)
        print(f"  {label:<15} {sh:>12.3f} {mn*100:>12.3f} {t:>8.2f}")

    sharpes = [v.get("sharpe", np.nan) for v in results.values()]
    if not all(np.isnan(sharpes)):
        best = list(results.keys())[int(np.nanargmax(sharpes))]
        print(f"\n  Highest L-S Sharpe in: {best}")
        if "narrow" in best.lower():
            print("  [OK]  Narrow-width stocks have the strongest PE signal -- as expected")
        else:
            print("  [~]  Narrow-width group does NOT have the strongest PE signal")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Test 4 — Width IC
# ---------------------------------------------------------------------------

def width_ic(interval_df: pd.DataFrame) -> dict:
    """
    Monthly Spearman IC between interval width and absolute prediction error.

    Positive IC means wider intervals → larger errors → width is a valid
    proxy for model uncertainty.

    Also computes IC between width and NEGATIVE realized return (to test
    whether uncertainty directly predicts lower returns, same period).
    """
    df = interval_df.dropna(subset=["y_true", "y_pred", "width"]).copy()
    df["abs_error"] = (df["y_true"] - df["y_pred"]).abs()

    def _monthly_ic(grp: pd.DataFrame, target_col: str) -> float:
        if len(grp) < 10:
            return np.nan
        r, _ = spearmanr(grp["width"], grp[target_col])
        return float(r)

    ic_error  = df.groupby("date").apply(_monthly_ic, "abs_error",  include_groups=False)
    ic_negret = df.groupby("date").apply(
        lambda g: _monthly_ic(g.assign(neg_ret=-g["y_true"]), "neg_ret"),
        include_groups=False,
    )

    def _summarise(ic_series: pd.Series, label: str) -> dict:
        ic = ic_series.dropna()
        mu = float(ic.mean())
        sd = float(ic.std(ddof=1))
        return {
            "label":    label,
            "mean_ic":  mu,
            "std_ic":   sd,
            "ir":       mu / sd if sd > 0 else np.nan,
            "n_months": len(ic),
            "pct_pos":  float((ic > 0).mean()),
        }

    return {
        "ic_vs_abs_error": _summarise(ic_error,  "IC(width, |error|)"),
        "ic_vs_neg_ret":   _summarise(ic_negret, "IC(width, -ret)"),
    }


def print_ic_results(ic_results: dict) -> None:
    print(f"\n{'='*60}")
    print("  WIDTH INFORMATION COEFFICIENTS")
    print(f"{'='*60}")
    for key, stats in ic_results.items():
        ir_str = f"{stats['ir']:.2f}" if not np.isnan(stats.get("ir", np.nan)) else "N/A"
        print(f"\n  {stats['label']}:")
        print(f"    Mean IC : {stats['mean_ic']:+.4f}")
        print(f"    Std IC  : {stats['std_ic']:.4f}")
        print(f"    IR      : {ir_str}")
        print(f"    % months positive: {stats['pct_pos']*100:.1f}%")
        print(f"    N months: {stats['n_months']}")

    ic_err = ic_results["ic_vs_abs_error"]["mean_ic"]
    if ic_err > 0.05:
        print(f"\n  [PASS]  IC(width, |error|) = {ic_err:.3f} > 0.05 -- gate passed")
    else:
        print(f"\n  [FAIL]  IC(width, |error|) = {ic_err:.3f} <= 0.05 -- width may not proxy uncertainty well")
    print(f"{'='*60}")
