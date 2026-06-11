"""
Coverage validation for conformal prediction intervals.

Computes empirical coverage — the fraction of realized returns that fall
inside their predicted interval — broken down by time period, volatility
decile, and (when market caps are available) size decile.

The expected coverage for a 90% conformal interval (α=0.10) is ≥ 90%.
Values below 90% indicate the intervals are too narrow; values significantly
above 90% indicate they are overly conservative.

See LEARNING.md §5.4 for interpretation guidance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stress period date ranges (inclusive both ends, pd.Period monthly)
# ---------------------------------------------------------------------------

STRESS_PERIODS = {
    "2008-09 Crisis":   ("2008-01", "2009-06"),
    "2020 COVID":       ("2020-01", "2020-06"),
    "2022 Rate Shock":  ("2022-01", "2022-12"),
}


# ---------------------------------------------------------------------------
# Core coverage computation
# ---------------------------------------------------------------------------

def empirical_coverage(interval_df: pd.DataFrame) -> float:
    """
    Fraction of rows where y_true ∈ [lower, upper].

    Parameters
    ----------
    interval_df : DataFrame with columns y_true, lower, upper
    """
    df = interval_df.dropna(subset=["y_true", "lower", "upper"])
    covered = (df["y_true"] >= df["lower"]) & (df["y_true"] <= df["upper"])
    return float(covered.mean())


def coverage_by_month(interval_df: pd.DataFrame) -> pd.Series:
    """Monthly empirical coverage — useful for spotting regime breakdowns."""
    df = interval_df.dropna(subset=["y_true", "lower", "upper"]).copy()
    df["covered"] = (df["y_true"] >= df["lower"]) & (df["y_true"] <= df["upper"])
    return df.groupby("date")["covered"].mean().rename("coverage")


# ---------------------------------------------------------------------------
# Decomposition tables
# ---------------------------------------------------------------------------

def coverage_by_period(
    interval_df: pd.DataFrame,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """
    Coverage table broken down by time period, matching spec Table 7.2.

    Reports coverage, mean interval width (in bps), and mean |residual|
    for the full test set and each stress period.
    """
    df = interval_df.dropna(subset=["y_true", "lower", "upper"]).copy()
    df["covered"]  = (df["y_true"] >= df["lower"]) & (df["y_true"] <= df["upper"])
    df["width_bps"]= df["width"] * 10_000
    df["abs_resid"]= (df["y_true"] - df["y_pred"]).abs()

    target_coverage = 1.0 - alpha
    rows = []

    def _row(label: str, mask: pd.Series) -> dict:
        sub = df[mask]
        if len(sub) == 0:
            return {}
        n_months = sub["date"].nunique()
        return {
            "Period":        label,
            "N months":      n_months,
            "Coverage":      f"{sub['covered'].mean()*100:.1f}%",
            "Width (bps)":   f"{sub['width_bps'].mean():.0f}",
            "Mean |resid|":  f"{sub['abs_resid'].mean()*10_000:.0f} bps",
            "Pass (>=target)": "Y" if sub["covered"].mean() >= target_coverage else "N",
        }

    # Full test set
    rows.append(_row("Full test set", pd.Series([True] * len(df), index=df.index)))

    # Stress periods
    for label, (start, end) in STRESS_PERIODS.items():
        s = pd.Period(start, freq="M")
        e = pd.Period(end,   freq="M")
        mask = (df["date"] >= s) & (df["date"] <= e)
        if mask.any():
            rows.append(_row(label, mask))

    # Normal periods (not in any stress period)
    stress_mask = pd.Series(False, index=df.index)
    for _, (start, end) in STRESS_PERIODS.items():
        s = pd.Period(start, freq="M")
        e = pd.Period(end,   freq="M")
        stress_mask |= (df["date"] >= s) & (df["date"] <= e)
    rows.append(_row("Normal periods", ~stress_mask))

    return pd.DataFrame([r for r in rows if r])


def coverage_by_volatility_decile(
    interval_df: pd.DataFrame,
    return_history: pd.DataFrame,
    n_deciles: int = 10,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """
    Coverage and interval width broken down by stock volatility decile.

    Volatility is measured as the trailing 12-month standard deviation of
    monthly returns, cross-sectionally sorted into deciles each month.

    Parameters
    ----------
    interval_df    : intervals DataFrame with [permno, date, y_true, lower, upper]
    return_history : panel with [permno, date, ret] spanning at least 12 months
                     before the test period — used to compute trailing vol
    n_deciles      : number of vol groups (default 10)
    alpha          : target miscoverage level
    """
    # Compute trailing 12-month realized volatility
    ret_panel = return_history.sort_values(["permno", "date"]).copy()
    ret_panel["trailing_vol"] = ret_panel.groupby("permno")["ret"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=6).std()
    )

    df = interval_df.merge(
        ret_panel[["permno", "date", "trailing_vol"]],
        on=["permno", "date"],
        how="left",
    ).dropna(subset=["y_true", "lower", "upper", "trailing_vol"])

    df["covered"]   = (df["y_true"] >= df["lower"]) & (df["y_true"] <= df["upper"])
    df["width_bps"] = df["width"] * 10_000
    df["vol_decile"]= df.groupby("date")["trailing_vol"].transform(
        lambda x: pd.qcut(x, n_deciles, labels=False, duplicates="drop")
    )
    df = df.dropna(subset=["vol_decile"])
    df["vol_decile"] = df["vol_decile"].astype(int)

    target = 1.0 - alpha
    rows = []
    for d in range(n_deciles):
        sub = df[df["vol_decile"] == d]
        if len(sub) == 0:
            continue
        rows.append({
            "Vol decile":    f"D{d+1} ({'low' if d==0 else 'high' if d==n_deciles-1 else ''})",
            "Coverage":      f"{sub['covered'].mean()*100:.1f}%",
            "Width (bps)":   f"{sub['width_bps'].mean():.0f}",
            "Pass":          "Y" if sub["covered"].mean() >= target else "N",
        })
    return pd.DataFrame(rows)


def coverage_by_size_decile(
    interval_df: pd.DataFrame,
    market_cap_panel: pd.DataFrame,
    n_deciles: int = 10,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """
    Coverage broken down by market-capitalisation decile.

    # CRSP-DEPENDENT: market_cap_panel must have columns [permno, date, me]
    # where me is market equity (market cap in USD millions).
    # If CRSP is unavailable, pass None and this function returns an empty DataFrame.

    Parameters
    ----------
    market_cap_panel : panel with [permno, date, me]; pass None to skip
    """
    if market_cap_panel is None:
        return pd.DataFrame(columns=["Size decile", "Coverage", "Width (bps)", "Pass"])

    df = interval_df.merge(
        market_cap_panel[["permno", "date", "me"]],
        on=["permno", "date"],
        how="left",
    ).dropna(subset=["y_true", "lower", "upper", "me"])

    df["covered"]    = (df["y_true"] >= df["lower"]) & (df["y_true"] <= df["upper"])
    df["width_bps"]  = df["width"] * 10_000
    df["size_decile"]= df.groupby("date")["me"].transform(
        lambda x: pd.qcut(x, n_deciles, labels=False, duplicates="drop")
    )
    df = df.dropna(subset=["size_decile"])
    df["size_decile"] = df["size_decile"].astype(int)

    target = 1.0 - alpha
    rows = []
    for d in range(n_deciles):
        sub = df[df["size_decile"] == d]
        if len(sub) == 0:
            continue
        rows.append({
            "Size decile":  f"D{d+1} ({'micro' if d==0 else 'mega' if d==n_deciles-1 else ''})",
            "Coverage":     f"{sub['covered'].mean()*100:.1f}%",
            "Width (bps)":  f"{sub['width_bps'].mean():.0f}",
            "Pass":         "Y" if sub["covered"].mean() >= target else "N",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Formatted report
# ---------------------------------------------------------------------------

def print_coverage_report(
    interval_df: pd.DataFrame,
    return_history: pd.DataFrame,
    market_cap_panel: pd.DataFrame | None,
    alpha: float = 0.10,
) -> None:
    target = (1.0 - alpha) * 100

    print(f"\n{'='*65}")
    print(f"  COVERAGE VALIDATION  (target: >= {target:.0f}%,  alpha={alpha})")
    print(f"{'='*65}")

    # Overall
    overall = empirical_coverage(interval_df)
    status  = "[PASS]" if overall >= 1.0 - alpha else "[FAIL]"
    print(f"\n  Overall coverage: {overall*100:.2f}%  {status}")

    spci_pct = interval_df["used_spci"].mean() * 100 if "used_spci" in interval_df else float("nan")
    print(f"  SPCI used for {spci_pct:.1f}% of observations "
          f"(remainder used fallback)")

    # By period
    print(f"\n  By period:")
    period_tbl = coverage_by_period(interval_df, alpha)
    print(period_tbl.to_string(index=False))

    # By volatility decile
    print(f"\n  By volatility decile:")
    vol_tbl = coverage_by_volatility_decile(interval_df, return_history, alpha=alpha)
    print(vol_tbl.to_string(index=False))

    # By size decile
    size_tbl = coverage_by_size_decile(interval_df, market_cap_panel, alpha=alpha)
    if not size_tbl.empty:
        print(f"\n  By size decile:")
        print(size_tbl.to_string(index=False))
    else:
        print(f"\n  Size-decile coverage: skipped (CRSP market cap not available)")

    print(f"\n{'='*65}")

    # Gate check
    if overall < 1.0 - alpha:
        print(f"  [!]  GATE FAILED: overall coverage {overall*100:.2f}% < {target:.0f}%")
        print(f"     Do not proceed to Phase 4 -- debug SPCI fitting.")
    else:
        print(f"  [OK]  Gate passed: proceed to Phase 4.")
