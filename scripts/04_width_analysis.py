"""
Phase 4 — Width-as-signal analysis.

Usage:
    python scripts/04_width_analysis.py [--fast-dev] [--config PATH]

Prerequisites:
    Phases 1–3 must have been run (needs test_intervals.parquet + test.parquet).

What this script does:
  1. Load test_intervals.parquet (Phase 3 output) and test.parquet (GKX controls)
  2. Run all four width-signal tests:
       Test 1: Univariate decile sort on interval width
       Test 2: Fama-MacBeth regression (forward return ~ width + controls)
       Test 3: Double sort (width tercile × point-estimate decile)
       Test 4: Width IC vs. |error| and vs. -return
  3. Save result tables to results/tables/

Verification gate:
  Width IC (vs. |error|) should be > 0.05
  If the width signal is real: FM coefficient on width should be significantly
  negative (t < -2.0) and the D1-D10 L-S Sharpe should be > 0.8.

Outputs (results/tables/):
  width_decile_sort.csv
  width_fm_regression.csv
  width_double_sort.csv
  width_ic.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.analysis.width_signal import (
    double_sort,
    fama_macbeth,
    print_double_sort,
    print_fm_result,
    print_ic_results,
    width_decile_sort,
    width_ic,
)
from src.data.splits import fast_dev_subsample
from src.utils.config import load_config
from src.utils.io import load_parquet, save_table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSRF Phase 4: width-signal analysis")
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
    print("CSRF Phase 4 — Width-Signal Analysis")
    print(f"  fast_dev: {cfg.fast_dev}")
    print("=" * 65)

    splits_dir  = Path(cfg.data.splits_dir)
    results_dir = cfg.results_dir

    # -- 1. Load data --------------------------------------------------------
    print("\n[1/5] Loading test intervals and GKX test panel...")

    intervals = load_parquet(splits_dir / "test_intervals.parquet")
    test_df   = load_parquet(splits_dir / "test.parquet")

    if cfg.fast_dev:
        intervals = fast_dev_subsample(intervals, cfg.data)
        test_df   = fast_dev_subsample(test_df,   cfg.data)

    with open(splits_dir / "column_meta.json") as f:
        meta = json.load(f)

    print(f"  Intervals: {len(intervals):,} rows  |  "
          f"Test panel: {len(test_df):,} rows")
    print(f"  Test period: {intervals['date'].min()} – {intervals['date'].max()}")
    print(f"  Unique stocks: {intervals['permno'].nunique():,}")

    # Interval width summary
    w_mean = intervals["width"].mean() * 10_000
    w_med  = intervals["width"].median() * 10_000
    print(f"\n  Interval width: mean={w_mean:.1f} bps, median={w_med:.1f} bps")

    # -- 2. Test 1: Univariate width decile sort ------------------------------
    print("\n[2/5] Test 1: Univariate width decile sort...")

    sort_results = width_decile_sort(intervals, n_deciles=cfg.portfolio.n_deciles)

    print("\n  Per-decile returns (D1=narrowest → D10=widest):")
    print(sort_results["decile_stats"].to_string(index=False))

    ls = sort_results["ls_stats"]
    print(f"\n  L-S (D1 − D10):")
    print(f"    Mean:   {ls['mean_monthly']*100:.3f}%/month  "
          f"({ls['ann_return']*100:.2f}% annualised)")
    print(f"    Sharpe: {ls['sharpe']:.3f}  (t={ls['tstat']:.2f}, "
          f"n={ls['n_months']}mo)")

    if ls["sharpe"] > 0.8:
        print("  [PASS]  L-S Sharpe > 0.8 -- width decile signal is economically meaningful")
    else:
        print("  [~]  L-S Sharpe <= 0.8 -- width signal may be weak")

    save_table(sort_results["decile_stats"], results_dir, "width_decile_sort.csv")
    ls_df = pd.DataFrame([ls])
    save_table(ls_df, results_dir, "width_ls_stats.csv")

    # -- 3. Test 2: Fama-MacBeth regression -----------------------------------
    print("\n[3/5] Test 2: Fama-MacBeth regression...")
    print("  (forward return ~ const + y_pred + width_pct + me + bm + mom12m + mom1m)")

    fm_result = fama_macbeth(
        interval_df   = intervals,
        gkx_panel     = test_df,
        forward_periods = 1,
        bandwidth       = cfg.portfolio.fm_bandwidth,
    )
    print_fm_result(fm_result)

    # Save FM summary table
    fm_df = pd.DataFrame({
        "variable": fm_result.params.index,
        "coeff":    fm_result.params.values,
        "tstat":    fm_result.tstats.values,
        "pvalue":   fm_result.pvalues.values,
    })
    save_table(fm_df, results_dir, "width_fm_regression.csv")

    # -- 4. Test 3: Double sort -----------------------------------------------
    print("\n[4/5] Test 3: Double sort (width tercile × PE decile)...")

    ds_results = double_sort(
        intervals,
        n_width_groups=3,
        n_pe_deciles=cfg.portfolio.n_deciles,
    )
    print_double_sort(ds_results)

    ds_df = pd.DataFrame([
        {"group": k, **v} for k, v in ds_results.items()
    ])
    save_table(ds_df, results_dir, "width_double_sort.csv")

    # -- 5. Test 4: Width IC --------------------------------------------------
    print("\n[5/5] Test 4: Width information coefficient...")

    ic_results = width_ic(intervals)
    print_ic_results(ic_results)

    ic_df = pd.DataFrame([
        {**{"key": k}, **v} for k, v in ic_results.items()
    ])
    save_table(ic_df, results_dir, "width_ic.csv")

    # -- Summary --------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  PHASE 4 SUMMARY")
    print("=" * 65)
    print(f"  Test 1 L-S Sharpe:        {ls['sharpe']:+.3f}  "
          f"(t={ls['tstat']:.2f})")

    width_t = float(fm_result.tstats.get("width_pct",
                    fm_result.tstats.iloc[2]))
    print(f"  Test 2 FM width t-stat:   {width_t:+.2f}  "
          f"({'[PASS] sig. negative' if width_t < -2 else '[FAIL] not significant'})")

    narrow_sh = ds_results.get("T1 (narrow)", {}).get("sharpe", np.nan)
    wide_sh   = ds_results.get("T3 (wide)",   {}).get("sharpe", np.nan)
    import numpy as np
    print(f"  Test 3 narrow/wide Sharpe: {narrow_sh:.3f} / {wide_sh:.3f}  "
          f"({'[PASS] narrow > wide' if narrow_sh > wide_sh else '[FAIL] narrow <= wide'})")

    ic_val = ic_results["ic_vs_abs_error"]["mean_ic"]
    print(f"  Test 4 IC(width, |err|):  {ic_val:+.4f}  "
          f"({'[PASS] > 0.05' if ic_val > 0.05 else '[FAIL] <= 0.05'})")

    print(f"\n  Results saved to: {results_dir}/tables/")
    print("\nPhase 4 complete.")
    print("Next step: python scripts/05_portfolio_eval.py")


if __name__ == "__main__":
    main()
