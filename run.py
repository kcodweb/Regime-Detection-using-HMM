"""
Command-line runner for the full pipeline; writes every figure and CSV to outputs/.

    python run.py            # uses the committed data snapshot (reproducible)
    python run.py --refresh  # re-downloads prices from yfinance first
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

import pipeline as pl
import plots

OUT = Path(__file__).resolve().parent / "outputs"


def save(fig, name):
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)


def main(refresh=False):
    OUT.mkdir(exist_ok=True)
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    pd.set_option("display.width", 140)

    # --- Phase 1: data ---
    prices, _ = pl.get_data(start="2012-01-01", end="2025-12-31", refresh=refresh)
    print(prices.tail())

    # --- Phase 2: features ---
    features = pl.build_features(prices)
    save(plots.sanity_check(prices, features), "01_data_features_sanity_check.png")

    # --- Phase 3: single in-sample HMM fit (for inspection only) ---
    X_scaled = StandardScaler().fit_transform(features[pl.FEATURE_COLS].values)
    model_full = pl.fit_hmm(X_scaled)
    mapping_full = pl.label_states_by_volatility(model_full, pl.FEATURE_COLS.index("vol_1m"))
    labels_full = pd.Series([mapping_full[s] for s in model_full.predict(X_scaled)], index=features.index)
    save(plots.regime_overlay(prices, labels_full,
                              "HMM regimes, full in-sample fit (inspection only: sees the whole history)"),
         "02_regimes_overlay_full_fit.png")

    order = [s for r in pl.REGIMES for s, lab in mapping_full.items() if lab == r]
    trans_df = pd.DataFrame(model_full.transmat_[order][:, order], index=pl.REGIMES, columns=pl.REGIMES)
    print("\nTransition matrix (full in-sample fit):")
    print(trans_df.round(3))
    trans_df.to_csv(OUT / "transition_matrix_full_fit.csv")

    # --- Phase 4: walk-forward validation (this is what the backtest actually uses) ---
    print("\nRunning walk-forward validation (train=504d, test=63d, rolling, filtered decoding)...")
    wf_regimes, wf_probs, wf_trans = pl.walk_forward_regimes(features, train_size=504, test_size=63)
    wf_smoothed = pl.smooth_regimes(wf_regimes, window=21)
    print(f"Out-of-sample regime-labeled days: {len(wf_regimes)}")
    print(f"Regime changes: raw {int((wf_regimes != wf_regimes.shift()).sum()) - 1}  ->  "
          f"smoothed {int((wf_smoothed != wf_smoothed.shift()).sum()) - 1}")
    save(plots.regime_overlay(prices, wf_smoothed,
                              "Walk-forward (out-of-sample) regimes, smoothed -- used in the backtest",
                              probs=wf_probs),
         "03_regimes_overlay_walkforward.png")

    wf_trans_avg = sum(wf_trans) / len(wf_trans)
    print("\nTransition matrix (average across walk-forward folds):")
    print(wf_trans_avg.round(3))
    wf_trans_avg.to_csv(OUT / "transition_matrix_walkforward_avg.csv")

    diag = pd.concat({"raw": pl.regime_diagnostics(prices, wf_regimes),
                      "smoothed": pl.regime_diagnostics(prices, wf_smoothed)}, names=["labels", "regime"])
    print("\nWhat happens over the NEXT month, by today's out-of-sample regime:")
    print(diag.round(4))
    diag.round(4).to_csv(OUT / "regime_diagnostics.csv")

    # --- Phase 5 & 6: backtest dynamic strategy vs benchmarks ---
    print("\nRunning dynamic strategy backtest (with transaction costs)...")
    dyn = pl.backtest_dynamic(prices, wf_smoothed, lookback=63, min_hold=21, max_hold=126, cost_bps=7)
    start, end = wf_smoothed.index[0], wf_smoothed.index[-1]
    static = pl.backtest_static(prices, {"stocks": 0.6, "bonds": 0.4}, start, end, cost_bps=7)
    eqw = pl.backtest_static(prices, {a: 1 / 3 for a in pl.ASSETS}, start, end, cost_bps=7)
    nifty = pl.backtest_static(prices, {"stocks": 1.0}, start, end, cost_bps=7)

    runs = {
        "Dynamic (regime-aware, net of costs)": (dyn.net, dyn.turnover),
        "Dynamic (regime-aware, gross, no costs)": (dyn.gross, None),
        "Static 60/40 (stocks/bonds)": (static.net, static.turnover),
        "Equal-Weight (1/3 each)": (eqw.net, eqw.turnover),
        "Nifty 50 buy & hold": (nifty.net, None),
    }
    common_idx = dyn.net.index
    for rets, _ in runs.values():
        common_idx = common_idx.intersection(rets.index)
    results_df = pd.DataFrame({
        name: pl.performance_stats(rets.loc[common_idx], None if to is None else to.loc[common_idx])
        for name, (rets, to) in runs.items()
    }).T

    print(f"\n=== PERFORMANCE SUMMARY ({common_idx[0].date()} to {common_idx[-1].date()}, INR) ===")
    print(results_df.round(4))
    results_df.round(4).to_csv(OUT / "performance_summary.csv")

    save(plots.equity_curves({"Dynamic regime-aware (net)": dyn.net.loc[common_idx],
                              "Static 60/40": static.net.loc[common_idx],
                              "Equal-weight": eqw.net.loc[common_idx]}),
         "04_equity_curves.png")
    save(plots.weights_over_time(dyn.weights.loc[common_idx], wf_smoothed), "05_dynamic_weights.png")

    print(f"\nAll figures and CSVs saved to {OUT.name}/. Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="re-download prices from yfinance")
    main(refresh=parser.parse_args().refresh)
