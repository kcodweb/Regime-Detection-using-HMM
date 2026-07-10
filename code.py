import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pipeline as pl

OUT = "figures"
import os
os.makedirs(OUT, exist_ok=True)

pd.set_option("display.width", 140)

# --- Phase 1: data ---
prices, source = pl.get_data(start="2012-01-01", end="2025-12-31")
prices.to_csv("figures/prices.csv")
print(prices.tail())

# --- Phase 2: features ---
features = pl.build_features(prices)
print("\nFeature sample:")
print(features.tail())

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(prices.index, prices["stocks"], label="stocks")
axes[0].set_title("Price (stocks proxy)")
axes[1].plot(features.index, features["vol_1m"], color="firebrick", label="1m annualized vol")
axes[1].set_title("Rolling volatility feature (sanity check vs known stress periods)")
for ax in axes:
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/01_data_features_sanity_check.png", dpi=130)
plt.close()

# --- Phase 3: single in-sample HMM fit (for the "wow it's working" plot) ---
from sklearn.preprocessing import StandardScaler
X = features[pl.FEATURE_COLS].values
X_scaled = StandardScaler().fit_transform(X)
model_full = pl.fit_hmm(X_scaled)
vol_idx = pl.FEATURE_COLS.index("vol_1m")
mapping_full = pl.label_states_by_volatility(model_full, X_scaled, vol_idx)
states_full = model_full.predict(X_scaled)
labels_full = pd.Series([mapping_full[s] for s in states_full], index=features.index)

colors = {"Bull": "#2ca02c", "Bear": "#ff7f0e", "Crisis": "#d62728"}
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(prices.loc[features.index, "stocks"], color="black", lw=1)
for regime, c in colors.items():
    mask = labels_full == regime
    ax.fill_between(features.index, prices.loc[features.index, "stocks"].min(),
                     prices.loc[features.index, "stocks"].max(),
                     where=mask.values, color=c, alpha=0.15, label=regime)
ax.set_title("HMM-Inferred Regimes Overlaid on Price (full in-sample fit, for inspection only)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/02_regimes_overlay_full_fit.png", dpi=130)
plt.close()

print("\nTransition matrix (full in-sample fit):")
trans_df = pd.DataFrame(model_full.transmat_,
                         index=[mapping_full[i] for i in range(3)],
                         columns=[mapping_full[i] for i in range(3)])
trans_df = trans_df.reindex(index=["Bull", "Bear", "Crisis"], columns=["Bull", "Bear", "Crisis"])
print(trans_df.round(3))
trans_df.to_csv(f"{OUT}/transition_matrix_full_fit.csv")

# --- Phase 4: walk-forward validation (this is what the backtest actually uses) ---
print("\nRunning walk-forward validation (train=504d, test=63d, rolling)...")
wf_regimes, wf_trans_mats = pl.walk_forward_regimes(features, train_size=504, test_size=63)
print(f"Out-of-sample regime-labeled days: {len(wf_regimes)}")
print(wf_regimes.value_counts())

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(prices.loc[wf_regimes.index, "stocks"], color="black", lw=1)
for regime, c in colors.items():
    mask = wf_regimes == regime
    ax.fill_between(wf_regimes.index, prices.loc[wf_regimes.index, "stocks"].min(),
                     prices.loc[wf_regimes.index, "stocks"].max(),
                     where=mask.values, color=c, alpha=0.15, label=regime)
ax.set_title("Walk-Forward (Out-of-Sample) Regimes Overlaid on Price -- used in backtest")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/03_regimes_overlay_walkforward.png", dpi=130)
plt.close()

# --- Phase 5 & 6: backtest dynamic strategy vs benchmarks ---
print("\nRunning dynamic strategy backtest (with transaction costs)...")
wf_regimes_smoothed = pl.smooth_regimes(wf_regimes, window=21)
print(f"Raw daily regime changes: {(wf_regimes != wf_regimes.shift()).sum()}  ->  "
      f"Smoothed regime changes: {(wf_regimes_smoothed != wf_regimes_smoothed.shift()).sum()}")

dyn_net, dyn_w, dyn_to = pl.backtest_dynamic(prices, wf_regimes_smoothed, lookback=63, min_hold=21, max_hold=126, cost_bps=7)
dyn_gross, _, _ = pl.backtest_dynamic(prices, wf_regimes_smoothed, lookback=63, min_hold=21, max_hold=126, cost_bps=0)

static_net, static_w, static_to = pl.backtest_static(prices, {"stocks": 0.6, "gold": 0.0, "bonds": 0.4}, cost_bps=7)
eqw_net, eqw_w, eqw_to = pl.backtest_static(prices, {"stocks": 1/3, "gold": 1/3, "bonds": 1/3}, cost_bps=7)

common_idx = dyn_net.index.intersection(static_net.index).intersection(eqw_net.index)
dyn_net, dyn_gross = dyn_net.loc[common_idx], dyn_gross.loc[common_idx]
static_net, eqw_net = static_net.loc[common_idx], eqw_net.loc[common_idx]
dyn_to = dyn_to.loc[common_idx]

results = {
    "Dynamic (regime-aware, net of costs)": pl.performance_stats(dyn_net),
    "Dynamic (regime-aware, gross, no costs)": pl.performance_stats(dyn_gross),
    "Static 60/40": pl.performance_stats(static_net),
    "Equal-Weight (1/3 each)": pl.performance_stats(eqw_net),
}
results_df = pd.DataFrame(results).T
results_df["turnover_annualized"] = np.nan
results_df.loc["Dynamic (regime-aware, net of costs)", "turnover_annualized"] = dyn_to.sum() / (len(dyn_to)/252)
results_df.loc["Static 60/40", "turnover_annualized"] = static_to.loc[common_idx].sum() / (len(common_idx)/252)
results_df.loc["Equal-Weight (1/3 each)", "turnover_annualized"] = eqw_to.loc[common_idx].sum() / (len(common_idx)/252)

print("\n=== PERFORMANCE SUMMARY ===")
print(results_df.round(4))
results_df.round(4).to_csv(f"{OUT}/performance_summary.csv")

# equity curve plot
fig, ax = plt.subplots(figsize=(12, 5))
(1 + dyn_net).cumprod().plot(ax=ax, label="Dynamic regime-aware (net)")
(1 + static_net).cumprod().plot(ax=ax, label="Static 60/40")
(1 + eqw_net).cumprod().plot(ax=ax, label="Equal-weight")
ax.set_title("Equity Curves: Dynamic Regime-Aware Strategy vs Static Benchmarks")
ax.set_ylabel("Growth of $1")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/04_equity_curves.png", dpi=130)
plt.close()

# weights over time for dynamic strategy
fig, ax = plt.subplots(figsize=(12, 4))
dyn_w.loc[common_idx].plot.area(ax=ax, alpha=0.8)
ax.set_title("Dynamic Strategy: Portfolio Weights Over Time")
ax.set_ylabel("Weight")
plt.tight_layout()
plt.savefig(f"{OUT}/05_dynamic_weights.png", dpi=130)
plt.close()

print("\nAll figures saved to figures/. Done.")
