"""
Is the edge real? Evidence beyond a single backtest.

  1. Ablation          -- the same allocation method with and without regimes.
  2. Random-regime null -- shuffle the regime runs (same lengths, same mix):
                           how often does random timing do as well as the HMM's?
  3. Bootstrap         -- block-bootstrap 95% intervals for Sharpe differences.
  4. Sensitivity       -- change one design choice at a time.

    python robustness.py   # ~3 min; writes outputs/robustness_*.csv and figures 06-07
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline as pl

N_NULL = 500      # random-regime backtests
N_BOOT = 2000     # bootstrap resamples
BLOCK = 21        # mean bootstrap block length (~1 month)
OUT = Path(__file__).resolve().parent / "outputs"


def sharpe(returns) -> float:
    r = np.asarray(returns)
    return r.mean() / r.std(ddof=1) * np.sqrt(252)


def _row_sharpes(R: np.ndarray) -> np.ndarray:
    return R.mean(axis=1) / R.std(axis=1, ddof=1) * np.sqrt(252)


def stationary_bootstrap_indices(T, n, block=BLOCK, seed=pl.RNG_SEED) -> np.ndarray:
    """
    Politis-Romano stationary bootstrap: resample days in blocks of random
    (geometric, mean `block`) length, wrapping around the end. Keeps the
    volatility clustering and short-range autocorrelation that resampling
    single days would destroy. Returns an (n, T) array of day indices.
    """
    rng = np.random.default_rng(seed)
    idx = np.empty((n, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, n)
    new_block = rng.random((n, T)) < 1 / block
    jumps = rng.integers(0, T, (n, T))
    for t in range(1, T):
        idx[:, t] = np.where(new_block[:, t], jumps[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def sharpe_difference(a: pd.Series, b: pd.Series, n=N_BOOT, block=BLOCK, seed=pl.RNG_SEED) -> dict:
    """Sharpe(a) - Sharpe(b), with a 95% interval from a paired bootstrap (both resampled on the same days)."""
    a, b = a.align(b, join="inner")
    idx = stationary_bootstrap_indices(len(a), n, block, seed)
    diffs = _row_sharpes(a.to_numpy()[idx]) - _row_sharpes(b.to_numpy()[idx])
    return dict(sharpe_diff=sharpe(a) - sharpe(b),
                ci_low=np.percentile(diffs, 2.5), ci_high=np.percentile(diffs, 97.5),
                share_not_better=(diffs <= 0).mean())


def shuffle_runs(regimes: pd.Series, rng) -> pd.Series:
    """
    The same regime runs with the same lengths, in random order: keeps how
    often and how long each regime lasts, destroys *when* it happens.
    """
    vals = regimes.to_numpy(dtype=object)
    starts = np.flatnonzero(np.r_[True, vals[1:] != vals[:-1]])
    runs = np.split(vals, starts[1:])
    return pd.Series(np.concatenate([runs[i] for i in rng.permutation(len(runs))]), index=regimes.index)


def random_regime_null(prices, regimes, n=N_NULL, seed=pl.RNG_SEED) -> np.ndarray:
    """Net Sharpe of the strategy run on `n` shuffled regime series."""
    rng = np.random.default_rng(seed)
    return np.array([sharpe(pl.backtest_dynamic(prices, shuffle_runs(regimes, rng)).net) for _ in range(n)])


def ablation(prices, regimes) -> tuple:
    """
    Each allocation method with and without the regime signal, plus the benchmarks,
    over the same dates. "Without" = same method, one setting, rebalanced every 21 days.
    Returns (table, {name: daily net returns}).
    """
    monthly = dict(min_hold=21, max_hold=21)
    start, end = regimes.index[0], regimes.index[-1]
    runs = {
        "Regime mean-variance (the strategy)": pl.backtest_dynamic(prices, regimes),
        **{f"Mean-variance, fixed gamma={g:g}, monthly": pl.backtest_dynamic(
               prices, regimes, allocator=pl.mean_variance(gammas={r: g for r in pl.REGIMES}), **monthly)
           for g in pl.REGIME_GAMMA.values()},
        "Regime risk parity (same-regime covariance)": pl.backtest_dynamic(
            prices, regimes, allocator=pl.risk_parity(same_regime=True)),
        "Risk parity, trailing covariance, monthly": pl.backtest_dynamic(
            prices, regimes, allocator=pl.risk_parity(), **monthly),
        "Equal-Weight (1/3 each)": pl.backtest_static(prices, {a: 1 / 3 for a in pl.ASSETS}, start, end),
        "Static 60/40 (stocks/bonds)": pl.backtest_static(prices, {"stocks": 0.6, "bonds": 0.4}, start, end),
    }
    idx = runs["Regime mean-variance (the strategy)"].net.index
    for bt in runs.values():
        idx = idx.intersection(bt.net.index)
    returns = {name: bt.net.loc[idx] for name, bt in runs.items()}

    rows = {}
    for name, bt in runs.items():
        stats = pl.performance_stats(bt.net.loc[idx], bt.turnover.loc[idx])
        vs_ew = sharpe_difference(returns[name], returns["Equal-Weight (1/3 each)"])
        rows[name] = dict(cagr=stats["cagr"], sharpe=stats["sharpe"], max_drawdown=stats["max_drawdown"],
                          turnover_annualized=stats["turnover_annualized"],
                          sharpe_vs_ew=vs_ew["sharpe_diff"], vs_ew_ci_low=vs_ew["ci_low"],
                          vs_ew_ci_high=vs_ew["ci_high"])
    return pd.DataFrame(rows).T, returns


def regime_value(returns: dict, null_sharpes: np.ndarray) -> pd.DataFrame:
    """Does the regime signal add value over the same method without it, and over random timing?"""
    fixed = [k for k in returns if k.startswith("Mean-variance, fixed")]
    best_fixed = max(fixed, key=lambda k: sharpe(returns[k]))   # best in hindsight: a conservative comparison
    strategy = returns["Regime mean-variance (the strategy)"]
    pairs = {
        f"Regime mean-variance vs best {best_fixed[len('Mean-variance, '):]}": (strategy, returns[best_fixed]),
        "Regime risk parity vs trailing risk parity": (returns["Regime risk parity (same-regime covariance)"],
                                                      returns["Risk parity, trailing covariance, monthly"]),
        "Regime mean-variance vs equal weight": (strategy, returns["Equal-Weight (1/3 each)"]),
        "Regime mean-variance vs static 60/40": (strategy, returns["Static 60/40 (stocks/bonds)"]),
    }
    table = pd.DataFrame({name: sharpe_difference(a, b) for name, (a, b) in pairs.items()}).T
    actual = sharpe(strategy)
    table.loc["Regime mean-variance vs random regime timing"] = dict(
        sharpe_diff=actual - null_sharpes.mean(),
        ci_low=actual - np.percentile(null_sharpes, 97.5), ci_high=actual - np.percentile(null_sharpes, 2.5),
        share_not_better=(null_sharpes >= actual).mean())
    return table


SENSITIVITY = {
    "HMM training window (days)": ("train_size", [252, 504, 756], 504),
    "Smoothing window (days)": ("smooth", [1, 10, 21, 42], 21),
    "Min hold (days)": ("min_hold", [5, 10, 21, 42], 21),
    "Estimation lookback (days)": ("lookback", [42, 63, 126, 252], 63),
    "Risk aversion (x default)": ("gamma_scale", [0.5, 1, 2], 1),
    "Cost (bps)": ("cost_bps", [0, 7, 15, 30], 7),
}


def sensitivity(prices, features, base_labels=None, grid=SENSITIVITY) -> pd.DataFrame:
    """
    One design choice at a time, everything else at its default; equal weight
    over the same dates for reference. `base_labels`: the raw walk-forward
    labels for the default 504-day window, if already computed.
    """
    wf_cache = {} if base_labels is None else {504: base_labels}
    rows = []
    for label, (key, values, default) in grid.items():
        for v in values:
            p = dict(train_size=504, smooth=21, min_hold=21, lookback=63, gamma_scale=1, cost_bps=7)
            p[key] = v
            if p["train_size"] not in wf_cache:
                wf_cache[p["train_size"]] = pl.walk_forward_regimes(features, train_size=p["train_size"])[0]
            regimes = pl.smooth_regimes(wf_cache[p["train_size"]], window=p["smooth"])
            gammas = {r: g * p["gamma_scale"] for r, g in pl.REGIME_GAMMA.items()}
            bt = pl.backtest_dynamic(prices, regimes, lookback=p["lookback"], min_hold=p["min_hold"],
                                     cost_bps=p["cost_bps"],
                                     allocator=pl.mean_variance(p["lookback"], gammas))
            ew = pl.backtest_static(prices, {a: 1 / 3 for a in pl.ASSETS},
                                    regimes.index[0], regimes.index[-1], cost_bps=p["cost_bps"])
            stats = pl.performance_stats(bt.net, bt.turnover)
            rows.append(dict(parameter=label, value=v, is_default=v == default, sharpe=stats["sharpe"],
                             cagr=stats["cagr"], max_drawdown=stats["max_drawdown"],
                             ew_sharpe=sharpe(ew.net.loc[bt.net.index.intersection(ew.net.index)])))
    return pd.DataFrame(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plots

    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    pd.set_option("display.width", 160)
    OUT.mkdir(exist_ok=True)

    prices, _ = pl.get_data()
    features = pl.build_features(prices)
    labels = pl.walk_forward_regimes(features)[0]
    regimes = pl.smooth_regimes(labels, window=21)

    print("\n[1] Ablation: each method with and without regimes")
    table, returns = ablation(prices, regimes)
    print(table.round(3))
    table.round(4).to_csv(OUT / "robustness_ablation.csv")

    print(f"\n[2+3] Random-regime null ({N_NULL} shuffles) and bootstrap intervals ({N_BOOT} resamples)")
    null = random_regime_null(prices, regimes)
    value = regime_value(returns, null)
    print(value.round(3))
    value.round(4).to_csv(OUT / "robustness_regime_value.csv")
    pd.Series(null, name="sharpe").round(4).to_csv(OUT / "robustness_null_sharpes.csv", index_label="shuffle")

    fig = plots.null_distribution(null, sharpe(returns["Regime mean-variance (the strategy)"]),
                                  sharpe(returns["Equal-Weight (1/3 each)"]))
    fig.savefig(OUT / "06_random_regime_null.png", dpi=130)
    plt.close(fig)

    print("\n[4] Sensitivity: one design choice at a time")
    sens = sensitivity(prices, features, base_labels=labels)
    print(sens.round(3).to_string(index=False))
    sens.round(4).to_csv(OUT / "robustness_sensitivity.csv", index=False)
    fig = plots.sensitivity_grid(sens)
    fig.savefig(OUT / "07_sensitivity.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved to {OUT.name}/.")


if __name__ == "__main__":
    main()
