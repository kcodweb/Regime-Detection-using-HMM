"""Figures shared by run.py and the notebook. Each function returns the matplotlib Figure."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REGIME_COLORS = {"Bull": "#2ca02c", "Bear": "#ff7f0e", "Crisis": "#d62728"}


def _shade_regimes(ax, regimes: pd.Series):
    """Shade each contiguous run of a regime across the full height of the axes."""
    starts = regimes.index[(regimes != regimes.shift()).to_numpy()]
    ends = list(starts[1:]) + [regimes.index[-1]]
    seen = set()
    for start, end in zip(starts, ends):
        label = regimes.loc[start]
        ax.axvspan(start, end, color=REGIME_COLORS[label], alpha=0.18, lw=0,
                   label=label if label not in seen else None)
        seen.add(label)


def sanity_check(prices: pd.DataFrame, features: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    axes[0].plot(prices.index, prices["stocks"], color="black", lw=1)
    axes[0].set_title("Nifty 50")
    axes[1].plot(features.index, features["vol_1m"], color="firebrick", lw=1)
    axes[1].set_title("1-month annualized volatility feature (should spike in known stress periods)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def regime_overlay(prices: pd.DataFrame, regimes: pd.Series, title: str, probs: pd.DataFrame = None):
    """Price with regimes shaded; if `probs` is given, a second panel shows the daily regime probabilities."""
    rows = 2 if probs is not None else 1
    fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(12, 5 + 2 * (rows - 1)),
                             gridspec_kw={"height_ratios": [3, 1.3][:rows]}, squeeze=False)
    ax = axes[0, 0]
    ax.plot(prices.loc[regimes.index, "stocks"], color="black", lw=1)
    _shade_regimes(ax, regimes)
    ax.set_title(title)
    handles = dict(zip(*reversed(ax.get_legend_handles_labels())))
    ax.legend([handles[r] for r in REGIME_COLORS if r in handles],
              [r for r in REGIME_COLORS if r in handles], loc="upper left")
    ax.margins(x=0)
    if probs is not None:
        axp = axes[1, 0]
        axp.stackplot(probs.index, probs.T.values, colors=[REGIME_COLORS[c] for c in probs.columns],
                      alpha=0.8, labels=probs.columns)
        axp.set_ylim(0, 1)
        axp.set_ylabel("P(regime)")
        axp.set_title("Filtered regime probabilities (each day uses only data up to that day)", fontsize=10)
    fig.tight_layout()
    return fig


def equity_curves(curves: dict):
    """Growth of 1 and drawdown for each {name: daily net returns}."""
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1.3]})
    for name, rets in curves.items():
        equity = (1 + rets).cumprod()
        axes[0].plot(equity.index, equity, lw=1.3, label=name)
        axes[1].plot(equity.index, equity / equity.cummax() - 1, lw=1)
    axes[0].set_title("Equity curves, net of costs (INR)")
    axes[0].set_ylabel("Growth of 1")
    axes[0].legend(loc="upper left")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.margins(x=0)
    fig.tight_layout()
    return fig


def weights_over_time(weights: pd.DataFrame, regimes: pd.Series = None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(weights.index, weights.T.values, labels=weights.columns, alpha=0.85)
    if regimes is not None:
        r = regimes.reindex(weights.index).ffill()
        colors = r.map(REGIME_COLORS).values
        ax.scatter(weights.index, np.full(len(r), 1.03), c=colors, marker="|", s=40, clip_on=False)
    ax.set_ylim(0, 1)
    ax.set_title("Dynamic strategy: portfolio weights (strip on top = smoothed regime)", pad=16)
    ax.set_ylabel("Weight")
    ax.legend(loc="lower left")
    ax.margins(x=0)
    fig.tight_layout()
    return fig
