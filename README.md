# Regime-Shift Detection using Hidden Markov Models
A regime-aware portfolio engine that detects whether the market is in a **Bull**, **Bear**,
or **Crisis** state using a Hidden Markov Model, then reallocates between stocks, gold, and
bonds using convex optimization (`cvxpy`) — validated with a strict walk-forward harness so
the backtest can't cheat by peeking into the future.

## Data

Pulled live via `yfinance`:

| Leg | Ticker | Notes |
|---|---|---|
| Stocks | `^NSEI` | Nifty 50 (NSE) |
| Gold | `GC=F` | Gold futures |
| Bonds | `IEF` | iShares 7-10Y Treasury Bond ETF — a real **price** series |
| Vol proxy | `^INDIAVIX` | falls back to `^VIX` only if unavailable |

**Why `IEF` and not a raw yield like `^TNX`?** A yield is not a tradable price. Using a yield
series directly as if it were a "bond price" gets the sign backwards — yields *rise* when
bond prices *fall*, so a naive substitution would make the strategy buy bonds exactly when
it should be selling them. An ETF price series (`IEF`) sidesteps that conversion issue
entirely and is treated identically to the stocks/gold legs everywhere in the pipeline
(log-returns, momentum, volatility, mean-variance inputs).

## Files

| File | Purpose |
|---|---|
| `Regime_Shift_Capstone.ipynb` | **Main deliverable.** Full pipeline top to bottom: data → features → regime detection → optimization → backtest → results. |
| `pipeline.py` | Same logic as reusable functions, if you'd rather import it from your own script than use the notebook. |
| `code.py` | Command-line runner that calls `pipeline.py` and saves all figures/CSVs to `figures/`. |
| `outputs/` | PNGs + CSVs from a run (regime overlays, equity curves, transition matrix, performance summary). |

## How to run it

### Option A — Notebook
```bash
pip install numpy pandas matplotlib scikit-learn hmmlearn cvxpy yfinance jupyter
jupyter notebook Regime_Shift_Notebook.ipynb
# Kernel -> Restart & Run All
```

### Option B — Script
```bash
pip install numpy pandas matplotlib scikit-learn hmmlearn cvxpy yfinance
python code.py
```
Outputs land in `figures/`: regime overlays, equity curve, weight-over-time chart, the
transition matrix, and `performance_summary.csv`.

## Key decisions

**Why 3 regimes (not 2 or 4)?** Two states can't separate "steadily falling" from "violently
crashing," which is exactly the distinction that matters for tail-risk management. Four or
more states starts fitting noise given only a handful of features and a few thousand daily
observations — `n_components=3` maps directly onto the project's Bull/Bear/Crisis framing
rather than being a free hyperparameter to tune.

**Why these features?** Momentum (1w/1m/1q rolling log-return) captures *direction*;
volatility (1w/1m rolling std, annualized) captures *turbulence*; VIX level + 1-week change
adds an independent, market-implied fear signal rather than relying purely on realized stats
from the same return series the strategy trades. `covariance_type="diag"` was chosen over
`"full"` because with 7 features and a training window of ~2 years, a full covariance matrix
per state has too many parameters relative to the data and overfits.

**Why label states by volatility rank instead of by mean return?** Mean return is noisier and
regime-dependent in sign only sometimes (a slow bear grind and a violent crash can have
similar mean returns but very different variances); ranking by volatility is more stable and
economically matches "Crisis = highest turbulence."

**Why the `IEF` bond ETF instead of a raw yield?** See the Data section above — a yield series
used as a price gets the direction of bond returns backwards. `IEF` is globally liquid,
reliably served by `yfinance`, and needs no duration/yield-to-price conversion.

**Why per-regime risk-aversion (`gamma`) instead of a totally different objective per
regime?** A single mean-variance formulation (`maximize mu'w - gamma * w'Σw`) that just
dials `gamma` up in Crisis and down in Bull is mathematically simpler, stays convex, and
avoids having to justify three unrelated objective functions — it naturally produces
"maximize Sharpe-like behavior in Bull" and "minimize-variance-like behavior in Crisis"
as the two extremes of the same convex program.

**Why walk-forward with a rolling (not expanding) window?** An expanding window means the
early folds have very little data (unstable HMM fits) while later folds are dominated by a
huge training set that drowns out recent regime shifts. A rolling 504-day (~2y) train / 63-day
(~3mo) test window keeps the HMM responsive to structural change while still giving it enough
data to fit 3 Gaussian states reliably.

**Why smooth the regime series, and why a hard `min_hold` floor?** A first attempt at
change-triggered rebalancing (regime changes → rebalance, with a short debounce) actually made
things *worse* — turnover went up, not down, because the walk-forward HMM decodes one day at
a time and can flip on individual noisy days more often than a short debounce can catch. The
fix has two parts: (1) `smooth_regimes()` applies a ~1-month rolling-majority-vote to the daily
regime series *before* the backtest ever sees it — each point only uses its own day and the
past, so this adds no lookahead, but it turns a noisy day-to-day signal into a steady one; (2)
`backtest_dynamic` also enforces a hard `min_hold=21`-day floor between any two rebalances, so
turnover is bounded even if the smoothed series still chatters near a regime boundary. A
`max_hold=126`-day safety refresh keeps weights from going stale if a regime persists for an
unusually long time. Together this is what actually cuts unnecessary turnover/cost — the
smoothing step matters more than the rebalancing rule itself.

**Why rebalance on regime change instead of a fixed monthly clock?** The original version
rebalanced every 21 trading days regardless of whether anything had actually changed --
pure cost drag with no informational justification, exactly what the brief warns about.
The static 60/40 and equal-weight benchmarks are still rebalanced monthly, since they have
no signal to react to -- that's simply the natural default schedule for a fixed-weight
portfolio, and it's held constant so the benchmarks' behavior isn't being tuned at the same
time as the strategy's.

**Why 7 bps transaction cost?** 7 bps sits in the middle of the
5–10 bps range given in the brief. It's applied only when a rebalance actually fires (see
above) rather than on a fixed clock, which is what keeps the cost drag proportionate to how
much the strategy is actually trading.

## Reproducing results
`get_data()` pulls live prices, so results will shift slightly run to run as new trading days
are added, and Yahoo Finance occasionally revises historical prints. The HMM's `random_state`
is fixed (`RNG_SEED = 42`) so, given the same price history, regime labeling is deterministic.
