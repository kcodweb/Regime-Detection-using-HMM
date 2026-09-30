# Regime-Shift Detection using Hidden Markov Models
[![tests](https://github.com/kcodweb/Regime-Detection-using-HMM/actions/workflows/tests.yml/badge.svg)](https://github.com/kcodweb/Regime-Detection-using-HMM/actions/workflows/tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kcodweb/Regime-Detection-using-HMM/blob/main/Regime_Shift_Notebook.ipynb)

A regime-aware portfolio engine that detects whether the Indian market is in a **Bull**,
**Bear**, or **Crisis** state using a Hidden Markov Model, then reallocates between stocks,
gold, and bonds using convex optimization (`cvxpy`) — validated with a strict walk-forward
harness so the backtest can't cheat by peeking into the future (and a test suite that checks
it doesn't), then stress-tested with ablations, a random-timing null, bootstrap intervals and
a sensitivity analysis to see whether the result is signal or luck.

## Results

Out-of-sample, 23 May 2014 – 30 Dec 2025, in INR, net of 7 bps costs:

| Strategy | CAGR | Vol | Sharpe | Sortino | Max DD | Calmar | Turnover / yr |
|---|---|---|---|---|---|---|---|
| **Dynamic (regime-aware)** | **13.0%** | 10.7% | 1.20 | 1.74 | −15.0% | 0.87 | 3.30 |
| Dynamic, before costs | 13.2% | 10.7% | 1.22 | 1.77 | −14.9% | 0.89 | — |
| Static 60/40 (stocks/bonds) | 10.4% | 9.7% | 1.07 | 1.52 | −17.5% | 0.59 | 0.26 |
| Equal weight (1/3 each) | 12.2% | 8.7% | **1.37** | **2.04** | **−12.5%** | **0.98** | 0.29 |
| Nifty 50 buy & hold | 12.3% | 16.3% | 0.79 | 1.09 | −38.4% | 0.32 | — |

![Equity curves](outputs/04_equity_curves.png)

- **The regime strategy beats static 60/40 and Nifty buy & hold** on every risk-adjusted
  measure, and has the highest CAGR of the group.
- **It does not beat equal weight on risk.** Equal weight has the better Sharpe, Sortino,
  Calmar and max drawdown. The CAGR lead over equal weight is a single episode: at the end of
  2024 the strategy was ~3% *behind*; 2025 — ~55% average gold weight while gold rose ~74% in
  INR — put it ~8% ahead.
- **Costs are small**: ~5 rebalances a year cost ~0.3% a year.
- **None of these differences is statistically significant**, and the Sharpe moves a lot with
  some design choices — see [Is the edge real?](#is-the-edge-real) below.
- Sharpe/Sortino use a 0% risk-free rate, so compare them across rows, not with published figures.

### Do the regimes mean anything?
Realized Nifty volatility and return over the **next** month, grouped by the out-of-sample
regime assigned today (`outputs/regime_diagnostics.csv`):

| Regime today | Share of days | Next-month vol | Next-month return |
|---|---|---|---|
| Bull | 42% | 11.0% | +0.8% |
| Bear | 39% | 15.5% | +0.9% |
| Crisis | 19% | 19.2% | +1.5% |

The labels forecast **volatility** out-of-sample; they do not forecast **returns** — the
month after a Crisis label has had the highest average return (rebounds after sell-offs).
That matches how the optimizer uses them (regime sets risk aversion, not expected return), and
it is also the cost of the approach: de-risking in Crisis gives up some of the rebound.

![Walk-forward regimes](outputs/03_regimes_overlay_walkforward.png)

## Is the edge real?

One backtest is one path through history. `robustness.py` runs four checks
(`outputs/robustness_*.csv`):

**1. Ablation: each method with and without the regime signal.** "Without" = one fixed
setting, rebalanced every 21 days. Risk parity gives each asset the same risk contribution and
needs no expected-return estimate; its regime version estimates the covariance from past days
that carried today's regime label, i.e. it uses the regime as the risk forecast the
diagnostics above say it is.

| Strategy | CAGR | Sharpe | Max DD | Turnover / yr | Sharpe vs equal weight (95% CI) |
|---|---|---|---|---|---|
| **Regime mean-variance** (the strategy) | 13.0% | 1.20 | −15.0% | 3.30 | −0.18 (−0.48, +0.14) |
| Mean-variance, fixed γ = 2 | 11.9% | 1.02 | −20.3% | 6.35 | −0.36 (−0.72, +0.02) |
| Mean-variance, fixed γ = 8 | 10.7% | 0.96 | −15.2% | 5.93 | −0.42 (−0.74, −0.08) |
| Mean-variance, fixed γ = 30 | 9.0% | 0.97 | −15.6% | 4.35 | −0.40 (−0.68, −0.09) |
| **Regime risk parity** | 10.7% | 1.32 | −12.6% | 0.75 | −0.05 (−0.18, +0.09) |
| Risk parity, trailing covariance | 10.1% | 1.29 | −12.4% | 0.93 | −0.09 (−0.22, +0.05) |
| Equal weight | 12.2% | 1.37 | −12.5% | 0.29 | — |
| Static 60/40 | 10.4% | 1.07 | −17.5% | 0.26 | −0.30 (−0.71, +0.15) |

**2–3. Does the regime signal add value?** Sharpe differences with 95% intervals from a
block bootstrap (2,000 resamples of ~1-month blocks), and against 500 random regime timings
(same regime runs and lengths, shuffled order):

| Comparison | ΔSharpe | 95% interval | Share of resamples / shuffles where it's not better |
|---|---|---|---|
| Regime mean-variance vs best fixed γ (picked in hindsight) | +0.18 | −0.16, +0.53 | 15% |
| Regime risk parity vs trailing risk parity | +0.04 | −0.07, +0.16 | 26% |
| Regime mean-variance vs random regime timing | +0.11 | −0.18, +0.40 | 23% |
| Regime mean-variance vs equal weight | −0.18 | −0.48, +0.14 | 86% |
| Regime mean-variance vs static 60/40 | +0.13 | −0.32, +0.55 | 31% |

![Random regime timing](outputs/06_random_regime_null.png)

**4. Sensitivity: one design choice at a time.**

![Sensitivity](outputs/07_sensitivity.png)

**What this says:**
- **The regime signal helps, but not provably.** It lifts mean-variance by ~0.18 Sharpe over
  the best fixed setting while halving turnover, and lifts risk parity a little; neither gain
  is significant at 95%.
- **The HMM's timing is only weakly better than random**: 23% of shuffled timings do as well.
  A good part of the result comes from *how much* time is spent in each regime, not *when*.
- **No significant difference from the benchmarks**: both of the strategy's intervals, against
  equal weight and 60/40, include zero. (The fixed-γ versions are significantly *worse* than
  equal weight.)
- **The headline is sensitive to design choices.** Across the 22 variations, Sharpe ranges from
  0.82 to 1.31. It moves most with the HMM training window (1.31 at one year, 0.83 at three)
  and the smoothing window, and the default 21-day smoothing happens to be the best of the four
  tried, so the headline is more likely optimistic than pessimistic. Only 1 of 22 variations
  beats equal weight, by 0.01. Risk aversion and costs barely matter.
- **Using the regime as a risk forecast works best among the regime strategies**: regime risk
  parity has the highest Sharpe and smallest drawdown at a quarter of the turnover, though at a
  lower CAGR, and it still doesn't beat equal weight. It was one of two fixes for the noisy
  63-day mean decided on before looking (the other, a 12-month estimation window, changed
  little: see the sensitivity panel), and the headline strategy was deliberately left as
  designed rather than swapped for whichever variant looked best afterwards.

## Data

Pulled via `yfinance`, snapshotted in `data/raw_prices.csv`:

| Leg | Ticker | Notes |
|---|---|---|
| Stocks | `^NSEI` | Nifty 50 (NSE), INR |
| Gold | `GC=F` | Gold futures, USD → converted to INR |
| Bonds | `IEF` | iShares 7-10Y Treasury Bond ETF, USD → converted to INR — a real **price** series |
| Vol proxy | `^INDIAVIX` | falls back to `^VIX` only if unavailable |
| FX | `INR=X` | INR per USD, used for the conversion |

**Why `IEF` and not a raw yield like `^TNX`?** A yield is not a tradable price. Using a yield
series directly as if it were a "bond price" gets the sign backwards — yields *rise* when
bond prices *fall*, so a naive substitution would make the strategy buy bonds exactly when
it should be selling them. An ETF price series (`IEF`) sidesteps that conversion issue
entirely and is treated identically to the stocks/gold legs everywhere in the pipeline.

**Why convert to INR?** The signal is Indian (Nifty, India VIX) and so is the investor it
describes. Mixing INR and USD returns in one portfolio silently ignores the currency: over
2012–2025 the rupee went from ~53 to ~90 per dollar, so for an Indian investor the dollar legs
earned that on top of their USD return — and the rupee tends to weaken exactly in the
sell-offs where the model moves into gold and bonds.

**FX data repair.** Yahoo's `INR=X` has a few one-day bad prints that fully reverse the next
day (up to 6% in Jan 2012). `clean_fx()` replaces a print more than 2% away from its centred
5-day median (4 prints in total); genuine moves persist, so they are kept. Only the FX series
is cleaned — big one-day moves in the traded assets (March 2020) are real.

## Files

| File | Purpose |
|---|---|
| `Regime_Shift_Notebook.ipynb` | **Main deliverable.** Walks through every phase top to bottom — data → features → regime detection → optimization → backtest → results → robustness — with the figures and commentary. |
| `pipeline.py` | All of the logic, one section per phase. The notebook and both scripts import it. |
| `robustness.py` | Ablation, random-regime null, bootstrap intervals and sensitivity analysis. |
| `plots.py` | The figures, shared by the notebook and the scripts. |
| `run.py` | Command-line runner for the main pipeline: saves figures 01–05 and the CSVs to `outputs/`. |
| `tests/` | Checks no-lookahead, portfolio accounting, smoothing, FX repair, the optimizers and the robustness statistics. Run by CI on every push. |
| `data/raw_prices.csv` | Raw price snapshot, so results reproduce exactly. |
| `outputs/` | Regime overlays, equity curves, weights, transition matrices, regime diagnostics, performance summary, robustness results. |

## How to run it

The quickest way is the **Open in Colab** badge at the top: the notebook's first cell fetches
the repo and installs what's missing. Locally:

```bash
pip install -r requirements.txt
python run.py              # main pipeline, ~1 min; uses data/raw_prices.csv
python run.py --refresh    # re-downloads prices from yfinance first
python robustness.py       # robustness checks, ~3 min
pytest                     # ~20 s
```

For the notebook: `pip install jupyter`, then `jupyter notebook Regime_Shift_Notebook.ipynb` and
Kernel → Restart & Run All from the repo root (~4 min, most of it the robustness section).

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
economically matches "Crisis = highest turbulence." The diagnostics table above confirms the
ranking carries over out-of-sample. Labels are relative to each 2-year training window, so in
a calm stretch "Crisis" can be fairly mild in absolute terms.

**Why fit the HMM from 5 seeds?** EM only finds a local optimum. With a single seed, several
walk-forward folds landed in poor optima whose states flip every few days: ~760 regime changes
over 11 years. Keeping the best log-likelihood of 5 fits (on the training window only) cuts
that to 128.

**Why filtered decoding instead of `model.predict()`?** `predict()` runs Viterbi over the whole
63-day test window, so the label it gives day *t* depends on days *t+1 … t+62* — lookahead in
every test window even though the model never trained on them (`predict_proba()` is
forward-backward and has the same problem). `filter_states()` runs the forward algorithm one
day at a time, giving P(state on day *t* | data up to day *t*). The tests check this two ways:
tampering with future data must not move any earlier label, and each day's output must be the
same whether the decoder sees the rest of the window or not (the old Viterbi decoding fails
this on real data).

**Why walk-forward with a rolling (not expanding) window?** An expanding window means the
early folds have very little data (unstable HMM fits) while later folds are dominated by a
huge training set that drowns out recent regime shifts. A rolling 504-day (~2y) train / 63-day
(~3mo) test window keeps the HMM responsive to structural change while still giving it enough
data to fit 3 Gaussian states reliably.

**Why smooth the regime series, and why a hard `min_hold` floor?** Even well-fitted, the daily
labels chatter around regime boundaries. `smooth_regimes()` applies a ~1-month rolling majority
vote (each point only uses its own day and the past, and a tied vote keeps the current regime),
cutting 128 changes to 55. `backtest_dynamic` also enforces a hard `min_hold=21`-day floor
between rebalances, so turnover is bounded even if the smoothed series still flips near a
boundary, and a `max_hold=126`-day safety refresh keeps weights from going stale if a regime
persists for an unusually long time.

**Why per-regime risk-aversion (`gamma`) instead of a totally different objective per
regime?** A single mean-variance formulation (`maximize mu'w - gamma * w'Σw`) that just
dials `gamma` up in Crisis and down in Bull is mathematically simpler, stays convex, and
avoids having to justify three unrelated objective functions — it naturally produces
"maximize Sharpe-like behavior in Bull" and "minimize-variance-like behavior in Crisis"
as the two extremes of the same convex program.

**Why rebalance on regime change instead of a fixed monthly clock?** Rebalancing every 21
trading days regardless of whether anything changed is pure cost drag with no informational
justification. The static 60/40 and equal-weight benchmarks are still rebalanced every 21
days, since they have no signal to react to — that's the natural default schedule for a
fixed-weight portfolio, and it's held constant so the benchmarks' behavior isn't being tuned
at the same time as the strategy's.

**How the backtest does its accounting.** Every strategy goes through the same `simulate()`:
portfolio returns are weighted sums of *simple* asset returns (a weighted sum of log returns is
not the portfolio's log return); weights decided at a day's close earn from the next day;
between rebalances the weights drift with prices; and each rebalance pays 7 bps on turnover =
`sum(|target − drifted weights|)`. So the benchmarks pay for their monthly resets too.

**Why 7 bps transaction cost?** 7 bps sits in the middle of the 5–10 bps range given in the
brief. It's applied only when a rebalance actually fires.

## Limitations and next steps
- **No statistically significant edge.** See [Is the edge real?](#is-the-edge-real): the
  regime signal helps, but none of the differences clears a 95% bar, and the Sharpe moves a lot
  with the training and smoothing windows. Treat the results as a well-tested prototype, not a
  proven strategy.
- **The optimizer is the weak link.** `mu` is a trailing 63-day mean, which is mostly noise, so
  mean-variance ends up chasing recent winners and hitting the 70% cap (see
  `outputs/05_dynamic_weights.png`). Using the regimes for risk instead (regime risk parity)
  gives a better Sharpe and drawdown at a lower return. The most promising next step is a
  **cash leg**: with one, the regime's volatility forecast could scale total risk up and down
  (volatility targeting), which a fully invested three-asset portfolio can't do.
- **One path, one period.** 11 years is only a handful of real crises, and one of them (2025
  gold) decides the CAGR ranking. More markets or a longer history would say more than any
  amount of resampling this one.
- **Not tuned — on purpose.** `gamma`, windows and holding periods are the original design
  choices; none were fitted to the backtest. The sensitivity analysis shows what tuning would
  have looked like — and why a tuned headline would not be trustworthy.

## Reproducing results
`run.py` and the notebook read `data/raw_prices.csv`, and every random step is seeded
(`RNG_SEED = 42`), so they reproduce the numbers above exactly. `python run.py --refresh`
re-downloads the same 2012–2025 range; results can shift slightly because Yahoo occasionally
revises old prints. To extend the sample, change `end` in `run.py` and refresh.
