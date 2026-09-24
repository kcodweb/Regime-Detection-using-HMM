"""
Regime-Shift: Macro-Aware Tactical Asset Allocation Engine
============================================================
Full pipeline: data -> features -> HMM regime detection -> walk-forward
validation -> convex portfolio optimization -> cost-aware backtest -> results.

Data: Nifty 50 (stocks), Gold futures (gold), a 7-10Y Treasury bond ETF
(bonds) and India VIX, pulled via yfinance. The two USD-quoted legs are
converted to INR so the whole portfolio is measured in one currency.
"""

from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42
ASSETS = ["stocks", "gold", "bonds"]
REGIMES = ["Bull", "Bear", "Crisis"]

DATA_DIR = Path(__file__).resolve().parent / "data"
SNAPSHOT = DATA_DIR / "raw_prices.csv"


# ---------------------------------------------------------------------------
# PHASE 1: Get and understand your data
# ---------------------------------------------------------------------------
TICKERS = {
    "stocks": "^NSEI",  # Nifty 50 (NSE), INR
    "gold": "GC=F",     # Gold futures, USD/oz
    "bonds": "IEF",     # iShares 7-10Y Treasury Bond ETF, USD -- a real bond PRICE series.
                        # (^TNX is a YIELD, not a price -- using it directly as if
                        # it were a bond price gets the sign backwards: yields rise
                        # when bond prices fall. IEF avoids that entirely.)
}
VIX_TICKER = "^INDIAVIX"
FX_TICKER = "INR=X"     # INR per 1 USD
USD_LEGS = ["gold", "bonds"]


def _extract_close(df: pd.DataFrame) -> pd.Series:
    """
    Robustly pull a 1-D Close price Series out of whatever shape yfinance
    hands back. Different yfinance versions return a flat single-level
    DataFrame, a MultiIndex with levels (field, ticker), or a MultiIndex
    with levels (ticker, field) depending on version / single-vs-multi
    ticker requests. This normalizes all of those to a plain Series.
    """
    if df is None or len(df) == 0:
        raise RuntimeError("Empty dataframe returned")

    if isinstance(df.columns, pd.MultiIndex):
        close = None
        for level in range(df.columns.nlevels):
            if "Close" in df.columns.get_level_values(level):
                close = df.xs("Close", axis=1, level=level)
                break
        if close is None:
            close = df.iloc[:, [0]]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

    close = close.dropna()
    if len(close) == 0:
        raise RuntimeError("No usable Close prices after extraction")
    return close


def download_raw(start="2012-01-01", end=None) -> pd.DataFrame:
    """Raw closes for every ticker on the union of their calendars (NaN where a market was shut)."""
    import yfinance as yf

    def close(tkr):
        return _extract_close(yf.download(tkr, start=start, end=end, progress=False, auto_adjust=True))

    frames = {name: close(tkr) for name, tkr in TICKERS.items()}
    try:
        frames["vix"] = close(VIX_TICKER)
    except Exception:
        # fall back to VIX (US) as a volatility proxy if India VIX unavailable
        frames["vix"] = close("^VIX")
    frames["usdinr"] = close(FX_TICKER)
    raw = pd.DataFrame(frames)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "Date"
    return raw


def clean_fx(fx: pd.Series, tol=0.02):
    """
    Yahoo's INR=X series has isolated bad prints: one-day jumps of 2-6% that
    fully reverse the next day (e.g. late Jan 2012, 2 Nov 2023). A print
    more than `tol` away from its centred 5-day median is replaced by that
    median; a genuine move persists, so the median follows it and it is kept.

    Applied to the FX series only. This repairs vendor errors in the P&L
    series; it never touches the HMM's inputs, and it is deliberately NOT
    applied to the traded assets, where big one-day moves (March 2020) are
    real and exactly what the model needs to see.
    """
    med = fx.rolling(5, center=True, min_periods=3).median()
    bad = (fx / med - 1).abs() > tol
    return fx.where(~bad, med), bad


def prepare_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Align the legs on common trading days and convert the USD legs to INR."""
    prices = raw[ASSETS + ["vix"]].dropna()
    fx, _ = clean_fx(raw["usdinr"].dropna())
    # FX trades on days NSE is shut and vice versa: carry the last FX print forward
    fx = fx.reindex(fx.index.union(prices.index)).ffill().reindex(prices.index)
    prices[USD_LEGS] = prices[USD_LEGS].mul(fx, axis=0)
    return prices.dropna()


def get_data(start="2012-01-01", end="2025-12-31", refresh=False):
    """
    Loads the committed raw snapshot (data/raw_prices.csv) so results are
    reproducible; `refresh=True` re-downloads it from yfinance instead.
    Raises if the download fails -- there is no synthetic fallback.
    """
    if refresh or not SNAPSHOT.exists():
        raw = download_raw(start, end)
        DATA_DIR.mkdir(exist_ok=True)
        raw.to_csv(SNAPSHOT)
        source = "yfinance (live)"
    else:
        raw = pd.read_csv(SNAPSHOT, index_col=0, parse_dates=True)
        source = f"snapshot {SNAPSHOT.relative_to(DATA_DIR.parent).as_posix()}"

    prices = prepare_prices(raw.loc[start:end])
    if prices.empty:
        raise RuntimeError("No overlapping dates across tickers after alignment")
    print(f"[data] Loaded {source}")
    print(f"[data] Range: {prices.index.min().date()} to {prices.index.max().date()} "
          f"({len(prices)} trading days); gold & bonds converted to INR")
    return prices, source


# ---------------------------------------------------------------------------
# PHASE 2: Feature engineering
# ---------------------------------------------------------------------------
FEATURE_COLS = ["mom_1w", "mom_1m", "mom_1q", "vol_1m", "vol_1w", "vix_level", "vix_chg_1w"]


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    ret = np.log(prices["stocks"]).diff()

    feat = pd.DataFrame(index=prices.index)
    # momentum over 1w / 1m / 1q (rolling cumulative log-return)
    feat["mom_1w"] = ret.rolling(5).sum()
    feat["mom_1m"] = ret.rolling(21).sum()
    feat["mom_1q"] = ret.rolling(63).sum()
    # volatility (annualized rolling std of returns)
    feat["vol_1m"] = ret.rolling(21).std() * np.sqrt(252)
    feat["vol_1w"] = ret.rolling(5).std() * np.sqrt(252)
    # VIX level and change (market fear proxy)
    feat["vix_level"] = prices["vix"]
    feat["vix_chg_1w"] = prices["vix"].pct_change(5)
    return feat.dropna()


# ---------------------------------------------------------------------------
# PHASE 3: HMM regime classifier
# ---------------------------------------------------------------------------
def fit_hmm(X_scaled: np.ndarray, n_states=3, seed=RNG_SEED, n_restarts=5):
    """EM only finds a local optimum, so fit from several seeds and keep the best log-likelihood."""
    from hmmlearn import hmm

    best, best_ll = None, -np.inf
    for k in range(n_restarts):
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=200,
            random_state=seed + k,
        )
        model.fit(X_scaled)
        ll = model.score(X_scaled)
        if ll > best_ll:
            best, best_ll = model, ll
    return best


def label_states_by_volatility(model, vol_col_idx):
    """Rank hidden states by mean volatility feature -> Bull (lowest) / Bear (mid) / Crisis (highest)."""
    order = np.argsort(model.means_[:, vol_col_idx])  # ascending vol
    return {state: REGIMES[rank] for rank, state in enumerate(order)}


def _emission_loglik(model, X):
    """Log-density of each row of X under each state's diagonal Gaussian, shape (T, n_states)."""
    var = np.array([np.diag(c) for c in model.covars_])
    diff = X[:, None, :] - model.means_[None, :, :]
    return -0.5 * (np.log(2 * np.pi * var).sum(axis=1) + (diff ** 2 / var).sum(axis=2))


def filter_states(model, X_hist, X_new):
    """
    Causal state probabilities P(state_t | x_1..x_t) for each row of X_new,
    via the forward algorithm only.

    model.predict() (Viterbi) and model.predict_proba() (forward-backward)
    both decode a whole sequence at once, so the label they give day t
    depends on days t+1, t+2, ... in the same sequence -- lookahead inside
    every test window. Here each day's probability is updated using that
    day's features and nothing later. X_hist (the training window) only
    sets the starting belief.
    """
    from scipy.special import logsumexp

    # last row of forward-backward == filtered probability on the final training day
    log_alpha = np.log(model.predict_proba(X_hist)[-1] + 1e-300)
    log_trans = np.log(model.transmat_ + 1e-300)
    out = np.empty((len(X_new), model.n_components))
    for t, ll in enumerate(_emission_loglik(model, X_new)):
        log_alpha = logsumexp(log_alpha[:, None] + log_trans, axis=0) + ll
        log_alpha -= logsumexp(log_alpha)
        out[t] = np.exp(log_alpha)
    return out


# ---------------------------------------------------------------------------
# PHASE 4: Walk-forward validation (no lookahead)
# ---------------------------------------------------------------------------
def walk_forward_regimes(features: pd.DataFrame, train_size=504, test_size=63, n_states=3):
    """
    Rolling train / test windows. Everything (scaler + HMM) is fit ONLY on
    the training slice; the test slice is only ever transformed, then
    labelled with filtered (forward-only) probabilities, so the regime for
    day t never uses data from after day t.

    Returns (labels, probs, transition_matrices): the daily regime label, the
    daily Bull/Bear/Crisis probabilities, and each fold's transition matrix
    relabelled to Bull/Bear/Crisis.
    """
    from sklearn.preprocessing import StandardScaler

    X = features[FEATURE_COLS].values
    idx = features.index
    n = len(features)
    vol_col_idx = FEATURE_COLS.index("vol_1m")

    probs = pd.DataFrame(np.nan, index=idx, columns=REGIMES)
    transition_matrices = []

    start = 0
    while start + train_size + 1 < n:
        train_end = start + train_size
        test_end = min(train_end + test_size, n)

        scaler = StandardScaler().fit(X[start:train_end])   # fit on TRAIN ONLY
        X_train_s = scaler.transform(X[start:train_end])
        X_test_s = scaler.transform(X[train_end:test_end])   # transform only

        model = fit_hmm(X_train_s, n_states=n_states)
        mapping = label_states_by_volatility(model, vol_col_idx)
        order = [s for r in REGIMES for s, lab in mapping.items() if lab == r]  # state id per regime

        p = filter_states(model, X_train_s, X_test_s)
        probs.iloc[train_end:test_end] = p[:, order]
        transition_matrices.append(
            pd.DataFrame(model.transmat_[np.ix_(order, order)], index=REGIMES, columns=REGIMES))
        start += test_size  # rolling: the train window slides forward with the test window

    probs = probs.dropna()
    labels = probs.idxmax(axis=1).astype(object)
    return labels, probs, transition_matrices


def smooth_regimes(regimes: pd.Series, window: int = 21) -> pd.Series:
    """
    Rolling-mode smoothing of the daily regime series: a ~1-month majority
    vote over each day and the days before it (no future data). On a tie
    the previous smoothed regime is kept, so a split vote doesn't cause a
    switch -- np.unique alone would break ties alphabetically ("Bear").
    """
    vals = regimes.to_numpy(dtype=object)
    out = np.empty(len(vals), dtype=object)
    for i in range(len(vals)):
        window_vals = vals[max(0, i - window + 1):i + 1]
        uniq, counts = np.unique(window_vals, return_counts=True)
        winners = uniq[counts == counts.max()]
        if len(winners) == 1:
            out[i] = winners[0]
        elif i > 0 and out[i - 1] in winners:
            out[i] = out[i - 1]
        else:
            out[i] = next(v for v in window_vals[::-1] if v in winners)
    return pd.Series(out, index=regimes.index)


def regime_diagnostics(prices: pd.DataFrame, regimes: pd.Series, horizon=21) -> pd.DataFrame:
    """
    Does the label say anything about what comes NEXT? Realized Nifty vol and
    return over the following `horizon` days, grouped by the regime assigned
    today. Uses future data on purpose -- this is evaluation, not a signal.
    """
    r = np.log(prices["stocks"]).diff()
    df = pd.DataFrame({
        "regime": regimes,
        "fwd_vol": r.rolling(horizon).std().shift(-horizon) * np.sqrt(252),
        "fwd_ret": r.rolling(horizon).sum().shift(-horizon),
    }).dropna()
    out = df.groupby("regime").agg(days=("fwd_vol", "size"),
                                   next_1m_vol=("fwd_vol", "mean"),
                                   next_1m_return=("fwd_ret", "mean"))
    out["share_of_days"] = out["days"] / out["days"].sum()
    return out.reindex(REGIMES)[["days", "share_of_days", "next_1m_vol", "next_1m_return"]]


# ---------------------------------------------------------------------------
# PHASE 5: Convex portfolio optimization per regime (cvxpy)
# ---------------------------------------------------------------------------
REGIME_GAMMA = {"Bull": 2.0, "Bear": 8.0, "Crisis": 30.0}  # risk-aversion: higher = more conservative
MAX_WEIGHT = 0.70


def optimize_weights(mu: np.ndarray, sigma: np.ndarray, regime: str) -> np.ndarray:
    import cvxpy as cp

    n = len(mu)
    w = cp.Variable(n)
    gamma = REGIME_GAMMA[regime]

    objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, cp.psd_wrap(sigma)))
    constraints = [cp.sum(w) == 1, w >= 0, w <= MAX_WEIGHT]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if w.value is None:
        return np.array([1 / n] * n)
    return np.clip(w.value, 0, None) / np.sum(np.clip(w.value, 0, None))


# ---------------------------------------------------------------------------
# PHASE 6: Backtest with transaction costs, vs benchmarks
# ---------------------------------------------------------------------------
Backtest = namedtuple("Backtest", "net gross weights turnover")


def asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily SIMPLE returns: a portfolio's return is the weighted sum of these (not of log returns)."""
    return prices[ASSETS].pct_change().dropna()


def simulate(rets: pd.DataFrame, targets: pd.DataFrame, cost_bps=7) -> Backtest:
    """
    Daily portfolio accounting shared by every strategy.

    `targets` has a row only on rebalance days: weights decided at that day's
    close, which earn returns from the next day on. The first row is the
    starting allocation (no cost charged for the initial buy). Between
    rebalances the weights drift with prices, and each rebalance pays
    `cost_bps` on turnover = sum(|target - drifted weights|).
    """
    targets = targets[rets.columns]
    dates = rets.index[rets.index >= targets.index[0]]
    R = rets.loc[dates].values
    is_rebal = dates.isin(targets.index)
    tgt = targets.reindex(dates).values

    w = tgt[0].astype(float)
    gross = np.zeros(len(dates))
    turnover = np.zeros(len(dates))
    weights = np.empty((len(dates), len(rets.columns)))
    weights[0] = w
    for i in range(1, len(dates)):
        gross[i] = w @ R[i]
        w = w * (1 + R[i]) / (1 + gross[i])       # drift with prices
        if is_rebal[i]:
            turnover[i] = np.abs(tgt[i] - w).sum()
            w = tgt[i].astype(float)
        weights[i] = w

    net = gross - turnover * cost_bps / 10000
    s = slice(1, None)  # day 0 is the setup day, before any return is earned
    return Backtest(net=pd.Series(net[s], index=dates[s]),
                    gross=pd.Series(gross[s], index=dates[s]),
                    weights=pd.DataFrame(weights[s], index=dates[s], columns=rets.columns),
                    turnover=pd.Series(turnover[s], index=dates[s]))


def dynamic_targets(rets: pd.DataFrame, regimes: pd.Series, lookback=63,
                    min_hold=21, max_hold=126) -> pd.DataFrame:
    """
    Rebalances on regime CHANGE rather than a fixed clock:
      - `regimes` should already be smoothed (see `smooth_regimes`) -- this
        function trusts whatever series it's given as the "confirmed" regime.
      - `min_hold`: hard floor -- once rebalanced, wait at least this many
        trading days before rebalancing again, even if the (smoothed) regime
        flips back and forth near a boundary.
      - `max_hold`: safety refresh -- if we haven't rebalanced in this many
        days (regime unchanged), re-estimate mu/Sigma and re-optimize once
        anyway, so weights don't go stale during a long, calm regime.
    mu/Sigma come from the trailing `lookback` days up to and including the
    decision day (all known at its close); the history before the first
    out-of-sample day is available, so the strategy is live from day one.
    """
    rows = {}
    current_regime = None
    days_since_rebal = 0
    for d, regime_today in regimes.items():
        days_since_rebal += 1
        trigger = current_regime is None or (
            days_since_rebal >= min_hold
            and (regime_today != current_regime or days_since_rebal >= max_hold)
        )
        if not trigger:
            continue
        window = rets.loc[:d].iloc[-lookback:]          # PAST DATA ONLY
        if len(window) < lookback:
            continue
        mu = window.mean().values * 252
        sigma = window.cov().values * 252
        rows[d] = optimize_weights(mu, sigma, regime_today)
        current_regime = regime_today
        days_since_rebal = 0
    return pd.DataFrame.from_dict(rows, orient="index", columns=rets.columns)


def backtest_dynamic(prices: pd.DataFrame, regimes: pd.Series, lookback=63,
                     min_hold=21, max_hold=126, cost_bps=7) -> Backtest:
    rets = asset_returns(prices)
    targets = dynamic_targets(rets, regimes, lookback, min_hold, max_hold)
    bt = simulate(rets.loc[:regimes.index[-1]], targets, cost_bps)
    years = len(bt.net) / 252
    print(f"[backtest] dynamic strategy rebalanced {len(targets)} times "
          f"over {len(bt.net)} trading days ({len(targets) / years:.1f}/year)")
    return bt


def backtest_static(prices: pd.DataFrame, weights: dict, start=None, end=None,
                    rebalance_every=21, cost_bps=7) -> Backtest:
    """Fixed-weight benchmark: drifts with prices, reset to target every `rebalance_every` days."""
    rets = asset_returns(prices).loc[start:end]
    rebal_dates = rets.index[::rebalance_every]
    targets = pd.DataFrame([weights] * len(rebal_dates), index=rebal_dates).reindex(columns=ASSETS).fillna(0.0)
    return simulate(rets, targets, cost_bps)


def performance_stats(returns: pd.Series, turnover: pd.Series = None) -> dict:
    """Sharpe/Sortino use a 0% risk-free rate: fine for comparing strategies, not as absolute figures."""
    years = len(returns) / 252
    equity = (1 + returns).cumprod()
    cagr = equity.iloc[-1] ** (1 / years) - 1
    ann_mean = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    downside_dev = np.sqrt((np.minimum(returns, 0) ** 2).mean()) * np.sqrt(252)
    max_dd = (equity / equity.cummax() - 1).min()

    return dict(cagr=cagr, ann_vol=ann_vol,
                sharpe=ann_mean / ann_vol if ann_vol > 0 else np.nan,
                sortino=ann_mean / downside_dev if downside_dev > 0 else np.nan,
                max_drawdown=max_dd,
                calmar=cagr / abs(max_dd) if max_dd != 0 else np.nan,
                turnover_annualized=turnover.sum() / years if turnover is not None else np.nan)
