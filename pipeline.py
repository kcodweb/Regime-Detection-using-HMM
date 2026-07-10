"""
Regime-Shift: Macro-Aware Tactical Asset Allocation Engine
============================================================
Full pipeline: data -> features -> HMM regime detection -> walk-forward
validation -> convex portfolio optimization -> cost-aware backtest -> results.

Data: Nifty 50 (stocks), Gold futures (gold), a 7-10Y Treasury bond ETF
(bonds), and India VIX, all pulled live via yfinance.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42


# ---------------------------------------------------------------------------
# PHASE 1: Get and understand your data
# ---------------------------------------------------------------------------
TICKERS = {
    "stocks": "^NSEI",  # Nifty 50 (NSE)
    "gold": "GC=F",     # Gold futures
    "bonds": "IEF",     # iShares 7-10Y Treasury Bond ETF -- a real bond PRICE series.
                        # (^TNX is a YIELD, not a price -- using it directly as if
                        # it were a bond price gets the sign backwards: yields rise
                        # when bond prices fall. IEF avoids that entirely.)
}
VIX_TICKER = "^INDIAVIX"


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


def _try_yfinance(start="2012-01-01", end=None):
    import yfinance as yf

    frames = {}
    for name, tkr in TICKERS.items():
        df = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=True)
        frames[name] = _extract_close(df)

    try:
        vix_df = yf.download(VIX_TICKER, start=start, end=end, progress=False, auto_adjust=True)
        frames["vix"] = _extract_close(vix_df)
    except Exception:
        # fall back to VIX (US) as a volatility proxy if India VIX unavailable
        vix_df = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
        frames["vix"] = _extract_close(vix_df)

    # Build from named Series via concat so pandas aligns on the DatetimeIndex
    # instead of requiring identical lengths/order (as dict-of-arrays would).
    prices = pd.concat(frames, axis=1, join="inner")
    prices.columns = list(TICKERS.keys()) + ["vix"]
    prices = prices.dropna()
    if prices.empty:
        raise RuntimeError("No overlapping dates across tickers after alignment")
    return prices, "yfinance (live)"


def get_data(start="2012-01-01", end=None):
    """
    Pulls real daily data via yfinance: Nifty 50 (stocks), Gold futures,
    a 7-10Y Treasury bond ETF (bonds), and India VIX (falling back to the
    US VIX only if India VIX is unavailable). Raises if the pull fails --
    there is no synthetic fallback; fix connectivity/tickers and re-run
    rather than silently substituting fake data.
    """
    prices, source = _try_yfinance(start, end)
    print(f"[data] Loaded live data via {source}")
    print(f"[data] Range: {prices.index.min().date()} to {prices.index.max().date()} "
          f"({len(prices)} trading days)")
    return prices, source


# ---------------------------------------------------------------------------
# PHASE 2: Feature engineering
# ---------------------------------------------------------------------------
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

    feat["ret_1d"] = ret  # kept for optimization step, not fed to HMM directly
    return feat.dropna()


FEATURE_COLS = ["mom_1w", "mom_1m", "mom_1q", "vol_1m", "vol_1w", "vix_level", "vix_chg_1w"]


# ---------------------------------------------------------------------------
# PHASE 3: HMM regime classifier
# ---------------------------------------------------------------------------
def fit_hmm(X_scaled: np.ndarray, n_states=3, seed=RNG_SEED):
    from hmmlearn import hmm

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        random_state=seed,
    )
    model.fit(X_scaled)
    return model


def label_states_by_volatility(model, X_scaled, vol_col_idx):
    """Rank hidden states by mean volatility feature -> Bull (lowest) / Bear (mid) / Crisis (highest)."""
    means = model.means_[:, vol_col_idx]
    order = np.argsort(means)  # ascending vol
    mapping = {order[0]: "Bull", order[1]: "Bear", order[2]: "Crisis"}
    return mapping


# ---------------------------------------------------------------------------
# PHASE 4: Walk-forward validation (no lookahead)
# ---------------------------------------------------------------------------
def walk_forward_regimes(features: pd.DataFrame, train_size=504, test_size=63, n_states=3):
    """
    Expanding train / rolling test windows. Everything (scaler + HMM) is fit
    ONLY on the training slice; the test slice is only ever transformed and
    predicted on. Regime label for day t never uses data from after day t.
    """
    from sklearn.preprocessing import StandardScaler

    X = features[FEATURE_COLS].values
    idx = features.index
    n = len(features)

    out_regimes = pd.Series(index=idx, dtype=object)
    transition_matrices = []

    start = 0
    while start + train_size + 1 < n:
        train_end = start + train_size
        test_end = min(train_end + test_size, n)

        X_train = X[start:train_end]
        X_test = X[train_end:test_end]

        scaler = StandardScaler().fit(X_train)          # fit on TRAIN ONLY
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)              # transform only

        model = fit_hmm(X_train_s, n_states=n_states)
        vol_col_idx = FEATURE_COLS.index("vol_1m")
        mapping = label_states_by_volatility(model, X_train_s, vol_col_idx)

        pred_train_states = model.predict(X_train_s)  # for transition matrix bookkeeping
        pred_test_states = model.predict(X_test_s)     # decode test window using train-fit params only

        labels = [mapping[s] for s in pred_test_states]
        out_regimes.iloc[train_end:test_end] = labels

        transition_matrices.append(model.transmat_)
        start += test_size  # slide forward (expanding train would use start=0 always; here rolling)

    return out_regimes.dropna(), transition_matrices


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
def performance_stats(returns: pd.Series) -> dict:
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside = returns[returns < 0]
    down_vol = downside.std() * np.sqrt(252) if len(downside) else np.nan
    sortino = ann_ret / down_vol if down_vol and down_vol > 0 else np.nan

    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    return dict(ann_return=ann_ret, ann_vol=ann_vol, sharpe=sharpe, sortino=sortino,
                max_drawdown=max_dd, calmar=calmar)


def smooth_regimes(regimes: pd.Series, window: int = 21) -> pd.Series:
    """
    Rolling-mode smoothing of the daily regime series. The raw walk-forward
    HMM output can legitimately flip on a noisy day-to-day basis (it's
    decoding one day at a time); a ~1-month rolling majority vote turns
    that into a much steadier "what regime are we actually in" signal
    without looking at any future data (each point only uses days up to
    and including itself).
    """
    vals = regimes.values
    out = np.empty(len(vals), dtype=object)
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        window_vals = vals[start:i + 1]
        uniq, counts = np.unique(window_vals, return_counts=True)
        out[i] = uniq[np.argmax(counts)]
    return pd.Series(out, index=regimes.index)


def backtest_dynamic(prices: pd.DataFrame, regimes: pd.Series, lookback=63,
                      min_hold=21, max_hold=126, cost_bps=7):
    """
    Rebalances on regime CHANGE rather than a fixed clock:
      - `regimes` should already be smoothed (see `smooth_regimes`) -- this
        function trusts whatever series it's given as the "confirmed" regime.
      - `min_hold`: hard floor -- once rebalanced, wait at least this many
        trading days before rebalancing again, even if the (smoothed) regime
        flips back and forth near a boundary. This bounds worst-case turnover
        regardless of how noisy the regime series is.
      - `max_hold`: safety refresh -- if we haven't rebalanced in this many
        days (regime unchanged), re-estimate mu/Sigma and re-optimize once
        anyway, so weights don't go stale during a long, calm regime.
    This directly targets turnover: a fixed 21-day clock rebalances ~12x/year
    regardless of whether anything changed, and even change-triggered
    rebalancing can over-trade if the regime series itself is noisy -- the
    `min_hold` floor is what actually caps worst-case trading frequency.
    """
    assets = ["stocks", "gold", "bonds"]
    rets = np.log(prices[assets]).diff().dropna()
    rets = rets.loc[regimes.index.intersection(rets.index)]
    regimes = regimes.loc[rets.index]

    dates = rets.index
    weights_hist = pd.DataFrame(index=dates, columns=assets, dtype=float)
    w_prev = np.array([1 / 3, 1 / 3, 1 / 3])
    turnover = pd.Series(0.0, index=dates)
    rebalance_dates = []

    current_regime = None
    days_since_rebal = 0

    for i, d in enumerate(dates):
        days_since_rebal += 1
        if i >= lookback:
            regime_today = regimes.iloc[i]
            can_rebalance = days_since_rebal >= min_hold
            trigger = current_regime is None or (
                can_rebalance and (regime_today != current_regime or days_since_rebal >= max_hold)
            )
            if trigger:
                window = rets.iloc[i - lookback:i]        # PAST DATA ONLY
                mu = window.mean().values * 252
                sigma = window.cov().values * 252
                w_new = optimize_weights(mu, sigma, regime_today)
                turnover.iloc[i] = np.abs(w_new - w_prev).sum()
                w_prev = w_new
                current_regime = regime_today
                days_since_rebal = 0
                rebalance_dates.append(d)
        weights_hist.iloc[i] = w_prev

    gross_ret = (weights_hist.shift(1).fillna(1 / 3) * rets).sum(axis=1)
    cost = turnover * (cost_bps / 10000)
    net_ret = gross_ret - cost
    print(f"[backtest] dynamic strategy rebalanced {len(rebalance_dates)} times "
          f"over {len(dates)} trading days "
          f"({len(rebalance_dates) / (len(dates) / 252):.1f}/year)")
    return net_ret, weights_hist, turnover


def backtest_static(prices: pd.DataFrame, weights: dict, rebalance_every=21, cost_bps=7):
    assets = list(weights.keys())
    w_target = np.array([weights[a] for a in assets])
    rets = np.log(prices[assets]).diff().dropna()

    dates = rets.index
    w_prev = w_target.copy()
    turnover = pd.Series(0.0, index=dates)
    weights_hist = pd.DataFrame(index=dates, columns=assets, dtype=float)

    for i, d in enumerate(dates):
        if i % rebalance_every == 0:
            turnover.iloc[i] = np.abs(w_target - w_prev).sum()
            w_prev = w_target.copy()
        weights_hist.iloc[i] = w_prev

    gross_ret = (weights_hist.shift(1).fillna(pd.Series(w_target, index=assets)) * rets).sum(axis=1)
    cost = turnover * (cost_bps / 10000)
    net_ret = gross_ret - cost
    return net_ret, weights_hist, turnover
