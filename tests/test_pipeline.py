import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import pipeline as pl

logging.getLogger("hmmlearn").setLevel(logging.ERROR)


def synthetic_prices(n=800, seed=0):
    """Nifty-like prices that alternate between a calm and a turbulent regime, plus a VIX that tracks it."""
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 120) % 2 == 0, 0.007, 0.02)
    stocks = 100 * np.exp(np.cumsum(rng.normal(0.0003, vol)))
    vix = 12 + 900 * vol + rng.normal(0, 0.5, n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"stocks": stocks, "gold": 100.0, "bonds": 100.0, "vix": vix}, index=idx)


@pytest.fixture(scope="module")
def features():
    return pl.build_features(synthetic_prices())


def test_filter_matches_forward_backward_on_each_prefix(features):
    X = StandardScaler().fit_transform(features[pl.FEATURE_COLS].values)
    X_hist, X_new = X[:300], X[300:340]
    model = pl.fit_hmm(X_hist, n_restarts=1)

    filtered = pl.filter_states(model, X_hist, X_new)
    # the last row of forward-backward on a prefix is the filtered probability at its end
    brute = np.array([model.predict_proba(np.vstack([X_hist, X_new[:k + 1]]))[-1] for k in range(len(X_new))])
    np.testing.assert_allclose(filtered, brute, atol=1e-8)


def test_walk_forward_labels_never_depend_on_later_data(features):
    cut = 300 + 30  # a day in the middle of the first test window
    kwargs = dict(train_size=300, test_size=60)
    labels, probs, _ = pl.walk_forward_regimes(features, **kwargs)

    tampered = features.copy()
    rng = np.random.default_rng(1)
    tampered.iloc[cut:] = tampered.iloc[cut:] * rng.uniform(0.2, 3.0, tampered.iloc[cut:].shape)
    labels_t, probs_t, _ = pl.walk_forward_regimes(tampered, **kwargs)

    before = features.index[cut - 1]
    # exact: smoothed (forward-backward) probabilities differ only in late decimals here
    pd.testing.assert_frame_equal(probs.loc[:before], probs_t.loc[:before], check_exact=True)
    pd.testing.assert_series_equal(labels.loc[:before], labels_t.loc[:before])
    assert not probs.loc[features.index[cut]:].equals(probs_t.loc[features.index[cut]:])


@pytest.mark.skipif(not pl.SNAPSHOT.exists(), reason="needs data/raw_prices.csv")
@pytest.mark.parametrize("fold_start", [0, 1008])
def test_decoding_is_prefix_consistent_on_real_data(fold_start):
    """Day k's output must be identical whether the decoder sees the test window up to k or all of it.
    The original model.predict() (Viterbi over the whole window) fails this on the 2018 fold."""
    prices, _ = pl.get_data()
    X = pl.build_features(prices)[pl.FEATURE_COLS].values
    scaler = StandardScaler().fit(X[fold_start:fold_start + 504])
    X_hist = scaler.transform(X[fold_start:fold_start + 504])
    X_new = scaler.transform(X[fold_start + 504:fold_start + 567])
    model = pl.fit_hmm(X_hist, n_restarts=1)

    full = pl.filter_states(model, X_hist, X_new)
    for k in range(len(X_new)):
        np.testing.assert_array_equal(pl.filter_states(model, X_hist, X_new[:k + 1])[k], full[k])


def test_smoothing_keeps_previous_regime_on_a_tie():
    s = pd.Series(["Crisis", "Crisis", "Bull", "Bull"], index=pd.bdate_range("2020-01-01", periods=4))
    out = pl.smooth_regimes(s, window=4)
    # day 4 is a 2-2 tie: stay in Crisis instead of picking alphabetically
    assert list(out) == ["Crisis", "Crisis", "Crisis", "Crisis"]


def test_simulate_drifts_weights_and_charges_turnover():
    idx = pd.bdate_range("2020-01-01", periods=4)
    rets = pd.DataFrame({"a": [0.0, 0.10, 0.0, 0.0], "b": [0.0, -0.10, 0.0, 0.0]}, index=idx)
    targets = pd.DataFrame({"a": [0.5, 0.5], "b": [0.5, 0.5]}, index=[idx[0], idx[2]])

    bt = pl.simulate(rets, targets, cost_bps=10)

    assert bt.gross.iloc[0] == pytest.approx(0.0)                    # +10% / -10% on 50/50
    np.testing.assert_allclose(bt.weights.iloc[0], [0.55, 0.45])     # drifted, not reset daily
    assert bt.turnover.iloc[1] == pytest.approx(0.10)                # |0.5-0.55| + |0.5-0.45|
    assert bt.net.iloc[1] == pytest.approx(-0.10 * 10 / 10000)
    np.testing.assert_allclose(bt.weights.iloc[1], [0.5, 0.5])


def test_static_benchmark_pays_for_rebalancing():
    prices = synthetic_prices(200)
    prices["bonds"] = np.linspace(100, 90, len(prices))
    bt = pl.backtest_static(prices, {"stocks": 0.6, "bonds": 0.4}, rebalance_every=21)
    assert bt.turnover.sum() > 0
    assert (bt.net <= bt.gross).all()


def test_portfolio_uses_simple_not_log_returns():
    idx = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"stocks": [100, 100, 150], "gold": [100, 100, 50], "bonds": [100] * 3}, index=idx)
    targets = pd.DataFrame({"stocks": [0.5], "gold": [0.5], "bonds": [0.0]}, index=idx[1:2])
    bt = pl.simulate(pl.asset_returns(prices), targets, cost_bps=0)
    assert bt.gross.iloc[0] == pytest.approx(0.0)  # +50% and -50% on 50/50 is flat, not log(1.5)/2 + log(0.5)/2


def test_clean_fx_repairs_reverting_spike_but_keeps_real_moves():
    idx = pd.bdate_range("2020-01-01", periods=12)
    fx = pd.Series([80.0] * 12, index=idx)
    fx.iloc[3] = 84.0          # one-day bad print that reverts
    fx.iloc[8:] = 84.0         # genuine level shift that persists
    cleaned, bad = pl.clean_fx(fx)
    assert list(bad[bad].index) == [idx[3]]
    assert cleaned.iloc[3] == pytest.approx(80.0)
    assert (cleaned.iloc[8:] == 84.0).all()


@pytest.mark.parametrize("regime", pl.REGIMES)
def test_optimizer_respects_constraints(regime):
    mu = np.array([0.15, 0.05, 0.03])
    sigma = np.diag([0.04, 0.02, 0.005])
    w = pl.optimize_weights(mu, sigma, regime)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all() and (w <= pl.MAX_WEIGHT + 1e-6).all()


def test_higher_risk_aversion_in_crisis_lowers_portfolio_variance():
    mu = np.array([0.20, 0.05, 0.03])
    sigma = np.diag([0.06, 0.02, 0.005])
    var = {r: pl.optimize_weights(mu, sigma, r) @ sigma @ pl.optimize_weights(mu, sigma, r) for r in pl.REGIMES}
    assert var["Bull"] > var["Bear"] > var["Crisis"]
