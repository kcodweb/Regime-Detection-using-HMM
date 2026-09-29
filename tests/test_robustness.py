import numpy as np
import pandas as pd
import pytest

import pipeline as pl
import robustness as rb


def test_optimizer_matches_direct_formulation():
    import cvxpy as cp

    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.normal(size=(60, 3)) * [0.012, 0.009, 0.004]
        mu, sigma = A.mean(0) * 252, np.cov(A.T) * 252
        regime = rng.choice(pl.REGIMES)
        w = cp.Variable(3)
        cp.Problem(cp.Maximize(mu @ w - pl.REGIME_GAMMA[regime] * cp.quad_form(w, cp.psd_wrap(sigma))),
                   [cp.sum(w) == 1, w >= 0, w <= pl.MAX_WEIGHT]).solve(solver=cp.CLARABEL)
        np.testing.assert_allclose(pl.optimize_weights(mu, sigma, regime), w.value, atol=1e-4)


def test_risk_parity_equalizes_risk_contributions():
    sigma = np.array([[0.04, 0.006, 0.0], [0.006, 0.02, 0.001], [0.0, 0.001, 0.005]])
    w = pl.risk_parity_weights(sigma)
    contributions = w * (sigma @ w)
    assert w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(contributions, contributions.mean(), rtol=1e-4)


def test_same_regime_risk_parity_uses_that_regimes_covariance():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    regimes = pd.Series(np.where((np.arange(400) // 50) % 2 == 0, "Bull", "Crisis"), index=idx)
    stock_vol = np.where(regimes == "Crisis", 0.04, 0.01)    # stocks only turbulent in Crisis
    rets = pd.DataFrame({"stocks": rng.normal(0, stock_vol), "gold": rng.normal(0, 0.01, 400),
                         "bonds": rng.normal(0, 0.01, 400)}, index=idx)
    allocate = pl.risk_parity(lookback=63, same_regime=True)
    w_bull = allocate(rets, regimes, "Bull")
    w_crisis = allocate(rets, regimes, "Crisis")
    assert w_crisis[0] < w_bull[0] / 2


def test_fixed_clock_when_min_hold_equals_max_hold():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rets = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (300, 3)), index=idx, columns=pl.ASSETS)
    regimes = pd.Series(np.random.default_rng(1).choice(pl.REGIMES, 300), index=idx)   # flips constantly
    targets = pl.dynamic_targets(rets, regimes, lookback=63, min_hold=21, max_hold=21)
    gaps = np.diff([idx.get_loc(d) for d in targets.index])
    assert (gaps == 21).all()


def test_bootstrap_indices_are_valid_and_reproducible():
    idx = rb.stationary_bootstrap_indices(500, 200, block=20, seed=3)
    assert idx.shape == (200, 500) and idx.min() >= 0 and idx.max() < 500
    np.testing.assert_array_equal(idx, rb.stationary_bootstrap_indices(500, 200, block=20, seed=3))
    continues = (idx[:, 1:] == (idx[:, :-1] + 1) % 500).mean()
    assert continues == pytest.approx(1 - 1 / 20, abs=0.01)   # blocks have mean length ~20


def test_sharpe_difference_of_a_series_with_itself_is_zero():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 1000), index=pd.bdate_range("2020-01-01", periods=1000))
    out = rb.sharpe_difference(r, r, n=200)
    assert out["sharpe_diff"] == 0 and out["ci_low"] == 0 and out["ci_high"] == 0


def test_bootstrap_interval_covers_a_clear_difference():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=2500)
    noise = rng.normal(0, 0.01, 2500)
    good = pd.Series(noise + 0.002, index=idx)      # same noise, much higher mean
    bad = pd.Series(noise, index=idx)
    out = rb.sharpe_difference(good, bad, n=500)
    assert out["ci_low"] > 0 and out["share_not_better"] == 0


def test_shuffle_runs_keeps_the_regime_mix_but_moves_it():
    idx = pd.bdate_range("2020-01-01", periods=120)
    labels = np.repeat(["Bull", "Crisis", "Bear", "Bull", "Crisis", "Bear"], [30, 10, 20, 25, 15, 20])
    regimes = pd.Series(labels, index=idx)
    shuffled = rb.shuffle_runs(regimes, np.random.default_rng(0))
    assert shuffled.index.equals(regimes.index)
    assert shuffled.value_counts().to_dict() == regimes.value_counts().to_dict()
    assert not shuffled.equals(regimes.astype(object))
