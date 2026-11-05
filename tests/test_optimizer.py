import numpy as np
import pytest

from impact_lab.capacity import CapacityParams, covariance, make_universe
from impact_lab.laws import SqrtLaw
from impact_lab.optimizer import OptimizerParams, cost_coef, optimize, reoptimised_capacity


@pytest.fixture(scope="module")
def setup():
    u = make_universe(np.random.default_rng(20261028))
    cov = covariance(u, 0.012)
    alpha = 0.1 * np.random.default_rng(1).standard_normal(len(u)) * u["sigma_d"] * np.sqrt(252)
    return u, cov, alpha.to_numpy()


def test_no_cost_matches_closed_form(setup):
    u, cov, alpha = setup
    n = len(u)
    p = OptimizerParams(gross_limit=None)
    w = optimize(alpha, cov, np.zeros(n), np.zeros(n), 0.5, 0.0, p, rebalances=12)
    S_inv = np.linalg.inv(p.risk_aversion * cov / 12)
    a_p, one = alpha / 12, np.ones(n)
    mu = (one @ S_inv @ a_p) / (one @ S_inv @ one)
    w_cf = S_inv @ (a_p - mu * one)
    assert abs(w_cf.sum()) < 1e-12
    assert np.max(np.abs(w - w_cf)) <= 1e-5


def test_prohibitive_cost_freezes_weights(setup):
    u, cov, alpha = setup
    n = len(u)
    w0 = np.zeros(n)
    w0[:2] = [0.1, -0.1]
    w = optimize(alpha, cov, w0, np.full(n, 1e3), 0.5, 5e-4, OptimizerParams())
    assert np.max(np.abs(w - w0)) <= 1e-4


@pytest.mark.parametrize("delta", [0.36, 0.5, 0.6])
def test_optimal_for_fitted_deltas(setup, delta):
    u, cov, alpha = setup
    c = cost_coef(1e9, u, SqrtLaw(delta=delta))
    w = optimize(alpha, cov, np.zeros(len(u)), c, delta, 5e-4, OptimizerParams())
    assert abs(w.sum()) <= 1e-6
    assert np.abs(w).sum() <= 2.0 + 1e-6


def test_reoptimised_turnover_falls_with_aum(setup):
    u, _, _ = setup
    models = {"true": SqrtLaw(), "naive": SqrtLaw(Y=0.45, delta=0.42)}
    r = reoptimised_capacity(np.array([1e8, 1e9, 1e10]), u, models, SqrtLaw(), CapacityParams(),
                             OptimizerParams(n_periods=15, burn_in=5), seed=3)
    for _, g in r.groupby("model"):
        t = g.sort_values("aum")["turnover"].to_numpy()
        assert (t[1:] <= t[:-1] * 1.01).all()
