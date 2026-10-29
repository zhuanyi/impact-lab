import numpy as np
import pytest

from impact_lab.experiments import recovery, seed_spread
from impact_lab.fit import binned_means, calibration_regression, fit_power
from impact_lab.laws import SqrtLaw


def test_recovery_square_root():
    r = recovery(20261028).iloc[0]
    assert r["n"] > 95_000
    assert abs(r["delta"] - 0.5) <= 0.05
    assert abs(r["Y"] / 0.5 - 1) <= 0.10
    assert abs(r["level_1pct"] / 0.05 - 1) <= 0.05


def test_sandwich_interval_coverage():
    spread = seed_spread(range(20), 100_000)
    assert spread["covered"].sum() >= 16


def test_noise_free_fit_and_bins():
    phi = np.logspace(-3, np.log10(0.2), 1000)
    f = fit_power(phi, 0.3 * phi**0.6)
    assert f.Y == pytest.approx(0.3, abs=1e-6)
    assert f.delta == pytest.approx(0.6, abs=1e-6)
    b = binned_means(phi, 0.3 * phi**0.6, n_bins=20)
    assert len(b) == 20
    assert b["n"].sum() == 1000


def test_level_and_draws():
    phi = np.logspace(-3, np.log10(0.2), 1000)
    rng = np.random.default_rng(0)
    f = fit_power(phi, 0.5 * phi**0.5 + 0.01 * rng.standard_normal(1000))
    lvl, se = f.level(0.01)
    assert lvl == pytest.approx(f.Y * 0.01**f.delta)
    assert se > 0
    d = f.draws(rng, 200)
    assert d.shape == (200, 2) and (d[:, 0] > 0).all()
    assert f.as_law() == SqrtLaw(Y=f.Y, delta=f.delta)


def test_calibration_regression_identity():
    rng = np.random.default_rng(0)
    pred = rng.uniform(1, 50, 5000)
    r = calibration_regression(pred + 2.0, pred)
    assert r["a"] == pytest.approx(2.0)
    assert r["b"] == pytest.approx(1.0)
    assert r["mean_resid"] == pytest.approx(2.0)
    assert list(r.index) == ["a", "b", "se_a", "se_b", "mean_resid", "n"]
