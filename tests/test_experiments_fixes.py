"""Acceptance tests for m2, m7, OOS split, tautology replacements. Implementer may not edit."""
import numpy as np
import pandas as pd
import pytest

from impact_lab.experiments import out_of_sample, oos_split, ratio_at, ratio_point
from impact_lab.fit import PowerFit, fit_power, initial_guess
from impact_lab.generate import generate_metaorders
from impact_lab.laws import SqrtLaw

SEED = 20261028


def test_ratio_at_window_mean_hand_computed():
    # fit z = 0.5 sqrt(phi): fitted = 0.015 and 0.01732051 at phi = 0.9e-3, 1.2e-3
    df = pd.DataFrame({"phi": [0.9e-3, 1.2e-3, 5e-2], "impact": [0.014, 0.017, 0.2],
                       "sigma_d": [1.0, 1.0, 1.0]})
    f = PowerFit(Y=0.5, delta=0.5, cov=np.eye(2) * 1e-6, n=3)
    expect = (0.015 + 0.01732051) / 2 / ((0.014 + 0.017) / 2)
    assert ratio_at(df, f, SqrtLaw(), 1e-3) == pytest.approx(expect, rel=1e-6)  # 1.04261


def test_ratio_point_hand_computed():
    f = PowerFit(Y=0.6, delta=0.5, cov=np.eye(2) * 1e-6, n=10)
    assert ratio_point(f, SqrtLaw(Y=0.5, delta=0.5), 1e-3) == pytest.approx(1.2)
    f2 = PowerFit(Y=0.5, delta=0.4, cov=np.eye(2) * 1e-6, n=10)
    assert ratio_point(f2, SqrtLaw(), 0.01) == pytest.approx(0.01**-0.1)


def test_initial_guess_is_not_the_answer():
    rng = np.random.default_rng(0)
    phi = np.exp(rng.uniform(np.log(1e-3), np.log(0.2), 50_000))
    z = 0.3 * phi**0.6 + 0.1 * rng.standard_normal(phi.size)
    Y0, d0 = initial_guess(phi, z)
    assert (Y0, d0) != (0.5, 0.5)
    assert 0.0 < d0 < 2.0 and Y0 > 0
    assert d0 == pytest.approx(0.6, abs=0.15)


def test_initial_guess_fallback_when_no_positive_bins():
    phi = np.logspace(-3, -1, 100)
    assert initial_guess(phi, -np.ones(100)) == (0.3, 1.0)


@pytest.mark.parametrize("x0", [(0.1, 1.0), (2.0, 0.2), (1.0, 1.5)])
def test_fit_power_converges_from_any_start(x0):
    rng = np.random.default_rng(1)
    phi = np.exp(rng.uniform(np.log(1e-3), np.log(0.2), 20_000))
    z = 0.5 * phi**0.5 + 0.05 * rng.standard_normal(phi.size)
    ref, alt = fit_power(phi, z), fit_power(phi, z, x0=x0)
    assert alt.Y == pytest.approx(ref.Y, abs=1e-5)
    assert alt.delta == pytest.approx(ref.delta, abs=1e-5)


def test_oos_split_is_a_partition():
    n = 20_000
    rng = np.random.default_rng(SEED)
    generate_metaorders(rng, n, SqrtLaw())
    calib = oos_split(rng, n)
    assert calib.dtype == bool and calib.shape == (n,)
    assert 0.48 < calib.mean() < 0.52
    regs, _ = out_of_sample(SEED, n)
    r = regs.set_index(["fit_on", "test_on"])["n"]
    for fit_on in ("flow", "alpha", "all"):
        assert r[(fit_on, "flow")] + r[(fit_on, "alpha")] == (~calib).sum()


def test_oos_fit_never_sees_test_rows(monkeypatch):
    import impact_lab.experiments as ex
    seen = []
    real = ex._fit
    monkeypatch.setattr(ex, "_fit", lambda df: (seen.append(df.index.to_numpy()), real(df))[1])
    n = 5_000
    rng = np.random.default_rng(SEED)
    generate_metaorders(rng, n, SqrtLaw())
    test_idx = np.flatnonzero(~oos_split(rng, n))
    ex.out_of_sample(SEED, n)
    assert len(seen) == 3
    for idx in seen:
        assert np.intersect1d(idx, test_idx).size == 0
