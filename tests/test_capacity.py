import numpy as np
import pandas as pd
import pytest

from impact_lab.capacity import (
    CapacityParams,
    Universe,
    capacity_band,
    capacity_curve,
    capacity_points,
    covariance,
    make_universe,
    net_alpha,
    trade_phi,
)
from impact_lab.experiments import alpha_bias, default_laws, fit_from_row
from impact_lab.fit import PowerFit
from impact_lab.laws import SqrtLaw

SEED = 20261028
CAP = CapacityParams()


def one_name(sigma=0.02, adv=1e8, half_spread=5e-4):
    return pd.DataFrame({"name": np.array([0], dtype=np.int64), "adv_usd": [adv],
                         "sigma_d": [sigma], "beta": [1.0], "half_spread": [half_spread]})


@pytest.fixture(scope="module")
def universe():
    return make_universe(np.random.default_rng(SEED))


def test_universe_and_covariance(universe):
    assert list(universe.columns) == ["name", "adv_usd", "sigma_d", "beta", "half_spread"]
    cov = covariance(universe, Universe().index_vol)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0
    bad = universe.assign(beta=5.0)
    with pytest.raises(ValueError):
        covariance(bad, 0.012)


def test_trade_phi_equal_dollar(universe):
    phi = trade_phi(np.array([1e9]), universe, CAP)
    dollars = phi[0] * universe["adv_usd"].to_numpy()
    np.testing.assert_allclose(dollars, 1e9 * 4.0 / (12 * len(universe)))


def test_net_alpha_shape(universe):
    aum = np.logspace(6, 12, 61)
    for law in default_laws().values():
        net = net_alpha(aum, universe, law, CAP)
        assert (np.diff(net) < 0).all()
    # aum -> 0 limit, checked at $1,000: impact vanishes, only the half-spread remains
    for law in default_laws().values():
        net0 = net_alpha(np.array([1e3]), universe, law, CAP)[0]
        assert abs(net0 - (CAP.gross_alpha - CAP.turnover * 5e-4)) <= 1e-4


def b_coef(cap, sigma, adv, Y=0.5):
    return cap.turnover * sigma * Y * np.sqrt(cap.turnover / (cap.rebalances * 1 * adv))


def test_capacity_closed_form_sqrt():
    u = one_name()
    be, opt = capacity_points(u, SqrtLaw(), CAP)
    a0 = ((CAP.gross_alpha - CAP.turnover * 5e-4) / b_coef(CAP, 0.02, 1e8)) ** 2
    assert be == pytest.approx(a0, rel=5e-3)
    assert opt / be == pytest.approx(4 / 9, rel=5e-3)


@pytest.mark.parametrize("delta", [1.0, 0.3])
def test_capacity_closed_form_general_delta(delta):
    be, opt = capacity_points(one_name(), SqrtLaw(delta=delta), CAP)
    assert opt / be == pytest.approx((1 + delta) ** (-1 / delta), rel=5e-3)


def test_capacity_points_nan_when_unprofitable():
    cap = CapacityParams(gross_alpha=0.001)
    assert np.isnan(capacity_points(one_name(), SqrtLaw(), cap)).all()


def test_capacity_bias(universe):
    ab = alpha_bias(SEED).set_index("subset")
    naive, flow = fit_from_row(ab.loc["all"]), fit_from_row(ab.loc["flow"])
    true_be, _ = capacity_points(universe, SqrtLaw(), CAP)
    naive_be, _ = capacity_points(universe, naive.as_law(), CAP)
    # The naive fit over-prices small orders but, with its flattened delta, under-prices the
    # 50%-ADV trades that set break-even, so it overstates capacity (SPEC deviation, README).
    assert naive_be >= 1.2 * true_be
    curve = capacity_curve(np.array([true_be]), universe, SqrtLaw(), CAP, phi_max=0.2)
    assert not curve["phi_in_sample"].iloc[0]
    grid = np.logspace(8, 11, 13)
    band = capacity_band(grid, universe, flow, CAP, np.random.default_rng(SEED), n_draws=500)
    lo, _, hi = band.attrs["break_even_q"]
    assert lo <= true_be <= hi
    assert ((band["lo"] <= band["mid"]) & (band["mid"] <= band["hi"])).all()


def test_band_keeps_draws_without_break_even():
    """Draws still profitable at $10tn count as +inf break-even instead of being dropped."""
    fit = PowerFit(Y=0.5, delta=0.5, cov=np.diag([0.25, 1e-6]), n=1000)
    band = capacity_band(np.logspace(8, 11, 5), one_name(), fit, CAP,
                         np.random.default_rng(0), n_draws=500)
    lo, _, hi = band.attrs["break_even_q"]
    assert band.attrs["break_even_outside"] > 25
    assert np.isfinite(lo)
    assert hi == np.inf
