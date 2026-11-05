"""Acceptance tests for M1 (capacity sensitivity). Implementer may not edit."""
import numpy as np
import pandas as pd
import pytest

from impact_lab.capacity import (
    CapacityParams,
    capacity_points,
    capacity_sensitivity,
    make_universe,
    net_alpha,
    trade_phi,
    trade_stats,
)
from impact_lab.laws import CrossoverLaw, SqrtLaw

SEED = 20261028


def two_names():
    # per-rebalance dollars per name at aum=1.2e9: 1.2e9 * 4 / (12 * 2) = 2e8
    return pd.DataFrame({"name": np.array([0, 1], dtype=np.int64),
                         "adv_usd": [4e8, 1e8], "sigma_d": [0.02, 0.01],
                         "beta": [1.0, 1.0], "half_spread": [5e-4, 5e-4]})


def test_trade_stats_hand_computed():
    s = trade_stats(1.2e9, two_names(), SqrtLaw(), CapacityParams())
    # phi = 0.5 and 2.0
    assert s["phi_min"] == pytest.approx(0.5)
    assert s["phi_median"] == pytest.approx(1.25)
    assert s["phi_mean"] == pytest.approx(1.25)
    assert s["phi_max"] == pytest.approx(2.0)
    assert s["share_phi_gt1"] == pytest.approx(0.5)
    c0 = 5e-4 + 0.02 * 0.5 * 0.5**0.5
    c1 = 5e-4 + 0.01 * 0.5 * 2.0**0.5
    assert s["cost_share_phi_gt1"] == pytest.approx(c1 / (c0 + c1))


def test_trade_stats_no_name_above_one():
    s = trade_stats(1e8, two_names(), SqrtLaw(), CapacityParams())
    assert s["share_phi_gt1"] == 0.0
    assert s["cost_share_phi_gt1"] == 0.0


def test_adv_weighting_equalises_phi():
    cap = CapacityParams(weighting="adv")
    phi = trade_phi(np.array([1.2e9]), two_names(), cap)
    # total per rebalance 1.2e9 * 4 / 12 = 4e8 over 5e8 of ADV
    np.testing.assert_allclose(phi[0], [0.8, 0.8])
    assert CapacityParams().weighting == "equal_dollar"
    with pytest.raises(ValueError):
        CapacityParams(weighting="bogus")


def test_eta_max_sets_per_name_horizon():
    u = two_names()
    law = CrossoverLaw(Y=0.5, eta_star=5e-2)
    cap = CapacityParams(eta_max=0.5)
    # phi = (0.5, 2.0) -> exec days (1, 4) -> eta = (0.5, 0.5)
    imp = law.impact(np.array([0.5, 2.0]), np.array([0.5, 0.5]))
    cost = u["half_spread"].to_numpy() + u["sigma_d"].to_numpy() * imp
    expect = cap.gross_alpha - cap.turnover * cost.mean()
    assert net_alpha([1.2e9], u, law, cap)[0] == pytest.approx(expect, rel=1e-12)
    assert CapacityParams().eta_max is None
    with pytest.raises(ValueError):
        CapacityParams(eta_max=0.0)


def test_sqrt_law_is_horizon_free():
    u = make_universe(np.random.default_rng(SEED))
    base = capacity_points(u, SqrtLaw(), CapacityParams())
    scaled = capacity_points(u, SqrtLaw(), CapacityParams(eta_max=0.25))
    assert scaled == pytest.approx(base, rel=1e-9)


def test_sensitivity_table():
    u = make_universe(np.random.default_rng(SEED))
    laws = {"true": SqrtLaw(), "stress": CrossoverLaw(eta_star=5e-2)}
    t = capacity_sensitivity(u, laws, CapacityParams(), eta_max_grid=(1.0, 0.25))
    assert list(t.columns) == ["variant", "law", "break_even", "dollar_optimal"]
    assert list(t["variant"].unique()) == ["base", "eta_max=1", "eta_max=0.25", "adv_weighted"]
    g = t.set_index(["variant", "law"])["break_even"]
    assert g[("base", "true")] == pytest.approx(1.15194e10, rel=1e-4)
    assert g[("eta_max=1", "true")] == pytest.approx(g[("base", "true")], rel=1e-9)
    assert g[("base", "stress")] == pytest.approx(1.2183e10, rel=1e-3)
    assert g[("eta_max=1", "stress")] == pytest.approx(1.239e10, rel=2e-3)
    assert g[("eta_max=0.25", "stress")] == pytest.approx(1.384e10, rel=2e-3)
    assert g[("adv_weighted", "true")] == pytest.approx(2.16e10, rel=2e-3)
