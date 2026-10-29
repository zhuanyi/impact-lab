import numpy as np
import pytest

from impact_lab.experiments import alpha_bias, misspecification, out_of_sample
from impact_lab.generate import generate_metaorders
from impact_lab.laws import CrossoverLaw, LogLaw

SEED = 20261028


@pytest.fixture(scope="module")
def miss():
    return misspecification(SEED)


def test_misspecification_fits(miss):
    fits = miss[0].set_index("law")
    assert fits.loc["log", "delta"] <= 0.45
    assert fits.loc["log", "ratio_at_0.001"] >= 1.3
    assert fits.loc["crossover_stress", "delta"] >= 0.55
    assert fits.loc["crossover_stress", "ratio_at_0.001"] >= 1.3
    assert abs(fits.loc["crossover_paper", "delta"] - 0.5) <= 0.08


def test_ratio_at_1e3_definition(miss):
    """ratio at 1e-3 = mean fitted / mean true impact / sigma over flow orders near 1e-3."""
    fits = miss[0].set_index("law")
    df = generate_metaorders(np.random.default_rng(SEED), 200_000, LogLaw())
    df = df[df["flow"]]
    m = (df["phi"] >= 0.8e-3) & (df["phi"] <= 1.25e-3)
    true = (df.loc[m, "impact"] / df.loc[m, "sigma_d"]).mean()
    pred = (fits.loc["log", "Y"] * df.loc[m, "phi"] ** fits.loc["log", "delta"]).mean()
    assert fits.loc["log", "ratio_at_0.001"] == pytest.approx(pred / true, rel=1e-12)


def test_ratio_at_sampled_edge_is_unbiased(miss):
    """At phi = 1e-3 (the sampled minimum) only the upper half of the window has orders; the
    ratio must still be ~1 for a correctly specified fit (it was 0.91 with predict(1e-3))."""
    fits = miss[0].set_index("law")
    sq = fits.loc["sqrt"]
    exact = sq["Y"] * 1e-3 ** sq["delta"] / (0.5 * 1e-3**0.5)
    assert sq["ratio_at_0.001"] == pytest.approx(exact, rel=0.01)


def test_misspecification_curves(miss):
    fits, curves = miss
    assert set(curves["law"]) == set(fits["law"])
    for _, c in curves.groupby("law"):
        assert c["in_sample"].sum() == 20
        assert (~c["in_sample"]).sum() == 5
        np.testing.assert_allclose(c["ratio"], c["fitted"] / c["true"])
    sq = curves[curves["law"] == "sqrt"]
    np.testing.assert_allclose(sq["elasticity_true"], 0.5, atol=1e-6)


def test_crossover_carries_duration_effect():
    law = CrossoverLaw(eta_star=5e-2)
    phi = np.full(2, 0.01)
    short, long_ = law.impact(phi, phi / np.array([0.05, 0.5]))
    assert long_ < short


def test_alpha_bias():
    ab = alpha_bias(SEED).set_index("subset")
    assert ab.loc["alpha", "delta"] <= 0.42
    assert ab.loc["alpha", "ratio_1pct"] >= 1.3
    assert ab.loc["all", "ratio_1pct"] >= 1.15
    assert 0.9 <= ab.loc["flow", "ratio_1pct"] <= 1.1
    assert ab["bias_removed"].iloc[0] >= 0.7
    assert ab.loc["flow", "mean_alpha_ret_bps"] == 0


def test_out_of_sample():
    regs, deciles = out_of_sample(SEED)
    r = regs.set_index(["fit_on", "test_on"])
    assert abs(r.loc[("flow", "flow"), "a"]) <= 1.5
    assert abs(r.loc[("flow", "flow"), "b"] - 1) <= 0.1
    assert r.loc[("flow", "alpha"), "mean_resid"] >= 4
    assert r.loc[("alpha", "flow"), "mean_resid"] <= -4
    assert abs(r.loc[("alpha", "alpha"), "mean_resid"]) <= 1.5
    assert len(regs) == 6
    assert len(deciles) == 60
