"""Golden values: recompute key metrics at the default seed and compare to the committed
reports/summary.csv (6 significant figures). Implementer may not edit."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impact_lab.capacity import CapacityParams, capacity_points, make_universe, trade_stats
from impact_lab.experiments import alpha_bias, default_laws, fit_from_row, misspecification
from impact_lab.laws import SqrtLaw

SEED = 20261028
SUMMARY = Path(__file__).resolve().parents[1] / "reports" / "summary.csv"


@pytest.fixture(scope="module")
def committed():
    s = pd.read_csv(SUMMARY)
    return dict(zip(s["metric"], s["value"], strict=True))


def check(committed, metric, value, rel=2e-5):
    assert committed[metric] == pytest.approx(value, rel=rel), metric


def test_alpha_bias_metrics(committed):
    ab = alpha_bias(SEED, 200_000).set_index("subset")
    for subset in ("all", "alpha", "flow"):
        check(committed, f"{subset}_delta", ab.loc[subset, "delta"])
        check(committed, f"{subset}_ratio_1pct", ab.loc[subset, "ratio_1pct"])
    check(committed, "bias_removed", ab["bias_removed"].iloc[0])


def test_misspecification_metrics(committed):
    laws = default_laws()
    fits, _ = misspecification(SEED, 200_000, {k: laws[k] for k in ("log", "crossover_stress")})
    f = fits.set_index("law")
    check(committed, "log_delta_hat", f.loc["log", "delta"])
    check(committed, "log_ratio_at_0.001", f.loc["log", "ratio_at_0.001"])
    check(committed, "crossover_stress_ratio_at_0.001", f.loc["crossover_stress", "ratio_at_0.001"])


def test_capacity_metrics(committed):
    cap = CapacityParams()
    u = make_universe(np.random.default_rng(SEED))
    ab = alpha_bias(SEED, 200_000).set_index("subset")
    laws = {"true": SqrtLaw(), "flow": fit_from_row(ab.loc["flow"]).as_law(),
            "naive": fit_from_row(ab.loc["all"]).as_law()}
    for lab, law in laws.items():
        be, opt = capacity_points(u, law, cap)
        check(committed, f"d1_{lab}_break_even", be)
        check(committed, f"d1_{lab}_dollar_optimal", opt)
    s = trade_stats(committed["d1_true_break_even"], u, SqrtLaw(), cap)
    check(committed, "d1_true_break_even_phi_mean", s["phi_mean"])
    check(committed, "d1_true_break_even_phi_max", s["phi_max"])
    check(committed, "d1_true_break_even_share_phi_gt1", s["share_phi_gt1"])
    check(committed, "d1_true_break_even_cost_share_phi_gt1", s["cost_share_phi_gt1"])
