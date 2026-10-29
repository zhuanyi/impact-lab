import numpy as np
import pandas as pd
import pytest
from tca_groundtruth.execution import ImpactParams
from tca_groundtruth.market import MarketParams, simulate_day
from tca_groundtruth.orders import apply_alpha, generate_orders

from impact_lab.fit import fit_power
from impact_lab.generate import (
    METAORDER_COLUMNS,
    MetaorderParams,
    alpha_window_return,
    generate_metaorders,
    p1_peak_impact,
)
from impact_lab.laws import CrossoverLaw, SqrtLaw

SEED = 20261028
DTYPES = {"order_id": "int64", "flow": "bool"}


def test_schema_and_invariants():
    law = CrossoverLaw(eta_star=5e-2)
    df = generate_metaorders(np.random.default_rng(SEED), 5_000, law)
    assert list(df.columns) == METAORDER_COLUMNS
    for c in df.columns:
        assert str(df[c].dtype) == DTYPES.get(c, "float64"), c
    assert len(df) == 5_000
    assert isinstance(df.index, pd.RangeIndex)
    pd.testing.assert_frame_equal(
        df, generate_metaorders(np.random.default_rng(SEED), 5_000, law))
    assert ((df["T"] > 0) & (df["T"] <= 1)).all()
    np.testing.assert_array_equal(df["eta"], df["phi"] / df["T"])
    assert (df["flow"] == (df["alpha_bps"] == 0)).all()
    assert (df.loc[df["flow"], "alpha_ret"] == 0).all()
    np.testing.assert_allclose(df["ret"], df["impact"] + df["alpha_ret"] + df["noise"],
                               rtol=0, atol=1e-15)
    np.testing.assert_array_equal(df["impact"],
                                  df["sigma_d"] * law.impact(df["phi"].to_numpy(),
                                                             df["eta"].to_numpy()))
    lo, hi = MetaorderParams().qty_pct_adv
    assert df["phi"].min() >= lo * (1 - 1e-5)
    assert df["phi"].max() <= hi * (1 + 1e-5)


@pytest.mark.parametrize("params,n", [
    (MetaorderParams(), 0),
    (MetaorderParams(flow_share=1.5), 10),
    (MetaorderParams(qty_pct_adv=(0.2, 0.001)), 10),
    (MetaorderParams(daily_vol=(0.04, 0.01)), 10),
])
def test_generator_rejects_bad_input(params, n):
    with pytest.raises(ValueError):
        generate_metaorders(np.random.default_rng(0), n, SqrtLaw(), params)


def test_noise_is_standard_normal_in_window_units():
    df = generate_metaorders(np.random.default_rng(SEED), 200_000, SqrtLaw())
    e = df["noise"] / (df["sigma_d"] * np.sqrt(df["T"]))
    assert abs(e.mean()) <= 0.01
    assert abs(e.std() - 1) <= 0.01


def test_alpha_closed_form_matches_p1_apply_alpha():
    rng = np.random.default_rng(SEED)
    od = generate_orders(rng, 50, 1e6)
    closed = alpha_window_return(od["alpha_bps"], od["alpha_halflife_min"],
                                 od["decision_minute"], od["arrival_minute"], od["end_minute"])
    for i, (_, o) in enumerate(od.iterrows()):
        bars = apply_alpha(simulate_day(rng, MarketParams()), o)
        sh = bars["alpha_shift"].to_numpy()
        t0, t1, td = int(o["arrival_minute"]), int(o["end_minute"]), int(o["decision_minute"])
        api = o["side"] * (sh[t1] - sh[t0]) / bars["mid_base"].to_numpy()[td]
        assert abs(api - closed[i]) <= 1e-12


def test_cross_check_with_p1_propagator():
    pure = p1_peak_impact(np.random.default_rng(11), 300, ImpactParams(gamma_perm=0.0))
    f = fit_power(pure["phi"], pure["z"])
    assert abs(f.delta - 0.5) <= 0.05
    assert abs(f.Y / 0.5 - 1) <= 0.10
    default = p1_peak_impact(np.random.default_rng(11), 300, ImpactParams())
    assert fit_power(default["phi"], default["z"]).delta >= 0.53
