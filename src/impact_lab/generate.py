"""Vectorised metaorder generator, plus a slow cross-check path through tca-groundtruth.

One row per metaorder: size, window, daily vol, flow label, alpha drift over the window, true
impact under a given law, noise, and the observed signed return arrival -> end (SPEC 3.2).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tca_groundtruth.execution import FillParams, ImpactParams, execute
from tca_groundtruth.market import MarketParams, simulate_day, u_profile
from tca_groundtruth.orders import OrderParams, apply_alpha, generate_orders

from impact_lab.laws import ImpactLaw

DAILY_VOLUME = 1e8   # shares; large enough that whole-share rounding of qty is negligible
LN2 = np.log(2.0)

METAORDER_COLUMNS = ["order_id", "phi", "T", "eta", "sigma_d", "flow", "alpha_bps", "alpha_ret",
                     "impact", "noise", "ret"]


@dataclass(frozen=True)
class MetaorderParams:
    qty_pct_adv: tuple[float, float] = (0.001, 0.2)   # log-uniform phi
    daily_vol: tuple[float, float] = (0.01, 0.04)     # uniform sigma_d per order
    flow_share: float = 0.5
    u_shape: float = 2.0                              # passed to u_profile
    n_minutes: int = 390


def order_params(p: MetaorderParams) -> OrderParams:
    """tca-groundtruth order parameters: its defaults except size range and flow share."""
    return OrderParams(qty_pct_adv=p.qty_pct_adv, flow_share=p.flow_share)


def window_fraction(arrival: np.ndarray, end: np.ndarray, n_minutes: int,
                    u_shape: float) -> np.ndarray:
    """Fraction of the day's volume (and variance) in minutes [arrival, end)."""
    cum = np.concatenate([[0.0], np.cumsum(u_profile(n_minutes, u_shape) / n_minutes)])
    return cum[np.asarray(end)] - cum[np.asarray(arrival)]


def alpha_window_return(alpha_bps, halflife_min, decision, arrival, end) -> np.ndarray:
    """Alpha drift realised from arrival to end; closed form of orders.apply_alpha."""
    a = np.asarray(alpha_bps, dtype=float) / 1e4
    h = np.asarray(halflife_min, dtype=float)
    td, t0, t1 = (np.asarray(x, dtype=float) for x in (decision, arrival, end))
    return a * (np.exp(-(t0 - td) * LN2 / h) - np.exp(-(t1 - td) * LN2 / h))


def _validate(n: int, p: MetaorderParams) -> None:
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0 <= p.flow_share <= 1:
        raise ValueError("flow_share must be in [0, 1]")
    for name in ("qty_pct_adv", "daily_vol"):
        lo, hi = getattr(p, name)
        if lo > hi:
            raise ValueError(f"{name} range is inverted: {(lo, hi)}")


def generate_metaorders(rng: np.random.Generator, n: int, law: ImpactLaw,
                        params: MetaorderParams | None = None) -> pd.DataFrame:
    """n independent metaorders with true impact from `law`; columns METAORDER_COLUMNS."""
    p = params or MetaorderParams()
    _validate(n, p)
    od = generate_orders(rng, n, daily_volume=DAILY_VOLUME, params=order_params(p),
                         n_minutes=p.n_minutes)
    sigma_d = rng.uniform(*p.daily_vol, n)
    eps = rng.standard_normal(n)
    phi = od["qty"].to_numpy() / DAILY_VOLUME
    arrival, end = od["arrival_minute"].to_numpy(), od["end_minute"].to_numpy()
    T = window_fraction(arrival, end, p.n_minutes, p.u_shape)
    eta = phi / T
    alpha_ret = alpha_window_return(od["alpha_bps"].to_numpy(), od["alpha_halflife_min"].to_numpy(),
                                    od["decision_minute"].to_numpy(), arrival, end)
    impact = sigma_d * law.impact(phi, eta)
    noise = sigma_d * np.sqrt(T) * eps
    return pd.DataFrame({
        "order_id": np.arange(n, dtype=np.int64),
        "phi": phi,
        "T": T,
        "eta": eta,
        "sigma_d": sigma_d,
        "flow": od["flow"].to_numpy(dtype=bool),
        "alpha_bps": od["alpha_bps"].to_numpy(dtype=float),
        "alpha_ret": alpha_ret,
        "impact": impact,
        "noise": noise,
        "ret": impact + alpha_ret + noise,
    })


def p1_peak_impact(rng: np.random.Generator, n: int, impact: ImpactParams,
                   qty_pct_adv=(0.001, 0.2), algo: str = "TWAP") -> pd.DataFrame:
    """Noise-free peak impact through tca-groundtruth's per-minute propagator (slow; n <= 1000).

    Columns phi (filled qty / day volume) and z (side * impact_shift / mid at the last fill,
    divided by the day's sigma).
    """
    market = MarketParams()
    od = generate_orders(rng, n, market.daily_volume, OrderParams(qty_pct_adv=qty_pct_adv),
                         market.n_minutes)
    fills = FillParams(passive_frac=0.0, max_participation=1.0)
    phis, zs = [], []
    for _, order in od.iterrows():
        order = order.copy()
        order["algo"] = algo
        bars = apply_alpha(simulate_day(rng, market), order)
        f, out = execute(rng, bars, order, impact, fills)
        if f.empty:
            continue
        t1 = int(f["minute"].max())
        sigma_day = float(np.sqrt((out["sigma_min"] ** 2).sum()))
        phis.append(f["qty"].sum() / out["volume"].sum())
        zs.append(int(order["side"]) * out["impact_shift"].to_numpy()[t1]
                  / out["mid"].to_numpy()[t1] / sigma_day)
    return pd.DataFrame({"phi": np.array(phis, dtype=float), "z": np.array(zs, dtype=float)})
