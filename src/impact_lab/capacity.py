"""Capacity: net alpha vs AUM at fixed turnover, for any impact law (SPEC 3.5).

Each rebalance trades an equal dollar amount in every name, executed over `exec_days` of
volume, and pays half-spread plus impact per unit traded:
net(A) = gross_alpha - turnover * mean_i(half_spread_i + sigma_i * impact(phi_i, phi_i / T)).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

from impact_lab.fit import PowerFit
from impact_lab.laws import ImpactLaw

AUM_BOUNDS = (1e6, 1e13)


@dataclass(frozen=True)
class Universe:
    n_names: int = 50
    adv_usd: tuple[float, float] = (2e7, 5e8)     # log-uniform daily dollar volume
    daily_vol: tuple[float, float] = (0.018, 0.04)
    beta: tuple[float, float] = (0.6, 1.4)
    index_vol: float = 0.012                       # daily
    half_spread_bps: float = 5.0


def make_universe(rng: np.random.Generator, u: Universe | None = None) -> pd.DataFrame:
    """Columns name, adv_usd, sigma_d (daily), beta, half_spread (return units)."""
    u = u or Universe()
    lo, hi = np.log(u.adv_usd)
    return pd.DataFrame({
        "name": np.arange(u.n_names, dtype=np.int64),
        "adv_usd": np.exp(rng.uniform(lo, hi, u.n_names)),
        "sigma_d": rng.uniform(*u.daily_vol, u.n_names),
        "beta": rng.uniform(*u.beta, u.n_names),
        "half_spread": np.full(u.n_names, u.half_spread_bps / 1e4),
    })


def covariance(universe: pd.DataFrame, index_vol: float) -> np.ndarray:
    """Annual one-factor covariance; ValueError if any idiosyncratic variance <= 0."""
    beta = universe["beta"].to_numpy()
    idio = universe["sigma_d"].to_numpy() ** 2 - beta**2 * index_vol**2
    if (idio <= 0).any():
        raise ValueError("sigma_d must exceed |beta| * index_vol for every name")
    return 252.0 * (np.outer(beta, beta) * index_vol**2 + np.diag(idio))


@dataclass(frozen=True)
class CapacityParams:
    gross_alpha: float = 0.06       # annual, illustrative
    turnover: float = 4.0           # annual traded value / AUM
    rebalances: int = 12            # per year
    exec_days: float = 1.0          # each rebalance trade executes over one day: T = 1, eta = phi


def trade_phi(aum: np.ndarray, universe: pd.DataFrame, cap: CapacityParams) -> np.ndarray:
    """(len(aum), n_names) trade size per name per rebalance as a fraction of ADV."""
    aum = np.atleast_1d(np.asarray(aum, dtype=float))
    per_name = aum * cap.turnover / (cap.rebalances * len(universe))
    return per_name[:, None] / universe["adv_usd"].to_numpy()[None, :]


def _net_from_impact(impact: np.ndarray, universe: pd.DataFrame, cap: CapacityParams):
    cost = universe["half_spread"].to_numpy() + universe["sigma_d"].to_numpy() * impact
    return cap.gross_alpha - cap.turnover * cost.mean(axis=-1)


def net_alpha(aum, universe: pd.DataFrame, law: ImpactLaw, cap: CapacityParams) -> np.ndarray:
    """Annual net alpha at each AUM."""
    phi = trade_phi(aum, universe, cap)
    return _net_from_impact(law.impact(phi, phi / cap.exec_days), universe, cap)


def _break_even_log(net_fn) -> float:
    """ln break-even AUM for net_fn(ln A); nan if no sign change inside AUM_BOUNDS."""
    lo, hi = np.log(AUM_BOUNDS)
    if net_fn(lo) <= 0 or net_fn(hi) > 0:
        return float("nan")
    return brentq(net_fn, lo, hi, xtol=1e-12)


def _points(net_fn) -> tuple[float, float]:
    be = _break_even_log(net_fn)
    if np.isnan(be):
        return float("nan"), float("nan")
    lo = np.log(AUM_BOUNDS[0])
    res = minimize_scalar(lambda x: -np.exp(x) * net_fn(x), bounds=(lo, be), method="bounded",
                          options={"xatol": 1e-8})
    return float(np.exp(be)), float(np.exp(res.x))


def capacity_points(universe: pd.DataFrame, law: ImpactLaw, cap: CapacityParams
                    ) -> tuple[float, float]:
    """(break_even, dollar_optimal) AUM; nan when net alpha is not positive at 1e6 or stays
    positive at 1e13."""
    return _points(lambda x: float(net_alpha(np.exp(x), universe, law, cap)[0]))


def capacity_curve(aum_grid, universe: pd.DataFrame, law: ImpactLaw, cap: CapacityParams,
                   phi_max: float = np.inf) -> pd.DataFrame:
    """Net and dollar alpha on a grid; phi_in_sample flags median trade phi <= phi_max."""
    aum = np.asarray(aum_grid, dtype=float)
    net = net_alpha(aum, universe, law, cap)
    med = np.median(trade_phi(aum, universe, cap), axis=1)
    return pd.DataFrame({"aum": aum, "net_alpha": net, "dollar_alpha": aum * net,
                         "median_phi": med, "phi_in_sample": med <= phi_max})


def capacity_band(aum_grid, universe: pd.DataFrame, fit: PowerFit, cap: CapacityParams,
                  rng: np.random.Generator, n_draws: int = 500,
                  q: tuple[float, float] = (0.05, 0.95)) -> pd.DataFrame:
    """Net-alpha quantiles over (Y, delta) draws from the fit's covariance.

    attrs["break_even_q"] = (lo, mid, hi) break-even quantiles over the same draws. A draw that
    is unprofitable at the smallest AUM counts as break-even 0 and one still profitable at the
    largest as +inf, so no draw is dropped; attrs["break_even_outside"] counts them.
    """
    aum = np.asarray(aum_grid, dtype=float)
    d = fit.draws(rng, n_draws)
    Y, delta = d[:, 0], d[:, 1]
    phi = trade_phi(aum, universe, cap)   # a power law ignores eta
    nets = _net_from_impact(Y[:, None, None] * phi[None] ** delta[:, None, None], universe, cap)
    lo, mid, hi = np.quantile(nets, [q[0], 0.5, q[1]], axis=0)
    out = pd.DataFrame({"aum": aum, "lo": lo, "mid": mid, "hi": hi})

    def be(y: float, dl: float) -> float:
        def f(x: float) -> float:
            p = trade_phi(np.exp(x), universe, cap)
            return float(_net_from_impact(y * p**dl, universe, cap)[0])
        x = _break_even_log(f)
        if np.isnan(x):
            return 0.0 if f(np.log(AUM_BOUNDS[0])) <= 0 else np.inf
        return float(np.exp(x))

    bes = np.array([be(y, dl) for y, dl in d])
    out.attrs["break_even_q"] = tuple(
        float(v) for v in np.quantile(bes, [q[0], 0.5, q[1]], method="inverted_cdf"))
    out.attrs["break_even_outside"] = int((~np.isfinite(bes) | (bes == 0)).sum())
    return out
