"""Cost-penalised mean-variance optimizer with power-law impact cost (SPEC 3.6).

Per rebalance period:
max alpha_p'w - (lam/2) w' S_p w - half_spread'|w - w0| - cost_coef'|w - w0|^(1+delta)
s.t. sum(w) == 0, ||w||_1 <= gross_limit; alpha_p = alpha / rebalances, S_p = cov / rebalances.
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from impact_lab.capacity import CapacityParams, covariance
from impact_lab.laws import SqrtLaw


@dataclass(frozen=True)
class OptimizerParams:
    risk_aversion: float = 2.0
    gross_limit: float | None = 2.0       # sum |w| <= gross_limit; None = no constraint
    signal_persistence: float = 0.9       # AR(1) per rebalance period
    alpha_scale: float = 0.10             # annual alpha per unit signal per unit annual vol
    n_periods: int = 60
    burn_in: int = 10


def optimize(alpha: np.ndarray, cov: np.ndarray, w0: np.ndarray, cost_coef: np.ndarray,
             delta: float, half_spread: float | np.ndarray, p: OptimizerParams,
             rebalances: int = 12) -> np.ndarray:
    """Optimal weights for one period; alpha and cov are annual. half_spread is a scalar or one
    value per name. RuntimeError unless optimal."""
    n = len(alpha)
    w = cp.Variable(n)
    dw = w - w0
    obj = (alpha / rebalances @ w
           - p.risk_aversion / 2 * cp.quad_form(w, cp.psd_wrap(cov / rebalances))
           - cp.sum(cp.multiply(np.broadcast_to(half_spread, (n,)), cp.abs(dw)))
           - np.asarray(cost_coef, dtype=float) @ cp.power(cp.abs(dw), 1 + delta, approx=False))
    cons = [cp.sum(w) == 0]
    if p.gross_limit is not None:
        cons.append(cp.norm1(w) <= p.gross_limit)
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != "optimal":
        raise RuntimeError(f"optimizer status {prob.status}")
    return np.asarray(w.value, dtype=float)


def cost_coef(aum: float, universe: pd.DataFrame, fit_or_law: SqrtLaw) -> np.ndarray:
    """Y * sigma_i * (aum / adv_i)**delta: the cost of |dw_i| is cost_coef_i * |dw_i|**(1+delta)
    (one-day execution, so phi_i = |dw_i| * aum / adv_i)."""
    return (fit_or_law.Y * universe["sigma_d"].to_numpy()
            * (aum / universe["adv_usd"].to_numpy()) ** fit_or_law.delta)


def signal_path(seed: int, n_names: int, p: OptimizerParams) -> np.ndarray:
    """(n_periods, n_names) stationary AR(1) signal with unit variance."""
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(n_names)
    out = np.empty((p.n_periods, n_names))
    rho = p.signal_persistence
    for t in range(p.n_periods):
        s = rho * s + np.sqrt(1 - rho**2) * rng.standard_normal(n_names)
        out[t] = s
    return out


def reoptimised_capacity(aum_grid, universe: pd.DataFrame, cost_models: dict[str, SqrtLaw],
                         true_law: SqrtLaw, cap: CapacityParams, p: OptimizerParams, seed: int,
                         index_vol: float = 0.012) -> pd.DataFrame:
    """Size trades with each cost model, charge them with the true cost.

    net_alpha: annualised mean of alpha_p'w - true cost after burn-in. turnover: annualised mean
    of sum |dw| (traded value / AUM, as CapacityParams.turnover).
    """
    cov = covariance(universe, index_vol)
    sig = signal_path(seed, len(universe), p)
    vol_ann = universe["sigma_d"].to_numpy() * np.sqrt(252.0)
    hs = universe["half_spread"].to_numpy()
    rows = []
    for name, model in cost_models.items():
        for aum in np.asarray(aum_grid, dtype=float):
            c_model = cost_coef(aum, universe, model)
            c_true = cost_coef(aum, universe, true_law)
            w0 = np.zeros(len(universe))
            net, turn = [], []
            for t in range(p.n_periods):
                alpha = p.alpha_scale * sig[t] * vol_ann
                w = optimize(alpha, cov, w0, c_model, model.delta, hs, p, cap.rebalances)
                dw = np.abs(w - w0)
                net.append(alpha / cap.rebalances @ w - hs @ dw
                           - c_true @ dw ** (1 + true_law.delta))
                turn.append(dw.sum())
                w0 = w
            rows.append({"model": name, "aum": aum,
                         "net_alpha": cap.rebalances * float(np.mean(net[p.burn_in:])),
                         "turnover": cap.rebalances * float(np.mean(turn[p.burn_in:]))})
    return pd.DataFrame(rows)
