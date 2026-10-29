"""Power-law fit z = Y * phi**delta by NLS on levels, with HC0 sandwich covariance.

Single-order returns are negative about 40% of the time at realistic noise, so logs of single
orders are undefined and binned log-log OLS is biased; bins are for display only (SPEC 1.5).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import least_squares

from impact_lab.laws import SqrtLaw


@dataclass(frozen=True)
class PowerFit:
    Y: float
    delta: float
    cov: np.ndarray          # 2x2, order (Y, delta), HC0 sandwich
    n: int

    def se(self) -> tuple[float, float]:
        s = np.sqrt(np.diag(self.cov))
        return float(s[0]), float(s[1])

    def predict(self, phi) -> np.ndarray:
        return self.Y * np.asarray(phi, dtype=float) ** self.delta

    def level(self, phi_ref: float = 0.01) -> tuple[float, float]:
        """(Y * phi_ref**delta, delta-method SE): cost of a reference-size order."""
        lvl = self.Y * phi_ref**self.delta
        grad = np.array([phi_ref**self.delta, lvl * np.log(phi_ref)])
        return float(lvl), float(np.sqrt(grad @ self.cov @ grad))

    def draws(self, rng: np.random.Generator, k: int) -> np.ndarray:
        """(k, 2) draws of (Y, delta) from N(estimate, cov); Y clipped above 0."""
        d = rng.multivariate_normal([self.Y, self.delta], self.cov, size=k)
        d[:, 0] = np.clip(d[:, 0], 1e-12, None)
        return d

    def as_law(self) -> SqrtLaw:
        return SqrtLaw(Y=self.Y, delta=self.delta)


def fit_power(phi: np.ndarray, z: np.ndarray) -> PowerFit:
    """NLS of z on Y * phi**delta; bounds Y > 0, 0 < delta < 2."""
    lphi = np.log(np.asarray(phi, dtype=float))
    z = np.asarray(z, dtype=float)

    def resid(p: np.ndarray) -> np.ndarray:
        return p[0] * np.exp(p[1] * lphi) - z

    sol = least_squares(resid, x0=[0.5, 0.5], bounds=([0.0, 0.0], [np.inf, 2.0]),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    J, r = sol.jac, sol.fun
    bread = np.linalg.inv(J.T @ J)
    meat = (J * r[:, None] ** 2).T @ J
    return PowerFit(Y=float(sol.x[0]), delta=float(sol.x[1]), cov=bread @ meat @ bread,
                    n=int(z.size))


def binned_means(phi, z, n_bins: int = 20) -> pd.DataFrame:
    """Quantile bins in ln phi: phi_mid (geometric mean), z_mean, z_se, n. Display only."""
    lphi = np.log(np.asarray(phi, dtype=float))
    b = pd.qcut(lphi, n_bins, labels=False)
    g = pd.DataFrame({"lphi": lphi, "z": np.asarray(z, dtype=float), "b": b}).groupby("b")
    out = pd.DataFrame({
        "phi_mid": np.exp(g["lphi"].mean()),
        "z_mean": g["z"].mean(),
        "z_se": g["z"].std(ddof=1) / np.sqrt(g["z"].count()),
        "n": g["z"].count().astype(np.int64),
    })
    return out.reset_index(drop=True)


def calibration_regression(realised: np.ndarray, predicted: np.ndarray) -> pd.Series:
    """OLS realised ~ a + b * predicted with HC1 SEs, plus mean residual realised - predicted.

    Inputs in bps.
    """
    realised = np.asarray(realised, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    res = sm.OLS(realised, sm.add_constant(predicted)).fit(cov_type="HC1")
    return pd.Series({
        "a": float(res.params[0]),
        "b": float(res.params[1]),
        "se_a": float(res.bse[0]),
        "se_b": float(res.bse[1]),
        "mean_resid": float(np.mean(realised - predicted)),
        "n": float(realised.size),
    })
