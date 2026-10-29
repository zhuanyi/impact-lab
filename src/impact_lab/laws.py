"""Metaorder impact laws in units of daily volatility: impact(phi, eta) -> I / sigma_d.

phi = Q / V_D (size as a fraction of daily volume), eta = phi / T (participation over a window
holding a fraction T of the day's volume).
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ImpactLaw(Protocol):
    def impact(self, phi: np.ndarray, eta: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class SqrtLaw:
    """impact = Y * phi**delta; eta is ignored (duration-free)."""

    Y: float = 0.5
    delta: float = 0.5

    def __post_init__(self) -> None:
        if self.Y <= 0:
            raise ValueError("Y must be > 0")
        if self.delta <= 0:
            raise ValueError("delta must be > 0")

    def impact(self, phi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi, dtype=float)
        return self.Y * phi**self.delta * np.ones_like(np.asarray(eta, dtype=float))


@dataclass(frozen=True)
class CrossoverLaw:
    """Linear-to-square-root crossover in participation (limits of Bucci et al. 2019 eq. 5).

    impact = Y * sqrt(phi) * sqrt(u / (1 + u)), u = eta / eta_star.
    eta >> eta_star: Y sqrt(phi); eta << eta_star: Y phi / sqrt(T eta_star), linear in phi at
    fixed T. The interpolation between the two limits is a modelling choice.
    """

    Y: float = 0.5
    eta_star: float = 1e-3

    def __post_init__(self) -> None:
        if self.Y <= 0:
            raise ValueError("Y must be > 0")
        if self.eta_star <= 0:
            raise ValueError("eta_star must be > 0")

    def impact(self, phi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi, dtype=float)
        u = np.asarray(eta, dtype=float) / self.eta_star
        return self.Y * np.sqrt(phi) * np.sqrt(u / (1.0 + u))


@dataclass(frozen=True)
class LogLaw:
    """impact = a * log10(1 + b * phi); eta is ignored.

    b from Zarinelli et al. (2015) s4.1. a = 0.05 / log10(1 + 4.65) matches SqrtLaw() at 1% ADV.
    """

    a: float = 0.0665
    b: float = 465.0

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError("a must be > 0")
        if self.b <= 0:
            raise ValueError("b must be > 0")

    def impact(self, phi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi, dtype=float)
        return self.a * np.log10(1.0 + self.b * phi) * np.ones_like(np.asarray(eta, dtype=float))


def local_elasticity(law: ImpactLaw, phi: np.ndarray, T: float = 0.25,
                     h: float = 1e-4) -> np.ndarray:
    """d ln impact / d ln phi at fixed T (eta = phi / T), central difference in ln phi."""
    phi = np.asarray(phi, dtype=float)
    up, dn = phi * np.exp(h), phi * np.exp(-h)
    return (np.log(law.impact(up, up / T)) - np.log(law.impact(dn, dn / T))) / (2.0 * h)
