import numpy as np
import pytest

from impact_lab.laws import CrossoverLaw, LogLaw, SqrtLaw, local_elasticity


def test_sqrt_value():
    assert SqrtLaw().impact(0.04, 1.0) == pytest.approx(0.1, abs=1e-12)


def test_crossover_limits():
    law = CrossoverLaw(eta_star=1e-3)
    phi = np.array([1e-3, 1e-2, 0.1])
    high = law.impact(phi, np.full(3, 1.0))
    np.testing.assert_allclose(high, SqrtLaw().impact(phi, 1.0), rtol=1e-3)
    eta = 1e-6
    low = law.impact(phi, np.full(3, eta))
    np.testing.assert_allclose(low, 0.5 * np.sqrt(phi) * np.sqrt(eta / 1e-3), rtol=1e-3)


def test_log_matches_sqrt_at_one_pct():
    assert LogLaw().impact(0.01, 1.0) == pytest.approx(0.05, rel=1e-3)


def test_local_elasticity():
    phi = np.logspace(-4, 0, 9)
    np.testing.assert_allclose(local_elasticity(SqrtLaw(delta=0.37), phi), 0.37, atol=1e-6)
    assert local_elasticity(LogLaw(), np.array([1e-6]))[0] > 0.99
    assert local_elasticity(LogLaw(), np.array([0.5]))[0] < 0.25


@pytest.mark.parametrize("make", [
    lambda: SqrtLaw(Y=0.0), lambda: SqrtLaw(delta=-0.1), lambda: CrossoverLaw(Y=-1.0),
    lambda: CrossoverLaw(eta_star=0.0), lambda: LogLaw(a=0.0), lambda: LogLaw(b=-1.0),
])
def test_constructors_reject_non_positive(make):
    with pytest.raises(ValueError):
        make()
