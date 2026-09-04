"""The residual was forced to change sign, so its slope could never mean anything.

The basis `[1, 1/log X, 1/(log X)^2]` carries an intercept column, so
least-squares residuals are orthogonal to it and SUM TO ZERO by construction.
They must change sign; `log|residual|` must dive; and the fitted slope is decided
by whichever sample landed nearest the crossing. Structural, not incidental --
true of any intercepted fit, at any range or sample count.

The recorded run used five points against a three-column basis: two degrees of
freedom, then a two-parameter log-log fit applied to them. Its own observation
invited the reader to seek an analytic derivation of the fitted terms.

Found by the rh-research-engine-da session.
"""

from __future__ import annotations

import numpy as np
import pytest

from rh_research_engine.experiments.counterterm_discovery import (
    BASIS_SIZE,
    MINIMUM_POINTS,
    _residual,
    run,
)


def test_the_residuals_sum_to_zero_by_construction():
    """The mechanism, tested directly rather than inferred from a sign count.

    Any ys at all: the intercept column forces it, so this holds for data that
    has nothing to do with zeta.
    """
    xs = np.array([2.0e3, 5.0e3, 1.0e4, 2.0e4, 4.0e4, 8.0e4])
    for ys in (
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        np.array([0.7, -0.2, 0.5, -0.9, 0.1, 0.3]),
        np.geomspace(1e-3, 1e-1, 6),
    ):
        residual = _residual(xs, ys)
        assert abs(residual.sum()) < 1e-12 * max(1.0, float(np.abs(ys).max()))
        assert np.any(residual > 0) and np.any(residual < 0), (
            "a zero-sum residual that is not identically zero must change sign"
        )


def test_a_forced_sign_change_makes_the_slope_unreadable():
    result = run(X_min=2000, X_max=20000, points=6, phases=4)
    assert result.metrics["corrected_sign_changes"] > 0
    assert result.metrics["slope_unreadable"] == 1.0
    assert result.metrics["residual_sum_magnitude"] < 1e-12
    assert any("FORCED" in o for o in result.observations)
    assert any("UNREADABLE" in o for o in result.observations)


def test_the_degrees_of_freedom_are_reported_and_floored():
    """Five points against a three-column basis left two, and nobody said so."""
    assert MINIMUM_POINTS == BASIS_SIZE + 3
    with pytest.raises(ValueError, match="degrees of freedom"):
        run(points=5)
    result = run(X_min=2000, X_max=20000, points=6, phases=4)
    assert result.metrics["degrees_of_freedom"] == float(
        result.metrics["samples"] - BASIS_SIZE
    )


def test_the_invitation_to_derive_the_noise_is_withdrawn():
    """"A lower corrected slope is a signal to seek a derivation" stood here
    while the slope was structural noise."""
    result = run(X_min=2000, X_max=20000, points=6, phases=4)
    assert any("NOT a signal to seek an analytic derivation" in o for o in result.observations)


def test_the_counterterm_coefficients_survive_and_carry_an_rmse():
    """The fit itself is honest as a conjecture generator; only the slope was not."""
    result = run(X_min=2000, X_max=20000, points=6, phases=4)
    assert result.metrics["fit_rmse"] > 0
    assert np.isfinite(result.metrics["counterterm_constant"])
    for name, value in result.metrics.items():
        assert not (isinstance(value, float) and value != value), name
