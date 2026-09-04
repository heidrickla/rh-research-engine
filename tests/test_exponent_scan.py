"""An estimator that has not been shown to return nothing on nothing.

`exponent_scan` had no tests at all, and reported a single fitted exponent with
no error bar. Over ten grids differing only in phase, inside the same range, it
returns exponents from 0.214 to 1.061 -- and its own observation explains that
theta bounds Re(rho), so any draw below 1/2 reads as evidence for something the
zeros on the critical line refute.

The signal is injectable so these controls can exist. A fitted exponent nobody
has watched recover a known one, or return nothing on a signal with no trend, is
a number with no scale attached.
"""

from __future__ import annotations

import math

import pytest

from rh_research_engine.experiments.exponent_scan import run
from rh_research_engine.math.loglog import geometric_phases, summarize


def test_a_pure_power_comes_back_with_its_own_exponent():
    """The positive control. Without it, no draw here has a known scale."""
    for exponent in (0.25, 0.5, 0.75, 1.5):
        result = run(signal=lambda x, e=exponent: x**e)
        assert abs(result.metrics["fitted_theta"] - exponent) < 1e-9, exponent
        assert result.metrics["theta_grid_spread"] < 1e-9, "a power law has no grid dependence"
        assert result.metrics["sign_changes"] == 0.0


def test_an_oscillation_with_no_trend_reports_no_exponent():
    """The negative control: noise has no exponent, and the bar must show it.

    A signal with a flat envelope has a true exponent of zero. What matters is
    not that the mean lands near zero but that the spread is wide enough to say
    the run constrains nothing -- which is exactly the state the real S_q is in.
    """
    result = run(signal=lambda x: math.cos(20.0 * math.log(x)))
    assert result.metrics["sign_changes"] > 0
    spread = result.metrics["theta_grid_spread"]
    assert spread > 0.05, spread
    assert result.metrics["unreadable"] == 1.0
    low, high = result.metrics["theta_min"], result.metrics["theta_max"]
    assert low < 0.0 < high or spread > abs(result.metrics["fitted_theta"]), (
        "the spread must be able to cover zero on a signal that has no trend"
    )
    assert any("constrains nothing" in o for o in result.observations)


def test_an_oscillating_power_law_shows_both_the_exponent_and_the_damage():
    """The realistic case: a real envelope exponent under a sign-changing factor.

    The mean can still land near the envelope's exponent while individual grids
    are far off. The spread is what says so, and the single-grid version of this
    experiment could not report it.
    """
    result = run(signal=lambda x: x**0.75 * math.cos(20.0 * math.log(x)))
    assert result.metrics["sign_changes"] > 0
    assert result.metrics["theta_grid_spread"] > 0.05
    assert result.metrics["theta_max"] - result.metrics["theta_min"] > 0.1


def test_the_grid_phase_actually_changes_the_grid():
    """Otherwise the spread is zero for a reason that has nothing to do with S_q.

    Averaging over a phase that is ignored would report spread 0.0 on the real
    signal and look like a stable measurement -- the failure mode this whole
    module exists to stop.
    """
    def signal(x: float) -> float:
        return math.cos(20.0 * math.log(x))

    grids = geometric_phases(100.0, 3000.0, 10, 4)
    assert grids[0] != grids[1], "the phases must produce different abscissae"
    fit = summarize([(grid, [signal(x) for x in grid]) for grid in grids])
    assert fit.spread > 1e-6, fit.spread


def test_zero_valued_points_are_dropped_and_counted():
    """A silent drop reports a fit over a narrower set than the parameters name."""
    def signal(x: float) -> float:
        return 0.0 if 800.0 < x < 900.0 else x**0.5

    result = run(signal=signal)
    assert result.metrics["points_dropped"] >= 1
    assert any("dropped" in o for o in result.observations)


def test_a_single_grid_is_refused():
    """One grid cannot show the grid dependence, so it may not be requested."""
    with pytest.raises(ValueError, match="phases must be >= 2"):
        run(phases=1, signal=lambda x: x**0.5)
    with pytest.raises(ValueError, match="points must be >= 3"):
        run(points=2, signal=lambda x: x**0.5)


def test_too_few_usable_points_is_refused_rather_than_fitted():
    """Three points is the minimum a slope and intercept can be read from."""
    with pytest.raises(ValueError, match="usable points"):
        summarize([([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])] * 2)


def test_the_real_signal_is_reported_as_constraining_nothing():
    """The finding, kept as a test so it cannot quietly stop being true.

    If S_q's fitted exponent ever becomes stable under grid phase, that is a
    result worth noticing -- and this failing is how it would be noticed.
    """
    result = run()
    assert result.metrics["sign_changes"] >= 1
    assert result.metrics["theta_grid_spread"] > 0.05
    assert result.metrics["unreadable"] == 1.0
    assert any("constrains nothing" in o for o in result.observations)
    assert any("must not be read as evidence" in o for o in result.observations)
