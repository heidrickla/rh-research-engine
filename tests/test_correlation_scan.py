"""The clamp reported Theta = 0.5 -- the RH endpoint -- from a refused input.

`correlation_scan` computed `0.5 + max(0.0, slope) / 2.0` inline. The same map
lives in `symbolic.exponents.screening_remainder_to_theta` and REFUSES a negative
exponent, because a Theta below 1/2 is not a stronger result but a provably
false one. The clamp turned every refusal into exactly 0.5, and both recorded
runs had negative slopes, so both recorded Theta = 0.5 beside a measured slope.
A reader sees the data landing on RH. It is the clamp.

Two implementations of one map, and the silent one wrote the record -- the
numeric form of this repository's one-name-resolution rule.

Found by the rh-research-engine-da session.
"""

from __future__ import annotations

import pytest

from rh_research_engine.experiments import correlation_scan
from rh_research_engine.experiments.correlation_scan import THETA_CEILING, _theta_from, run
from rh_research_engine.symbolic.exponents import (
    ImpossibleBoundError,
    screening_remainder_to_theta,
)


@pytest.mark.parametrize("slope", [-0.3216179776210677, -0.2422, -1e-9, -5.0])
def test_a_negative_slope_is_refused_rather_than_clamped(slope):
    """The exact recorded values are here: both real runs took this branch."""
    theta, reason = _theta_from(slope)
    assert theta is None, f"{slope} must be refused, not mapped"
    assert "REFUSED" in reason
    with pytest.raises(ImpossibleBoundError):
        screening_remainder_to_theta(slope)


def test_the_clamp_is_gone_at_the_value_it_produced():
    """0.5 is reachable only from exactly zero, never from anything below it."""
    theta, _ = _theta_from(0.0)
    assert theta == 0.5
    for slope in (-1e-12, -0.001, -0.5):
        assert _theta_from(slope)[0] is None


def test_theta_has_a_ceiling_as_well_as_a_floor():
    """Theta <= 1 unconditionally, and the inline map had no ceiling.

    Phase noise pushing the slope positive produced 2.789 and 4.442 in testing,
    and the shared function's guard only checks below THETA_FLOOR.
    """
    assert _theta_from(0.9)[0] == pytest.approx(0.95)
    for slope in (1.01, 4.578, 7.885):
        theta, reason = _theta_from(slope)
        assert theta is None, slope
        assert "ceiling" in reason
    assert screening_remainder_to_theta(7.885).theta_upper > THETA_CEILING, (
        "the shared function still passes it, which is why the ceiling lives here"
    )


def test_the_shared_map_is_imported_rather_than_reimplemented():
    """A second implementation is how the clamp got in, so name the dependency.

    Checked ON THE SYNTAX TREE, not on the source text. The first version of
    this test searched for the substring `0.5 + max(` and failed against the
    module's own docstring, which quotes the clamp in order to explain it -- a
    check on the notation rather than on the code, which is the failure this
    repository keeps rediscovering.
    """
    import ast
    import inspect

    assert correlation_scan.screening_remainder_to_theta is screening_remainder_to_theta

    tree = ast.parse(inspect.getsource(correlation_scan))
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
            continue
        left = node.left
        if isinstance(left, ast.Constant) and left.value == 0.5:
            raise AssertionError(
                f"line {node.lineno} recomputes `0.5 + ...`: the Theta map has one "
                "implementation, in symbolic.exponents"
            )


def test_an_unreadable_slope_yields_no_theta_at_all():
    """`metrics` is what downstream reads, so a caveat in prose is not enough.

    The energy slope is a cancellation residual and moves by whole units on
    grid phase alone. A first repair here mapped it anyway and printed the
    result beside a note saying it was not a bound.
    """
    result = run(X_min=1000, X_max=5000, points=5, phases=4)
    assert result.metrics["energy_slope_spread"] > 0.05
    assert result.metrics["theta_refused"] == 1.0
    assert "heuristic_theta_from_energy" not in result.metrics
    assert any("REFUSED" in o for o in result.observations)


def test_the_energy_slope_is_reported_unreadable_and_the_remainder_is_not():
    """The contrast is the finding: one of the three is a real number."""
    result = run(X_min=1000, X_max=5000, points=5, phases=4)
    assert result.metrics["energy_dynamic_range"] < 1e-2
    assert result.metrics["remainder_dynamic_range"] > 0.5
    assert result.metrics["remainder_slope_spread"] < 0.05
    assert result.metrics["unreadable_slopes"] >= 1
    assert any("UNREADABLE" in o and "energy" in o for o in result.observations)
    assert any("Readable" in o and "remainder" in o for o in result.observations)


def test_no_metric_is_nan():
    """NaN serialises to JSON null and fails validation on the way back in.

    Four metrics here were written as `float("nan")` on the empty branch, so a
    run with too few usable points recorded something unreadable.
    """
    result = run(X_min=1000, X_max=5000, points=5, phases=4)
    for name, value in result.metrics.items():
        assert not (isinstance(value, float) and value != value), name


def test_too_few_points_is_refused():
    with pytest.raises(ValueError, match="at least 3"):
        run(points=2)
