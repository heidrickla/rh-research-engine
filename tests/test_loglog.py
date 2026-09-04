"""The predicate has to fire on the case it was built for, at the value it has.

`DYNAMIC_RANGE_LIMIT` was first written as 1e-3. The cancellation minimum it
exists to detect measures 9.4e-6 against a median of 1.1e-3 -- a ratio of
8.6e-3, which 1e-3 would have passed. A guard tuned tighter than its own
motivating example is decoration, so the motivating example is a test.

The cancellation-minimum case, and this predicate, came from the
rh-research-engine-da session, against a sign-change check that had already been
written, shipped and believed.
"""

from __future__ import annotations

import math

import pytest

from rh_research_engine.math.loglog import (
    DYNAMIC_RANGE_LIMIT,
    SPREAD_LIMIT,
    dynamic_range,
    geometric_phases,
    summarize,
)


def test_the_cancellation_minimum_fires_at_the_measured_ratio():
    """The exact numbers that set the limit, so it cannot drift past them."""
    values = [1.1e-3, 1.2e-3, 9.4e-6, 1.0e-3, 1.3e-3]
    ratio, index = dynamic_range(values)
    assert index == 2
    assert abs(ratio - 9.4e-6 / 1.1e-3) < 1e-9
    assert ratio < DYNAMIC_RANGE_LIMIT, "the motivating case must fire"
    assert ratio > 1e-3, (
        "and it must fire at 1e-2 specifically: this ratio passes a 1e-3 limit, "
        "which is what the first version of this constant would have done"
    )


def test_the_surviving_fit_does_not_fire():
    """`screening_remainder` sits near 0.99, two orders clear of the limit."""
    ratio, _ = dynamic_range([1.38, 1.39, 1.38, 1.40, 1.38])
    assert ratio > 0.9
    assert ratio > DYNAMIC_RANGE_LIMIT * 10


def test_a_sign_change_drives_the_ratio_down_too():
    """Which is why the ratio subsumes the sign-change count rather than joining it."""
    values = [1.0, 0.4, -1e-8, -0.5, -1.0]
    ratio, index = dynamic_range(values)
    assert index == 2
    assert ratio < DYNAMIC_RANGE_LIMIT


def test_the_median_is_not_the_mean():
    """The outlier would otherwise drag the scale it is measured against."""
    ratio, _ = dynamic_range([1e-9, 1.0, 1.0, 1.0, 1e6])
    # A mean-based ratio would be 1e-9/200000 and would exaggerate; the median
    # keeps the comparison against a typical sample.
    assert abs(ratio - 1e-9) < 1e-15


def test_an_empty_or_all_zero_sample_does_not_divide_by_zero():
    assert dynamic_range([]) == (0.0, -1)
    ratio, index = dynamic_range([0.0, 0.0])
    assert ratio == 0.0 and index == 0


def test_a_clean_power_law_is_readable():
    grids = geometric_phases(10.0, 1000.0, 8, 4)
    fit = summarize([(grid, [x**0.75 for x in grid]) for grid in grids])
    assert abs(fit.slope - 0.75) < 1e-9
    assert fit.spread < 1e-12
    assert fit.sign_changes == 0
    assert fit.dynamic_range > 0.01
    assert fit.unreadable_because == []


def test_each_reason_is_reported_separately():
    """A bare "unreadable" is another verdict nobody can act on."""
    grids = geometric_phases(10.0, 1000.0, 8, 4)
    fit = summarize([(grid, [math.cos(9.0 * math.log(x)) for x in grid]) for grid in grids])
    reasons = fit.unreadable_because
    assert reasons, "an oscillation with crossings must be unreadable"
    assert any("grid-phase spread" in r for r in reasons) or any(
        "dynamic range" in r for r in reasons
    )
    assert all(isinstance(r, str) and len(r) > 20 for r in reasons)


def test_the_spread_limit_separates_the_two_real_cases():
    """-0.0093 +/- 0.0014 survives; the energy slope's spread of 2.03 does not."""
    assert 0.0014 < SPREAD_LIMIT < 2.03


def test_one_grid_cannot_show_grid_dependence():
    with pytest.raises(ValueError, match="at least two grids"):
        summarize([([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])])
    with pytest.raises(ValueError, match="phases must be >= 2"):
        geometric_phases(10.0, 100.0, 5, 1)
    with pytest.raises(ValueError, match="points must be >= 3"):
        geometric_phases(10.0, 100.0, 2, 4)


def test_dropped_points_are_counted_not_skipped():
    grids = [
        ([1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 3.0, 4.0]),
        ([1.1, 2.1, 3.1, 4.1], [1.0, 2.0, 3.0, 4.0]),
    ]
    fit = summarize(grids)
    assert fit.dropped == 1


def test_the_phases_are_distinct_and_stay_in_range():
    grids = geometric_phases(100.0, 3000.0, 6, 4)
    assert len({tuple(g) for g in grids}) == 4
    for grid in grids:
        assert grid == sorted(grid)
        assert grid[0] >= 100.0
    # The integral variant may collide two offsets onto one abscissa at small X,
    # so it deduplicates; the caller sees a shorter grid rather than a repeat.
    integral = geometric_phases(2.0, 12.0, 6, 4, integral=True)
    for grid in integral:
        assert len(set(grid)) == len(grid)
        assert all(float(x).is_integer() for x in grid)


def test_readable_and_sign_determined_are_different_questions():
    """The gap a falsification test falls through.

    `baez_duarte`'s envelope slope: spread 0.019 against a limit of 0.05, so it
    is readable -- and the range straddles zero, so its sign is not determined.
    A caller that checked only readability and then tested `slope > 0` would
    announce a refutation of RH on 5 of 18 binnings.
    """
    from rh_research_engine.math.loglog import LogLogSummary

    straddling = LogLogSummary(
        slope=-0.01105, intercept=0.0, spread=0.01917,
        lowest=-0.05738, highest=0.02824,
        sign_changes=0, dynamic_range=0.9, dropped=0, phases=18,
    )
    assert straddling.unreadable_because == [], "the magnitude is quotable"
    assert not straddling.sign_determined, "the sign is not"

    for low, high in ((0.01, 0.05), (-0.05, -0.01)):
        agreed = LogLogSummary(
            slope=0.5 * (low + high), intercept=0.0, spread=0.001,
            lowest=low, highest=high,
            sign_changes=0, dynamic_range=0.9, dropped=0, phases=18,
        )
        assert agreed.sign_determined, (low, high)

    touching = LogLogSummary(
        slope=0.01, intercept=0.0, spread=0.001, lowest=0.0, highest=0.02,
        sign_changes=0, dynamic_range=0.9, dropped=0, phases=18,
    )
    assert not touching.sign_determined, "a bound of exactly zero is not a sign"

