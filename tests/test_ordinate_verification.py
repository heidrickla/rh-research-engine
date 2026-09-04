"""Stored ordinates are read back from a disk on a machine with no ECC.

`verify_ordinates` evaluates `Z` at each one and compares the residual to what
that ordinate's own float64 precision allows. These tests break it on purpose:
a check that has only ever been shown passing has not been shown to work.

The interesting cases are the two edges. It must catch a displacement that
changes the value, and it must NOT flag one that does not -- an ordinate a few
ulps from the zero is the zero in float64, and flagging it would be a statement
about the representation rather than about the data.
"""

from __future__ import annotations

import numpy as np
import pytest

from rh_research_engine.symbolic.riemann_siegel import (
    ORDINATE_ULP_TOLERANCE,
    first_zero_ordinates,
    verify_ordinates,
)

COUNT = 2_000


@pytest.fixture(scope="module")
def ordinates() -> np.ndarray:
    return first_zero_ordinates(COUNT)


def _displace(values: np.ndarray, index: int, ulps: float) -> np.ndarray:
    moved = values.copy()
    moved[index] = moved[index] + ulps * np.spacing(moved[index])
    return moved


def test_genuine_ordinates_pass_with_room_to_spare(ordinates):
    """If the clean case sat near the threshold, the threshold would be a coin toss."""
    check = verify_ordinates(ordinates)
    assert check.ok
    assert check.count == COUNT
    assert check.failed.size == 0
    # Measured at 22.1 over the first 20,000; the margin is the point.
    assert check.worst < ORDINATE_ULP_TOLERANCE / 2, (
        f"worst clean ordinate is {check.worst:.1f} against a tolerance of "
        f"{ORDINATE_ULP_TOLERANCE} -- the margin has gone"
    )


@pytest.mark.parametrize("ulps", [2048, 4096, 2**16, 2**20])
def test_a_displaced_ordinate_is_caught_and_named(ordinates, ulps):
    """And the failure names WHICH one, or the report cannot be acted on.

    The smallest case is twice the threshold, not the threshold itself, which
    lands ON the boundary -- see the test below. Asserting a catch at exactly
    the tolerance would claim a guarantee better than the one measured.
    """
    check = verify_ordinates(_displace(ordinates, 777, ulps))
    assert not check.ok
    assert list(check.failed) == [777], (
        "the corrupted ordinate must be the only one flagged"
    )


def test_exactly_at_the_tolerance_is_a_coin_toss_and_is_recorded_as_one(ordinates):
    """A displacement of exactly `tolerance` ulps gives a ratio of about
    `tolerance`, so whether it trips depends on the local `|Z'|`. Measured over
    20,000 zeros it caught 49.3%. That is the honest boundary, and writing it
    down stops the next reader assuming the guarantee starts one bit lower.
    """
    indices = np.arange(0, 400, 4)
    moved = ordinates.copy()
    moved[indices] += ORDINATE_ULP_TOLERANCE * np.spacing(moved[indices])
    caught = verify_ordinates(moved).failed.size
    assert 10 < caught < len(indices) - 10, (
        f"{caught}/{len(indices)} caught at exactly the tolerance -- expected "
        "roughly half; if this is now all or nothing the bound has moved"
    )


@pytest.mark.parametrize("ulps", [1, 2, 64, 167])
def test_a_displacement_below_the_tolerance_is_deliberately_not_flagged(ordinates, ulps):
    """Not a gap. A value this close IS the zero to float64.

    A gate that failed here would fail on the same data recomputed by a build
    with different rounding, which is a statement about the compiler.
    """
    check = verify_ordinates(_displace(ordinates, 777, ulps))
    assert check.ok


@pytest.mark.parametrize("bit", [11, 12, 20, 32])
def test_every_mantissa_bit_that_matters_is_caught_everywhere(ordinates, bit):
    """Flipping bit k moves the ordinate by 2**k ulps, so the ratio is ~2**k."""
    flipped = (ordinates.view(np.int64) ^ (np.int64(1) << bit)).view(np.float64)
    check = verify_ordinates(flipped)
    assert check.failed.size == check.count, (
        f"bit {bit} moves an ordinate by {2**bit} ulps and every one should fail"
    )


def test_the_tolerance_could_not_have_been_a_constant(ordinates):
    """The residual a genuine zero produces grows with height. A fixed bound
    generous at the bottom is impossibly tight at the top, and vice versa --
    which is why the allowance is derived per ordinate from `|Z'| * ulp`.
    """
    low, high = ordinates[:200], ordinates[-200:]
    from rh_research_engine.symbolic.riemann_siegel import z_function

    raw_low = float(np.median(np.abs(z_function(low))))
    raw_high = float(np.median(np.abs(z_function(high))))
    assert raw_high > 5 * raw_low, (
        "if the raw residual did not grow with height, a constant would do"
    )
    # ...while the normalised ratio does not drift anything like as much.
    check = verify_ordinates(ordinates)
    ratio_low = float(np.median(check.ratio[:200]))
    ratio_high = float(np.median(check.ratio[-200:]))
    assert 0.2 < ratio_high / ratio_low < 5.0


def test_the_check_says_nothing_about_completeness(ordinates):
    """Every ordinate present can be genuine while zeros are missing between
    them. `ZeroCount` answers that; this must not be read as having done so.
    """
    with_a_hole = np.delete(ordinates, slice(500, 600))
    check = verify_ordinates(with_a_hole)
    assert check.ok
    assert check.count == COUNT - 100


def test_an_empty_set_is_not_a_passing_set_by_accident(ordinates):
    check = verify_ordinates(np.empty(0))
    assert check.count == 0
    assert check.ok
    assert check.worst == 0.0


def test_a_value_that_is_not_a_zero_at_all_is_caught(ordinates):
    """The blunt case: a midpoint between two zeros, where |Z| is at its largest."""
    midpoints = (ordinates[:-1] + ordinates[1:]) / 2
    check = verify_ordinates(midpoints)
    assert check.failed.size == check.count
