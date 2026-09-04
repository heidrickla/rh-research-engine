"""Bulk computation of the Z function and the zeros of zeta.

The point of this module is speed, and the only thing that makes speed
admissible is that the answers are the same. So every check here is a
comparison against something independent: `mpmath.siegelz` for the function,
`mpmath.zetazero` for the ordinates, and Turing's method -- which counts the
zeros without locating any of them -- for whether the list is complete.
"""

from __future__ import annotations

import mpmath
import numpy as np
import pytest
import sympy as sp

from rh_research_engine.symbolic.functions import ZeroCount
from rh_research_engine.symbolic.riemann_siegel import (
    CROSSOVER,
    SAME_ZERO,
    SEARCH_CROSSOVER,
    first_zero_ordinates,
    theta,
    z_function,
    zero_ordinates,
)


def test_theta_agrees_with_mpmath():
    """The phase has to be right or every zero is in the wrong place.

    Compared RELATIVELY, because theta grows like `t log t`: at t = 50000 it is
    about 2e5, where float64's own spacing is 3e-11. An absolute threshold here
    tests the width of a double, not this code.
    """
    for height in (20.0, 100.0, 1000.0, 50000.0, 10**6):
        mine = float(theta(height))
        reference = float(mpmath.siegeltheta(height))
        assert abs(mine - reference) <= 4e-16 * abs(reference) + 1e-12, height


def test_the_phase_is_what_bounds_the_accuracy_at_height():
    """Z's error grows with the height, and this measures why.

    `theta` enters Z inside a cosine, so an absolute error in the phase is an
    absolute error in Z. Theta grows like `t log t`, so float64's spacing AT THE
    PHASE -- not the truncation of any series -- is the floor.

    Asserting that the spacing grows is not the same as showing it is the
    limit, so this compares the two: the error tracks the spacing within a small
    factor over three decades, which a series-truncation floor would not do.

    This used to end "past roughly 1e8 the phase would need carrying in
    extended precision". Retracted: that rests on a phase budget nothing
    states, and the quantity that matters is whether a zero POSITION moves
    enough to shift a statistic -- measured at about 1e-2, against an actual
    position error of 1.16e-10. See `theta`'s own docstring.
    """
    generator = np.random.default_rng(11)
    for low, high in ((1e5, 1e6), (1e6, 1e7), (1e7, 1e8)):
        samples = generator.uniform(low, high, 12)
        reference = np.array([float(mpmath.siegelz(float(point))) for point in samples])
        error = np.abs(z_function(samples) - reference).max()
        spacing = float(np.spacing(float(mpmath.siegeltheta(high))))
        assert error < 40 * spacing, (low, high, error, spacing)
        assert error > spacing / 40, (
            "an error far BELOW the phase resolution would mean the phase is "
            "not the limit and this docstring is wrong",
            (low, high, error, spacing),
        )


@pytest.mark.parametrize(
    ("low", "high", "tolerance"),
    [
        (10.0, 400.0, 1e-11),
        (400.0, 2000.0, 1e-10),
        (2000.0, 10000.0, 1e-8),
        (10000.0, 100000.0, 1e-8),
    ],
)
def test_z_agrees_with_mpmath_across_the_bands(low, high, tolerance):
    """Including at the heights where the published coefficients blow up.

    `Psi` has removable singularities at `p = 1/4` and `p = 3/4`, and its
    twelfth derivative near one of those is a difference of enormous numbers:
    evaluated from the symbolic derivatives, the error at these very points
    reached 3e12. They are put in the sample deliberately.
    """
    generator = np.random.default_rng(7)
    samples = generator.uniform(low, high, 40)
    scale = np.sqrt(samples / (2 * np.pi))
    samples = np.concatenate(
        [
            samples,
            2 * np.pi * (np.floor(scale) + 0.25) ** 2,
            2 * np.pi * (np.floor(scale) + 0.75) ** 2,
        ]
    )
    samples = samples[(samples >= low) & (samples < high)]
    assert len(samples) > 40, "the band should have been sampled"

    reference = np.array([float(mpmath.siegelz(float(point))) for point in samples])
    error = np.abs(z_function(samples) - reference)
    assert error.max() < tolerance, f"worst {error.max():.3e}"


def test_the_search_crossover_is_good_enough_to_bracket_but_not_to_report():
    """Why there are two crossovers rather than one.

    Below `SEARCH_CROSSOVER` both methods are used; between it and `CROSSOVER`
    the cheap path is Riemann-Siegel and the accurate path is Euler-Maclaurin,
    and the difference between them is the reason the located roots get
    polished. If the cheap path were as good, the polish would be dead code.
    """
    assert SEARCH_CROSSOVER < CROSSOVER
    band = np.linspace(SEARCH_CROSSOVER + 10, CROSSOVER - 10, 60)
    cheap = z_function(band, crossover=SEARCH_CROSSOVER)
    accurate = z_function(band, crossover=CROSSOVER)
    gap = np.abs(cheap - accurate)
    # Good enough to decide a sign at a bracket end, not good enough to report.
    assert gap.max() < 1e-5
    assert gap.max() > 1e-13, "the polish step would be pointless"


@pytest.mark.parametrize("height", [100.0, 500.0, 1420.0, 5000.0])
def test_the_located_zeros_are_exactly_as_many_as_turing_counts(height):
    """Completeness is checked, not assumed.

    A grid spaced by a fraction of the AVERAGE gap cannot see a pair closer
    than the grid, and the first version of this missed four zeros in ten
    thousand -- silently. `ZeroCount` locates none of them, so it is an
    independent answer to how many there are.
    """
    found = zero_ordinates(height)
    assert len(found) == int(ZeroCount(sp.Float(height)))
    assert np.all(np.diff(found) > SAME_ZERO), "no zero counted twice"
    assert found[-1] <= height


def test_a_short_list_raises_rather_than_being_returned():
    """The one outcome that cannot be allowed.

    Every downstream sum over zeros would be missing a term and would still
    look like an answer, so an incomplete list must not be returnable. Forced
    by refusing the passes that would have repaired it.
    """
    with pytest.raises(RuntimeError, match="ZeroCount says"):
        zero_ordinates(1420.0, max_passes=0)


def test_the_ordinates_agree_with_mpmath():
    """Located, not assumed -- and the same zeros mpmath finds.

    Checked at every index rather than a sample, over a range small enough that
    the mpmath side is affordable: `mpmath.zetazero` costs about 160 ms each,
    which is the reason this module exists.
    """
    found = first_zero_ordinates(40)
    worst = 0.0
    for index in range(1, 41):
        reference = float(mpmath.im(mpmath.zetazero(index)))
        worst = max(worst, abs(found[index - 1] - reference))
    assert worst < 1e-10, f"worst {worst:.3e}"


def test_the_first_zero_is_the_first_zero():
    assert abs(first_zero_ordinates(1)[0] - 14.134725141734693) < 1e-11


def test_z_vanishes_at_every_located_ordinate():
    """The cheap completeness check, over far more zeros than mpmath affords."""
    found = zero_ordinates(2000.0)
    assert len(found) > 1300
    assert np.abs(z_function(found)).max() < 1e-8


def test_asking_for_more_zeros_than_the_first_guess_reaches_still_works():
    """The height needed is estimated, so the estimate has to be checked."""
    found = first_zero_ordinates(600)
    assert len(found) == 600
    assert np.all(np.diff(found) > 0)
    reference = float(mpmath.im(mpmath.zetazero(600)))
    assert abs(found[-1] - reference) < 1e-9


def test_the_seeded_root_find_returns_exactly_what_zetazero_returns():
    """`NthZetaZero` is seeded from the bulk scan instead of searching blind.

    That is only admissible if the value is the same value, so it is compared
    bit for bit rather than to a tolerance. The seed is checked inside
    `_locate_zero` too -- a root-find given a starting point can converge to
    the neighbouring zero, and a list with one ordinate duplicated and one
    missing would still be the right length.
    """
    from rh_research_engine.symbolic.functions import NthZetaZero

    for index in (1, 2, 3, 17, 40):
        mine = complex(NthZetaZero(sp.Integer(index)))
        assert mine == complex(mpmath.zetazero(index)), index
        assert mine.real == 0.5


def test_the_seeded_root_find_reaches_the_working_precision():
    """A refinement that stops at the seed's accuracy has refined nothing.

    The seed is float64. With `findroot`'s default tolerance the secant
    iteration declared victory as soon as it matched the seed, so at forty
    digits it returned a value agreeing with `zetazero` in sixteen of them --
    while the cache was recording that it had been computed at forty. The
    tolerance is tied to the working precision now, and this is the check that
    it actually is.
    """
    from rh_research_engine.symbolic.functions import NthZetaZero

    original = mpmath.mp.dps
    try:
        for digits in (15, 30, 40):
            mpmath.mp.dps = digits
            # SymPy memoises expression construction, so without this the
            # object built at fifteen digits comes straight back and the guard
            # inside `_locate_zero` is never reached. Documented there.
            sp.core.cache.clear_cache()
            mine = sp.im(NthZetaZero(sp.Integer(1)))
            reference = mpmath.im(mpmath.zetazero(1))
            difference = abs(mpmath.mpf(str(sp.N(mine, digits + 5))) - reference)
            assert difference < mpmath.mpf(10) ** (-(digits - 2)), (
                digits,
                difference,
            )
    finally:
        mpmath.mp.dps = original
        sp.core.cache.clear_cache()


def test_sympy_caching_is_what_hides_a_precision_change():
    """The behaviour above, isolated, so it is a known fact and not a surprise.

    `NthZetaZero(1)` at fifteen digits and the same call at forty return the
    IDENTICAL object, because SymPy memoises expression construction. The
    precision guard inside `_locate_zero` is real and protects direct callers;
    through the symbolic layer it cannot be reached without clearing the cache.
    """
    from rh_research_engine.symbolic.functions import NthZetaZero

    original = mpmath.mp.dps
    try:
        mpmath.mp.dps = 15
        sp.core.cache.clear_cache()
        first = NthZetaZero(sp.Integer(1))
        mpmath.mp.dps = 40
        assert NthZetaZero(sp.Integer(1)) is first, "SymPy stopped memoising"
        sp.core.cache.clear_cache()
        assert NthZetaZero(sp.Integer(1)) is not first
    finally:
        mpmath.mp.dps = original
        sp.core.cache.clear_cache()


def test_gram_points_are_where_theta_is_a_multiple_of_pi():
    """`theta(g_n) = n pi`, and mpmath agrees."""
    from rh_research_engine.symbolic.riemann_siegel import gram_points

    indices = np.arange(0, 40)
    points = gram_points(indices)
    assert np.all(np.diff(points) > 0), "Gram points are increasing"
    for index, point in zip(indices, points, strict=True):
        assert abs(float(theta(point)) / np.pi - index) < 1e-11, index
        assert abs(point - float(mpmath.grampoint(int(index)))) < 1e-9, index


def test_the_main_sum_does_not_depend_on_the_order_of_its_input():
    """The sorted path takes a SLICE where the general one takes a mask.

    `count >= n` is a contiguous suffix wherever the heights are sorted, and a
    slice is a view where a boolean mask copies three arrays per term -- 126 of
    them at height 10^5. The optimisation is only admissible if the answer is
    identical, and an unsorted array taking the sliced path would silently get
    the wrong terms, which reads as a bad zero rather than a bad index.
    """
    generator = np.random.default_rng(3)
    ascending = np.sort(generator.uniform(3000.0, 90000.0, 400))
    forwards = z_function(ascending)
    backwards = z_function(ascending[::-1])[::-1]
    assert np.allclose(forwards, backwards, rtol=0, atol=1e-12)

    shuffled = ascending.copy()
    generator.shuffle(shuffled)
    order = np.argsort(shuffled)
    assert np.allclose(z_function(shuffled)[order], forwards, rtol=0, atol=1e-12)


def test_the_scan_reaches_a_height_a_uniform_grid_could_not():
    """Bracketing by Gram points is one Z evaluation per zero.

    A grid fine enough not to step over a close pair spends two dozen, which at
    height 10^5 is 3.7 million points and did not finish in ten minutes. This
    height is kept modest for the suite; the same path reaches 466659 zeros
    below 3 x 10^5 in about four minutes.
    """
    found = zero_ordinates(20000.0)
    assert len(found) == int(ZeroCount(sp.Float(20000.0)))
    assert np.all(np.diff(found) > SAME_ZERO)
    assert np.abs(z_function(found)).max() < 1e-7


def test_theta_is_exact_below_where_the_series_is_not():
    """The asymptotic expansion is not the definition, and below t = 20 it shows.

    The corpus records `theta(t) = -t log(pi)/2 + arg Gamma(it/2 + 1/4)`, exact
    everywhere. The implementation used the asymptotic series, which is wrong
    by 2.1e-2 at t = 1 and divides by t -- so at t = 0 it returned NaN rather
    than a number.

    Nothing had noticed because the lowest zero is at 14.135 and every consumer
    starts there. An integral running from zero found it.
    """
    import mpmath

    from rh_research_engine.symbolic.riemann_siegel import (
        _THETA_TERMS,
        THETA_ASYMPTOTIC_FLOOR,
        theta,
    )

    points = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 14.134725, 19.9])
    computed = theta(points)
    for value, mine in zip(points, computed, strict=True):
        exact = float(mpmath.siegeltheta(float(value)))
        assert abs(mine - exact) < 1e-12, f"theta({value}) is {mine}, not {exact}"

    # And the series it replaced really is wrong there, or this tests nothing.
    def series(t):
        out = t / 2 * np.log(t / (2 * np.pi)) - t / 2 - np.pi / 8
        power = t
        for numerator, denominator in _THETA_TERMS:
            out = out + numerator / (denominator * power)
            power = power * t * t
        return out

    assert abs(series(np.float64(1.0)) - float(mpmath.siegeltheta(1.0))) > 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        assert not np.isfinite(series(np.float64(0.0)))
    # Above the floor the series is what is used, and it agrees.
    high = THETA_ASYMPTOTIC_FLOOR + 30.0
    assert abs(theta(high) - series(high)) < 1e-12


def test_the_two_routes_agree_at_the_threshold():
    """A step where they meet would be a discontinuity in a smooth function.

    Compared AT THE SAME POINT, not at two points either side of it: theta
    has slope log(t/2pi)/2, which is 0.58 there, so probing 1e-7 apart shows
    a difference of 1.2e-7 that is the function and not a seam. The first
    version of this test did exactly that and read it as a jump.
    """
    from rh_research_engine.symbolic.riemann_siegel import (
        _THETA_TERMS,
        THETA_ASYMPTOTIC_FLOOR,
        theta,
    )

    at = np.float64(THETA_ASYMPTOTIC_FLOOR)
    series = at / 2 * np.log(at / (2 * np.pi)) - at / 2 - np.pi / 8
    power = at
    for numerator, denominator in _THETA_TERMS:
        series = series + numerator / (denominator * power)
        power = power * at * at

    assert abs(theta(np.array([at]))[0] - series) < 1e-12


def test_z_at_the_origin_is_zeta_at_one_half():
    """`Z(0) = zeta(1/2)`, which used to be NaN.

    `theta(0)` divided by zero, so the whole product was NaN and any integral
    starting at the origin came back NaN -- not wrong, absent.
    """
    import mpmath

    from rh_research_engine.symbolic.riemann_siegel import z_function

    value = float(np.atleast_1d(z_function(np.array([0.0])))[0])
    assert value == pytest.approx(float(mpmath.zeta(0.5)), abs=1e-12)
    assert value == pytest.approx(-1.4603545088, abs=1e-9)


def test_z_is_accurate_below_the_first_zero():
    """The region the engine had never evaluated, now that an integral does."""
    import mpmath

    from rh_research_engine.symbolic.riemann_siegel import z_function

    points = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 13.0])
    for value, mine in zip(points, z_function(points), strict=True):
        exact = float(
            (
                mpmath.zeta(mpmath.mpc(0.5, float(value)))
                * mpmath.exp(1j * mpmath.siegeltheta(float(value)))
            ).real
        )
        assert abs(mine - exact) < 1e-12, f"Z({value}) is {mine}, not {exact}"


def test_this_module_reaches_the_real_zero_finder():
    """The session cache in `conftest` must not apply here.

    Every other test file gets `zero_ordinates` memoised, because it is pure
    and it dominates the suite. This module is the one that tests the finder,
    so a cached answer here would be a zero-finder test that never runs the
    zero finder -- and returning a short list quietly is the one outcome
    `zero_ordinates` exists to rule out.

    Asserted through the module attribute rather than the name imported at the
    top of this file, because the name is bound at import time and the point is
    to be independent of when that happened.
    """
    from conftest import is_memoised
    from rh_research_engine.symbolic import riemann_siegel

    assert not is_memoised(riemann_siegel.zero_ordinates), (
        "conftest is memoising zero_ordinates for this module, so these tests "
        "are reading a cache instead of exercising the finder"
    )
    assert not is_memoised(zero_ordinates)


def test_theta_error_tracks_the_width_of_a_double():
    """`theta` agrees with mpmath to about one ulp, over nine decades.

    The claim in its docstring, gated. SAMPLED ACROSS EACH DECADE with `ulp`
    taken PER POINT -- both matter. The error depends on where the rounding
    falls, so a fixed height says nothing; and taking `ulp` at the bottom of a
    range while sampling across it inflates the ratio tenfold, which is how a
    draft of this test came to assert `1 < worst < 60` when the truth is about
    one.
    """
    import mpmath as mp

    mp.mp.dps = 40
    generator = np.random.default_rng(4)
    for exponent in (4, 8, 12):
        ratios = []
        for height in generator.uniform(10.0**exponent, 10.0 ** (exponent + 1), 8):
            ours = float(theta(np.array([height]))[0])
            # Differenced IN mpmath. Converting the exact value to float first
            # measures the conversion, not the function.
            exact = mp.siegeltheta(mp.mpf(height))
            error = float(abs(mp.mpf(ours) - exact))
            ulp = float(mp.mpf(2) ** (mp.floor(mp.log(abs(exact), 2)) - 52))
            ratios.append(error / ulp)
        worst = max(ratios)
        assert 0.1 < worst < 5.0, (exponent, worst)


def test_the_float64_ceiling_is_where_it_is_claimed_to_be():
    """The smooth precision ceiling, gated as a POSITION error across heights.

    An earlier version of this read one `theta` at `T = 10^6` and compared the
    phase in radians against a tolerance stated in mean gaps. Those are
    different quantities, and the test stayed green wherever the ceiling
    actually was. Raised in review.

    The conversion: Gram points sit where `theta = n pi`, spaced by the mean
    gap `2 pi / l`, so a phase error `d` moves a zero by `d / theta'(t)`, which
    as a fraction of a mean gap is `d / pi`. That is what the 2% is about --
    the jitter at which the measured pair correlation starts to move.

    Asserted at both ends, because a ceiling nobody has stood on either side of
    is not a ceiling: safe far below it, and genuinely past it above.
    """
    import mpmath as mp

    mp.mp.dps = 40
    tolerance = 0.02

    def shift_in_gaps(height: float) -> float:
        phase = mp.siegeltheta(mp.mpf(height))
        ulp = float(mp.mpf(2) ** (mp.floor(mp.log(abs(phase), 2)) - 52))
        return ulp / np.pi

    # Every height this engine can reach is far inside it.
    assert shift_in_gaps(1e6) < 1e-8
    # Odlyzko's deepest table that could ever be compared against.
    assert shift_in_gaps(2.67e11) < 1e-3

    # And the ceiling is real: past it, float64 alone loses the positions.
    assert shift_in_gaps(1e13) < tolerance
    assert shift_in_gaps(1e14) > tolerance

    # Bracketed where the docstring says, within a factor of three.
    crossing = 3.96e13
    assert shift_in_gaps(crossing / 3) < tolerance < shift_in_gaps(crossing * 3)


def test_a_band_is_exactly_the_slice_of_a_full_run():
    """`zeros_in_band` must not be a cheaper answer to a different question.

    It enters the Gram scan at height instead of walking up from the bottom,
    which is the whole point -- and the only thing that makes that safe is that
    the result is identical, not merely similar.
    """
    from rh_research_engine.symbolic.riemann_siegel import zeros_in_band

    full = zero_ordinates(100_000.0)
    expected = full[(full > 90_000.0) & (full <= 100_000.0)]
    band = zeros_in_band(90_000.0, 100_000.0)
    assert len(band) == len(expected)
    assert np.array_equal(band, expected)


def test_a_band_that_comes_back_short_is_refused():
    """The count check is what makes entering at height safe.

    `ZeroCount` is monotone, so a band holds `N(high) - N(low)`. A band that
    finds fewer is a MISSED zero -- the failure mode at height -- and it must
    raise rather than return a short list, exactly as a full run does.

    Reached by substituting the count through our own module, not by breaking
    the finder: the point is that the refusal fires, and a seam in our code is
    how to make it fire on demand.

    THE OFFSET MUST NOT BE CONSTANT. This first added 500 to every count, which
    CANCELS in `ZeroCount(high) - ZeroCount(low)` -- so the expected number was
    never inflated at all and the test was really asserting that two passes are
    too few to repair this band. It passed for years and stopped passing the
    moment the repair got faster, which is the tell. Inflate only the top.
    """
    import rh_research_engine.symbolic.functions as functions
    from rh_research_engine.symbolic.riemann_siegel import zeros_in_band

    real = functions.ZeroCount
    try:
        functions.ZeroCount = lambda value: real(value) + (500 if float(value) > 95_000 else 0)
        # Sanity: the substitution really does move the difference.
        assert int(functions.ZeroCount(sp.Float(100_000.0))) - int(
            functions.ZeroCount(sp.Float(90_000.0))
        ) == int(real(sp.Float(100_000.0))) - int(real(sp.Float(90_000.0))) + 500
        with pytest.raises(RuntimeError, match="ZeroCount says"):
            zeros_in_band(90_000.0, 100_000.0, max_passes=2)
    finally:
        functions.ZeroCount = real

    # And with the real count it succeeds, so the test above is not passing
    # because the band is simply broken.
    assert len(zeros_in_band(90_000.0, 100_000.0)) > 15_000


@pytest.mark.parametrize(
    ("low", "high"),
    [(100.0, 50.0), (5.0, 100.0), (1000.0, 1000.0)],
)
def test_a_band_that_is_not_a_band_is_refused(low, high):
    from rh_research_engine.symbolic.riemann_siegel import zeros_in_band

    with pytest.raises(ValueError, match="a band runs from"):
        zeros_in_band(low, high)


def test_a_short_gram_block_names_its_own_deficit():
    """Rosser's rule turns the repair from a guess into an assignment.

    `g_n` is a good Gram point when `(-1)**n Z(g_n) > 0`, and between two
    consecutive good ones spanning `k` Gram intervals there are exactly `k`
    zeros. So a block that has yielded fewer says how many are missing and
    exactly where -- where the width test only says a gap looks odd.

    A HINT AND NOT AN AUTHORITY. Rosser's rule has known exceptions, so
    `ZeroCount` still decides and the width-based pass stays behind this one.
    """
    from rh_research_engine.symbolic.riemann_siegel import (
        _dedupe,
        _gram_scan,
        _gram_structure,
        _rosser_deficits,
    )

    low, high = 2_000_000.0, 2_002_000.0
    indices, points, values = _gram_structure(high, low)
    assert len(points) > 100
    found = np.sort(_dedupe(_gram_scan(high, low=low)))

    lows, highs, deficits = _rosser_deficits(indices, points, values, found)
    assert len(lows) > 0, "Gram's law fails often at this height; blocks must be short"
    assert np.all(deficits > 0)
    assert np.all(highs > lows)
    # Every reported block really is short: the count inside is below expected.
    for start, stop, short in zip(lows[:5], highs[:5], deficits[:5], strict=True):
        inside = int(np.searchsorted(found, stop) - np.searchsorted(found, start))
        assert short > 0 and inside >= 0


def test_the_block_repair_is_not_load_bearing_on_its_own():
    """Disabling it must not change the ANSWER, only the time taken.

    It is an optimisation over a fallback that already works, and a test that
    let a wrong answer through here would be testing the optimisation instead
    of the result.
    """
    import rh_research_engine.symbolic.riemann_siegel as rs

    real = rs._repair_by_block
    try:
        rs._repair_by_block = lambda *args, **kwargs: np.array([])
        without = rs.zeros_in_band(2_000_000.0, 2_001_000.0)
    finally:
        rs._repair_by_block = real
    with_blocks = rs.zeros_in_band(2_000_000.0, 2_001_000.0)

    assert len(with_blocks) == len(without)
    assert np.allclose(with_blocks, without, atol=1e-6)


def test_a_cluster_hiding_in_a_normal_gap_is_still_found():
    """The refinement's gap test cannot see a cluster, so it must relax.

    At `t = 2002828.4` four zeros sit inside 0.5920, which is 1.194 mean gaps.
    Two of them were missed by the Gram scan and then never looked for: the
    enclosing gap is under the 1.3 threshold that decides what gets rescanned,
    so no amount of density would have found them, and the band refused with
    40346 against ZeroCount's 40348.

    The threshold now relaxes when a pass finds nothing, which is safe only
    because `ZeroCount` is authoritative -- the loop stops the moment the count
    is met and refuses if it never is, so a wider net can only cost time.
    """
    from rh_research_engine.symbolic.riemann_siegel import zeros_in_band

    band = zeros_in_band(2_002_827.0, 2_002_830.0)
    inside = band[(band > 2_002_828.4) & (band < 2_002_829.1)]
    assert len(inside) == 4, (
        f"expected the four-zero cluster, got {len(inside)}: {inside}"
    )
    assert float(np.diff(np.sort(inside)).min()) < 0.25, np.sort(inside)


def test_a_band_reaches_heights_a_full_run_would_not():
    """The reason for the function: a band high up costs its own size.

    Walking from the bottom to `t = 2x10^6` is about three and a half million
    zeros; this band is nine thousand of them. Checked against `ZeroCount`
    internally, and spot-checked here against mpmath's own Z -- an ordinate
    that is not a zero would show up as a Z far from nothing.
    """
    from rh_research_engine.symbolic.riemann_siegel import zeros_in_band

    band = zeros_in_band(2_000_000.0, 2_020_000.0)
    assert len(band) > 5_000
    assert band.min() > 2_000_000.0 and band.max() <= 2_020_000.0
    # Sorted and without duplicates, which a mis-entered scan would break.
    assert np.all(np.diff(band) > 0)

    mpmath.mp.dps = 25
    for point in band[:: len(band) // 4][:4]:
        assert abs(float(mpmath.siegelz(mpmath.mpf(float(point))))) < 1e-6, point
