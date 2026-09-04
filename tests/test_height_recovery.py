"""Recovering the height of the zeros from their pair correlation.

The claim under test is not "the curve fits". It is that the curve's ONE free
parameter, `l = log(t/2 pi)`, can be read back out of the zeros and lands where
the heights say it should -- which a curve that merely had the right shape
could not do, and which Montgomery's universal law cannot even be asked.

So what is checked here is mostly the ways that could be true without any
arithmetic: an estimator that returns ascending numbers for ascending input,
bands whose precision improves up the range, and a verdict that reports a fit
which determined nothing as agreement.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from rh_research_engine.contracts.epistemic import RIGOROUS, Confidence
from rh_research_engine.symbolic.height_recovery import (
    MAX_BAND_ELL_SPAN,
    HeightRecovery,
    curve_bank,
    ell_grid,
    equal_count_bands,
    fit_ell,
    narrow_ell_bands,
    recover_height,
)

#: The prime limit the tests build curves at.
#:
#: Measured, not assumed cheap: at 10000 primes the recovered slope is
#: 1.1399 +/- 0.1602 and at 100000 it is 1.1400 +/- 0.1599 -- the same
#: measurement, and 53 s against 72 s. `test_the_prime_truncation_is_converged`
#: in `test_conrey_snaith` is what says why.
FAST_PRIMES = 10_000


EDGES = np.linspace(0.0, 3.0, 31)


@pytest.fixture(scope="session")
def bank():
    """The curves at every `l` on the grid, built once for the whole file.

    They do not depend on the data, and building them is most of the cost of
    everything below -- three tests were each paying for their own.
    """
    return curve_bank(EDGES, ell_grid(), prime_limit=FAST_PRIMES)


@pytest.fixture(scope="session")
def recovery(bank):
    """One `recover_height` for the file.

    `band_width` is 0.4 rather than the 0.5 default because partial bins are
    now dropped: `T = 200000` lands inside the fourth bin at 0.5, leaving three
    bands where a weighted regression needs four. Narrowing the bins is the
    right lever -- widening them is what the mixture bias punishes.
    """
    from rh_research_engine.symbolic.riemann_siegel import zero_ordinates

    ordinates = np.asarray(zero_ordinates(200000.0), dtype=float)
    return recover_height(ordinates, band_width=0.4, prime_limit=FAST_PRIMES, bank=bank)


def test_a_bank_must_be_the_bank_it_claims_to_be(bank):
    """Shape is not identity, and the first version checked only shape.

    Thirty bins over [0, 4] and thirty over [0, 3] are the same shape and
    different curves; so are two banks on different `l` grids or prime limits.
    Accepting any same-shaped array compares each measured column against a
    curve averaged over some other interval, silently. Raised in review.
    """
    grid = ell_grid()

    wrong_shape = bank._replace(curves=bank.curves[:, :-1])
    with pytest.raises(ValueError, match="this fit wants"):
        wrong_shape.matching(EDGES, grid, FAST_PRIMES)

    # Same shape, different window -- the case shape alone cannot catch.
    other_window = bank._replace(edges=np.linspace(0.0, 4.0, 31))
    with pytest.raises(ValueError, match="different curves"):
        other_window.matching(EDGES, grid, FAST_PRIMES)

    with pytest.raises(ValueError, match="different l grid"):
        bank._replace(grid=grid + 0.1).matching(EDGES, grid, FAST_PRIMES)

    with pytest.raises(ValueError, match="prime_limit"):
        bank._replace(prime_limit=999).matching(EDGES, grid, FAST_PRIMES)

    # And the real one passes.
    assert bank.matching(EDGES, grid, FAST_PRIMES).shape == (len(grid), 30)


# --------------------------------------------------------------------------
# The estimator, against an answer that is known exactly.
# --------------------------------------------------------------------------


def test_the_fitter_recovers_an_ell_it_is_given(bank):
    """Fed the curve at a known `l`, it must return that `l`.

    The positive control for the estimator itself, separate from any data:
    if this failed, every number in this module would be measuring the grid
    and the parabola rather than the zeros.
    """
    grid = ell_grid()
    for truth in (8.5, 10.5, 12.5):
        exact = curve_bank(EDGES, np.array([truth]), prime_limit=FAST_PRIMES).curves[0]
        got = fit_ell(exact, grid=grid, bank=bank)
        # The residual is the parabola's discretisation, not the data's.
        assert abs(got - truth) < 0.02, (truth, got)


def test_the_grid_reaches_past_both_ends_of_what_is_reachable():
    """A grid that stopped at the truth would find it by construction."""
    grid = ell_grid()
    # T = 10^6 is l = 12.0, and the lowest zeros retained are near l = 4.
    assert grid.min() < 7.0 and grid.max() > 14.0


def test_the_answer_is_not_quantised_by_the_grid(bank):
    """The mesh is COARSER than what the fit resolves, so this has to hold.

    The step is 0.25 and the regression resolves about 0.11, so without the
    parabola through the three lowest points every fitted `l` would be a grid
    point and the slope would be a staircase. Fed a curve at an `l` deliberately
    between two grid points, the fit has to land on it and not on either.

    This test replaced one that asserted `step < 0.05 or True` -- which is not
    a test, and which I wrote in the file whose subject is gates that cannot
    fail.
    """
    grid = ell_grid()
    step = float(grid[1] - grid[0])

    between = 10.37  # a long way from any multiple of 0.25
    assert min(abs(between - value) for value in grid) > 0.1 * step
    exact = curve_bank(EDGES, np.array([between]), prime_limit=FAST_PRIMES).curves[0]
    got = fit_ell(exact, grid=grid, bank=bank)
    assert abs(got - between) < 0.02, (between, got)
    assert min(abs(got - value) for value in grid) > 0.1 * step, (
        "the fit returned a grid point, so the refinement is not working"
    )


# --------------------------------------------------------------------------
# The finding.
# --------------------------------------------------------------------------


def test_the_zeros_say_what_height_they_are_at(recovery):
    """Fitted `l` against true `l`: slope one, and not zero.

    Both halves matter. Consistent with one says the fit tracks the height;
    far from zero says it determined anything at all. Over 1747146 zeros below
    T = 10^6 this is 0.94 +/- 0.13, 0.5 sigma from one and 7.2 from zero, with
    the band grid's arbitrary phase carried as a systematic; here, on the
    smaller sample the suite can afford, it is looser.
    """
    assert abs(recovery.slope - 1.0) < 2 * recovery.slope_error, (
        recovery.slope,
        recovery.slope_error,
    )
    # Two sigma, not three, and the difference is the sample this suite can
    # afford. Narrow bands over T = 2e5 reach l = 10.4 and the low ones hold
    # few zeros, so the lever arm is short; the full T = 10^6 run reaches 7.0
    # sigma. `tracks_the_height` demands three and is correctly False here --
    # asserted below, because a property that never returns False is not a
    # property.
    assert abs(recovery.slope) > 2 * recovery.slope_error, (
        recovery.slope,
        recovery.slope_error,
    )
    # Unbiased: the fitted heights are not systematically off.
    assert abs(recovery.bias) < 2 * recovery.bias_error, (
        recovery.bias,
        recovery.bias_error,
    )
    assert recovery.confidence is Confidence.NUMERICAL
    assert not recovery.tracks_the_height, (
        "this sample reaches about 2 sigma from zero; the verdict wants 3, and "
        "it must say so rather than passing on a fit this loose"
    )


def test_a_flat_relation_does_not_produce_this_slope(recovery):
    """The control against an estimator that just returns ascending numbers.

    If the slope came from the fit drifting upward for any reason at all,
    permuting which fitted value belongs to which band would leave it intact.
    """
    # Again the affordable sample: the full run gives 0.0000.
    assert recovery.null_p < 0.05, recovery.null_p

    from rh_research_engine.symbolic.height_recovery import _weighted_line

    true = np.array(recovery.true_ell)
    fitted = np.array(recovery.fitted_ell)
    sigma = np.array(recovery.fitted_error)

    # The null draws each band from ITS OWN measured sigma about a flat
    # relation. Permuting fitted values between bands -- what the first version
    # did -- assumes they are exchangeable, and unequal sigmas make that false.
    generator = np.random.default_rng(3)
    centre = float(np.average(fitted, weights=1 / sigma**2))
    null = np.array(
        [
            _weighted_line(true, centre + generator.normal(0, sigma), sigma)[0]
            for _ in range(2000)
        ]
    )
    assert abs(null.mean()) < 0.2, null.mean()
    assert recovery.slope > null.mean() + 2 * null.std()


# --------------------------------------------------------------------------
# The ways this could report more than it measured.
# --------------------------------------------------------------------------


def test_a_fit_that_determined_nothing_is_not_agreement():
    """`tracks_the_height` needs the error bar to exclude zero as well.

    A slope of exactly one whose error bar covers zero is a fit that measured
    nothing, and reporting it as agreement is the failure this property exists
    to prevent.
    """
    common = dict(
        band_size=1000,
        true_ell=[8.0, 9.0, 10.0],
        fitted_ell=[8.0, 9.0, 10.0],
        bias=0.0,
        bias_error=0.1,
        null_p=0.001,
    )
    determined = HeightRecovery(slope=1.0, slope_error=0.1, **common)
    assert determined.tracks_the_height

    useless = HeightRecovery(slope=1.0, slope_error=0.9, **common)
    assert not useless.tracks_the_height, (
        "a slope of one whose error bar covers zero determined nothing"
    )

    wrong = HeightRecovery(slope=0.2, slope_error=0.02, **common)
    assert not wrong.tracks_the_height, "sharply measured, and not one"


@pytest.mark.parametrize("confidence", sorted(RIGOROUS, key=str))
def test_a_fit_against_a_conjecture_cannot_be_filed_as_a_proof(confidence):
    with pytest.raises(ValidationError) as caught:
        HeightRecovery(
            band_size=1000,
            slope=1.0,
            slope_error=0.1,
            bias=0.0,
            bias_error=0.1,
            null_p=0.001,
            confidence=confidence,
        )
    assert "ratios conjecture" in str(caught.value)


def test_too_few_usable_bands_cannot_carry_a_standard_error():
    """A range too short in `l` yields too few narrow bands to regress."""
    # 100000 zeros over a tiny height range: one band's worth of `l`.
    with pytest.raises(ValueError, match="standard error"):
        recover_height(np.linspace(400.0, 402.0, 100_000), prime_limit=FAST_PRIMES)


def test_bands_too_thin_to_measure_are_dropped_not_widened():
    """Widening is what produced the bias review found.

    A band that cannot hold enough zeros should be absent, not quietly wider
    than its neighbours and fitted as a mixture against a single curve.
    """
    ordinates = np.sort(np.random.default_rng(2).uniform(400.0, 200000.0, 60_000))
    bands = narrow_ell_bands(ordinates, width=0.5, minimum=5_000)
    ell_spans = [
        float(np.log(b.max() / (2 * np.pi)) - np.log(b.min() / (2 * np.pi))) for b in bands
    ]
    assert bands, "no band survived"
    assert max(ell_spans) <= 0.5 + 1e-9, ell_spans
    assert all(len(b) >= 5_000 for b in bands)


def test_the_bands_are_disjoint_and_hold_equal_counts():
    """Equal COUNT, so the noise floor is the same in every band.

    With equal spans of height the top bands would hold far more zeros, the fit
    would get more precise up the range, and a trend in the fitted `l` could be
    precision rather than height. Disjoint because nested ranges in this
    repository once correlated at 0.98 by construction.
    """
    ordinates = np.sort(np.random.default_rng(1).uniform(400.0, 200000.0, 500_000))
    pieces = equal_count_bands(ordinates, 8)
    assert len({len(piece) for piece in pieces}) == 1
    for earlier, later in zip(pieces[:-1], pieces[1:], strict=True):
        assert earlier.max() <= later.min(), "bands overlap"


def test_a_wide_band_is_fitted_as_the_wrong_height(bank):
    """Why `MAX_BAND_ELL_SPAN` exists, and it is measured rather than asserted.

    A band's density has expectation `(1/N) sum_n R_2(u; l_n)` -- a mixture --
    while the fit compares it against ONE curve at the band's mean `l`. `R_2`
    is nonlinear in `l`, so a wide band is fitted at the wrong height. The
    first version of this module used equal-COUNT bands whose lowest spanned
    `l = 4.15..9.47`, and that band carried the most leverage in the
    regression. Raised in review.

    Measured here by feeding the fitter a synthetic mixture and asking for its
    mean `l` back: the bias runs -0.008 at width 0.5 and -0.104 at 5.32.
    """
    from rh_research_engine.symbolic import conrey_snaith as cs
    from rh_research_engine.symbolic.level_spacing import curve_bin_average

    grid = ell_grid()

    def single(value):
        return curve_bin_average(
            EDGES,
            lambda u: cs.pair_correlation(u, value, prime_limit=FAST_PRIMES),
            samples=cs.BIN_SAMPLES,
        )

    def bias_at(width):
        ells = np.linspace(10.0 - width / 2, 10.0 + width / 2, 15)
        # Weighted by the density of zeros, which grows with height.
        weights = np.exp(ells)
        weights /= weights.sum()
        mixture = sum(w * single(float(e)) for e, w in zip(ells, weights, strict=True))
        mean_ell = float((ells * weights).sum())
        return (
            fit_ell(mixture, EDGES, grid=grid, prime_limit=FAST_PRIMES, bank=bank)
            - mean_ell
        )

    narrow = abs(bias_at(MAX_BAND_ELL_SPAN))
    wide = abs(bias_at(5.32))
    assert wide > 10 * narrow, (narrow, wide)
    # And at the width the module uses, the bias is far inside a band's own
    # uncertainty, which runs about 0.2-0.7.
    assert narrow < 0.05, narrow


def test_the_bands_are_not_equally_precise(recovery):
    """Which is why the regression is weighted rather than least-squares.

    Equal zero counts would make the density noise similar, but they do not
    make `l` equally well determined, and these bands are not equal in count
    either. Pooling one residual spread across all of them -- what the first
    version did -- assumes a homoskedasticity the estimator does not have.
    Raised in review.
    """
    errors = np.array(recovery.fitted_error)
    assert len(errors) == len(recovery.fitted_ell)
    assert np.all(errors > 0)

    # NEITHER A TUNED RATIO NOR A FIXTURE ACCIDENT. `errors.max()/errors.min()
    # > 1.5` was calibrated against a band set that no longer exists. Replacing
    # it with "weighting changes this fixture's answer" was no better: once the
    # band sigmas were averaged over chunk placements they came out close
    # enough here that weighted and pooled agree to 1e-3, which says something
    # about four bands at T = 200000 and nothing about the estimator.
    #
    # So assert the ESTIMATOR weights, on errors chosen to make it show. On the
    # real T < 10^6 bands the measured sigmas span 0.088 to 0.699, a factor of
    # eight, and that is where it matters.
    from rh_research_engine.symbolic.height_recovery import _weighted_line

    ell = np.array(recovery.true_ell)
    fitted = np.array(recovery.fitted_ell)
    lopsided = np.full_like(errors, 0.5)
    lopsided[0] = 0.01  # one band known far better than the rest
    weighted, _, _ = _weighted_line(ell, fitted, lopsided)
    pooled, _, _ = _weighted_line(ell, fitted, np.full_like(errors, 0.5))
    assert abs(weighted - pooled) > 1e-3, (
        f"weighted {weighted:.4f} and pooled {pooled:.4f} agree even when one "
        "band is given 2500x the weight, so `_weighted_line` is not weighting"
    )
    # And the real bands do differ, which is why it is not least squares.
    assert errors.max() > errors.min()


def test_the_band_grid_phase_is_measured_rather_than_assumed_harmless(bank):
    """`anchors > 1` reports the spread the arbitrary grid position creates.

    Edges anchor at `ell.min()`, which is an accident of which zero is lowest.
    On the full T < 10^6 set, shifting within one width moves the slope across
    0.875..1.037 -- a systematic worth +/-0.068 that sat outside the quoted
    error until it was measured.
    """
    from rh_research_engine.symbolic.riemann_siegel import zero_ordinates

    # T = 200000 at width 0.35 keeps four bands at every phase tried; at 0.4
    # one phase drops to three and the regression rightly refuses.
    ordinates = np.asarray(zero_ordinates(200_000.0), dtype=float)
    single = recover_height(ordinates, band_width=0.35, prime_limit=FAST_PRIMES, bank=bank)
    assert single.slope_anchor_error is None, (
        "one anchor measures no systematic, and None must not be reported as 0.0"
    )
    assert single.slope_over_anchors is None

    several = recover_height(
        ordinates, band_width=0.35, prime_limit=FAST_PRIMES, bank=bank, anchors=3
    )
    assert several.slope_anchor_error is not None
    assert several.slope_anchor_error > 0.0, (
        "three grid positions giving byte-identical slopes would mean the "
        "phase is not reaching the banding at all"
    )
    assert several.slope_over_anchors is not None
    # The default anchor's own slope is preserved, not overwritten by the mean.
    assert several.slope == single.slope


def test_shifting_the_grid_actually_changes_the_bands(ordinates_for_phase):
    """Otherwise the systematic above is measuring nothing."""
    from rh_research_engine.symbolic.height_recovery import narrow_ell_bands

    base = narrow_ell_bands(ordinates_for_phase, width=0.35)
    moved = narrow_ell_bands(ordinates_for_phase, width=0.35, phase=0.5)
    base_means = [float(np.log(b / (2 * np.pi)).mean()) for b in base]
    moved_means = [float(np.log(b / (2 * np.pi)).mean()) for b in moved]
    assert base_means != moved_means


@pytest.fixture(scope="session")
def ordinates_for_phase():
    from rh_research_engine.symbolic.riemann_siegel import zero_ordinates

    return np.asarray(zero_ordinates(200_000.0), dtype=float)


def test_every_band_is_narrow(recovery):
    """The constant is only worth having if the bands honour it."""
    assert max(recovery.band_widths) <= MAX_BAND_ELL_SPAN + 1e-9, recovery.band_widths
