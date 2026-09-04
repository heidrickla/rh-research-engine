"""The spacing distribution, and what it is and is not allowed to claim.

The measurement half is cheap to get subtly wrong -- an unfolding that stretches
every gap by a tenth produced a stable, convincing, entirely spurious deficit in
the pair correlation once already -- so the unfolding is checked directly rather
than inferred from the answer looking plausible.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from rh_research_engine.contracts.epistemic import NON_DEDUCTIVE, RIGOROUS, Confidence
from rh_research_engine.symbolic.level_spacing import (
    REPULSION_WINDOW,
    LevelSpacingCheck,
    check_level_spacing,
    wigner_surmise,
)


def _record(**overrides) -> dict:
    payload = {
        "height": 1000.0,
        "zeros": 649,
        "bins": 25,
        "upper": 3.0,
        "mean_spacing": 1.0,
        "mean_deviation": 0.03,
        "worst_deviation": 0.1,
        "worst_at": 0.8,
        "repulsion": 0.0005,
        "poisson_repulsion": 0.0952,
        "deviation_from_exact": 0.031,
        "curve_error": 0.0018,
        "noise_floor": 0.011,
    }
    payload.update(overrides)
    return payload


# --- the curve ------------------------------------------------------------


def test_the_surmise_is_a_probability_density():
    """It integrates to 1 and has mean 1, which is what fixes its constants.

    Both are properties of the CURVE, not of the zeros, so getting them wrong
    would shift every reported deviation by the same amount and look like a
    finding about zeta.
    """
    grid = np.linspace(0.0, 12.0, 200001)
    values = wigner_surmise(grid)
    assert np.trapezoid(values, grid) == pytest.approx(1.0, abs=1e-6)
    assert np.trapezoid(grid * values, grid) == pytest.approx(1.0, abs=1e-6)


def test_the_surmise_vanishes_quadratically_at_zero():
    """Level repulsion, in the curve: `p(s) ~ s^2`, not `p(0) > 0`.

    This is the whole difference from Poisson, whose density is largest at
    zero. A curve that did not vanish there would agree with the zeros
    everywhere except the one place the statistic exists to look at.
    """
    assert wigner_surmise(np.array([0.0]))[0] == 0.0
    small = wigner_surmise(np.array([0.01, 0.02]))
    assert small[1] / small[0] == pytest.approx(4.0, rel=1e-3)


# --- the measurement ------------------------------------------------------


def test_the_unfolding_leaves_mean_spacing_at_one():
    """The check that catches the normalisation error before it becomes a result.

    `theta(gamma)/pi` has unit mean spacing by construction. The plausible
    alternative, `gamma log(gamma/2pi)/(2pi)`, does not: it stretches every gap
    by `(L+1)/L`, which is around a tenth at these heights and shows up here as
    a mean of 1.1 rather than 1.
    """
    result = check_level_spacing(5000.0, bins=20)
    assert result.mean_spacing == pytest.approx(1.0, abs=1e-3)
    assert result.zeros > 4000


def test_the_zeros_repel_by_two_orders_of_magnitude():
    """The robust half, and the one a normalisation error cannot fake.

    Agreement with a fitted curve can survive a wrong normalisation. "Almost
    no spacing is small, where independent points would make a tenth of them
    small" cannot: it is a fact about the raw ordering, and stretching every
    gap uniformly does not create it.
    """
    result = check_level_spacing(5000.0, bins=20)
    assert result.poisson_repulsion == pytest.approx(1 - np.exp(-REPULSION_WINDOW))
    assert result.repulsion < result.poisson_repulsion / 50
    assert result.repulsion_factor > 50


def test_too_few_zeros_raises_rather_than_reporting_noise():
    """A 25-bin histogram of a hundred spacings is noise with a shape.

    Its mean deviation is a number, and reporting it beside the same number
    computed from a hundred thousand spacings would put a measurement and an
    artefact in the same column.
    """
    with pytest.raises(ValueError) as caught:
        check_level_spacing(100.0)
    assert "too few" in str(caught.value)


def test_the_curve_is_named_in_the_record():
    """A deviation without the curve it is from is not a measurement.

    "GUE" alone would not distinguish the surmise from the exact distribution,
    and they differ by more than the finite-height effect being looked for --
    so a reader told only "deviation 0.02 from GUE" would be misled about
    which of the two the number is about.
    """
    from rh_research_engine.symbolic.level_spacing import curve_bin_average

    result = check_level_spacing(5000.0, bins=20)
    assert "surmise" in result.model.lower()

    # The AVERAGE of the surmise over each bin, not its value at the centre.
    # A histogram bin is an average; comparing it against a point value
    # compares two different quantities, and the gap is a floor under every
    # residual that does not shrink with the sample.
    edges = np.linspace(0.0, result.upper, result.bins + 1)
    assert result.predicted == [
        pytest.approx(value) for value in curve_bin_average(edges, wigner_surmise)
    ]
    at_centres = wigner_surmise(np.array(result.centres))
    assert np.max(np.abs(np.array(result.predicted) - at_centres)) > 1e-4, (
        "the two must differ, or this test is not testing anything"
    )


# --- what the record refuses ----------------------------------------------


@pytest.mark.parametrize("confidence", sorted(RIGOROUS, key=str))
def test_a_measured_spacing_cannot_claim_a_rigorous_confidence(confidence):
    """A finite sample against an asymptotic conjecture, via a surmise for it.

    Two layers of "not a proof", and the record refuses at construction rather
    than relying on either being remembered.
    """
    with pytest.raises(ValidationError) as caught:
        LevelSpacingCheck(**_record(confidence=confidence))
    assert "never a proof" in str(caught.value)


def test_the_default_confidence_is_numerical():
    record = LevelSpacingCheck(**_record())
    assert record.confidence is Confidence.NUMERICAL
    assert record.confidence in NON_DEDUCTIVE


# --- the exact law --------------------------------------------------------


def test_the_fredholm_determinant_converges_and_then_stops_moving():
    """Bornemann's claim, tested rather than trusted.

    Gauss-Legendre Nystrom converges super-exponentially for an analytic
    kernel, and the sine kernel is entire. An unconverged determinant looks
    exactly like a converged one, so the node count is chosen from a measured
    plateau and this pins the plateau.
    """
    from rh_research_engine.symbolic.level_spacing import GUE_NODES, gap_probability

    coarse = gap_probability(1.0, nodes=5)
    settled = gap_probability(1.0, nodes=15)
    assert abs(coarse - settled) < 1e-6, "five nodes should already be close"
    for nodes in (20, GUE_NODES, 40, 60):
        assert abs(gap_probability(1.0, nodes=nodes) - settled) < 1e-13


def test_the_gap_probability_starts_at_one_and_falls_at_unit_rate():
    """`E(0) = 1` and `E'(0) = -1`, the second because the density is 1.

    Both are properties of the discretisation as much as the mathematics: a
    quadrature mapped to the wrong interval would break the slope while
    leaving `E(0)` alone.
    """
    from rh_research_engine.symbolic.level_spacing import gap_probability

    assert gap_probability(0.0) == 1.0
    slope = (gap_probability(1e-4) - gap_probability(0.0)) / 1e-4
    assert slope == pytest.approx(-1.0, abs=1e-8)


def test_the_exact_density_has_both_moments_right():
    """`integral p = 1` AND `integral s p = 1`, and the first one is the check
    that matters.

    The first version of the stencil clamped `s - h` to zero, leaving three
    unequally spaced points below `s = h` and returning about -499 near the
    origin. Its FIRST moment still came back as 1.0000000, because the bad
    values sit where `s` is nearly zero and are multiplied away; only the
    zeroth moment noticed, at 0.0005 instead of 1.
    """
    from rh_research_engine.symbolic.level_spacing import exact_gue_density

    grid = np.linspace(0.0, 9.0, 3001)
    density = exact_gue_density(grid)
    assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=1e-6)
    assert np.trapezoid(grid * density, grid) == pytest.approx(1.0, abs=1e-6)
    assert density.min() >= 0.0, "a density is non-negative"


def test_the_exact_density_repels_with_the_right_constant():
    """`p(s) -> (pi^2/3) s^2`, where the surmise gives `32/pi^2` -- 1.4% low.

    The two constants are close, which is why the surmise is a good
    approximation and why "GUE" alone is not enough to name a curve by.
    """
    from rh_research_engine.symbolic.level_spacing import exact_gue_density

    small = np.array([0.02, 0.04])
    ratios = exact_gue_density(small) / small**2
    assert ratios[0] == pytest.approx(np.pi**2 / 3, rel=2e-3)
    assert ratios[1] == pytest.approx(np.pi**2 / 3, rel=3e-3)
    assert 32 / np.pi**2 < np.pi**2 / 3


def test_the_curve_error_is_a_twelfth_of_the_residual_not_the_bulk():
    """The correction this file first got wrong.

    It recorded that the flattening floor was "consistent with the surmise's
    own error". Measured, that error is 0.0018 against a residual of about
    0.02 -- so the surmise accounts for under a tenth of it, and the guess was
    wrong in the direction that made the data look better explained than it
    was.
    """
    result = check_level_spacing(30000.0)
    assert result.curve_error == pytest.approx(0.0018, abs=3e-4)
    assert result.curve_error < result.deviation_from_exact / 8


def test_the_noise_floor_falls_like_one_over_root_n():
    """Four times the spacings, half the floor.

    The floor is what decides whether a residual is a finding, so its scaling
    is worth pinning: a floor that did not fall with N would be a constant
    mislabelled as noise.
    """
    from rh_research_engine.symbolic.level_spacing import sampling_noise_floor

    centres = np.linspace(0.06, 2.94, 25)
    small = sampling_noise_floor(10000, centres, 0.12)
    large = sampling_noise_floor(40000, centres, 0.12)
    assert large == pytest.approx(small / 2, rel=1e-9)


def test_the_residual_shape_survives_bands_that_share_no_data():
    """The finding, and the test that does not assume how noise combines.

    Subtracting a noise floor from a mean absolute deviation assumes something
    about how the two add. This assumes nothing: noise is independent between
    disjoint samples, so a signed residual with the same shape in bands
    sharing no zeros is not the noise.

    The null is MEASURED rather than taken to be zero. With 25 bins the
    correlation of two independent noise vectors has a standard deviation near
    0.2, so "greater than zero" would not be a bar at all.
    """
    from rh_research_engine.symbolic.level_spacing import residual_shape

    result = residual_shape([(0.0, 8000.0), (8000.0, 20000.0)], trials=60)
    assert result.spacings[0] > 250 and result.spacings[1] > 250
    assert min(result.correlations) > 0.8
    assert result.null_worst < min(result.correlations)
    assert result.is_a_shape


def test_nested_ranges_would_have_correlated_by_construction():
    """Why the bands are disjoint, kept as an executable note.

    The first version compared the zeros below 30000 against those below
    80000 -- one set a third of the other -- and reported r = 0.98 that was
    partly guaranteed. Here the overlap is total, and the correlation it
    manufactures is the point: it is not evidence of anything about zeta.
    """
    import numpy as np

    from rh_research_engine.symbolic.level_spacing import (
        exact_gue_density,
        unfold,
    )
    from rh_research_engine.symbolic.riemann_siegel import zero_ordinates

    edges = np.linspace(0.0, 3.0, 26)
    centres = (edges[:-1] + edges[1:]) / 2
    exact = exact_gue_density(centres)

    everything = np.array([float(g) for g in zero_ordinates(20000.0)])
    subset = everything[everything <= 8000.0]

    def residual(values):
        counts, _ = np.histogram(
            np.diff(unfold(values)), bins=edges, density=True
        )
        return counts - exact

    nested = float(np.corrcoef(residual(subset), residual(everything))[0, 1])
    assert nested > 0.8, (
        "a subset against its superset correlates whatever the mathematics "
        "does, which is exactly why this comparison proves nothing"
    )


def test_a_residual_shape_cannot_claim_a_rigorous_confidence():
    """It detects a correction to a limit no computation reaches."""
    from rh_research_engine.symbolic.level_spacing import ResidualShape

    with pytest.raises(ValidationError) as caught:
        ResidualShape(confidence=Confidence.PROVED)
    assert "limit no computation reaches" in str(caught.value)


def test_the_noise_floor_does_not_rest_on_an_assumption_the_data_violates():
    """The multinomial formula assumes independent spacings. They are not.

    Consecutive gaps between zeros are correlated -- that is what pair
    correlation measures -- so the formula behind `sampling_noise_floor` rests
    on an assumption the data breaks, and its number decides whether a residual
    is a finding.

    A moving-block bootstrap resamples contiguous RUNS, so local correlation
    survives into the replicates, and it agrees. It comes out slightly SMALLER
    than the formula, which is the right direction: a rigid spectrum fluctuates
    less than independent points, so the formula errs toward calling a real
    residual noise.
    """
    from rh_research_engine.symbolic.level_spacing import (
        bootstrap_noise_floor,
        sampling_noise_floor,
        unfold,
    )
    from rh_research_engine.symbolic.riemann_siegel import zero_ordinates

    ordinates = np.array([float(g) for g in zero_ordinates(20000.0)])
    spacings = np.diff(unfold(ordinates))
    centres = np.linspace(0.06, 2.94, 25)

    formula = sampling_noise_floor(len(spacings), centres, 0.12)
    for block in (1, 50):
        resampled = bootstrap_noise_floor(
            spacings, centres, 0.12, block=block, replicates=60
        )
        assert resampled == pytest.approx(formula, rel=0.25), (
            f"block {block}: the two estimates should agree to a quarter"
        )
        assert resampled <= formula * 1.05, (
            "the formula must not be the optimistic one of the two"
        )


def test_the_shape_survives_two_halves_of_a_single_band():
    """Height held fixed, and the shape is still there.

    Comparing bands at different heights leaves open that the residual is some
    artefact varying with height rather than a property of the zeros. Two
    halves of one band are disjoint and at essentially the same height, so
    that reading is closed off.
    """
    from rh_research_engine.symbolic.level_spacing import (
        exact_gue_density,
        unfold,
    )
    from rh_research_engine.symbolic.riemann_siegel import zero_ordinates

    ordinates = np.array([float(g) for g in zero_ordinates(20000.0)])
    band = ordinates[ordinates > 6000.0]
    middle = len(band) // 2

    edges = np.linspace(0.0, 3.0, 26)
    exact = exact_gue_density((edges[:-1] + edges[1:]) / 2)
    residuals = [
        np.histogram(np.diff(unfold(part)), bins=edges, density=True)[0] - exact
        for part in (band[:middle], band[middle:])
    ]
    assert np.corrcoef(residuals[0], residuals[1])[0, 1] > 0.8


def test_a_bin_average_is_not_the_density_at_the_bin_centre():
    """The bug that floored every residual, pinned.

    `np.histogram(..., density=True)` returns the MEAN density over a bin.
    Comparing it against the density AT the centre compares two different
    quantities, and for a curved density they differ by about `w^2 p''/24` --
    of order 1e-3 per bin at these widths.

    It does not shrink with the sample, so it is a constant floor under every
    residual and it dominates once the sample is large. The null test found it
    as a bias of +0.44 of the noise term at 500000 draws; refining the sampler
    did not move it; bin averages took it to -0.035.
    """
    from rh_research_engine.symbolic.level_spacing import (
        exact_gue_bin_average,
        exact_gue_density,
    )

    edges = np.linspace(0.0, 3.0, 26)
    centres = (edges[:-1] + edges[1:]) / 2
    average = exact_gue_bin_average(edges)
    at_centres = exact_gue_density(centres)

    # Both are the same curve, so they integrate to the same thing.
    width = edges[1] - edges[0]
    assert np.sum(average) * width == pytest.approx(
        np.sum(at_centres) * width, abs=1e-5
    )
    # And they are NOT equal bin by bin, which is the whole point.
    discrepancy = float(np.sum((average - at_centres) ** 2))
    assert discrepancy > 1e-5, "if these agreed there would be no bug to fix"

    # Sized against the noise term it would otherwise be mistaken for.
    noise_term = float(np.sum(np.maximum(average, 0.0) / (500000 * width)))
    assert 0.2 < discrepancy / noise_term < 1.0, (
        "at half a million spacings the centre-vs-average error is a "
        "substantial fraction of the noise, which is how it hid"
    )


def test_the_amplitude_estimator_is_unbiased_on_samples_with_no_systematic():
    """Draws from the exact law have nothing in them, and must measure nothing.

    `sum r^2 - sum sigma^2` estimates the squared systematic without needing a
    template -- projecting onto a shape taken from some band would bias every
    band overlapping it upward, which is the direction that fakes a decay.

    An estimator with a positive bias would report an amplitude in every band
    and a decay from the band sizes alone, so this is the check the whole
    measurement rests on.
    """
    from rh_research_engine.symbolic.level_spacing import exact_gue_bin_average

    edges = np.linspace(0.0, 3.0, 26)
    width = edges[1] - edges[0]
    exact = exact_gue_bin_average(edges)

    grid = np.linspace(0.0, 9.0, 20001)
    from rh_research_engine.symbolic.level_spacing import exact_gue_density

    cumulative = np.cumsum(exact_gue_density(grid)) * (grid[1] - grid[0])
    cumulative /= cumulative[-1]

    generator = np.random.default_rng(5)
    size = 200000
    squares = []
    for _ in range(20):
        draws = np.interp(generator.random(size), cumulative, grid)
        counts, _ = np.histogram(draws, bins=edges, density=True)
        variance = np.maximum(exact, 0.0) / (size * width)
        squares.append(np.sum((counts - exact) ** 2) - np.sum(variance))

    noise_term = float(np.sum(np.maximum(exact, 0.0) / (size * width)))
    bias = float(np.mean(squares)) / noise_term
    assert abs(bias) < 0.2, f"the estimator carries a bias of {bias:+.3f}"
