"""Measuring the zeros against the pair correlation the corpus asserts.

The corpus's first indexed formula is `1 - (sin(pi u)/(pi u))^2`, and nothing
had ever held it against anything. What is under test is that the comparison is
against the CORPUS's expression rather than a copy of it, that the unfolding is
the right one, and that agreement cannot be filed as a proof of a conjecture.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from rh_research_engine.contracts.epistemic import RIGOROUS, Confidence
from rh_research_engine.symbolic import pair_correlation as module
from rh_research_engine.symbolic.pair_correlation import (
    SKIP_LOWEST,
    PairCorrelationCheck,
    check_pair_correlation,
    measured_density,
    montgomery_density,
    unfold,
)


@pytest.fixture(scope="session")
def lower_order_check():
    """One `check_pair_correlation(..., lower_order=True)` per height, shared.

    Three tests below want the same measurement at `T = 5000`, and it costs a
    Euler product per quadrature point. The results are read-only, so sharing
    them is a cache and not shared state between tests.
    """
    cache: dict[float, object] = {}

    def get(height: float):
        if height not in cache:
            cache[height] = check_pair_correlation(height, lower_order=True, prime_limit=100_000)
        return cache[height]

    return get


def test_the_curve_comes_from_the_index(tmp_path, monkeypatch):
    """A check against a retyped copy of the formula is a check on the copy."""
    values = montgomery_density(np.array([0.0, 0.5, 1.0, 2.0]))
    assert values[0] == 0.0, "level repulsion: the density vanishes at zero"
    assert abs(values[1] - (1 - (np.sin(np.pi / 2) / (np.pi / 2)) ** 2)) < 1e-12
    assert abs(values[2] - 1.0) < 1e-12

    # With the expression gone from the index, this must stop rather than fall
    # back on something it remembers.
    empty = tmp_path / "index.json"
    empty.write_text(json.dumps([{"expression": "x + 1"}]), encoding="utf-8")
    monkeypatch.setattr(module, "INDEX_PATH", empty)
    with pytest.raises(LookupError, match="no longer records"):
        montgomery_density(np.array([0.5]))


def test_a_missing_index_is_reported_as_such(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "INDEX_PATH", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="ingest"):
        montgomery_density(np.array([0.5]))


def test_an_expression_in_the_wrong_variable_is_refused(tmp_path, monkeypatch):
    """It has to be a density in the spacing, not something shaped like one."""
    wrong = tmp_path / "index.json"
    wrong.write_text(
        json.dumps([{"expression": "1 - (sin(pi*u)/((pi*u)))**2 + x"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "INDEX_PATH", wrong)
    with pytest.raises(LookupError, match="not in u alone"):
        montgomery_density(np.array([0.5]))


def test_the_unfolding_is_theta_over_pi_and_not_the_lookalike():
    """The trap, kept as a test because it was completely convincing.

    `gamma log(gamma/2 pi)/(2 pi)` differs from `theta(gamma)/pi` by a term
    linear in gamma -- irrelevant to a shift, fatal to a SPACING, since the
    derivatives differ by `1/(2 pi)`. Every gap comes out `(L+1)/L` too wide,
    and the measured correlation then sits about a tenth below the curve at
    every point: stable, clean, and entirely an artefact.
    """
    ordinates = np.array([1000.0, 1001.0, 5000.0, 5001.0])
    correct = unfold(ordinates)
    lookalike = ordinates * np.log(ordinates / (2 * np.pi)) / (2 * np.pi)

    for low, high in ((0, 1), (2, 3)):
        right = correct[high] - correct[low]
        wrong = lookalike[high] - lookalike[low]
        scale = np.log(ordinates[low] / (2 * np.pi))
        # The derivative ratio is exact for an infinitesimal gap; over a gap
        # of 1 the second-order term is about 2e-5, which is what this allows.
        assert abs(wrong / right - (scale + 1) / scale) < 1e-3
        assert wrong > right * 1.08, "the lookalike stretches every gap"


def test_the_measured_correlation_follows_the_asserted_one():
    """The measurement the module exists for."""
    result = check_pair_correlation(5000.0)
    assert result.zeros > 4000
    assert result.confidence is Confidence.NUMERICAL
    assert result.mean_deviation < 0.08, result.mean_deviation
    # Level repulsion: almost no pairs at a fraction of a mean spacing.
    assert result.measured[0] < 0.05
    # And the density settles near one further out.
    assert 0.85 < np.mean(result.measured[-10:]) < 1.15


def test_the_agreement_improves_with_height():
    """Montgomery's is asymptotic, so this is the direction that matters."""
    low = check_pair_correlation(5000.0)
    high = check_pair_correlation(30000.0)
    assert high.zeros > low.zeros
    assert high.mean_deviation < low.mean_deviation, (
        low.mean_deviation,
        high.mean_deviation,
    )


def test_a_height_with_too_few_zeros_is_refused():
    """The unfolding is asymptotic, so a short run cannot say anything."""
    with pytest.raises(ValueError, match="cannot say anything"):
        check_pair_correlation(300.0)


def test_the_density_is_normalised_per_zero():
    """A density that counted pairs twice would agree with nothing."""
    spaced = np.arange(0.0, 500.0)
    centres, density, _, _ = measured_density(spaced, window=3.0, bins=30)
    # A perfectly rigid sequence puts all its mass at the integers.
    assert density[np.argmin(np.abs(centres - 0.5))] == 0.0
    near_one = density[np.argmin(np.abs(centres - 1.05))]
    assert near_one > 5.0, "a rigid ladder should spike at unit spacing"


@pytest.mark.parametrize("confidence", sorted(RIGOROUS, key=str))
def test_agreement_with_a_conjecture_cannot_be_filed_as_a_proof(confidence):
    with pytest.raises(ValidationError) as caught:
        PairCorrelationCheck(
            height=5000.0,
            zeros=4320,
            window=3.0,
            bins=30,
            mean_deviation=0.02,
            worst_deviation=0.06,
            worst_at=0.45,
            confidence=confidence,
        )
    assert "conjecture" in str(caught.value)


def test_the_lowest_zeros_are_skipped():
    """They are normalised worst and a short run has most of them."""
    assert SKIP_LOWEST > 0
    result = check_pair_correlation(5000.0)
    from rh_research_engine.symbolic.riemann_siegel import zero_ordinates

    assert result.zeros == len(zero_ordinates(5000.0)) - SKIP_LOWEST


def test_the_curve_is_averaged_over_the_bin_and_not_read_at_its_centre():
    """A histogram bin is a mean. This module compared it against a point.

    The gate that fails if `check_pair_correlation` goes back to
    `montgomery_density(centres)`. What makes it worth a test rather than a
    comment is the SIZE: the bias is 2.7e-3 at the worst bin, and the
    lower-order terms it now has to resolve are of the same order. A floor that
    does not shrink with the sample, sitting exactly where the signal is.
    """
    from rh_research_engine.symbolic.conrey_snaith import BIN_SAMPLES
    from rh_research_engine.symbolic.pair_correlation import montgomery_bin_average

    edges = np.linspace(0.0, 3.0, 31)
    centres = 0.5 * (edges[:-1] + edges[1:])
    at_centre = montgomery_density(centres)
    over_bin = montgomery_bin_average(edges, samples=BIN_SAMPLES)

    difference = np.abs(at_centre - over_bin)
    assert difference.max() > 2e-3, difference.max()
    # It is worst where the curve bends hardest, which is the repulsion end.
    assert centres[difference.argmax()] < 0.2

    # And the check reports the averaged one.
    result = check_pair_correlation(5000.0)
    assert np.allclose(result.predicted, over_bin, atol=1e-12)


def test_the_pairs_own_heights_are_measured_not_assumed():
    """The pooled sample spans a range of `l`, and the curve depends on it.

    Each zero contributes very nearly the same number of pairs -- which is
    exactly why assuming it would never be caught being wrong. The weights come
    off the same loop that builds the histogram.
    """
    from rh_research_engine.symbolic.riemann_siegel import zero_ordinates

    ordinates = np.asarray(zero_ordinates(5000.0), dtype=float)[SKIP_LOWEST:]
    unfolded = unfold(zero_ordinates(5000.0))[SKIP_LOWEST:]
    statistics = measured_density(unfolded, ordinates=ordinates, window=3.0, bins=30)

    assert statistics.ell_weights.sum() == pytest.approx(1.0)
    assert np.all(statistics.ell_weights >= 0)
    # The range really is spanned rather than collapsed onto one node.
    assert statistics.ell_nodes.max() - statistics.ell_nodes.min() > 1.5
    # With no ordinates there are no weights, rather than made-up ones.
    bare = measured_density(unfolded, window=3.0, bins=30)
    assert bare.ell_weights.sum() == 0.0


def test_refining_the_l_bands_converges_by_quartering():
    """The averaging is a midpoint rule, so halving the band width quarters
    the error -- which is what makes eight bands a derivable choice rather
    than a round number.
    """
    from rh_research_engine.symbolic.conrey_snaith import BIN_SAMPLES
    from rh_research_engine.symbolic.pair_correlation import lower_order_bin_average
    from rh_research_engine.symbolic.riemann_siegel import zero_ordinates

    ordinates = np.asarray(zero_ordinates(5000.0), dtype=float)[SKIP_LOWEST:]
    unfolded = unfold(zero_ordinates(5000.0))[SKIP_LOWEST:]
    edges = np.linspace(0.0, 3.0, 13)

    curves = {}
    for bands in (2, 4, 8):
        statistics = measured_density(
            unfolded, ordinates=ordinates, window=3.0, bins=12, ell_bands=bands
        )
        curves[bands] = lower_order_bin_average(
            edges,
            statistics.ell_nodes,
            statistics.ell_weights,
            samples=BIN_SAMPLES,
            prime_limit=100_000,
        )
    coarse = np.abs(curves[2] - curves[8]).mean()
    fine = np.abs(curves[4] - curves[8]).mean()
    assert fine < coarse / 3, (coarse, fine)


def test_the_zeros_follow_the_arithmetic_curve_and_not_the_universal_one(
    lower_order_check,
):
    """The measurement the lower-order form exists for.

    Montgomery's curve is universal -- random matrices and quantum billiards
    obey it -- so following it says the zeros are a spectrum. Conrey-Snaith
    carries the primes, has NO free parameters, and the data follows it
    instead. What makes the comparison readable is `curve_separation` against
    `noise_floor`: if the two curves were closer together than the sampling
    noise, neither deviation would be evidence about anything.
    """
    result = lower_order_check(5000.0)

    # The curves are far enough apart for the data to tell them apart at all.
    assert result.curves_are_distinguishable, (
        result.curve_separation,
        result.noise_floor,
    )
    # And it does tell them apart, in the direction where the arithmetic is.
    assert result.lower_order_deviation < result.mean_deviation, (
        result.lower_order_deviation,
        result.mean_deviation,
    )


def test_the_noise_floor_is_measured_and_not_the_one_for_spacings():
    """`level_spacing.sampling_noise_floor` is a ruler for a different quantity.

    It evaluates the GUE nearest-neighbour SPACING density and assumes bin
    counts are multinomial over N independent draws. This histogram counts
    about 3N pairs from N zeros, every zero in several of them, correlated.
    Using it here understates the noise by about 1.6x at every height -- which
    is exactly enough to turn "the fit is at the noise" into "something
    remains at 1.7x the noise".

    The gate that fails if `check_pair_correlation` goes back to it.
    """
    from rh_research_engine.symbolic.level_spacing import sampling_noise_floor

    result = check_pair_correlation(5000.0)
    centres = np.array(result.centres)
    borrowed = sampling_noise_floor(result.zeros, centres, centres[1] - centres[0])

    assert result.noise_floor > 1.4 * borrowed, (result.noise_floor, borrowed)


def test_the_noise_floor_predicts_a_known_answer():
    """The positive control for the estimator itself.

    A Poisson process has `R_2 = 1` identically, so every departure from 1 is
    the estimator and nothing else -- which makes the true mean absolute
    deviation directly observable, and the floor's prediction checkable rather
    than merely plausible.
    """
    from rh_research_engine.symbolic.pair_correlation import pair_sampling_noise_floor

    rng = np.random.default_rng(5)
    observed = []
    predicted = []
    for _ in range(12):
        points = np.cumsum(rng.exponential(1.0, size=20_000))
        density = measured_density(points, window=3.0, bins=30).density
        observed.append(np.abs(density - 1.0).mean())
        predicted.append(pair_sampling_noise_floor(points, window=3.0, bins=30))
    ratio = np.mean(observed) / np.mean(predicted)
    assert 0.85 < ratio < 1.15, (ratio, np.mean(observed), np.mean(predicted))


def test_a_sample_too_short_to_estimate_its_own_scatter_is_refused():
    """A floor computed from chunks of nothing is an estimate of its own error."""
    from rh_research_engine.symbolic.pair_correlation import pair_sampling_noise_floor

    with pytest.raises(ValueError, match="too few"):
        pair_sampling_noise_floor(np.arange(100.0), window=3.0, bins=30)
    with pytest.raises(ValueError, match="scatter"):
        pair_sampling_noise_floor(np.arange(5000.0), window=3.0, bins=30, chunks=1)


def test_the_universal_curve_is_excluded_harder_as_the_sample_grows(
    lower_order_check,
):
    """The sharpest form of the finding, and the one a fudge cannot fake.

    A systematic error stays put while the noise falls, so its ratio to the
    noise GROWS with the sample. A correct curve's residual falls with the
    noise, so its ratio stays flat at about one. Montgomery goes 1.6x -> 4.8x
    between T = 5000 and T = 2e5; Conrey-Snaith sits at 1.08x and 1.07x.
    """
    low = lower_order_check(5000.0)
    high = lower_order_check(200000.0)

    montgomery_low = low.mean_deviation / low.noise_floor
    montgomery_high = high.mean_deviation / high.noise_floor
    assert montgomery_high > 2 * montgomery_low, (montgomery_low, montgomery_high)

    # And the arithmetic curve does not do that -- it tracks the noise.
    for result in (low, high):
        ratio = result.lower_order_deviation / result.noise_floor
        assert 0.6 < ratio < 1.6, (result.height, ratio)


def test_the_arithmetic_curve_wins_by_more_at_greater_height(lower_order_check):
    """Both curves are asymptotic, so the direction is what carries meaning.

    A lower-order form that were merely a fudge closer to this particular
    sample would not improve RELATIVE to Montgomery as the sample changes.
    """
    low = lower_order_check(5000.0)
    high = lower_order_check(30000.0)
    assert (
        high.lower_order_deviation / high.mean_deviation
        < low.lower_order_deviation / low.mean_deviation
    ), (
        low.lower_order_deviation / low.mean_deviation,
        high.lower_order_deviation / high.mean_deviation,
    )


def test_the_lower_order_curve_is_absent_unless_asked_for():
    """It costs a Euler product per quadrature point, so it is opt-in -- and
    an absent measurement must read as absent rather than as zero deviation.
    """
    plain = check_pair_correlation(5000.0)
    assert plain.lower_order == []
    assert plain.lower_order_deviation is None
    assert plain.curve_separation is None


def test_the_reduced_quadrature_stays_inside_its_budget():
    """`BIN_SAMPLES` was cut from 21 to 9 to make CI cheaper. This is what
    stops that from having quietly moved the published deviations.

    Neither existing test can see it. The bin-average test derives BOTH its
    expected value and the production one from `BIN_SAMPLES`, so it is
    invariant to the count; the refinement test varies `ELL_BANDS` instead. So
    cutting this to five -- whose error is 1.4e-4, above the 9.2e-5 that
    `ELL_BANDS` already contributes -- would leave the suite green while the
    numbers in `docs/research/pair-correlation-lower-order.md` drifted.

    Raised by review on the commit that made the cut, which is the right place
    for it to have been caught and not where I caught it.
    """
    from rh_research_engine.symbolic.conrey_snaith import (
        BIN_SAMPLES,
        QUADRATURE_BUDGET,
        QUADRATURE_REFERENCE_SAMPLES,
    )
    from rh_research_engine.symbolic.pair_correlation import lower_order_bin_average

    edges = np.linspace(0.0, 3.0, 31)
    # Two bands spanning the l the reachable heights actually reach.
    nodes, weights = np.array([4.5, 6.5]), np.array([0.5, 0.5])

    def curve(samples):
        return lower_order_bin_average(
            edges, nodes, weights, samples=samples, prime_limit=100_000
        )

    reference = curve(QUADRATURE_REFERENCE_SAMPLES)
    chosen = np.abs(curve(BIN_SAMPLES) - reference).mean()
    assert chosen < QUADRATURE_BUDGET, (BIN_SAMPLES, chosen, QUADRATURE_BUDGET)

    # And the budget has teeth: the next count down is outside it, so this is
    # a bound the choice actually had to clear rather than one anything passes.
    coarse = np.abs(curve(5) - reference).mean()
    assert coarse > QUADRATURE_BUDGET, (coarse, QUADRATURE_BUDGET)


def test_the_height_weights_count_zeros_and_not_pairs():
    """The weights must be the retained ZEROS' height distribution.

    `density` divides by `len(unfolded)`, so its expectation is
    `(1/N) sum_n R_2(.; l_n)` -- every retained zero weighted once. Weighting a
    band by the pairs observed near it weights each height by a random quantity
    that fluctuates with the spacings under test, making the curve compared
    against a different, data-dependent statistic from the density returned.

    Checked against the band fractions computed straight off the ordinates, so
    it fails on pair weighting rather than merely on the weights summing to one
    -- which they do either way, and which is all the neighbouring test saw.
    """
    from rh_research_engine.symbolic.riemann_siegel import zero_ordinates

    raw = zero_ordinates(5000.0)
    ordinates = np.asarray(raw, dtype=float)[SKIP_LOWEST:]
    unfolded = unfold(raw)[SKIP_LOWEST:]
    bands = 8
    statistics = measured_density(
        unfolded, ordinates=ordinates, window=3.0, bins=30, ell_bands=bands
    )

    ell = np.log(ordinates / (2 * np.pi))
    edges = np.linspace(ell.min(), ell.max(), bands + 1)
    band = np.clip(np.searchsorted(edges, ell, side="right") - 1, 0, bands - 1)
    expected = np.bincount(band, minlength=bands) / len(ordinates)

    assert np.allclose(statistics.ell_weights, expected, atol=1e-12), (
        statistics.ell_weights,
        expected,
    )
