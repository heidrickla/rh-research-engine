"""The decay rate, and the ways a fit like this manufactures a result.

Three of them, each with a test: an estimator with a positive bias reports an
amplitude in every band and a decay from the band sizes alone; overlapping
bands agree by construction; and an exponent quoted without its range reads as
asymptotic when it is not.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from pydantic import ValidationError

from rh_research_engine.contracts.epistemic import RIGOROUS
from rh_research_engine.symbolic.level_spacing import (
    exact_gue_bin_average,
    exact_gue_density,
)
from rh_research_engine.symbolic.spacing_decay import (
    EFFECTIVE_NOT_ASYMPTOTIC,
    DecayFit,
    amplitude,
    measure_decay,
)

EDGES = np.linspace(0.0, 3.0, 26)


@lru_cache(maxsize=1)
def _inverse_cdf() -> tuple[np.ndarray, np.ndarray]:
    """The exact law's CDF on a grid, computed ONCE.

    Cached at module scope because it is twenty thousand points and each is
    three Fredholm determinants -- rebuilt per call it dominated this file's
    runtime at three minutes.
    """
    grid = np.linspace(0.0, 9.0, 20001)
    cumulative = np.cumsum(exact_gue_density(grid)) * (grid[1] - grid[0])
    return grid, cumulative / cumulative[-1]


def _drawn(count: int, seed: int = 0) -> np.ndarray:
    """Spacings drawn FROM the exact law: no systematic, by construction."""
    grid, cumulative = _inverse_cdf()
    generator = np.random.default_rng(seed)
    return np.interp(generator.random(count), cumulative, grid)


# --- the estimator --------------------------------------------------------


def _noise_scale(count: int) -> float:
    """`sqrt(sum sigma^2)`: the size of a residual made only of noise.

    The right yardstick for an amplitude, which is `sqrt(sum s^2)` over the
    same bins. Comparing it against a per-bin sigma would be comparing a
    vector norm against one of its components -- out by a factor of five at
    25 bins.
    """
    variance = np.maximum(exact_gue_bin_average(EDGES), 0.0) / (
        count * (EDGES[1] - EDGES[0])
    )
    return float(np.sqrt(np.sum(variance)))


def test_the_estimator_measures_nothing_where_there_is_nothing():
    """The property the whole measurement rests on.

    A positive bias would report an amplitude in every band, and since the
    bias scales with the noise it would fall with band size -- producing a
    decay out of nothing but how many zeros each band happened to hold.

    This is not hypothetical. It is how the bin-average bug was found: the
    estimator was comparing a histogram against the density at the bin
    centres, which floors every residual, and the null test showed it as a
    bias of +0.44 of the noise term.
    """
    for count in (20000, 100000):
        values = [amplitude(_drawn(count, seed), EDGES)[0] for seed in range(4)]
        scale = _noise_scale(count)
        assert np.mean(values) < scale, (
            f"at N = {count} the estimator finds {np.mean(values):.5f} in "
            f"samples that contain nothing (noise scale {scale:.5f})"
        )
        # And far under what the real bands measure, which is the comparison
        # that matters: those run 0.10 to 0.24.
        assert np.mean(values) < 0.03


def test_a_planted_systematic_is_recovered():
    """And it must measure something where there IS something.

    An estimator that returns zero on everything would pass the test above.
    Spacings drawn from the exact law, then a fraction of them shifted, give a
    known departure -- and the amplitude has to grow with it.
    """
    drawn = _drawn(100000, seed=3)
    plain, _ = amplitude(drawn, EDGES)

    generator = np.random.default_rng(11)
    nudged = drawn.copy()
    chosen = generator.random(len(nudged)) < 0.05
    nudged[chosen] += 0.3
    disturbed, _ = amplitude(nudged, EDGES)

    assert disturbed > 10 * max(plain, 1e-6), (
        f"a planted departure measured {disturbed:.5f} against {plain:.5f} "
        "for the undisturbed sample"
    )


# --- the fit --------------------------------------------------------------


def test_overlapping_bands_are_refused():
    """A band inside another shares its zeros and agrees by construction.

    The same error as the nested-range correlation that read 0.98 and meant
    nothing. A fit through such points measures the sharing.
    """
    ordinates = np.cumsum(_drawn(50000)) * 0.5 + 1000.0
    with pytest.raises(ValueError) as caught:
        measure_decay(ordinates, [(1e3, 3e4), (1e4, 1e5)])
    assert "overlap" in str(caught.value)


def test_a_band_with_no_measurable_amplitude_is_refused_not_fitted(monkeypatch):
    """Zero is not a data point on a log-log fit.

    A band whose systematic sits under its own noise carries no amplitude.
    Fitting it as zero would drag the exponent; clamping it to something small
    would put a floor under the fit and bend it. Refusing says which band and
    why.

    Reached through the module's own seam. Feeding it synthetic ordinates does
    not work -- `measure_decay` unfolds what it is given, so spacings built to
    follow the exact law arrive at the histogram unfolded a second time and no
    longer do. That is not a fixture worth constructing to reach one branch.
    """
    from rh_research_engine.symbolic import spacing_decay

    monkeypatch.setattr(spacing_decay, "amplitude", lambda spacings, edges: (0.0, 0.01))
    ordinates = np.sort(np.random.default_rng(2).uniform(1e3, 2e5, 60000))

    with pytest.raises(ValueError) as caught:
        spacing_decay.measure_decay(ordinates, [(1e3, 5e4), (5e4, 2e5)])
    assert "no systematic above its noise" in str(caught.value)


def test_a_fitted_decay_cannot_claim_a_rigorous_confidence():
    """Weighted least squares through six points, about an unreachable limit."""
    for confidence in sorted(RIGOROUS, key=str):
        with pytest.raises(ValidationError) as caught:
            DecayFit(confidence=confidence)
        assert "no computation reaches" in str(caught.value)


def test_the_caveat_about_the_lever_cannot_be_dropped():
    """"Decays like (log T)^-1.6" detached from its range reads as asymptotic.

    It is not: over a lever of 1.8 a two-term expansion fits as well as a
    single power. The sentence travels with the record or the record is not
    built.
    """
    for attempt in ("", "decays like 1/(log T)^1.6", "asymptotically"):
        with pytest.raises(ValidationError) as caught:
            DecayFit(caveat=attempt)
        assert "not the asymptotic one" in str(caught.value)
    assert DecayFit().caveat == EFFECTIVE_NOT_ASYMPTOTIC


def test_steeper_than_inverse_log_needs_two_sigma():
    """A comparison, not a preference between two fits.

    An exponent above 1 is not by itself evidence against 1/log T; the claim
    needs the separation to exceed what the fit's own uncertainty allows.
    """
    assert DecayFit(exponent=1.61, exponent_error=0.17).steeper_than_inverse_log
    assert not DecayFit(exponent=1.61, exponent_error=0.40).steeper_than_inverse_log
    assert not DecayFit(exponent=1.05, exponent_error=0.17).steeper_than_inverse_log
    assert not DecayFit().steeper_than_inverse_log, "no fit is not a verdict"
