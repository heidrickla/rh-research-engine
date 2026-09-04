"""Moments of |zeta(1/2+it)|: the constants, the integrator, and a refusal.

The refusal is the substance. Fitting the moment polynomial and reading off its
leading coefficient is the obvious way to test the conjecture at k = 3, and it
does not work at reachable heights -- which is demonstrable, because at k = 2
the coefficient is a theorem.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from pydantic import ValidationError

from rh_research_engine.contracts.epistemic import RIGOROUS, Confidence
from rh_research_engine.symbolic.moments import (
    MomentFit,
    _extract,
    arithmetic_factor,
    fit_moment,
    keating_snaith_constant,
    measured_moment,
    moment_constant,
    random_matrix_factor,
    second_moment_asymptotic,
)

# --- the constants --------------------------------------------------------


def test_the_random_matrix_factor_reproduces_the_tabulated_constants():
    """`g_k` is 1, 2, 42, 24024 -- the numbers Keating and Snaith identified.

    Exact rationals, not floats: `g_3 = 42` is a check that a rounding would
    blur into "about 42", and the whole point of the random-matrix half is
    that it is a closed form.
    """
    assert [int(keating_snaith_constant(k)) for k in (1, 2, 3, 4)] == [
        1, 2, 42, 24024,
    ]
    assert random_matrix_factor(1) == 1
    assert random_matrix_factor(2) == sp.Rational(1, 12)
    for k in (1, 2, 3, 4):
        assert random_matrix_factor(k).is_Rational

    with pytest.raises(ValueError):
        random_matrix_factor(0)


def test_the_arithmetic_factor_matches_its_closed_forms():
    """`a_1 = 1` identically and `a_2 = 1/zeta(2)`, both derivable.

    The Euler product is the definition; these are two places it collapses. If
    the inner sum or the exponent were wrong the product would still converge
    to something, and only a closed form catches which something.
    """
    assert arithmetic_factor(1) == pytest.approx(1.0, abs=1e-12)
    assert arithmetic_factor(2) == pytest.approx(float(6 / sp.pi**2), rel=1e-6)


def test_the_moment_constants_match_the_two_theorems():
    """`c_1 = 1` (Hardy-Littlewood) and `c_2 = 1/(2 pi^2)` (Ingham)."""
    assert moment_constant(1) == pytest.approx(1.0, abs=1e-12)
    assert moment_constant(2) == pytest.approx(float(1 / (2 * sp.pi**2)), rel=1e-6)


# --- the integrator -------------------------------------------------------


def test_the_integrator_matches_the_k_equals_one_theorem():
    """Against the FULL asymptotic, lower-order term included.

    `log(T/2pi) + 2 gamma - 1`, not `log T`. The residual should be the size of
    the known error term, which is `O(T^{-2/3})` for the mean -- about 5e-4 at
    T = 10^4.
    """
    for height in (5000.0, 20000.0):
        measured = measured_moment(1, height)
        residual = abs(measured - second_moment_asymptotic(height))
        # The theorem's error term is O(T^{1/3+eps}) for the integral, so
        # O(T^{-2/3+eps}) for the mean. A fixed tolerance would be too loose
        # at one end and too tight at the other -- 7.9e-3 at T = 5000 against
        # 3.2e-4 at 10^4 -- so the bar scales the way the theorem does.
        assert residual < 5 * height ** (-2 / 3), (
            f"at T = {height:g} the residual is {residual:.2e}, past what the "
            "error term allows"
        )


def test_the_leading_term_alone_is_badly_wrong_where_the_answer_is_known():
    """Which is why the test above compares against the full statement.

    At k = 1 the gap between `log T` and the truth is a constant 1.68 -- about
    eighteen per cent at these heights. Any k >= 3 discrepancy of that size,
    measured against a leading term, would therefore say nothing at all.
    """
    height = 20000.0
    measured = measured_moment(1, height)
    assert abs(measured - float(np.log(height))) == pytest.approx(1.684, abs=0.01)
    assert abs(measured - second_moment_asymptotic(height)) < 5e-3


def test_the_quadrature_has_converged_at_the_default_density():
    """Twenty points per unit is measured, not assumed.

    Simpson on a smooth oscillation converges fast, and the oscillation
    density grows only like `log t` -- so doubling the grid should move
    nothing.
    """
    coarse = measured_moment(2, 5000.0, per_unit=20)
    fine = measured_moment(2, 5000.0, per_unit=80)
    assert coarse == pytest.approx(fine, rel=1e-8)


def test_a_moment_needs_a_positive_range_and_a_positive_index():
    with pytest.raises(ValueError):
        measured_moment(0, 1000.0)
    with pytest.raises(ValueError):
        measured_moment(1, 0.0)


# --- the refusal ----------------------------------------------------------


def test_too_few_heights_cannot_determine_the_polynomial():
    """A degree-`k^2` fit needs more than `k^2` points. Refused, not attempted."""
    heights = np.geomspace(1000.0, 40000.0, 6)
    with pytest.raises(ValueError) as caught:
        _extract(3, heights, np.ones(6))
    assert "degree-9" in str(caught.value)


def test_the_extraction_is_not_usable_and_the_record_says_so():
    """The finding: fitting cannot recover `c_k` at reachable heights.

    Demonstrated rather than asserted. The same fit is run at k = 2, where the
    coefficient is Ingham's theorem, on the same heights -- and it is out by
    hundreds of per cent, because `log(T/2pi)` spans a factor of about 1.5
    here and a degree-4 polynomial cannot be separated across that.

    So `calibration_error` travels with every fit. An extracted coefficient
    without the demonstrated error of the method that produced it is a number
    nobody can weigh.
    """
    heights = np.geomspace(800.0, 20000.0, 7)
    result = fit_moment(2, heights)

    assert result.calibration_error > 0.10
    assert not result.extraction_is_usable
    # And the ratio to the leading term is well off 1, consistently.
    assert all(1.05 < ratio < 1.25 for ratio in result.leading_ratio)


def test_a_measured_moment_cannot_claim_a_rigorous_confidence():
    """A finite integral against a conjecture about the limit."""
    for confidence in sorted(RIGOROUS, key=str):
        with pytest.raises(ValidationError) as caught:
            MomentFit(k=2, confidence=confidence)
        assert "the conjecture is open" in str(caught.value) or "open" in str(
            caught.value
        )
    assert MomentFit(k=2).confidence is Confidence.NUMERICAL


def test_usability_needs_a_calibration_that_was_actually_run():
    """Zero is not a small error; it is no measurement.

    A default of 0.0 passing "under ten per cent" would mark every fit usable
    on a record where the calibration never ran.
    """
    assert not MomentFit(k=3).extraction_is_usable
    assert not MomentFit(k=3, calibration_error=0.0).extraction_is_usable
    assert MomentFit(k=3, calibration_error=0.02).extraction_is_usable
    assert not MomentFit(k=3, calibration_error=0.53).extraction_is_usable


# --- the full polynomial, which is the sharp test -------------------------


def test_the_proven_polynomial_agrees_where_the_leading_term_does_not():
    """The whole degree-4 polynomial against one term of it.

    Both are about the same moment and the same theorem. Against the full
    polynomial the measurement agrees to 1e-4; against its leading term alone,
    in the same variable, it is out by more than a hundred per cent. So the
    discrepancy was lower-order terms, and a comparison against a leading term
    was never going to see the conjecture.
    """
    from rh_research_engine.symbolic.moments import (
        PROVEN_FOURTH_MOMENT,
        evaluate_polynomial,
    )

    for height in (20000.0, 80000.0):
        measured = measured_moment(2, height)
        full = evaluate_polynomial(PROVEN_FOURTH_MOMENT, height)
        leading = evaluate_polynomial((PROVEN_FOURTH_MOMENT[0], 0, 0, 0, 0), height)
        assert abs(measured - full) / full < 1e-3
        assert abs(measured - leading) / leading > 1.0


def test_the_transcribed_leading_coefficient_matches_the_one_computed_here():
    """A transcription check on the polynomial, from a number derived here.

    The coefficients came from a paper; the leading one is also `c_2` from the
    Euler product in this module. They must agree, or one of the two is wrong
    and the comparison is against something that is not the fourth moment.
    """
    from rh_research_engine.symbolic.moments import (
        NAIVE_RMT_FOURTH_MOMENT,
        PROVEN_FOURTH_MOMENT,
    )

    # The paper states its numbers are obtained "via truncation", not
    # rounding -- so the printed 0.050660 is the true 0.0506606 with the tail
    # cut, and sits BELOW it by up to one unit in the last place. Asserting
    # `approx` at rounding tolerance fails on a correct transcription.
    computed = moment_constant(2)
    assert 0.0 <= computed - PROVEN_FOURTH_MOMENT[0] < 1e-6, (
        f"transcribed {PROVEN_FOURTH_MOMENT[0]} against computed {computed}"
    )
    # And the two polynomials SHARE it -- that is what makes the comparison
    # isolate the lower-order terms rather than the whole prediction.
    assert NAIVE_RMT_FOURTH_MOMENT[0] == PROVEN_FOURTH_MOMENT[0]
    assert NAIVE_RMT_FOURTH_MOMENT[1:] != PROVEN_FOURTH_MOMENT[1:]


def test_the_data_follows_the_theorem_and_not_naive_random_matrix_theory():
    """The result: the universal part is right, the subleading part is not.

    Two polynomials with the same leading coefficient. The measurement sits
    1e-4 from the proven one and about 6% from the RMT one, and the 6% does
    not shrink with height -- so it is not a finite-height effect but a real
    failure of RMT at subleading order. The gap between them is where the
    arithmetic lives.
    """
    from rh_research_engine.symbolic.moments import check_fourth_moment

    result = check_fourth_moment(np.array([20000.0, 80000.0]))
    assert result.follows_the_proven_polynomial
    assert max(result.proven_error) < 1e-3
    assert min(result.rmt_error) > 0.05
    # Not shrinking: the RMT departure at the top is no smaller than below.
    assert result.rmt_error[-1] > result.rmt_error[0] * 0.9


def test_the_polynomial_source_travels_with_the_record():
    """Transcribed coefficients without their provenance are folklore."""
    from rh_research_engine.symbolic.moments import (
        MOMENT_POLYNOMIAL_SOURCE,
        FourthMomentCheck,
    )

    assert "Hiary" in MOMENT_POLYNOMIAL_SOURCE
    assert FourthMomentCheck().source == MOMENT_POLYNOMIAL_SOURCE
    assert not FourthMomentCheck().follows_the_proven_polynomial, (
        "no measurement is not a verdict"
    )
