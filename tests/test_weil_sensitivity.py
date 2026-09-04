"""What the Weil form would have caught, and the three ways of not measuring it.

The threshold is the number that makes `weil_positivity`'s null result mean
anything, and each test here corresponds to a way the measurement was actually
got wrong: a removal that removed nothing, a sign read below the noise floor,
and a control that would have made any change look like a violation.
"""

from __future__ import annotations

import numpy as np
import pytest

from rh_research_engine.experiments.weil_positivity import classify, noise_floor
from rh_research_engine.experiments.weil_sensitivity import (
    PAIR_LIMIT,
    STRIP,
    WEIGHT_CUTOFF,
    _fit_with_error,
    _shuffle_control,
    is_negative,
    perturbed_form,
    quadruple,
    run,
    secular_threshold,
    threshold,
)


@pytest.fixture(scope="module")
def zeros():
    from rh_research_engine.symbolic.riemann_siegel import first_zero_ordinates

    return first_zero_ordinates(2000)


def test_the_three_verdicts_are_distinguishable():
    """UNRESOLVED must not collapse into either of the claims about zeta.

    Forced through hand-built spectra rather than by hunting for a singular
    matrix, so both branches run on every machine. The parameters that produce
    each one are a fact about float64, not about the code being tested.
    """
    assert classify(np.array([1e-3, 1.0]), 2)[0] == "POSITIVE"
    assert classify(np.array([-1e-3, 1.0]), 2)[0] == "REFUTED"
    floor = noise_floor(np.array([0.0, 1.0]), 2)
    assert classify(np.array([0.5 * floor, 1.0]), 2)[0] == "UNRESOLVED"
    assert classify(np.array([-0.5 * floor, 1.0]), 2)[0] == "UNRESOLVED", (
        "a negative eigenvalue below the floor is a fact about the arithmetic"
    )
    assert floor > 0


def test_the_floor_carries_the_entry_error_and_not_only_the_eigensolver():
    """Which of the two terms dominates depends on the width, so test both.

    The first floor was `size * eps * lambda_max` alone -- the eigensolver's
    backward error and nothing about the error already in the entries, which
    come from a fifty-thousand-term prime sum and a quadrature. At size 12 and
    sigma = 0.2 that gave 1.5e-17 and let a spurious refutation at -3.3e-15
    stand.

    The terms cross at `lambda_max = ENTRY_ERROR/eps`. A first version of this
    test asserted the entry error always dominates, which is false at large
    lambda_max -- the two regimes are the point. A second version pinned the
    crossover near 9, which was a consequence of the constant rather than a
    property of the floor, and broke when the constant was measured again.
    """
    from rh_research_engine.experiments.weil_positivity import ENTRY_ERROR

    eps = float(np.finfo(float).eps)
    crossover = ENTRY_ERROR / eps

    assert noise_floor(np.array([0.0, 4.0]), 2) == 2 * (eps * 4.0 + ENTRY_ERROR)
    assert noise_floor(np.array([0.0, 1.0]), 8) == 4 * noise_floor(np.array([0.0, 1.0]), 2)

    # sigma = 0.2, lambda_max 5.6e-3: far below the crossover, entries dominate.
    assert 5.6e-3 < crossover / 100
    # sigma = 0.03 at size 12, lambda_max 23.8: within a factor of two of the
    # crossover, so the two terms are comparable and neither can be neglected.
    assert 0.5 < 23.8 / crossover < 2.0, crossover

    # And at the width that produced the false refutation, the floor now covers it.
    assert noise_floor(np.array([-3.3e-15, 5.6e-3]), 12) > 3.3e-15
    assert noise_floor(np.array([-3.3e-15, 5.6e-3]), 12) < 1e-10, "and is not a blanket"


def test_a_collision_at_zero_displacement_is_a_double_zero(zeros):
    """The identity that says the quadruple formula is the right one.

    At eta = 0 the four gamma_rho = +/- mu +/- i eta collapse to two, twice
    over, so the contribution must be exactly twice a single on-line pair.
    """
    d = np.log(np.outer([1.0, 2.0, 3.0], [1.0, 0.5, 1 / 3]))
    mu, sigma = 17.5, 0.03
    pair = 2.0 * np.exp(-(sigma**2) * mu**2) * np.cos(mu * d)
    assert np.allclose(quadruple(mu, 0.0, d, sigma), 2.0 * pair, rtol=0, atol=1e-14)


def test_the_quadruple_is_even_in_the_displacement(zeros):
    """Reversing eta exchanges the two zeros, so it cannot change anything."""
    d = np.log(np.outer([1.0, 2.0, 5.0], [1.0, 0.5, 0.2]))
    for eta in (0.01, 0.2, 0.49):
        a = quadruple(17.5, eta, d, 0.03)
        b = quadruple(17.5, -eta, d, 0.03)
        assert np.allclose(a, b, rtol=0, atol=1e-13)


def test_the_removal_actually_removes(zeros):
    """The bug that changed every threshold in a scaling table by up to 3x.

    Matching ordinates by value against hard-coded literals missed by two bits
    and removed nothing, which builds the SAME matrix as no removal at all --
    the quadruple added without deleting the pairs it replaces. So assert the
    two differ, which is exactly what a no-op removal cannot do.
    """
    size, sigma = 8, 0.03
    collided = perturbed_form(zeros, size, sigma, (0, 1), 0.0)
    d = np.log(np.outer(np.arange(1.0, size + 1), 1.0 / np.arange(1.0, size + 1)))
    mu = float(zeros[[0, 1]].mean())
    not_removed = perturbed_form(zeros, size, sigma) + quadruple(mu, 0.0, d, sigma)
    assert not np.allclose(collided, not_removed), "the pair was not actually removed"

    with pytest.raises(ValueError, match="removal took"):
        perturbed_form(zeros, size, sigma, (0, 0), 0.0)


def test_moving_a_zero_along_the_line_never_breaks_positivity(zeros):
    """The control that separates off-line-ness from mere change.

    h = |phi_hat|^2 >= 0 for real ordinates however they are arranged, so no
    displacement along the critical line can make the form negative. If one
    does, the threshold below is measuring sensitivity to perturbation rather
    than to a failure of RH.
    """
    size, sigma = 12, 0.03
    base = perturbed_form(zeros, size, sigma)

    # THE ROUND TRIP IS WHAT GIVES THIS TEETH. Every assertion below holds on a
    # build that ignores `along` entirely, because the form stays positive; a
    # first attempt asserted only that the matrix had changed, which the
    # removal already guarantees. Taking one ordinate out and putting it back
    # unmoved must reproduce the original matrix exactly, and cannot pass
    # unless the displaced ordinate is genuinely re-entered.
    replaced = perturbed_form(zeros, size, sigma, (0,), along=float(zeros[0]))
    assert np.allclose(replaced, base, rtol=0, atol=1e-12)

    for step in (0.05, 0.5, 2.0, 5.0):
        moved = perturbed_form(zeros, size, sigma, (0,), along=float(zeros[0]) + step)
        assert np.abs(moved - base).max() > 1e-6 * np.abs(base).max(), step
        assert not is_negative(moved, size), step


def test_a_negative_below_the_floor_is_not_a_detection():
    """`is_negative` must read the floor, not the sign.

    Nothing else here would notice: at the widths the threshold is measured on,
    the margin stands 1e10 times the floor and the distinction never arises. It
    arises exactly where the bisection would otherwise report a spectacularly
    sensitive search, which is the case this whole verdict exists for.
    """
    floor = noise_floor(np.array([0.0] * 7 + [1.0]), 8)
    assert not is_negative(np.diag([1.0] * 7 + [-0.5 * floor]), 8)
    assert is_negative(np.diag([1.0] * 7 + [-10.0 * floor]), 8)


def test_a_collision_on_the_line_never_breaks_positivity(zeros):
    """A double zero is still a zero on the line, so it cannot refute anything."""
    for size in (8, 12, 16):
        assert not is_negative(perturbed_form(zeros, size, 0.03, (0, 1), 0.0), size)


def test_the_threshold_is_a_real_displacement_and_is_found(zeros):
    """Bracketed on both sides: below eta* positive, above it negative."""
    size, sigma = 16, 0.03
    eta = threshold(zeros, size, sigma, (0, 1))
    assert eta is not None and 0.0 < eta < STRIP
    assert not is_negative(perturbed_form(zeros, size, sigma, (0, 1), eta * 0.9), size)
    assert is_negative(perturbed_form(zeros, size, sigma, (0, 1), eta * 1.1), size)


def test_a_basis_too_small_to_detect_anything_says_so(zeros):
    """None, not zero and not an eta outside the strip.

    Re(rho) = 1/2 + eta and zeta has no zeros with Re >= 1, so a threshold at
    1/2 would name a displacement zeta cannot make. Returning the bisection's
    upper bound there would report the least sensitive possible basis as though
    it had a finite reach.
    """
    assert threshold(zeros, 4, 0.03, (0, 1)) is None


def test_a_larger_basis_detects_a_smaller_displacement(zeros):
    """Monotone in the direction the search is supposed to improve."""
    found = [threshold(zeros, size, 0.03, (0, 1)) for size in (8, 12, 16, 20)]
    assert all(e is not None for e in found)
    assert found == sorted(found, reverse=True), found


def test_the_run_reports_a_threshold_and_clean_controls(zeros):
    result = run(size=16, sigma=0.03, pairs=6, ordinates=zeros)
    assert result.metrics["controls_fired"] == 0.0
    assert result.metrics["measured_pairs"] >= 3
    assert 0.0 < result.metrics["smallest_threshold"] < STRIP
    assert result.metrics["margin_over_floor"] > 1e4


def test_a_singular_setting_is_cut_rather_than_reported(zeros):
    """The verdict that must not read as "no violation found".

    At sigma = 0.2 the form's condition number passes 1e15 by size 12, every
    perturbation registers as negative, and an unguarded bisection would report
    a spectacularly sensitive search. It has instead run out of arithmetic.
    """
    result = run(size=16, sigma=0.2, pairs=4, ordinates=zeros)
    assert result.metrics["measured_pairs"] == 0.0
    assert "smallest_threshold" not in result.metrics
    assert any("CUT" in o for o in result.observations)


def test_the_shuffle_control_can_say_no():
    """The control that carries the claim must be able to fail.

    Adding any second regressor to a small fit moves the first coefficient, so
    "it moved towards the prediction" is not evidence. The permutation keeps
    both marginals and destroys only the pairing -- so on data where the second
    column carries nothing, permuting it must be as good as the real thing.
    """
    generator = np.random.default_rng(11)
    height_squared = np.linspace(300.0, 4000.0, 14)
    noise = generator.normal(scale=0.05, size=height_squared.size)

    # A second column with no relationship to the target at all.
    irrelevant = generator.normal(size=height_squared.size)
    design = np.column_stack([height_squared, irrelevant, np.ones_like(height_squared)])
    target = 0.00045 * height_squared + noise
    assert _shuffle_control(design, target, 0.00045, 400) > 0.2, (
        "a column carrying nothing must not beat its own permutations"
    )

    # And a second column that genuinely carries the correction.
    informative = -0.0004 * height_squared + generator.normal(scale=0.1, size=height_squared.size)
    design = np.column_stack([height_squared, informative, np.ones_like(height_squared)])
    target = 0.00045 * height_squared + 0.9 * informative + noise
    assert _shuffle_control(design, target, 0.00045, 400) < 0.1


def test_the_coefficients_carry_standard_errors():
    """Three coefficients were recorded bare, in the session that fixed three
    other experiments for exactly that."""
    design = np.column_stack([np.linspace(1.0, 10.0, 12), np.ones(12)])
    target = 2.0 * design[:, 0] + 1.0
    coefficients, errors = _fit_with_error(design, target)
    assert abs(coefficients[0] - 2.0) < 1e-9
    assert errors[0] >= 0.0
    noisy = target + np.random.default_rng(3).normal(scale=0.5, size=12)
    _, noisy_errors = _fit_with_error(design, noisy)
    assert noisy_errors[0] > errors[0], "noise must widen the bar"


def test_the_pair_count_saturates_for_the_bisection_only(zeros):
    """The count stops rising, and the reason is the bisection, not the data.

    Beyond 16 every further pair is dropped for having no threshold inside the
    critical strip, so asking for more changes nothing -- which is what makes
    the default a measurement rather than a preference.

    What it is a measurement OF was stated wrongly here. The bisection brackets
    inside `[0, STRIP]` and cannot speak above it, so a pair needing eta = 1.026
    and a pair needing 0.6 both come back as no-threshold. The saturation is the
    censoring, and the companion test below shows the closed form does not
    saturate at the same place.
    """
    counts = {
        requested: run(size=20, sigma=0.03, pairs=requested, ordinates=zeros).metrics[
            "measured_pairs"
        ]
        for requested in (PAIR_LIMIT, PAIR_LIMIT + 4)
    }
    assert len(set(counts.values())) == 1, counts
    fewer = run(size=20, sigma=0.03, pairs=12, ordinates=zeros).metrics["measured_pairs"]
    assert fewer < counts[PAIR_LIMIT], (fewer, counts)


def test_the_recorded_height_coefficient_matches_the_prediction(zeros):
    """The number the whole experiment is about, with its bar and its control."""
    result = run(size=20, sigma=0.03, ordinates=zeros)
    predicted = result.metrics["predicted_height_coefficient"]
    coefficient = result.metrics["height_coefficient"]
    error = result.metrics["height_coefficient_error"]
    assert abs(coefficient - predicted) < 2 * error, (coefficient, error, predicted)
    assert result.metrics["shuffled_gap_p_value"] < 0.05
    # And the honest caveat: the two fits are not separated by their own bars.
    assert result.metrics["gap_separation_sigma"] < 2.0
    assert any("DO NOT ESTABLISH" in o for o in result.observations)


def test_the_two_routes_agree_where_the_expansion_is_valid(zeros):
    """A bisected sign and a closed-form root, sharing only the ordinates.

    `threshold` drives `eta` until the smallest eigenvalue crosses the floor.
    `secular_threshold` never forms a perturbed matrix at all: it solves the
    second-order problem on the constrained hyperplane. Different code path,
    different failure modes, so agreement is evidence and a gap is a bug in one
    of them.

    The tolerance tightens with size ON PURPOSE. The closed form is the
    SECOND-ORDER theory, exact as `eta -> 0`, so it must drift where `eta*` is
    large and must not where it is small. A single loose tolerance across all
    sizes would pass a routine that had the expansion order wrong.
    """
    for size, tolerance in ((14, 2e-2), (20, 3e-3), (28, 1e-4)):
        bisected = threshold(zeros, size, 0.03, (0, 1))
        closed = secular_threshold(zeros, size, 0.03, (0, 1))
        assert bisected is not None and closed is not None, size
        assert abs(bisected - closed) / closed < tolerance, (size, bisected, closed)


def test_the_threshold_is_conditional_on_the_support(zeros):
    """The correction: the recorded 0.0434 was a property of size 20, not of the method.

    Four orders of magnitude between size 20 and size 56, and every row of it
    far above the noise floor -- so this is not the margin running out. A
    threshold quoted without its support is not a statement about what the
    criterion detects.
    """
    found = {size: threshold(zeros, size, 0.03, (0, 1)) for size in (20, 28, 40, 56)}
    assert all(v is not None for v in found.values()), found
    assert found[20] / found[56] > 1e3, found
    ordered = [found[size] for size in (20, 28, 40, 56)]
    assert ordered == sorted(ordered, reverse=True), found


def test_the_support_and_cutoff_are_recorded_with_the_threshold(zeros):
    """A threshold whose support is not in the record can be misread as general.

    It was. The run must carry `t = log(size)/2` and the prime cutoff beside
    every threshold it reports.
    """
    for size in (14, 16):
        result = run(size=size, sigma=0.03, pairs=6, ordinates=zeros)
        assert result.metrics["prime_cutoff"] == float(size)
        assert result.metrics["support"] == pytest.approx(np.log(size) / 2.0)


def test_the_closed_form_speaks_where_the_bisection_is_censored(zeros):
    """`blind` was never a property of the basis.

    At size 10 the pair at height 48.9 has no threshold inside the strip, and
    the bisection can only decline. The closed form returns 1.026 -- a
    displacement zeta cannot make, which is a finding, where "the bracket ran
    out" is not.
    """
    assert threshold(zeros, 10, 0.03, (8, 9)) is None
    closed = secular_threshold(zeros, 10, 0.03, (8, 9))
    assert closed is not None and closed > STRIP, closed


def _unconstrained_threshold(zeros, size, sigma, mu_pair):
    """`secular_threshold` with the hyperplane dropped -- the comparison, not the code."""
    keep = zeros[zeros < WEIGHT_CUTOFF / sigma]
    mu = float(keep[list(mu_pair)].mean())
    matrix = perturbed_form(zeros, size, sigma, mu_pair, 0.0)
    u = np.log(np.arange(1, size + 1, dtype=float))
    v = np.exp(1j * mu * u)
    values, vectors = np.linalg.eigh(matrix)
    assert values.min() > 0
    root = vectors @ np.diag(values**-0.5) @ vectors.T
    response = np.vstack([(u * v).real, (u * v).imag]) @ root
    largest = float(np.linalg.eigvalsh(response @ response.T).max())
    return float(np.sqrt(1.0 / (4.0 * np.exp(-(sigma**2) * mu**2) * largest)))


def test_the_constraint_binds_weakly_and_less_as_the_basis_grows(zeros):
    """How much the hyperplane is worth, measured rather than predicted.

    Two predictions preceded this test and both were wrong: that dropping a
    constraint would visibly shrink the threshold, and then that dropping it
    entirely would change nothing measurable. It moves the fourth significant
    figure at size 20 and the eighth at size 40, and the first version of this
    test asserted agreement to 1e-6 and failed in the full suite.

    So the assertion is on the SHAPE that was actually measured -- a real shift
    that shrinks as the basis grows -- bounded loosely enough to be stable and
    tightly enough that a constraint which began to bind hard, or which stopped
    mattering at small sizes, would break it.
    """
    for size, ceiling in ((14, 3e-2), (20, 2e-3), (28, 1e-4)):
        constrained = secular_threshold(zeros, size, 0.03, (0, 1))
        free = _unconstrained_threshold(zeros, size, 0.03, (0, 1))
        shift = abs(constrained - free) / constrained
        assert shift < ceiling, (size, shift, ceiling)
        # Constraining can only raise a minimum, so `free` above `constrained`
        # would mean the hyperplane was never applied.
        assert constrained >= free, (size, constrained, free)

    def shift_at(size):
        constrained = secular_threshold(zeros, size, 0.03, (0, 1))
        return abs(constrained - _unconstrained_threshold(zeros, size, 0.03, (0, 1))) / constrained

    assert shift_at(40) < shift_at(14) / 100, (shift_at(40), shift_at(14))
