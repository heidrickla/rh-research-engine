"""A certificate is only worth what its inputs are worth.

Every test here is a way the certified path was actually wrong while reporting
balls of radius 1e-34: float64 logarithms fed in as exact, a sieve that marked
nothing, a square that returned nan, and a comparison run at lower precision
than the thing it was checking. The balls were tight around the wrong number,
which is the numerical form of a clean parse of the wrong thing.

The anchor is the OTHER SIDE of the explicit formula. `weil_certified` sums over
primes; `mp.zetazero` gives the ordinates themselves to any precision, and at
sigma = 0.2 the weights fall off fast enough that fifteen of them settle an entry
to 1e-70. That comparison is what caught every bug listed above -- none of them
were visible from inside the prime side, where the balls looked immaculate.
"""

from __future__ import annotations

import numpy as np
import pytest

flint = pytest.importorskip("flint", reason="ball arithmetic needs python-flint")
mp = pytest.importorskip("mpmath")

from rh_research_engine.experiments.weil_certified import (  # noqa: E402
    _square,
    certified_lower_bound,
    certified_verdict,
    form_ball,
    is_positive_definite,
    kernel_ball,
    negative_certificate,
    prime_powers,
    run,
)
from rh_research_engine.experiments.weil_positivity import von_mangoldt  # noqa: E402

SIGMA = "0.2"


DIGITS = 60


@pytest.fixture(autouse=True)
def _restore_global_precision():
    """mpmath's dps and flint's prec are GLOBAL, and this file raises both.

    An earlier version set `mp.mp.dps = 60` in the module fixture and left it
    there. Nothing here failed; everything downstream got slower, because
    `weil_positivity`'s refining quadrature runs on mpmath -- 224s became 2054s
    and the whole suite went from 25 minutes to 47. A test that changes global
    state and does not put it back makes every later test's cost depend on what
    ran before it, which is the timing counterpart of a gate that reads its
    environment.
    """
    saved_dps, saved_prec = mp.mp.dps, flint.ctx.prec
    yield
    mp.mp.dps, flint.ctx.prec = saved_dps, saved_prec


@pytest.fixture(scope="module")
def ordinates():
    """Fifteen ordinates at 60 digits. The weight of the last is 2e-74."""
    with mp.workdps(DIGITS):
        return [mp.im(mp.zetazero(n)) for n in range(1, 16)]


def zero_side(numerator: int, denominator: int, ordinates) -> mp.mpf:
    """The ordinates keep their value, but the ARITHMETIC uses the current dps.

    So the width has to be asked for here too: at the default 15 digits this
    sum would be rounded far above the 1e-34 ball it is compared against.
    """
    with mp.workdps(DIGITS):
        d = mp.log(mp.mpf(numerator) / denominator)
        s = mp.mpf(SIGMA)
        return 2 * mp.fsum(mp.e ** (-s * s * g * g) * mp.cos(g * d) for g in ordinates)


@pytest.mark.parametrize("a,b", [(1, 1), (2, 1), (3, 2), (8, 7), (12, 7)])
def test_the_certified_entry_encloses_the_sum_over_exact_zeros(a, b, ordinates):
    """The check that found every bug in this module, and none from inside it.

    `(2, 1)` is here because `d = log 2` put a prime power exactly at the centre
    of the Gaussian, so `log n - d` straddled zero and `**2` returned nan -- one
    non-finite entry among exact ones, immune to precision and to evaluation
    budget alike, because it was never a convergence problem.
    """
    ball = kernel_ball(a, b, SIGMA)
    exact = zero_side(a, b, ordinates)
    with mp.workdps(DIGITS):
        middle = mp.mpf(ball.mid().str(45, radius=False))
        radius = mp.mpf(ball.rad().str(10, radius=False))
        assert radius > 0, "a point is not an enclosure"
        assert radius < mp.mpf("1e-30"), radius
        assert abs(exact - middle) <= radius


def test_the_entry_is_compared_above_its_own_precision(ordinates):
    """The check must be finer than the ball, or it is a check on the check.

    Rounding the zero side to double before comparing reports a 1e-34 enclosure
    as failing, which is how two earlier containment tests "failed" against
    perfectly good balls.
    """
    ball = kernel_ball(3, 2, SIGMA)
    exact = zero_side(3, 2, ordinates)
    with mp.workdps(DIGITS):
        radius = mp.mpf(ball.rad().str(10, radius=False))
        assert abs(exact - mp.mpf(float(exact))) > radius, (
            "float64 is coarser than the ball, so a float comparison cannot decide this"
        )


def test_the_sieve_finds_exactly_the_prime_powers():
    """A bytearray slice assigned `bytes(n)` writes n ZEROS and marks nothing.

    Every integer then counted as a prime, which moved entries by 1e-2 -- large,
    obvious against the zeros, and invisible against the float path, which uses
    a different sieve.
    """
    mine = prime_powers(5000)
    n, lam = von_mangoldt(5000)
    assert len(mine) == len(n)
    assert [value for value, _ in mine] == [int(v) for v in n]
    assert all(
        abs(float(np.log(base)) - float(weight)) < 1e-12
        for (_, base), weight in zip(mine, lam, strict=True)
    )
    assert (4, 2) in mine and (9, 3) in mine, "prime powers, not just primes"
    assert (6, 2) not in mine and (6, 3) not in mine


def test_the_tails_are_what_make_a_crude_truncation_honest(ordinates):
    """Both tails can be deleted at production parameters with nothing to show.

    They are exp(-900) and exp(-23000) there, so a mutation that drops either
    passes every other test in this file -- a bound nobody has watched work.
    Truncate hard enough that the tail is the whole story, and the enclosure has
    to carry it: the radius must grow to cover what was left out, and the true
    value must still be inside.
    """
    truth = zero_side(2, 1, ordinates)
    coarse = kernel_ball(2, 1, SIGMA, limit=6)
    narrow = kernel_ball(2, 1, SIGMA, span_widths=1.0)

    with mp.workdps(DIGITS):
        # Primes only to 6, so everything from 7 upward lives in the bound.
        middle = mp.mpf(coarse.mid().str(45, radius=False))
        radius = mp.mpf(coarse.rad().str(10, radius=False))
        assert radius > mp.mpf("1e-6"), "the dropped primes must show up as width"
        assert abs(truth - middle) <= radius
        assert abs(truth - middle) > mp.mpf("1e-30"), (
            "the centre must actually have moved, or this proves nothing"
        )

        # Integrate only to |r| = 5, where the integrand is still order exp(-1).
        middle = mp.mpf(narrow.mid().str(45, radius=False))
        radius = mp.mpf(narrow.rad().str(10, radius=False))
        assert radius > mp.mpf("0.1"), radius
        assert abs(truth - middle) <= radius


def test_a_tail_bound_refuses_a_cutoff_it_cannot_bound():
    """Both bounds assume the truncation is past the Gaussian; neither may guess."""
    with pytest.raises(ValueError, match="not past the Gaussian"):
        kernel_ball(2, 1, SIGMA, limit=3)
    with pytest.raises(ValueError, match="span >= 5"):
        kernel_ball(2, 1, SIGMA, span_widths=0.5)


def test_squaring_a_ball_that_straddles_zero():
    """`**` is the general power and is undefined for a non-positive base."""
    flint.ctx.prec = 128
    straddling = flint.arb(2).log() - flint.arb(2).log()
    assert not (straddling**2).is_finite()
    assert _square(straddling).is_finite()
    assert _square(flint.arb(-3)) == flint.arb(9)


def test_the_ratio_enters_as_integers_not_as_a_float():
    """`cos(r d)` runs to r = 30/sigma, so a float64 d is 1e-14 in the phase."""
    exact = kernel_ball(8, 7, SIGMA)
    # The same ratio, reduced differently, must give the identical enclosure.
    assert exact.overlaps(kernel_ball(16, 14, SIGMA))
    assert exact.overlaps(kernel_ball(24, 21, SIGMA))


def test_cholesky_proves_positive_definiteness_and_refuses_to_guess():
    """`pivot > 0` in Arb is true only when the whole ball is above zero."""
    flint.ctx.prec = 128
    good = flint.arb_mat([[flint.arb(4), flint.arb(1)], [flint.arb(1), flint.arb(3)]])
    assert is_positive_definite(good, 0.0, 128)
    assert not is_positive_definite(good, 10.0, 128), "the shift must be able to fail"

    indefinite = flint.arb_mat([[flint.arb(1), flint.arb(2)], [flint.arb(2), flint.arb(1)]])
    assert not is_positive_definite(indefinite, 0.0, 128)

    straddling = flint.arb(0).union(flint.arb(1))
    unknown = flint.arb_mat([[straddling, flint.arb(0)], [flint.arb(0), flint.arb(1)]])
    assert not is_positive_definite(unknown, 0.0, 128), (
        "a pivot that might be zero is not proved positive"
    )


def test_the_lower_bound_is_a_real_bound():
    """`lambda_min >= t` exactly when `A - tI` still factors."""
    flint.ctx.prec = 128
    matrix = flint.arb_mat([[flint.arb(4), flint.arb(1)], [flint.arb(1), flint.arb(3)]])
    bound = certified_lower_bound(matrix, 128)
    true_min = float(np.linalg.eigvalsh(np.array([[4.0, 1.0], [1.0, 3.0]]))[0])
    assert 0 < bound <= true_min
    assert bound > true_min - 1e-9, bound
    assert is_positive_definite(matrix, bound, 128)


def test_refutation_takes_one_vector_and_no_eigenvalue():
    """A quadratic form enclosing below zero is a complete refutation."""
    flint.ctx.prec = 128
    indefinite = flint.arb_mat([[flint.arb(1), flint.arb(2)], [flint.arb(2), flint.arb(1)]])
    value = negative_certificate(indefinite, [1.0, -1.0], 128)
    assert value < 0
    assert negative_certificate(indefinite, [1.0, 1.0], 128) > 0
    verdict, _, _ = certified_verdict(indefinite, 128)
    assert verdict == "REFUTED"


def test_unproved_is_not_refuted():
    """The third verdict, which is about this computation and not about zeta."""
    flint.ctx.prec = 128
    straddling = flint.arb(0).union(flint.arb(1))
    unknown = flint.arb_mat([[straddling, flint.arb(0)], [flint.arb(0), flint.arb(1)]])
    verdict, _, _ = certified_verdict(unknown, 128)
    assert verdict == "UNRESOLVED"


def test_a_small_form_is_certified_positive():
    matrix = form_ball(4, "0.03")
    verdict, lower, _ = certified_verdict(matrix, 128)
    assert verdict == "POSITIVE"
    assert lower > 0
    assert is_positive_definite(matrix, lower, 128)


def test_the_run_reports_a_proved_bound():
    result = run(size=4, sigma="0.03")
    assert result.metrics["violated"] == 0.0
    assert result.metrics["unresolved"] == 0.0
    assert result.metrics["certified_lower_bound"] > 0
    assert result.assumptions, "the tail bounds rest on stated inequalities"


def test_the_general_digamma_bound_is_false_and_the_one_used_is_not():
    """The assumption this record used to declare does not hold as stated.

    `Re psi(z) <= log|z|` on `Re z > 0` was listed as "the standard bound". For
    fixed x and large y the margin behaves as `(x/2 - 1/12)/y^2`, so it is
    NEGATIVE for every `0 < x < 1/6` and the sign flips exactly at `Re z = 1/6`.
    Nothing computed was wrong -- the tail is applied only on `z = 1/4 + i r/2`,
    where `1/4 > 1/6` -- but a false inequality in the `assumptions` list of a
    rigorous record is the one thing that list exists to prevent.

    Found by the rh-research-engine-da session.
    """
    with mp.workdps(40):
        for x in ("0.05", "0.10", "0.16"):
            z = mp.mpc(mp.mpf(x), mp.mpf("1e4"))
            assert mp.log(abs(z)) - mp.re(mp.digamma(z)) < 0, (
                f"the general bound must FAIL at Re z = {x}"
            )
        # And it holds above the threshold, so 1/6 really is where it turns.
        for x in ("0.17", "0.25", "0.5"):
            z = mp.mpc(mp.mpf(x), mp.mpf("1e4"))
            assert mp.log(abs(z)) - mp.re(mp.digamma(z)) > 0, x
        # The asymptotic that predicts the threshold, to six decimals.
        for x in ("0.05", "0.25"):
            xx = mp.mpf(x)
            z = mp.mpc(xx, mp.mpf("1e4"))
            margin = (mp.log(abs(z)) - mp.re(mp.digamma(z))) * mp.mpf("1e4") ** 2
            assert abs(margin - (xx / 2 - mp.mpf(1) / 12)) < mp.mpf("1e-6"), x


def test_the_bound_as_used_holds_with_margin():
    """`|Re psi(1/4 + i r/2)| <= log r` for r >= 3, both signs, and it is wide.

    The tail bound needs the absolute value, because `Re psi` is free to be
    negative; the upper margin tends to log 2 and the lower stays above 1.49.
    """
    with mp.workdps(40):
        upper, lower = [], []
        for r in (3, 5, 10, 50, 150, 1000, 10_000, 100_000):
            value = mp.re(mp.digamma(mp.mpc(mp.mpf(1) / 4, mp.mpf(r) / 2)))
            upper.append(mp.log(mp.mpf(r)) - value)
            lower.append(mp.log(mp.mpf(r)) + value)
        assert min(upper) > mp.mpf("0.69"), min(upper)
        assert min(lower) > mp.mpf("1.49"), min(lower)
        assert abs(upper[-1] - mp.log(2)) < mp.mpf("1e-8"), "the margin tends to log 2"


def test_the_digamma_enclosure_widens_with_precision():
    """The mechanism behind NOT_COMPUTED, which was first recorded as a budget.

    `acb.digamma` on a ball with a real radius returns a WORSE enclosure at
    higher precision. The integrand contains that call, so the integrator must
    subdivide further at 256 bits than at 128 and runs out -- raising precision
    makes the enclosure worse before the arithmetic makes it better.
    """
    radii = []
    for precision in (53, 128, 256):
        flint.ctx.prec = precision
        centre, half = flint.arb("2"), flint.arb("0.05")
        ball = (centre - half).union(centre + half)
        value = (flint.acb(0.25) + flint.acb(0, 1) * ball / 2).digamma()
        radii.append(float(value.real.rad()))
    flint.ctx.prec = 128
    assert radii[0] < radii[1] < radii[2], radii
    assert radii[2] > 10 * radii[0], radii


def test_the_evaluation_budget_scales_with_precision():
    """A fixed cap is what made precision look non-monotone.

    This module recorded that 192 bits proves the form positive while 256 bits
    "computes nothing". The mechanism -- acb.digamma enclosing worse on the same
    box at higher precision -- is real, but the conclusion was not: the
    integrator subdivides further and needs a bigger budget, which is cheap.
    Measured minimums were 100k, 200k, 400k, 1.6M, 3.2M at 128 to 512 bits, and
    the budget must cover each with margin.
    """
    from rh_research_engine.experiments.weil_certified import eval_budget

    measured_minimum = {128: 100_000, 192: 200_000, 256: 400_000,
                        384: 1_600_000, 512: 3_200_000}
    for precision, needed in measured_minimum.items():
        assert eval_budget(precision) >= 2 * needed, (precision, eval_budget(precision))
    # Monotone, and doubling per 64 bits.
    assert eval_budget(192) == 2 * eval_budget(128)
    assert eval_budget(256) == 2 * eval_budget(192)
    # Never below the floor for a low precision.
    assert eval_budget(53) >= 200_000

