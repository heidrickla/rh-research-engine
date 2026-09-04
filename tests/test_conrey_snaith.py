"""The pair correlation with its lower-order terms.

What is under test is that the assembly is RIGHT, not that it runs: three
controls with known answers (`l -> infinity` gives Montgomery, `x -> 0` gives
level repulsion, the conjugate identity gives the two-sided sum), the two
numerical decisions taken by measurement rather than choice, and the finding
that resolved the transcription page's one open problem.

Every check here was written by breaking the thing it checks first.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

from rh_research_engine.symbolic import conrey_snaith as cs

#: Small enough to keep the file quick; the truncation test is what says the
#: difference from the module default does not matter.
FAST_PRIMES = 100_000


def montgomery(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x == 0, 0.0, 1 - (np.sin(np.pi * x) / (np.pi * x)) ** 2)


# --------------------------------------------------------------------------
# The three controls. These are what say the assembly is right rather than
# merely plausible -- none of them involves a zero.
# --------------------------------------------------------------------------


def test_the_limit_in_l_is_montgomerys_curve():
    """`R_2/l^2 -> 1 - (sin pi x/pi x)^2`: the leading term is the universal law.

    IT CONSTRAINS THE LEADING TERM AND NOTHING ELSE, which is worth saying
    because the transcription page called this "the only check available that
    does not involve the zeros ... what says the assembly is right rather than
    merely plausible". Deleting `B` from (183) entirely -- one of the two
    lower-order terms, the whole prime sum -- leaves this test PASSING. It has
    to: every lower-order term vanishes in the limit, so the limit cannot see
    any of them.

    What catches a dropped lower-order term is `test_the_zeros_repel_at_
    coincidence` and `test_the_departure_has_an_exponent_so_it_is_not_noise`,
    both of which look at finite `l`. Checked by deleting `B` and watching
    which of the three fail.
    """
    x = np.array([0.25, 0.5, 1.0, 1.75, 2.5])
    target = montgomery(x)
    previous = None
    for ell in (40.0, 160.0, 640.0):
        error = np.abs(cs.pair_correlation(x, ell, prime_limit=FAST_PRIMES) - target)
        if previous is not None:
            # Fourfold per fourfold in l, i.e. the departure is O(1/l^2). The
            # margin is loose because at integer x it is O(1/l^4) and converges
            # faster; what is under test is that it converges at all.
            assert error.max() < previous.max() / 3, (previous.max(), error.max())
        previous = error
    assert previous.max() < 1e-4, previous.max()


def test_the_zeros_repel_at_coincidence():
    """`R_2/l^2 -> 0` as `x -> 0`, quadratically, and the value AT zero is the
    limit rather than whatever the pole does.

    `x = 0` is where `zeta(1+x)` is the pole, so the value there is substituted.
    A substituted value that did not match the limit would be a hole in the
    curve wearing the shape of level repulsion.
    """
    for ell in (6.0, 12.0):
        assert cs.pair_correlation([0.0], ell, prime_limit=FAST_PRIMES)[0] == 0.0
        x = np.array([4e-3, 2e-3, 1e-3])
        quadratic = cs.pair_correlation(x, ell, prime_limit=FAST_PRIMES) / x**2
        # A quadratic vanishing means R_2/x^2 tends to a constant; if the
        # substitution at 0 were wrong this would still hold, which is why the
        # value at 0 is asserted separately above.
        assert np.ptp(quadratic) / np.mean(quadratic) < 1e-3, quadratic
        assert quadratic.mean() > 0


def test_the_conjugate_identity_is_the_two_sided_sum():
    """`pair_correlation` evaluates only `+i delta` and doubles the real part.

    That is an identity, because every factor has real Taylor coefficients --
    but it is an identity about the transcription, so it is checked against the
    form that evaluates both signs rather than assumed.
    """
    x = np.array([0.3, 1.0, 2.7])
    one_sided = cs.pair_correlation(x, 12.0, prime_limit=FAST_PRIMES)
    two_sided = cs.pair_correlation_two_sided(x, 12.0, prime_limit=FAST_PRIMES)
    assert np.abs(one_sided - two_sided).max() < 1e-15


# --------------------------------------------------------------------------
# The transcription itself.
# --------------------------------------------------------------------------


def test_the_telescoped_euler_product_is_the_papers_one():
    """`A` is written `prod [1 - (1-p^-s)^2/(p-1)^2]`; (179) writes it as a
    ratio. They are equal by algebra, which is a claim, so it is checked.
    """
    mp.mp.dps = 50
    for value in (0.05, 0.5):
        s = mp.mpc(0, mp.mpf(value))
        telescoped = cs._arithmetic_product_exact(s, 10_000)
        literal = mp.mpf(1)
        for prime in cs._primes(10_000):
            p = int(prime)
            u = mp.power(p, -(1 + s))
            q = mp.mpf(1) / p
            literal *= (1 - u) * (1 - 2 * q + u) / (1 - q) ** 2
        assert abs(telescoped - literal) < mp.mpf(10) ** -45


def test_the_page_control_table_is_reproduced():
    """The numbers `docs/research/pair-correlation-lower-order.md` printed.

    The prototype that produced them is gone; this is what makes the page's
    table a record rather than a memory.
    """
    expected = {
        0.5: (0.55714, 0.58111, 0.59093),
        1.0: (1.02111, 1.00408, 1.00038),
    }
    for x, row in expected.items():
        got = [
            float(cs.pair_correlation([x], ell, prime_limit=FAST_PRIMES)[0])
            for ell in (10.0, 20.0, 40.0)
        ]
        assert np.allclose(got, row, atol=5e-6), (x, got, row)


# --------------------------------------------------------------------------
# The two numerical decisions, each measured rather than chosen.
# --------------------------------------------------------------------------


def test_the_prime_truncation_is_converged():
    """Refutes the transcription page's stated hypothesis for its `1e-3`.

    The page recorded "the likely cause is truncating the Euler product ... at
    1e5 primes, but that is a guess and has not been tested". Tested: a
    thousandfold in the limit moves the curve by less than 1e-4.
    """
    x = np.array([0.5, 1.5, 3.0])
    coarse = cs.pair_correlation(x, 12.0, prime_limit=1_000)
    fine = cs.pair_correlation(x, 12.0, prime_limit=FAST_PRIMES)
    finer = cs.pair_correlation(x, 12.0, prime_limit=1_000_000)
    assert np.abs(coarse - finer).max() < 1e-4, np.abs(coarse - finer).max()
    assert np.abs(fine - finer).max() < 1e-5, np.abs(fine - finer).max()


def test_the_euler_product_is_summed_as_logarithms_not_multiplied():
    """`np.prod` is sequential and loses `n * eps`; `np.sum` is pairwise.

    This is the check that fails if someone rewrites `arithmetic_product` the
    obvious way. Three orders of relative error in `A` is the difference
    between the small-separation end of the curve being signal and being
    rounding, so it is asserted against an mpmath reference rather than left as
    a comment about style.
    """
    mp.mp.dps = 60
    primes = cs._primes(FAST_PRIMES)
    log_p = np.log(primes)
    inverse_square = 1.0 / (primes - 1.0) ** 2
    for value in (2e-4, 5e-3):
        s = 1j * value
        reference = cs._arithmetic_product_exact(mp.mpc(0, mp.mpf(value)), FAST_PRIMES)
        sequential = np.prod(1 - (1 - np.exp(-s * log_p)) ** 2 * inverse_square)
        summed = cs.arithmetic_product(np.array([s]), prime_limit=FAST_PRIMES)[0]

        def relative(got, reference=reference):
            return float(abs(mp.mpc(got) - reference) / abs(reference))

        assert relative(summed) < 1e-15, relative(summed)
        assert relative(summed) < relative(sequential) / 100, (
            relative(summed),
            relative(sequential),
        )


def test_below_the_floor_it_refuses_instead_of_returning_a_shape():
    """And says the true reason, because the plausible one is wrong.

    Raising `dps` does not help: the limit is the complex128 `A`, not zeta.
    The refusal names `exact=True`, so the check is that `exact=True` actually
    delivers there.
    """
    with pytest.raises(ValueError, match="below 0.001"):
        cs.pair_correlation([1e-5], 12.0, prime_limit=FAST_PRIMES)
    with pytest.raises(ValueError, match=r"does NOT help"):
        cs.pair_correlation([1e-5], 12.0, prime_limit=FAST_PRIMES, dps=100)

    # The escape hatch the refusal points at has to exist and be right.
    exact = cs.pair_correlation([1e-4], 12.0, prime_limit=10_000, exact=True, dps=50)[0]
    coarser = cs.pair_correlation([1e-4], 12.0, prime_limit=10_000, exact=True, dps=30)[0]
    assert abs(exact / 1e-8) > 0.1, exact
    assert abs(exact - coarser) / abs(exact) < 1e-6


def test_more_precision_does_not_rescue_the_small_separations(monkeypatch):
    """The measurement behind the refusal's wording, on the branch it refuses.

    If the limit were zeta's precision, `dps` would move these. It does not
    move them in a single digit, which is why the refusal tells the caller to
    change `A` rather than to turn `dps` up.

    The floor is lowered through OUR own module constant. A refusal nobody has
    watched fail is a refusal nobody has watched fail -- and the branch it
    guards has to be run to know what it is guarding.
    """
    monkeypatch.setattr(cs, "MINIMUM_SEPARATION", 0.0)
    x = np.array([1e-5])
    values = [
        cs.pair_correlation(x, 12.0, prime_limit=10_000, dps=dps, exact=False)[0]
        for dps in (30, 80)
    ]
    assert values[0] == values[1], values

    # And it really is wrong there, which is what the floor exists for. Not
    # "less accurate": the wrong SIGN, on a density, which is the plausible
    # shape made of rounding rather than a degraded version of the answer.
    truth = cs.pair_correlation(x, 12.0, prime_limit=10_000, exact=True, dps=50)[0]
    assert truth > 0, truth
    assert values[0] < 0, values[0]
    assert abs(values[0] - truth) > 10 * abs(truth), (values[0], truth)


# --------------------------------------------------------------------------
# The finding: the page's "unidentified noise" is the formula's own terms.
# --------------------------------------------------------------------------


def test_the_departure_has_an_exponent_so_it_is_not_noise():
    """`l^2 (R_2/l^2 - Montgomery)` settles to a constant. Noise does not.

    This is what resolved the page's open problem. It is asserted at
    `x = 0.5`, where the exponent is 2; at integer `x` Montgomery's `sin^2`
    vanishes and the departure is `O(1/l^4)` instead, which is the other half
    of why the departure "jumped about" over `l = 10..40`.
    """
    x = np.array([0.5])
    scaled = [
        float(ell**2 * (cs.pair_correlation(x, ell, prime_limit=FAST_PRIMES)[0] - montgomery(x)[0]))
        for ell in (160.0, 320.0, 640.0)
    ]
    # Settling, and to something definite rather than to zero.
    assert abs(scaled[2] - scaled[1]) < abs(scaled[1] - scaled[0]) / 3
    assert -6.30 < scaled[2] < -6.28, scaled

    integer_x = np.array([2.0])
    faster = [
        float(
            ell**2
            * (
                cs.pair_correlation(integer_x, ell, prime_limit=FAST_PRIMES)[0]
                - montgomery(integer_x)[0]
            )
        )
        for ell in (160.0, 640.0)
    ]
    # O(1/l^4) means l^2 times it still falls, by sixteenfold per fourfold.
    assert abs(faster[1]) < abs(faster[0]) / 8, faster


def test_the_reachable_heights_are_all_pre_asymptotic():
    """Why the full form is carried rather than a leading correction.

    `l = log(t/2 pi)`, so `T = 10^6` is `l = 12` and Odlyzko's deepest table is
    `l = 48`. Over that whole range the departure has not sorted into its
    asymptotic order -- `l^2` times it is still moving. Fitting a power of `l`
    to anything measurable would be fitting to the pre-asymptotic regime.
    """
    assert 11.9 < cs.ell(1e6) < 12.1
    x = np.array([2.0])
    scaled = [
        float(ell**2 * (cs.pair_correlation(x, ell, prime_limit=FAST_PRIMES)[0] - montgomery(x)[0]))
        for ell in (12.0, 24.0, 48.0)
    ]
    # Not settling: each is at least a third away from the next. Past l = 160
    # the same quantity converges (previous test), so this is a statement about
    # the reachable range and not about the curve.
    assert abs(scaled[1] - scaled[0]) > abs(scaled[0]) / 3, scaled
    assert abs(scaled[2] - scaled[1]) > abs(scaled[1]) / 3, scaled
