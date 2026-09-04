"""Pair correlation of the zeros WITH its lower-order terms, from the ratios
conjecture.

`pair_correlation.py` measures the zeros against Montgomery's
`1 - (sin pi u / pi u)^2`. That curve is UNIVERSAL -- it describes random
matrices and quantum billiards too -- so agreement with it says the zeros are a
spectrum and says nothing about primes. The arithmetic is in the departure from
it, and Conrey and Snaith derived the departure in closed form.

Conrey and Snaith, *Correlations of eigenvalues and Riemann zeros*,
arXiv:0803.2795, equations (178)-(184) and (210), (212). Transcribed, NOT
independently derived; see `docs/research/pair-correlation-lower-order.md` for
the transcription with equation numbers beside each line.

    A(x) = prod_p (1 - p^-(1+x))(1 - 2/p + p^-(1+x)) / (1 - 1/p)^2   (179), (210)
    B(x) = sum_p ( log p / (p^(1+x) - 1) )^2                         (180), (212)
    P1(x) = e^(-l x) A(x) zeta(1+x) zeta(1-x)                        (182)
    P2(x) = (zeta'/zeta)'(1+x) - B(x)                                (183)
    R_2(u, v) = l^2 + [P1+P2](i delta) + [P1+P2](-i delta)           (181), (184)

with `l = log(t/2 pi)`, `delta = u - v`, and unfolded separation
`x = delta l / 2 pi`.

WHAT THE `~1e-3` TURNED OUT TO BE. The page that transcribed these formulas
recorded, as an open problem, "residual noise at ~1e-3, source unidentified",
and said it had to be tracked down before any comparison against data because
the measured deviations are of order 1e-2. It is not noise. It is this
formula's own lower-order terms, and three measurements say so:

- **It does not move with the prime truncation.** Taking the product and the
  sum from 1e3 primes to 1e6 -- a thousandfold -- moves `R_2/l^2` by at most
  5.6e-5, and from 1e5 to 1e6 by under 1e-6. The page's stated hypothesis was
  the truncation. It is refuted.
- **It does not move with the working precision.** dps 25, 50 and 100 agree in
  every printed digit, bit for bit.
- **It has an exponent, and noise does not.** `l^k (R_2/l^2 - Montgomery)`
  settles to a definite constant: `k = 2` generically (at `x = 0.5`, to
  -6.2921), and `k = 4` at integer `x`, where Montgomery's `sin^2` vanishes.

WHY IT LOOKED LIKE NOISE ANYWAY, WHICH IS THE FINDING. The page sampled
`l = 10, 20, 40`. Over that range the `1/l^2` and `1/l^4` terms are still
comparable, so the departure changes sign with `l` and does not scale -- it
sorts into asymptotic order only past `l ~ 160`. And `l = log(t/2 pi)`, so
`l = 10..40` is not an unlucky choice: it is every height a zero has ever been
computed at. `T = 10^6` is `l = 12.0`; Odlyzko's tables at zero index 10^12 and
10^22 are `l = 24.4` and `l = 48.4`. **At every height the zeros are known at,
no single term of this expansion dominates.** That is precisely why the full
form is worth carrying rather than Montgomery plus a leading correction -- and
it is why the departure has to be evaluated, never fitted to a power of `l`.

THE CANCELLATION IS WORSE THAN THE PAGE'S TABLE SHOWS, AND IT IS NOT ABOUT
FLOAT64. The page tabulated `zeta(1+x)zeta(1-x)` against `(zeta'/zeta)'(1+x)`,
each diverging like `1/x^2`, and concluded "doing this in numpy would produce a
plausible curve made of rounding". The conclusion is right and the reason is
one step short, which matters because the reason is what tells you where it
fails.

There are TWO cancellations, not one. `P1 + P2` cancels the `1/delta^2` poles
down to about `-l^2`; then `l^2 + (P1+P2)` cancels again, down to `R_2` itself,
which vanishes like `x^2` as `x -> 0` because the zeros repel. Composing them,
`A` must be known to a RELATIVE precision of about `110 x^4`. That is 1.1e-10 at
`x = 1e-3` and 1.1e-14 at `x = 1e-4` -- so the usable floor depends on `x`, and
no amount of `dps` fixes it, because the limit is in `A` and not in zeta.
Measured: raising the zeta precision from dps 30 to 50 to 80 changes the broken
small-`x` values in not one digit.

WHICH IS WHY `A` IS A SUM OF LOGARITHMS AND NOT `np.prod`. `np.prod` is
sequential, so 9592 factors accumulate about `n * eps ~ 1e-12` of relative
error; `np.sum` is pairwise, so the same terms as logarithms accumulate
`log2(n) * eps ~ 1e-15`. Measured side by side against an mpmath reference,
that is 1.9e-13 against 1.7e-16 -- three orders, for the same arithmetic, and
exactly the three orders that decide whether the first histogram bin is signal
or rounding. The small terms use the series for `log(1-t)` rather than forming
`1-t` first, since `t ~ 1/p^2` is where most of the primes are.

With that, complex128 carries `A` to about 1e-16 relative and the curve is
right down to `x ~ 1e-4`. `MINIMUM_SEPARATION` refuses below it rather than
returning the plausible curve the page warned about, and `exact=True` routes
`A` and `B` through mpmath for anyone who needs to go lower.

The zeta factors are computed in mpmath regardless, because they are four calls
per point and cost nothing next to the primes. The split is measured in
`tests/test_conrey_snaith.py`, not assumed.

WHAT AGREEMENT WOULD BE WORTH. The ratios conjecture is a conjecture. A
measurement following this curve is evidence for it and is not a proof of it,
which is why nothing here returns a rigorous confidence.
"""

from __future__ import annotations

from functools import lru_cache

import mpmath as mp
import numpy as np
from sympy import primerange

#: Primes carried in the Euler product for `A` and the prime sum for `B`.
#:
#: `B` is the binding one. Its terms are `(log p / (p^(1+x) - 1))^2`, so the
#: tail past `P` is about `(log P + 1)/P` by partial summation, and it enters
#: `R_2/l^2` divided by `l^2`. Measured rather than estimated: moving the limit
#: from 1e5 to 1e6 moves `R_2/l^2` by 8.0e-7 at `l = 12`, so the tail past 1e6
#: is of order 1e-7 -- four orders below the departure being measured.
#:
#: `A` converges faster. Written telescoped (see `arithmetic_product`) its
#: terms are `1 - O(1/p^2)`, so its tail past 1e6 is about 4e-7 on a quantity
#: of size 1.
ARITHMETIC_PRIME_LIMIT = 1_000_000

#: Working precision for the zeta factors.
#:
#: The cancellation costs about six digits at `delta = 1e-3` and grows like
#: `1/delta^2`. Thirty leaves twenty-four, which is past anything a computable
#: height reaches; see the module docstring for the measurement.
WORKING_DPS = 30

#: Points per bin when averaging this curve over a histogram bin.
#:
#: Fewer than `level_spacing.curve_bin_average`'s default of 101, because each
#: point here is a product over a million primes rather than an arithmetic
#: expression. The curve is smooth and the bins are narrow, so 9-point
#: Simpson-grade quadrature is far past what the comparison needs. What
#: matters is that BOTH curves are averaged the same way -- see
#: `pair_correlation.check_pair_correlation`.
#:
#: Fixed against the errors ALREADY in the budget rather than by taste.
#: Refining to 41 points, the curve moves by 1.4e-4 at 5 samples, 3.5e-5 at 9,
#: and 4.3e-6 at 21. Nine sits below the 9.2e-5 that `ELL_BANDS` contributes,
#: so it is not the limiting term; five does not. It is two orders below the
#: smallest thing this is used to measure -- a 3.6e-3 residual against a
#: 3.6e-3 noise floor -- and it is 2.4x cheaper than 21, which is most of the
#: cost of the whole comparison.
BIN_SAMPLES = 9

#: The error `BIN_SAMPLES` has to stay under, as a NUMBER rather than as the
#: paragraph above.
#:
#: It is what `pair_correlation.ELL_BANDS` already contributes, so a quadrature
#: below it is not the limiting term and one above it is. Written here because
#: a budget recorded only in a comment is a budget nothing enforces: the
#: bin-average test derives both of its sides from `BIN_SAMPLES` and the
#: refinement test varies `ELL_BANDS`, so dropping this to five -- whose 1.4e-4
#: exceeds the budget -- would leave both of them green while the published
#: deviations quietly moved. `test_the_reduced_quadrature_stays_inside_its_
#: budget` compares the chosen count against the refined reference.
QUADRATURE_BUDGET = 9.2e-5

#: The refined reference that budget is measured against. Enough points that
#: its own error is a fortieth of the thing being bounded.
QUADRATURE_REFERENCE_SAMPLES = 41

#: x-values evaluated at once. The arithmetic factors form an
#: `len(x) x len(primes)` array, so this bounds it to about 80 MB.
_CHUNK = 64

#: Smallest unfolded separation the complex128 arithmetic factors can carry.
#:
#: `A` needs about `110 x^4` of relative precision (see the module docstring),
#: so with complex128's 1e-16 the curve's own relative error runs about
#: `1e-16/(110 x^4)`. Measured against an all-mpmath reference at dps 50, it is
#: 3.9e-16 at `x = 0.5`, 3.0e-10 at `x = 1e-2` and 3.6e-7 at `x = 1e-3` -- the
#: predicted fourth-power growth, over nine orders.
#:
#: Set at 1e-3, where the curve is still right to seven figures, and two
#: decades above where it actually breaks: 3.2e-3 relative at `x = 1e-4`, and
#: at `x = 1e-5` the fast path returns -5.1e-8 where the truth is +2.8e-10 --
#: the wrong sign and 180 times the magnitude, which is exactly the plausible
#: shape made of rounding that the transcription page warned about.
#:
#: A refusal threshold, not a record of where it broke. Below it,
#: `pair_correlation` raises unless asked for `exact=True`. Nothing in the
#: histogram comparison approaches it -- thirty bins over a window of three put
#: the lowest quadrature point at 5e-3 -- so the floor costs nothing and exists
#: so that a caller who does go lower is told rather than shown a curve.
MINIMUM_SEPARATION = 1e-3


@lru_cache(maxsize=4)
def _primes(limit: int) -> np.ndarray:
    return np.array(list(primerange(2, limit)), dtype=np.float64)


def arithmetic_product(s: np.ndarray, *, prime_limit: int = ARITHMETIC_PRIME_LIMIT) -> np.ndarray:
    """`A(s)`, equations (179) and (210), over an array of complex `s`.

    Written telescoped. Expanding (179) with `q = 1/p` and `u = q p^-s`,

        (1-u)(1-2q+u)/(1-q)^2 = 1 - q^2 (1 - p^-s)^2 / (1-q)^2

    exactly, so the product is `prod_p [1 - (1 - p^-s)^2/(p-1)^2]`. This is the
    same statement and not an approximation of it -- the two forms are checked
    against each other in the tests, where they agree to 1e-49 at dps 50 -- but
    it shows the `O(1/p^2)` tail in the open rather than leaving it inside a
    ratio of factors that each tend to something else.

    ACCUMULATED AS A SUM OF LOGARITHMS. `np.prod` over 9592 factors is
    sequential and loses about `n * eps`; `np.sum` is pairwise and loses
    `log2(n) * eps`. That is 1.9e-13 against 1.7e-16 measured, and the module
    docstring explains why three orders of relative error in THIS quantity
    decides whether the small-separation end of the curve exists.
    """
    primes = _primes(prime_limit)
    s = np.atleast_1d(np.asarray(s, dtype=np.complex128))
    log_p = np.log(primes)
    inverse_square = 1.0 / (primes - 1.0) ** 2
    out = np.empty(s.shape, dtype=np.complex128)
    for lo in range(0, len(s), _CHUNK):
        block = s[lo : lo + _CHUNK, None]
        term = (1.0 - np.exp(-block * log_p)) ** 2 * inverse_square
        # log(1 - t). numpy has no complex log1p, and `t ~ 1/p^2` is where
        # most of the primes are, so forming `1 - t` there would round the
        # whole term away before the logarithm ever saw it.
        small = np.abs(term) < 1e-4
        logs = np.where(
            small,
            -(term + term**2 / 2.0 + term**3 / 3.0),
            np.log(np.where(small, 1.0, 1.0 - term)),
        )
        out[lo : lo + _CHUNK] = np.exp(np.sum(logs, axis=1))
    return out


def prime_sum(s: np.ndarray, *, prime_limit: int = ARITHMETIC_PRIME_LIMIT) -> np.ndarray:
    """`B(s) = sum_p (log p / (p^(1+s) - 1))^2`, equations (180) and (212)."""
    primes = _primes(prime_limit)
    s = np.atleast_1d(np.asarray(s, dtype=np.complex128))
    log_p = np.log(primes)
    out = np.empty(s.shape, dtype=np.complex128)
    for lo in range(0, len(s), _CHUNK):
        block = s[lo : lo + _CHUNK, None]
        out[lo : lo + _CHUNK] = np.sum((log_p / (np.exp((1.0 + block) * log_p) - 1.0)) ** 2, axis=1)
    return out


def zeta_log_derivative_prime(s: mp.mpc) -> mp.mpc:
    """`(zeta'/zeta)'(s)`, as `zeta''/zeta - (zeta'/zeta)^2`.

    Not `mp.diff` of the logarithmic derivative: that would evaluate zeta at
    displaced points and lose digits to the difference, on the one quantity in
    this assembly that the cancellation is about.
    """
    zeta = mp.zeta(s)
    first = mp.zeta(s, 1, 1)
    second = mp.zeta(s, 1, 2)
    return second / zeta - (first / zeta) ** 2


def ell(height: float) -> float:
    """`l = log(t / 2 pi)`, the density of zeros at height `t`, up to `2 pi`."""
    return float(np.log(np.asarray(height, dtype=float) / (2.0 * np.pi)))


def pair_correlation(
    x: np.ndarray,
    ell_value: float,
    *,
    prime_limit: int = ARITHMETIC_PRIME_LIMIT,
    dps: int = WORKING_DPS,
    exact: bool = False,
) -> np.ndarray:
    """`R_2/l^2` at unfolded separation `x`, equations (181) and (184).

    ONLY `s = +i delta` IS EVALUATED. `A`, `B`, `zeta` and `(zeta'/zeta)'` all
    have real Taylor coefficients, so each satisfies `f(conj s) = conj f(s)`,
    and `s = -i delta` is the conjugate of `s = +i delta`. Hence

        [P1+P2](i delta) + [P1+P2](-i delta) = 2 Re [P1+P2](i delta)

    identically. Taking the real part of a two-sided sum would give the same
    number while silently discarding whatever imaginary part a transcription
    error had introduced; this way the realness is structural, and the tests
    check the identity against the two-sided form rather than assuming it.

    `x = 0` is the removable singularity where `zeta(1+x)` is the pole. The
    density there is zero -- the zeros repel, which is the whole content of the
    curve at that end -- and the tests check that the limit really is zero
    rather than taking the substitution on trust.

    With `exact`, `A` and `B` go through mpmath instead of complex128. That is
    about 150x slower and it is what `MINIMUM_SEPARATION` points at.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    ell_value = float(ell_value)
    if ell_value <= 0:
        raise ValueError(
            f"l = {ell_value} is not positive, so it is not log(t/2 pi) for any "
            "height above 2 pi and the unfolding it describes does not exist"
        )
    if np.any(x < 0):
        raise ValueError(
            "the pair correlation is a density in a separation, so a negative "
            f"separation is not a point of it; got {x[x < 0].min()}"
        )
    too_small = (x > 0) & (x < MINIMUM_SEPARATION)
    if np.any(too_small) and not exact:
        raise ValueError(
            f"separation {x[too_small].min():.3g} is below "
            f"{MINIMUM_SEPARATION:g}, where the complex128 Euler product for A "
            "no longer carries the two cancellations this assembly makes: the "
            "curve there would be rounding wearing a plausible shape. Raising "
            "`dps` does NOT help -- the limit is in A, not in zeta. Pass "
            "`exact=True` to route A and B through mpmath instead"
        )

    delta = 2.0 * np.pi * x / ell_value
    s = 1j * delta
    positive = x > 0

    if exact:
        factor_a = [None] * len(x)
        factor_b = [None] * len(x)
    else:
        factor_a = arithmetic_product(s, prime_limit=prime_limit)
        factor_b = prime_sum(s, prime_limit=prime_limit)

    previous, mp.mp.dps = mp.mp.dps, dps
    try:
        total = np.zeros(x.shape, dtype=float)
        for index in np.flatnonzero(positive):
            s_mp = mp.mpc(0, mp.mpf(float(delta[index])))
            if exact:
                a_k = _arithmetic_product_exact(s_mp, prime_limit)
                b_k = _prime_sum_exact(s_mp, prime_limit)
            else:
                a_k = mp.mpc(factor_a[index])
                b_k = mp.mpc(factor_b[index])
            p1 = mp.exp(-mp.mpf(ell_value) * s_mp) * a_k * mp.zeta(1 + s_mp) * mp.zeta(1 - s_mp)
            p2 = zeta_log_derivative_prime(1 + s_mp) - b_k
            total[index] = float(2 * mp.re(p1 + p2))
    finally:
        mp.mp.dps = previous

    out = 1.0 + total / ell_value**2
    # Level repulsion. `total` was left at zero above, which would read as
    # R_2/l^2 = 1; the density at coincidence is 0.
    return np.where(positive, out, 0.0)


def _arithmetic_product_exact(s: mp.mpc, prime_limit: int) -> mp.mpc:
    """`A(s)` in mpmath, in the same telescoped form as `arithmetic_product`."""
    total = mp.mpf(1)
    for prime in _primes(prime_limit):
        total *= 1 - (1 - mp.power(int(prime), -s)) ** 2 / mp.mpf(int(prime) - 1) ** 2
    return total


def _prime_sum_exact(s: mp.mpc, prime_limit: int) -> mp.mpc:
    """`B(s)` in mpmath, for the same reason."""
    total = mp.mpf(0)
    for prime in _primes(prime_limit):
        p = int(prime)
        total += (mp.log(p) / (mp.power(p, 1 + s) - 1)) ** 2
    return total


def pair_correlation_two_sided(
    x: np.ndarray,
    ell_value: float,
    *,
    prime_limit: int = ARITHMETIC_PRIME_LIMIT,
    dps: int = WORKING_DPS,
) -> np.ndarray:
    """(181) evaluated at both `+i delta` and `-i delta`, for the tests.

    This is what `pair_correlation` would be without the conjugate identity.
    It exists so that identity is checked rather than asserted, and it is not
    the function to call: it does twice the work for the same number.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    delta = 2.0 * np.pi * x / float(ell_value)
    previous, mp.mp.dps = mp.mp.dps, dps
    try:
        out = np.empty(x.shape, dtype=float)
        for index, d in enumerate(delta):
            total = mp.mpf(0)
            for sign in (1, -1):
                s_np = np.array([1j * sign * d])
                s_mp = mp.mpc(0, sign * mp.mpf(float(d)))
                a_k = mp.mpc(arithmetic_product(s_np, prime_limit=prime_limit)[0])
                b_k = mp.mpc(prime_sum(s_np, prime_limit=prime_limit)[0])
                total += (
                    mp.exp(-mp.mpf(ell_value) * s_mp) * a_k * mp.zeta(1 + s_mp) * mp.zeta(1 - s_mp)
                )
                total += zeta_log_derivative_prime(1 + s_mp) - b_k
            out[index] = float(mp.re(total))
    finally:
        mp.mp.dps = previous
    return 1.0 + out / float(ell_value) ** 2
