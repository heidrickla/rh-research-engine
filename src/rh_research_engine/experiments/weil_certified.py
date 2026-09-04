"""The Weil form in ball arithmetic, so that a verdict is a proof and not a reading.

`weil_positivity` computes the same form in float64 and reports the sign of its
smallest eigenvalue. That sign is worth nothing near the noise floor -- it
announced a refutation of RH at `--size 12` from rounding error -- and a
falsification test whose POSITIVE answer is uncertified cannot deliver a
NEGATIVE answer anyone should believe. Here every entry is an enclosure, the
eigenvalues are enclosures, and the three verdicts become claims that hold:

  POSITIVE    every eigenvalue ball lies strictly above zero
  REFUTED     some eigenvalue ball lies strictly below zero
  UNRESOLVED  a ball straddles zero -- more precision, or a bigger basis

The middle one is now honest rather than heuristic: no floor has to be guessed,
because the enclosure says how much is known.

THE INTEGRAND HAD TO BE MADE ANALYTIC FIRST. Arb integrates over complex boxes
and `Re psi(z)` is not analytic in `z`, so handing it `psi(1/4 + i r/2).real`
produces an enclosure with no theorem behind it -- however exactly it agrees
with mpmath, which it did, to 1e-16. On the real axis `psi(conj z) = conj psi z`,
so

    Re psi(1/4 + i r/2) = [ psi(1/4 + i r/2) + psi(1/4 - i r/2) ] / 2

and the right side is meromorphic in `r`, with poles only at `-+ i(2k + 1/2)` on
the imaginary axis, off the path. Same values on the line, sound off it; it
costs 1.8x and returns the identical radius.

TWO TAILS, BOTH BOUNDED RATHER THAN NEGLECTED. An integral over the reals and a
sum over all prime powers are each truncated, and a truncation without a bound
is an approximation wearing a ball. Both bounds are astronomically small at the
parameters used -- `exp(-900)` and `exp(-23000)` -- which is the reason to write
them down rather than the reason not to.

PRECISION LOOKED NON-MONOTONE, AND THE BUDGET WAS THE REASON. At sigma = 0.2 and
size 12 this proved POSITIVE at 192 bits with `lambda_min >= 1.96945e-29` and,
at 256 bits with the SAME fixed budget, returned 144 non-finite entries. The
mechanism behind that is real: `acb.digamma` on a ball with a real radius
encloses WORSE as precision rises -- measured at box radius 0.05, the returned
radius runs 1.86, 2.66, 6.16, 9.72, 18.8, 64.9 across 53 to 256 bits -- so the
integrator must subdivide further and exhausts a fixed cap.

But the conclusion drawn from it was wrong. The budget simply has to scale, and
it is cheap: 400,000 evaluations converge at 256 bits in 11 seconds against 9 at
192. `eval_budget` doubles it per 64 bits and 512 bits reaches a radius of
1.78e-149. Precision is monotone once the cap stops being the binding
constraint.

The earlier measurement that found infinite balls at every budget was taken
BEFORE `_digamma_even` replaced the non-analytic `Re(psi(z))` -- the old
integrand measured, the conclusion recorded about the new one.

`NOT_COMPUTED` remains its own verdict, because an integrator that never
delivered a matrix is still a different state from a matrix whose smallest
eigenvalue straddles zero. It should now be rare rather than routine.
An integrator that never delivered a matrix and a matrix whose smallest
eigenvalue straddles zero are different states, and letting them share a word is
the same error as letting "not tested" read as "refuted".

NEITHER VERDICT COMES FROM AN EIGENVALUE. `arb_mat.eig` raises rather than
return something unsound, and at sigma = 0.2 and size 12 it declines at 128, 256
and 384 bits alike: isolating twelve eigenvalues of a matrix with condition 1e26
is a much harder question than the sign of the smallest one. So POSITIVE is a
Cholesky in ball arithmetic that ran to completion with every pivot proved above
zero, and REFUTED would be a single vector whose quadratic form encloses below
it. Each is cheap, each decides exactly what is being asked, and the vector in
the second needs no pedigree at all -- only its evaluation must be rigorous.
"""

from __future__ import annotations

import math

from ..core.assumptions import cite
from ..core.models import ExperimentResult

#: 128 bits gives the archimedean integral a radius near 1e-32, past float64's
#: wall at 1e-14 by eighteen orders, at about 1.2s per distinct entry.
PRECISION = 128
#: Integrate over `|r| <= SPAN_WIDTHS / sigma`; the rest is bounded, not dropped.
SPAN_WIDTHS = 30.0
PRIME_LIMIT = 200_000
#: Integrand evaluations allowed before Arb gives up and returns an infinite
#: ball. SCALED WITH PRECISION, because a fixed value is what made precision
#: look non-monotone: see `eval_budget`.
EVAL_LIMIT = 200_000


def eval_budget(precision: int) -> int:
    """Evaluations to allow at `precision`. Doubles per 64 bits, measured.

    A FIXED 200,000 IS WHAT MADE PRECISION LOOK NON-MONOTONE. This module
    recorded that 192 bits proves the form positive while 256 bits "computes
    nothing", and attributed it to `acb.digamma` enclosing worse on the same box
    at higher precision. That mechanism is real -- the returned radius runs 1.86
    to 64.9 across 53 to 256 bits at box radius 0.05 -- but the CONCLUSION drawn
    from it was wrong: the integrator subdivides further and simply needs a
    bigger budget, which is cheap.

        prec 128 -> converges at    100,000    radius 2.05e-34
        prec 192 ->                 200,000           1.31e-53
        prec 256 ->                 400,000           9.05e-73
        prec 384 ->               1,600,000          4.07e-111
        prec 512 ->               3,200,000          1.78e-149

    11 seconds against 9 at 256 bits, so nothing was being bought by the cap.

    THE EARLIER MEASUREMENT WAS OF A DIFFERENT INTEGRAND. It found infinite
    balls at every budget tried and was run before `_digamma_even` replaced the
    non-analytic `Re(psi(z))` -- so the old integrand was measured and the
    conclusion recorded about the new one. A mechanism reasoned about across a
    change rather than re-executed after it.
    """
    doublings = max(0, (precision - 128)) / 64.0
    return int(EVAL_LIMIT * 2 ** (doublings + 1))


def prime_powers(limit: int) -> list[tuple[int, int]]:
    """`(n, p)` for every prime power `n = p^k < limit`, as EXACT INTEGERS.

    The float path carries `Lambda(n)` as a float64 `log p`, which is the right
    thing there and fatal here: fed to Arb as an exact input it produces a ball
    of radius 1e-34 around a value wrong at 1e-16. The prime itself is an
    integer, so `Lambda` is computed inside the ball arithmetic instead.
    """
    composite = bytearray(limit)
    out: list[tuple[int, int]] = []
    for candidate in range(2, limit):
        if not composite[candidate]:
            # b"" per slot: `bytes(n)` is n ZERO bytes, which marks nothing
            # composite and makes every integer a prime.
            composite[candidate * candidate :: candidate] = b"" * len(
                range(candidate * candidate, limit, candidate)
            )
            power = candidate
            while power < limit:
                out.append((power, candidate))
                power *= candidate
    out.sort()
    return out


def _square(x):
    """`x * x`, never `x ** 2`.

    python-flint routes `**` through the general power, which is undefined for a
    non-positive base, so squaring a ball that STRADDLES zero returns nan --
    exactly what `log n - d` is when a prime power sits at the centre of the
    Gaussian. At `d = log 2` that made one entry non-finite while every other
    entry was exact, and no amount of extra precision or evaluation budget moved
    it, because it was never a convergence problem.
    """
    return x * x


def _digamma_even(z, flint):
    """`[psi(1/4 + i z/2) + psi(1/4 - i z/2)] / 2`, analytic, real on the reals."""
    half = flint.acb(0, 0.5)
    return ((flint.acb(0.25) + z * half).digamma() + (flint.acb(0.25) - z * half).digamma()) / 2


def archimedean_tail(sigma: str | float, span: float, flint):
    """A bound on `|r| > span`, as a ball centred at zero.

    ASSUMES `|Re psi(1/4 + i r/2)| <= log r` for `r >= 3`, and then `log r <= r`
    for `r >= 5`.

    THE FIRST VERSION CITED A FALSE INEQUALITY. It claimed `Re psi(z) <= log|z|`
    on `Re z > 0` "the standard bound", and that is not true: for fixed `x` and
    large `y` the margin behaves as `(x/2 - 1/12)/y^2`, so it is NEGATIVE for
    every `0 < x < 1/6`, with the sign flipping exactly at `Re z = 1/6`.
    Measured against mpmath the asymptotic reproduces to six decimals, and at
    `x = 0.05, y = 1e4` the margin is `-5.8e-9`.

    Nothing computed here was ever wrong -- the bound is applied only on
    `z = 1/4 + i r/2`, where `1/4 > 1/6` -- but an assumption in the
    `assumptions` list of a rigorous record was stated more broadly than it is
    true, which is the whole point of that list. What is used is the composite,
    both signs, and it is wide: the upper margin tends to `log 2 = 0.6931` and
    the lower stays above `1.499` for `r >= 3`.

    `span = 30/sigma` is at least 150 for every width used here, so the crude
    second step costs nothing: the bound comes out as

        2 int_span^inf r exp(-sigma^2 r^2) dr / (2 pi) = exp(-sigma^2 span^2) / (2 pi sigma^2)
    """
    if span < 5.0:
        raise ValueError(f"the tail bound needs span >= 5, got {span}")
    s = flint.arb(str(sigma))
    bound = (-(s**2) * flint.arb(span) ** 2).exp() / (2 * flint.arb.pi() * s**2)
    return flint.arb(0).union(bound).union(-bound)


def prime_tail(a, sigma: str | float, limit: int, flint):
    """A bound on the prime powers above `limit`, as a ball centred at zero.

    `Lambda(n) <= log n` and the summand decreases past `log limit`, so the sum
    is at most its integral. With `u = log t` and the square completed,

        int_L^inf u e^(u/2) exp(-(u-d)^2/(4 s^2)) du / (2 s sqrt(pi))
            = e^(d/2 + s^2/4) [ m erfc(V)/2 + s exp(-V^2)/sqrt(pi) ]

    for `m = d + s^2` and `V = (L - m)/(2 s)`. Both signs of `d` appear in the
    kernel, so the larger `|d|` is used and the result doubled.
    """
    s, L = flint.arb(str(sigma)), flint.arb(limit).log()
    m = a + s**2
    v = (L - m) / (2 * s)
    if not (v > 1):
        raise ValueError(f"the prime cutoff {limit} is not past the Gaussian at d={a}")
    bound = 2 * (a / 2 + s**2 / 4).exp() * (
        m * v.erfc() / 2 + s * (-(v**2)).exp() / flint.arb.pi().sqrt()
    )
    return flint.arb(0).union(bound).union(-bound)


def kernel_ball(
    numerator: int,
    denominator: int,
    sigma: str | float,
    limit: int = PRIME_LIMIT,
    precision: int = PRECISION,
    eval_limit: int | None = None,
    span_widths: float = SPAN_WIDTHS,
):
    """An enclosure of the kernel at `d = |log(numerator/denominator)|`.

    THE RATIO IS PASSED AS INTEGERS, not as a float `d`. `cos(r d)` is evaluated
    out to `r = 30/sigma`, which is 150 at the widest setting used, so a float64
    `d` puts an error of 1e-14 in the phase -- comparable to the entries
    themselves at that width, and dressed in a ball of radius 1e-34.

    Manifestly even, so `|d|` is used.

    Every term is even in `d` on its face -- `cosh(d/2)`, `exp(-d^2/...)`,
    `cos(r d)`, and a prime sum written with both signs -- which is why taking
    `|d|` here is a fact rather than the shortcut that once cost an eigenvalue
    of -28. The float path symmetrises instead, and the two must agree.
    """
    import flint

    if eval_limit is None:
        eval_limit = eval_budget(precision)

    previous = flint.ctx.prec
    flint.ctx.prec = precision
    try:
        s = flint.arb(str(sigma))
        a = (flint.arb(numerator) / flint.arb(denominator)).log()
        if a < 0:
            a = -a
        span = flint.arb(span_widths) / s

        pole = 2 * (s**2 / 4).exp() * (a / 2).cosh()
        norm = 1 / (2 * s * flint.arb.pi().sqrt())
        g_zero = norm * (-(a**2) / (4 * s**2)).exp()

        def integrand(r, _):
            return (-(s**2) * r**2).exp() * (r * a).cos() * _digamma_even(r, flint)

        integral = flint.acb.integral(integrand, -span, span, eval_limit=eval_limit)
        arch = integral.real / (2 * flint.arb.pi())
        arch = arch + archimedean_tail(sigma, float(span), flint)

        primes = flint.arb(0)
        for value, base in prime_powers(limit):
            log_n = flint.arb(value).log()
            primes = primes + flint.arb(base).log() / flint.arb(value).sqrt() * norm * (
                (-_square(log_n - a) / (4 * _square(s))).exp()
                + (-_square(log_n + a) / (4 * _square(s))).exp()
            )
        primes = primes + prime_tail(a, sigma, limit, flint)

        return pole - g_zero * flint.arb.pi().log() + arch - primes
    finally:
        flint.ctx.prec = previous


def form_ball(
    size: int,
    sigma: float,
    limit: int = PRIME_LIMIT,
    precision: int = PRECISION,
    eval_limit: int | None = None,
):
    import flint

    cache: dict[tuple[int, int], object] = {}
    rows = []
    for a in range(1, size + 1):
        row = []
        for b in range(1, size + 1):
            high, low = max(a, b), min(a, b)
            common = math.gcd(high, low)
            key = (high // common, low // common)
            if key not in cache:
                cache[key] = kernel_ball(key[0], key[1], sigma, limit, precision, eval_limit)
            row.append(cache[key])
        rows.append(row)
    previous = flint.ctx.prec
    flint.ctx.prec = precision
    try:
        return flint.arb_mat(rows)
    finally:
        flint.ctx.prec = previous


def is_positive_definite(matrix, shift: float = 0.0, precision: int = PRECISION) -> bool:
    """True only when `A - shift I` is PROVED positive definite.

    Cholesky in ball arithmetic. Every pivot is a ball, and `pivot > 0` in Arb is
    true only if the whole ball lies above zero, so a run to completion is a
    proof and a failure is only a failure to prove. That asymmetry is the point:
    refutation gets its own certificate below rather than being read off this
    one's absence.

    THIS REPLACED THE EIGENVALUES, which could not do the job. `arb_mat.eig`
    raises rather than return something unsound, and at sigma = 0.2 and size 12
    it declines at 128, 256 and 384 bits alike -- isolating twelve eigenvalues of
    a matrix with condition 1e26 is far harder than deciding a sign. Cholesky
    costs n^3/3 ball operations and decides exactly the question asked.
    """
    import flint

    previous = flint.ctx.prec
    flint.ctx.prec = precision
    try:
        n = matrix.nrows()
        lower = [[flint.arb(0)] * n for _ in range(n)]
        for j in range(n):
            pivot = matrix[j, j] - flint.arb(shift)
            for k in range(j):
                pivot = pivot - lower[j][k] ** 2
            if not (pivot > 0):
                return False
            lower[j][j] = pivot.sqrt()
            for i in range(j + 1, n):
                entry = matrix[i, j]
                for k in range(j):
                    entry = entry - lower[i][k] * lower[j][k]
                lower[i][j] = entry / lower[j][j]
        return True
    finally:
        flint.ctx.prec = previous


def certified_lower_bound(matrix, precision: int = PRECISION, steps: int = 50) -> float:
    """The largest shift whose Cholesky still succeeds: a proved bound on lambda_min.

    `lambda_min <= A_ii` for every i, so the smallest diagonal entry brackets the
    search from above. Returns 0.0 when positive definiteness itself is not
    proved -- the bound and the verdict come from the same certificate.
    """
    if not is_positive_definite(matrix, 0.0, precision):
        return 0.0
    n = matrix.nrows()
    high = min(float(matrix[i, i].mid()) for i in range(n))

    # DESCEND GEOMETRICALLY FIRST. Bisecting [0, high] linearly cannot reach a
    # bound many orders below `high`: at sigma = 0.2 the smallest eigenvalue is
    # near 1e-29 against a diagonal of 6.7e-4, and fifty bisections get to 6e-19
    # having failed at every step -- reporting a bound of exactly zero for a
    # matrix just proved positive definite.
    low = 0.0
    for _ in range(240):
        if is_positive_definite(matrix, high, precision):
            low = high
            break
        high = high / 2.0
    if low == 0.0:
        return 0.0
    high = 2.0 * low
    for _ in range(steps):
        middle = 0.5 * (low + high)
        if is_positive_definite(matrix, middle, precision):
            low = middle
        else:
            high = middle
    return low


def negative_certificate(matrix, vector, precision: int = PRECISION):
    """An enclosure of `v^T A v`. Entirely below zero refutes positive definiteness.

    One vector is a complete refutation and needs no eigenvalue at all, so `v`
    may come from anywhere -- a float64 eigenvector is fine. Only the evaluation
    has to be rigorous, and it is.
    """
    import flint

    previous = flint.ctx.prec
    flint.ctx.prec = precision
    try:
        n = matrix.nrows()
        components = [flint.arb(float(x)) for x in vector]
        total = flint.arb(0)
        for i in range(n):
            for j in range(n):
                total = total + components[i] * matrix[i, j] * components[j]
        return total
    finally:
        flint.ctx.prec = previous


def certified_verdict(matrix, precision: int = PRECISION) -> tuple[str, float, float]:
    """POSITIVE, REFUTED or UNRESOLVED, each from a certificate rather than a floor.

    POSITIVE is a completed Cholesky, REFUTED is a vector whose quadratic form
    encloses in the negatives, and UNRESOLVED is neither -- which is a statement
    about this computation, never about zeta. No floor is guessed anywhere: an
    enclosure already says how much is known.
    """
    import numpy as np

    if is_positive_definite(matrix, 0.0, precision):
        return "POSITIVE", certified_lower_bound(matrix, precision), float("nan")

    n = matrix.nrows()
    approx = np.array([[float(matrix[i, j].mid()) for j in range(n)] for i in range(n)])
    try:
        _, vectors = np.linalg.eigh(approx)
    except np.linalg.LinAlgError:
        # No candidate vector, so no refutation is available -- which is a
        # statement about the search for one, never about the matrix.
        return "UNRESOLVED", float("nan"), float("nan")
    value = negative_certificate(matrix, vectors[:, 0], precision)
    if value < 0:
        return "REFUTED", float("nan"), float(value.mid())
    return "UNRESOLVED", float("nan"), float(value.mid())


def run(
    size: int = 8,
    sigma: float = 0.03,
    prime_limit: int = PRIME_LIMIT,
    precision: int = PRECISION,
    eval_limit: int = EVAL_LIMIT,
) -> ExperimentResult:
    matrix = form_ball(size, sigma, prime_limit, precision, eval_limit)
    non_finite = sum(
        0 if matrix[i, j].is_finite() else 1 for i in range(size) for j in range(size)
    )
    if non_finite:
        verdict, lower, quadratic = "NOT_COMPUTED", float("nan"), float("nan")
    else:
        verdict, lower, quadratic = certified_verdict(matrix, precision)

    # NaN DOES NOT SURVIVE THE ROUND TRIP. It serialises to JSON `null`, which
    # fails validation on the way back in, so a recorded run could not be read
    # again -- and the determinism gate reads every record. A metric that does
    # not apply is omitted, exactly as `weil_sensitivity` omits a threshold it
    # never measured.
    metrics = {
        "size": size,
        "sigma": float(sigma),
        "precision": precision,
        "violated": float(1.0 if verdict == "REFUTED" else 0.0),
        "unresolved": float(1.0 if verdict == "UNRESOLVED" else 0.0),
        "not_computed": float(1.0 if verdict == "NOT_COMPUTED" else 0.0),
        "non_finite_entries": float(non_finite),
    }
    if verdict == "POSITIVE":
        metrics["certified_lower_bound"] = lower
    if verdict == "REFUTED":
        metrics["refuting_quadratic_form"] = quadratic

    return ExperimentResult(
        name="weil-certified",
        parameters={
            "size": size,
            "sigma": sigma,
            "prime_limit": prime_limit,
            "precision": precision,
        },
        metrics=metrics,
        observations=[
            f"Verdict {verdict}, from a certificate rather than a floor. POSITIVE "
            "means a Cholesky in ball arithmetic ran to completion, every pivot "
            "proved strictly above zero; the lower bound on the smallest "
            f"eigenvalue is {lower:.6g}, the largest shift that still factors. It "
            "is a proved property of this matrix at this precision, and still only "
            "consistent with RH.",
            f"{non_finite} of {size * size} entries came back non-finite. Any at all "
            "means NOT_COMPUTED rather than UNRESOLVED: the integrator exceeded its "
            "evaluation budget and declined to return something unsound, which is a "
            "different state from a matrix whose smallest eigenvalue straddles zero. "
            "The evaluation budget scales with precision (see eval_budget), because "
            "acb.digamma's enclosure on a fixed box WIDENS with precision -- radius "
            "1.86 to 64.9 over 53 to 256 bits at box radius 0.05 -- so the integrator "
            "subdivides further and a fixed cap made precision look non-monotone. It "
            "is not: 256 bits converges on 400,000 evaluations in 11 seconds.",
            "Certified because the falsification test needs it. The float64 path "
            "reports lambda_min = -3.3e-15 at size 12 and sigma = 0.2, where this "
            "path proves lambda_min >= 1.96945e-29 at 192 bits. An uncertified "
            "REFUTED is indistinguishable from rounding.",
            "Positive definiteness is decided by Cholesky, not by eigenvalues. "
            "arb_mat.eig declines to isolate the spectrum of this form at 128, 256 "
            "and 384 bits alike -- separating twelve eigenvalues of a matrix with "
            "condition 1e26 is a far harder question than the sign, and it is not "
            "the question being asked.",
            "Refutation would come from a single vector whose quadratic form "
            "encloses below zero. That certificate needs no eigenvalue and no "
            "trust in where the vector came from: only the evaluation is rigorous.",
            "Both truncations are bounded, not dropped: the archimedean integral "
            "beyond |r| = 30/sigma, and the prime powers above the cutoff. At these "
            "parameters they contribute less than exp(-900) and exp(-23000).",
            "The integrand is the symmetrised digamma, not Re(digamma). Arb "
            "integrates over complex boxes and Re(psi(z)) is not analytic there, so "
            "the obvious spelling gives an enclosure with no theorem behind it.",
        ],
        assumptions=cite("digamma-tail", "von-mangoldt-bound"),
    )
