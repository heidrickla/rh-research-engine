"""Moments of `|zeta(1/2+it)|`, and what a reachable height can settle.

WHY MOMENTS. Keating and Snaith conjecture

    (1/T) int_0^T |zeta(1/2+it)|^{2k} dt  ~  c_k (log T)^{k^2},
    c_k = a_k * prod_{j=0}^{k-1} j!/(j+k)!

where the product is the random-matrix factor -- the moments of the
characteristic polynomial of a random unitary matrix, via the Barnes G
function -- and

    a_k = prod_p (1 - 1/p)^{k^2} sum_{m>=0} (Gamma(m+k)/(m! Gamma(k)))^2 p^{-m}

is an Euler product over the primes. **That factorisation is the point.** Every
other statistic in this engine measures the zeros against a universal law, and
universality means the agreement carries no arithmetic: GUE describes random
matrices and quantum billiards too. Here the arithmetic is a separate factor,
written down, and the two can be told apart.

k = 1 AND k = 2 ARE THEOREMS, which is the other reason. `c_1 = 1` (Hardy and
Littlewood) and `c_2 = 1/(2 pi^2)` (Ingham). So the measurement has a case
where the answer is known sitting under the case where it is not -- the same
shape as `finite_field_zeta`, and this engine keeps needing it.

THE LEADING TERM IS NOT THE ASYMPTOTIC, and at k = 1 that is checkable. The
theorem is not `log T` but `log(T/2pi) + 2 gamma - 1`, and at T = 5 x 10^4 the
two differ by 1.68 in 9.14 -- eighteen per cent. Measured against the full
statement the integrator agrees to 6e-4, which is the size of the known error
term; measured against the leading term alone it is out by eighteen per cent at
the one k where the truth is known. A discrepancy of that size at k = 3 would
therefore say nothing at all.

SO CAN THE LEADING COEFFICIENT BE EXTRACTED? The moment is `T` times a
polynomial of degree `k^2` in `log T`, whose leading coefficient is `c_k`.
Fitting that polynomial and reading it off is the obvious move, and **it does
not work here**: over T from 5 x 10^3 to 1.6 x 10^5, `log(T/2pi)` spans a
factor of 1.5, and a degree-4 fit across it returns `c_2 = 0.0777` against the
true 0.0507. Fifty-three per cent out, at a k where the answer is known.

That is a result, not a failure to produce one, and it is the reason
`MomentFit` carries `calibration_error`: the same extraction is run at k = 2
on the same heights, every time, so no extracted `c_3` is ever reported without
the demonstrated error rate of the method that produced it beside it.

AND THE FULL POLYNOMIAL IS THE SHARP TEST, which is the other half of this
file. At k = 2 the whole degree-4 polynomial is proven -- Ivic, and separately
Conrey -- and against it the measurement agrees to 1.2e-4 where the leading
term alone is out by 155%. That settles what the discrepancy was: lower-order
terms, not the conjecture.

Better still, naive random matrix theory predicts a DIFFERENT degree-4
polynomial with the SAME leading coefficient. The data follows the proven one
to 1e-4 and sits 6.4% from the RMT one, and the 6.4% does not shrink with
height. So the universal part is right and the subleading part is not, and the
gap between two polynomials sharing a leading term is the arithmetic, measured.

k = 3 CANNOT BE TESTED HERE, and the reason is the polynomial rather than the
computation. Conrey, Farmer, Keating, Rubinstein and Snaith conjecture the full
`P_3`; Hiary and Odlyzko use those coefficients without printing them. Until
they are transcribed from that source, the only available comparison at k = 3
is against a leading term -- which the k = 1 and k = 2 calibrations above show
says nothing.
"""

from __future__ import annotations

from math import factorial

import numpy as np
import sympy as sp
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..contracts.epistemic import RIGOROUS, Confidence

#: Primes carried in the Euler product for `a_k`.
#:
#: The product converges like `1 + O(k^2/p^2)`, so the tail past 200000 moves
#: `a_2` by under 1e-8 -- checked against the closed form `a_2 = 1/zeta(2)`,
#: which is what makes this number verifiable rather than chosen.
ARITHMETIC_PRIME_LIMIT = 200_000

#: Terms of the inner sum. It is `sum_m binomial(m+k-1, m)^2 p^{-m}`, so at
#: p = 2 and k = 4 the terms fall like `m^6 2^-m` -- forty is far past enough.
ARITHMETIC_SUM_TERMS = 60


def random_matrix_factor(k: int) -> sp.Rational:
    """`prod_{j=0}^{k-1} j!/(j+k)!`, exactly.

    The random-matrix half of `c_k`. Equivalently `g_k/(k^2)!` with `g_k` the
    Keating-Snaith constant, and `g_k` is what the literature tabulates:
    1, 2, 42, 24024 for k = 1..4. Kept as a `Rational` because it is one, and
    because `g_3 = 42` exactly is a check a float would blur.
    """
    if k < 1:
        raise ValueError(f"k = {k}: a moment index is a positive integer")
    out = sp.Integer(1)
    for j in range(k):
        out *= sp.Rational(factorial(j), factorial(j + k))
    return out


def keating_snaith_constant(k: int) -> sp.Integer:
    """`g_k = (k^2)! prod j!/(j+k)!` -- the tabulated form, for checking."""
    return sp.Integer(factorial(k * k)) * random_matrix_factor(k)


def arithmetic_factor(
    k: int, *, prime_limit: int = ARITHMETIC_PRIME_LIMIT
) -> float:
    """`a_k`, the Euler product over primes.

    `Gamma(m+k)/(m! Gamma(k))` is `binomial(m+k-1, m)`, an integer, so the
    inner sum is exact term by term and only the truncation of the product
    costs anything.

    Verifiable, which is why it is computed rather than tabulated: `a_1 = 1`
    identically, and `a_2 = 1/zeta(2) = 6/pi^2` in closed form, because the
    inner sum is `(1+x)/(1-x)^3` at `x = 1/p` and the product telescopes to
    `prod (1 - p^-2)`.
    """
    if k < 1:
        raise ValueError(f"k = {k}: a moment index is a positive integer")
    weights = np.array(
        [float(sp.binomial(m + k - 1, m)) ** 2 for m in range(ARITHMETIC_SUM_TERMS)]
    )
    total = 1.0
    for prime in sp.primerange(2, prime_limit):
        powers = float(prime) ** -np.arange(ARITHMETIC_SUM_TERMS)
        total *= (1.0 - 1.0 / prime) ** (k * k) * float(weights @ powers)
    return total


def moment_constant(k: int, *, prime_limit: int = ARITHMETIC_PRIME_LIMIT) -> float:
    """`c_k = a_k * prod j!/(j+k)!`. Known: `c_1 = 1`, `c_2 = 1/(2 pi^2)`."""
    return arithmetic_factor(k, prime_limit=prime_limit) * float(
        random_matrix_factor(k)
    )


def measured_moment(k: int, height: float, *, per_unit: int = 20) -> float:
    """`(1/T) int_0^T |Z(t)|^{2k} dt`, by Simpson on a uniform grid.

    `|zeta(1/2+it)| = |Z(t)|`, so this is the moment and not a proxy for it.

    Twenty points per unit is not a guess: the integral is stable to 3e-10
    between 20 and 40 points per unit at T = 2 x 10^4, and to 1e-12 by 160.
    Simpson on a smooth oscillation converges fast, and the density of
    oscillations grows only like `log t`.

    The lower limit is a real zero, not an approximation to one. `Z(0)` used
    to be NaN, because `theta` was computed from a series that divides by `t`;
    an integral starting here is what found that.
    """
    if k < 1:
        raise ValueError(f"k = {k}: a moment index is a positive integer")
    if height <= 0:
        raise ValueError(f"T = {height:g}: the range must be positive")
    from .riemann_siegel import z_function

    count = int(height * per_unit)
    count += count % 2
    grid = np.linspace(0.0, float(height), count + 1)
    values = np.abs(z_function(grid)) ** (2 * k)
    step = float(grid[1] - grid[0])
    total = (
        step
        / 3
        * (
            values[0]
            + values[-1]
            + 4 * values[1:-1:2].sum()
            + 2 * values[2:-2:2].sum()
        )
    )
    return float(total / height)


def second_moment_asymptotic(height: float) -> float:
    """`log(T/2pi) + 2 gamma - 1`: the k = 1 theorem, lower-order term included.

    The control on the integrator. Comparing against `log T` instead drops a
    constant of 1.68, which at T = 5 x 10^4 is eighteen per cent -- so an
    agreement at the per-cent level against the LEADING term would mean
    nothing, and against this it means the quadrature is right.
    """
    return float(
        np.log(height / (2 * np.pi)) + 2 * float(sp.EulerGamma.evalf()) - 1
    )


class MomentFit(BaseModel):
    """Measured moments, and the leading coefficient a fit claims to see."""

    model_config = ConfigDict(extra="forbid")

    k: int
    heights: list[float] = Field(default_factory=list)
    measured: list[float] = Field(default_factory=list)
    #: `measured / (c_k (log T)^{k^2})`. Converges to 1, slowly: at k = 1,
    #: where `c_k` is known exactly, it is still 0.86 at T = 1.6 x 10^5.
    leading_ratio: list[float] = Field(default_factory=list)
    #: What a polynomial fit says `c_k` is.
    extracted: float = 0.0
    #: `c_k` from the conjecture, for k where it is known to be a theorem.
    predicted: float = 0.0
    #: **The same extraction run at k = 2, where the answer is a theorem, on
    #: the same heights.** Relative error of the method, measured rather than
    #: assumed -- so no extracted coefficient is ever read without it.
    calibration_error: float = 0.0
    confidence: Confidence = Confidence.NUMERICAL

    @field_validator("confidence")
    @classmethod
    def _reject_rigorous_confidence(cls, value: Confidence) -> Confidence:
        if value in RIGOROUS:
            raise ValueError(
                f"a measured moment may not claim {value.value!r}: it is a "
                "finite integral against a conjecture about the limit, and for "
                "k >= 3 the conjecture itself is open"
            )
        return value

    @property
    def extraction_is_usable(self) -> bool:
        """Is the fitted coefficient worth reading at all?

        Only if the same method, on the same heights, recovers the k = 2
        theorem to better than ten per cent. Over T up to 1.6 x 10^5 it does
        not: it returns 0.0777 against 0.0507, out by 53%.
        """
        return 0.0 < self.calibration_error < 0.10


def _extract(k: int, heights: np.ndarray, values: np.ndarray) -> float:
    """Leading coefficient of the degree-`k^2` polynomial in `log(T/2pi)`."""
    degree = k * k
    if len(heights) <= degree:
        raise ValueError(
            f"{len(heights)} heights cannot determine a degree-{degree} "
            f"polynomial; k = {k} needs more than {degree}"
        )
    return float(np.polyfit(np.log(heights / (2 * np.pi)), values, degree)[0])


def fit_moment(
    k: int,
    heights: np.ndarray,
    *,
    per_unit: int = 20,
    prime_limit: int = ARITHMETIC_PRIME_LIMIT,
) -> MomentFit:
    """Measure the moment across heights and try to read off `c_k`.

    The k = 2 calibration is run every time, on the same heights, and travels
    with the record. An extracted coefficient without the demonstrated error
    of the method that produced it is a number nobody can weigh.
    """
    heights = np.asarray(heights, dtype=float)
    measured = np.array(
        [measured_moment(k, height, per_unit=per_unit) for height in heights]
    )
    predicted = moment_constant(k, prime_limit=prime_limit)
    ratio = measured / (predicted * np.log(heights) ** (k * k))

    calibration = np.array(
        [measured_moment(2, height, per_unit=per_unit) for height in heights]
    )
    truth = moment_constant(2, prime_limit=prime_limit)
    error = abs(_extract(2, heights, calibration) - truth) / truth

    return MomentFit(
        k=k,
        heights=[float(value) for value in heights],
        measured=[float(value) for value in measured],
        leading_ratio=[float(value) for value in ratio],
        extracted=_extract(k, heights, measured),
        predicted=predicted,
        calibration_error=float(error),
    )


# --- the full polynomial, which is where the arithmetic shows -------------
#
# THIS IS THE SHARP TEST, and the leading coefficient is not it. Both
# polynomials below have the SAME leading coefficient, 0.050660. They differ
# only in lower-order terms -- and the data tells them apart at the fourth
# decimal, which is the whole argument about where the arithmetic lives,
# measured.

#: The fourth-moment polynomial, PROVEN. Ivic and, separately, Conrey.
#:
#: Degree 4 in `x = log(T/2pi)`, and for the AVERAGED moment `(1/T) int_0^T`,
#: not for the integrand. The two differ: at k = 1 the integrand polynomial is
#: `x + 2 gamma` and its average is `x + 2 gamma - 1`, same leading term and a
#: shifted constant.
#:
#: Transcribed from Hiary and Odlyzko, "The zeta function on the critical line:
#: numerical evidence for moments and random matrix theory models", Math. Comp.
#: 81 (2012), equation (2.4). Checked, not trusted: its leading coefficient
#: agrees with `moment_constant(2)` computed here from the Euler product, and
#: the polynomial as a whole reproduces the measured fourth moment to 1.2e-4.
PROVEN_FOURTH_MOMENT = (0.050660, 0.496227, 0.937279, 1.35334, -0.040924)

#: The same moment as naive random matrix theory predicts it, with the
#: identification `N -> log(T/2pi)`. Hiary-Odlyzko equation (2.6).
#:
#: Same leading coefficient, different lower terms -- and that is the point of
#: keeping it. RMT gets the universal part right and the subleading part
#: wrong, so the gap between these two polynomials IS the arithmetic, written
#: down. The data sits 6.4% from this one and 1e-4 from the other, and the
#: 6.4% does not shrink with height.
NAIVE_RMT_FOURTH_MOMENT = (0.050660, 0.405284, 1.16519, 1.41849, 0.607927)

#: Where the polynomials came from, carried in the record.
MOMENT_POLYNOMIAL_SOURCE = (
    "Hiary and Odlyzko, Math. Comp. 81 (2012) 1723-1752, equations (2.4) and "
    "(2.6); the proven coefficients are due to Ivic and to Conrey"
)


def evaluate_polynomial(coefficients: tuple[float, ...], height: float) -> float:
    """A moment polynomial at `x = log(T/2pi)`, highest power first.

    The VARIABLE matters as much as the coefficients. In `log(T/2pi)` the
    leading term alone accounts for 214 of a measured 546 at T = 2 x 10^4 --
    out by 155%. In `log T` the same leading term gives 487, out by 12%.
    Neither is the conjecture; the polynomial is.
    """
    return float(np.polyval(list(coefficients), np.log(height / (2 * np.pi))))


class FourthMomentCheck(BaseModel):
    """The measured fourth moment against two polynomials that share a leader.

    Only the lower-order terms differ, so the comparison isolates exactly the
    part random matrix theory does not supply.
    """

    model_config = ConfigDict(extra="forbid")

    heights: list[float] = Field(default_factory=list)
    measured: list[float] = Field(default_factory=list)
    proven: list[float] = Field(default_factory=list)
    naive_rmt: list[float] = Field(default_factory=list)
    #: Relative departure from each, per height.
    proven_error: list[float] = Field(default_factory=list)
    rmt_error: list[float] = Field(default_factory=list)
    source: str = MOMENT_POLYNOMIAL_SOURCE
    confidence: Confidence = Confidence.NUMERICAL

    @field_validator("confidence")
    @classmethod
    def _reject_rigorous_confidence(cls, value: Confidence) -> Confidence:
        if value in RIGOROUS:
            raise ValueError(
                f"a measured moment may not claim {value.value!r}: the fourth "
                "moment polynomial is a theorem, but this is a finite "
                "quadrature compared against it, not a derivation of it"
            )
        return value

    @property
    def follows_the_proven_polynomial(self) -> bool:
        """Closer to the theorem than to naive RMT everywhere, and far closer
        where the error term is smallest.

        Two conditions, because one is not enough either way. "Closer at every
        height" alone would pass on a marginal preference; "ten times closer"
        alone would fail at the lowest height, where `E_2(T)/T` is still 5e-3
        and swamps the distinction -- and that would be the error term being
        read as a verdict about the polynomials.

        A conjunction over heights rather than an average: agreement at one
        height and not another would be the interesting failure, and a mean
        would hide it.
        """
        if not self.proven_error:
            return False
        everywhere = all(
            proven < rmt
            for proven, rmt in zip(self.proven_error, self.rmt_error, strict=True)
        )
        return everywhere and self.proven_error[-1] < self.rmt_error[-1] / 10


def check_fourth_moment(heights: np.ndarray, *, per_unit: int = 20) -> FourthMomentCheck:
    """Measure the fourth moment and hold it against both polynomials."""
    heights = np.asarray(heights, dtype=float)
    measured = np.array(
        [measured_moment(2, height, per_unit=per_unit) for height in heights]
    )
    proven = np.array(
        [evaluate_polynomial(PROVEN_FOURTH_MOMENT, height) for height in heights]
    )
    naive = np.array(
        [evaluate_polynomial(NAIVE_RMT_FOURTH_MOMENT, height) for height in heights]
    )
    return FourthMomentCheck(
        heights=[float(value) for value in heights],
        measured=[float(value) for value in measured],
        proven=[float(value) for value in proven],
        naive_rmt=[float(value) for value in naive],
        proven_error=[float(value) for value in np.abs(measured - proven) / proven],
        rmt_error=[float(value) for value in np.abs(measured - naive) / naive],
    )
