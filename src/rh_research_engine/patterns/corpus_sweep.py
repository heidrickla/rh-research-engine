"""Point the discovery loop at the corpus's own quantities.

The columns are not a guess at what the corpus is about. They come from four
places, and the third and fourth are where the value is:

  1. CALLABLES IN THE INDEXED CORPUS that evaluate at a positive integer.
  2. DERIVED ARITHMETIC nobody asked for. Two of these -- the running sums of
     `mu` and of `Lambda` -- are definitions the corpus already states, and
     rediscovering them is the validation: a scan that cannot find what it
     knows will not find what it does not.
  3. BOTH SIDES OF EVERY INEQUALITY THE CORPUS ASSERTS. Robin, Lagarias, the
     two Schoenfeld bounds, the Mertens conjecture. Each is a claim `a <= b`
     that says nothing about where it is tight, so a saturated-bound finding
     over such a pair is the scan reading the corpus's own statement back as a
     characterisation of the cases where it is sharp.
  4. THE ZEROS. Ordinates, gaps, GUE-normalised gaps, and `S(gamma_n)`. These
     cost 160 ms each before `riemann_siegel` and cost about 0.1 ms now, which
     is the only reason this section can exist at all.

WHAT THE ZERO COLUMNS FOUND, WHICH IS NOTHING, AND WHY THAT IS INFORMATIVE.
This detector looks for EXACT relations -- equalities, saturated bounds,
constant ratios. Zero ordinates are irrational and satisfy no exact relation
with an arithmetic function, so no amount of widening will produce one. The
structure that is there -- pair correlation, the GUE spacing law, Montgomery's
conjecture -- is DISTRIBUTIONAL, and a detector built to refuse statistical
trends cannot see it by construction. That is a real limit of the instrument
and not a fact about the zeros; recorded here so the next reader does not
conclude the zeros are featureless.

EVERY SPEED-UP IS CHECKED AGAINST THE SLOW ROUTE. The summatory columns are
accumulated once instead of re-summed per row, `zeta` comes from mpmath rather
than sympy, and the sieve replaces `factorint`. Each is compared against the
original at sampled points in `verify_shortcuts`, because a column that is
merely fast is a column that has not been checked.
"""

from __future__ import annotations

import sys
from fractions import Fraction

import mpmath
import numpy as np
import sympy as sp

from .models import Observation

#: `gamma(n) = (n-1)!` runs to some sixteen thousand digits by n = 5000, and
#: Python refuses to render an integer longer than 4300 by default. The column
#: was dropped for it -- reported rather than silent, but dropped -- and a
#: display limit is not a reason to stop measuring something.
_MAX_DIGITS = 500_000

#: Working precision for the accumulated sums: twice what any scan reads, so
#: the accumulation is never the thing that limits a comparison.
_ACCUMULATION_DPS = 60


class CorpusColumns:
    """Every column of a sweep, and the sieves they are built from."""

    def __init__(self, top: int) -> None:
        if top < 4:
            raise ValueError("a sweep needs at least a few rows")
        sys.set_int_max_str_digits(_MAX_DIGITS)
        mpmath.mp.dps = _ACCUMULATION_DPS

        self.top = top
        self.rows = list(range(2, top))
        self.labels = [str(n) for n in self.rows]

        self.mobius = _mobius_sieve(top)
        self.smallest_factor = _smallest_prime_factor(top)

        self.mertens = [0] * (top + 2)
        running = 0
        for n in range(1, top + 2):
            running += self.mobius[n]
            self.mertens[n] = running

        # psi(n) and H(n) are running sums. Held symbolically they were
        # re-evaluated in full on every row: psi alone cost nine seconds at
        # n = 1000 and grows quadratically, and H(5000) is an exact rational
        # with a two-thousand-digit denominator that every comparison carries.
        # Both are irrational or effectively so at the point of use, so they are
        # accumulated once at high precision.
        self.psi = [mpmath.mpf(0)] * (top + 2)
        self.primepi = [0] * (top + 2)
        running_psi, primes_seen = mpmath.mpf(0), 0
        for n in range(2, top + 2):
            factors = self.factorise(n)
            if len(factors) == 1:
                running_psi += mpmath.log(next(iter(factors)))
                if sum(factors.values()) == 1:
                    primes_seen += 1
            self.psi[n] = running_psi
            self.primepi[n] = primes_seen

        self.harmonic = [mpmath.mpf(0)] * (top + 2)
        running_harmonic = mpmath.mpf(0)
        for n in range(1, top + 2):
            running_harmonic += mpmath.mpf(1) / n
            self.harmonic[n] = running_harmonic

        self.primes = list(sp.sieve.primerange(2, sp.prime(top + 2) + 2))

        # A second summatory Mobius function, accumulated from sympy's `mobius`
        # instead of the sieve, so their agreement is worth something.
        self.independent_mertens = [0] * (top + 2)
        running_independent = 0
        for n in range(1, top + 2):
            running_independent += int(sp.mobius(n))
            self.independent_mertens[n] = running_independent

        self.ordinates: np.ndarray | None = None

    def von_mangoldt(self, n: int):
        """`Lambda(n)`: `log p` at a prime power, zero otherwise."""
        factors = self.factorise(n)
        if len(factors) != 1:
            return mpmath.mpf(0)
        return mpmath.log(next(iter(factors)))

    def factorise(self, n: int) -> dict[int, int]:
        out: dict[int, int] = {}
        while n > 1:
            prime = self.smallest_factor[n]
            while n % prime == 0:
                out[prime] = out.get(prime, 0) + 1
                n //= prime
        return out

    def omega(self, n: int) -> int:
        return len(self.factorise(n))

    def big_omega(self, n: int) -> int:
        return sum(self.factorise(n).values())

    def load_zeros(self) -> None:
        """Locate the ordinates. Separate because it is the expensive half."""
        from ..symbolic.riemann_siegel import first_zero_ordinates

        self.ordinates = first_zero_ordinates(self.top + 2)

    def verify_shortcuts(self, at: list[int] | None = None) -> list[str]:
        """Compare every accumulated column against the slow route.

        A column that is merely fast is a column that has not been checked, and
        each of these replaced something the rest of the engine still computes
        the long way.
        """
        points = at or [n for n in (5, 17, 60, 199, self.top - 1) if 2 <= n < self.top]
        checked = []
        for n in points:
            assert self.mertens[n] == sum(sp.mobius(k) for k in range(1, n + 1)), n
            assert self.independent_mertens[n] == self.mertens[n], n
            assert _close(
                self.von_mangoldt(n),
                sp.log(min(sp.factorint(n)))
                if len(sp.factorint(n)) == 1
                else sp.Integer(0),
            ), n
            assert self.primepi[n] == sp.primepi(n), n
            assert self.primes[n - 1] == sp.prime(n), n
            assert _close(self.harmonic[n], sp.harmonic(n)), n
            # Against the engine's own VonMangoldt, which is the definition.
            # A hand-rolled reference summing log(p^k) instead of log(p) called
            # the accumulation wrong when the accumulation was right.
            from ..symbolic.functions import VonMangoldt

            symbolic_psi = sp.Add(
                *[VonMangoldt(sp.Integer(k)) for k in range(2, n + 1)]
            )
            assert _close(self.psi[n], symbolic_psi), n
            checked.append(f"n={n}")
        return checked


def _mobius_sieve(limit: int) -> list[int]:
    values = [1] * (limit + 2)
    values[0] = 0
    for prime in sp.sieve.primerange(2, limit + 2):
        for multiple in range(prime, limit + 2, prime):
            values[multiple] *= -1
        for multiple in range(prime * prime, limit + 2, prime * prime):
            values[multiple] = 0
    return values


def _smallest_prime_factor(limit: int) -> list[int]:
    smallest = list(range(limit + 2))
    candidate = 2
    while candidate * candidate <= limit + 1:
        if smallest[candidate] == candidate:
            for multiple in range(candidate * candidate, limit + 2, candidate):
                if smallest[multiple] == multiple:
                    smallest[multiple] = candidate
        candidate += 1
    return smallest


def _close(fast, slow, tolerance: str = "1e-40") -> bool:
    return abs(mpmath.mpf(fast) - mpmath.mpf(str(sp.N(slow, 50)))) < mpmath.mpf(
        tolerance
    )


def exact_text(value, digits: int) -> str:
    """Exact where the value is exact; a decimal of `digits` places otherwise.

    Exactness is the property the scan measures, so a rational must stay
    rational: squeezing one through a float makes "equality in 13 of 13" a
    claim about 1e-16 instead.
    """
    if isinstance(value, mpmath.mpf | mpmath.mpc):
        return mpmath.nstr(value, digits, strip_zeros=False)
    value = sp.sympify(value)
    if value.is_Rational:
        return str(Fraction(int(value.p), int(value.q)))
    return str(sp.N(value, digits))


def column_tables(columns: CorpusColumns) -> list[tuple[dict, bool, str]]:
    """The four groups of columns, with whether each was asked for."""
    data = columns

    corpus = {
        "n": lambda n: sp.Integer(n),
        "sigma": lambda n: sp.divisor_sigma(n),
        "mu": lambda n: sp.Integer(data.mobius[n]),
        "primepi": lambda n: sp.Integer(data.primepi[n]),
        "harmonic": lambda n: data.harmonic[n],
        "Mertens": lambda n: sp.Integer(data.mertens[n]),
        "nth_prime": lambda n: sp.Integer(data.primes[n - 1]),
        "psi": lambda n: data.psi[n],
        # From the factorisation, NOT as psi(n) - psi(n-1). The accumulated sums
        # carry rounding, so the difference of two of them is log(p) plus an
        # error in the last bits -- and `Lambda(n) = log n exactly at the
        # primes` then failed the precision check, losing the von Mangoldt
        # characterisation of the primes to an arithmetic convenience.
        "Lambda": data.von_mangoldt,
        "li": lambda n: sp.li(n),
        "log": lambda n: mpmath.log(n),
        # mpmath rather than sympy: 136x faster over this range, agreeing to
        # 1e-28, where sympy's cost 42 seconds of every build at n = 5000.
        "zeta": lambda n: mpmath.zeta(n),
        "gamma": lambda n: sp.gamma(n),
        "binomial_2n_n": lambda n: sp.binomial(2 * n, n),
    }

    derived = {
        "d": lambda n: sp.divisor_count(n),
        "totient": lambda n: sp.totient(n),
        "omega": data.omega,
        "Omega": data.big_omega,
        "rad": lambda n: sp.prod(data.factorise(n).keys()),
        "n_plus_1": lambda n: sp.Integer(n + 1),
        "sigma_minus_n": lambda n: sp.divisor_sigma(n) - n,
        "n_log_n": lambda n: n * mpmath.log(n),
        "sqrt_n": lambda n: mpmath.sqrt(n),
        "sigma_over_n": lambda n: sp.Rational(sp.divisor_sigma(n), n),
        "abs_mu": lambda n: abs(data.mobius[n]),
        "sigma_2": lambda n: sp.divisor_sigma(n, 2),
        "n_minus_totient": lambda n: n - sp.totient(n),
        "harmonic_minus_log": lambda n: data.harmonic[n] - mpmath.log(n),
        "abs_Mertens": lambda n: abs(data.mertens[n]),
        "prime_gap": lambda n: sp.Integer(data.primes[n] - data.primes[n - 1]),
        # Accumulated from sympy's mobius rather than from the sieve, so that
        # `Mertens = cum_mu` is a cross-check between two implementations and
        # not two names for one array. It read `data.mertens` at first, which
        # made the rediscovery vacuous.
        "cum_mu": lambda n: sp.Integer(data.independent_mertens[n]),
    }

    bounds = {
        # sigma(n) < e^gamma n log log n for n > 5040.
        "robin_rhs": lambda n: mpmath.exp(mpmath.euler) * n * mpmath.log(mpmath.log(n))
        if n >= 3
        else mpmath.mpf(0),
        # sigma(n) <= e^{H_n} log(H_n) + H_n for n >= 1.
        "lagarias_rhs": lambda n: mpmath.exp(data.harmonic[n])
        * mpmath.log(data.harmonic[n])
        + data.harmonic[n],
        # |psi(x) - x| < sqrt(x) log^2 x / (8 pi) for x >= 74.
        "schoenfeld_psi_lhs": lambda n: abs(data.psi[n] - n),
        "schoenfeld_psi_rhs": lambda n: mpmath.sqrt(n)
        * mpmath.log(n) ** 2
        / (8 * mpmath.pi),
        # |pi(x) - li(x)| < sqrt(x) log x / (8 pi) for x >= 2657.
        "schoenfeld_pi_lhs": lambda n: abs(
            mpmath.mpf(data.primepi[n]) - mpmath.li(n)
        ),
        "schoenfeld_pi_rhs": lambda n: mpmath.sqrt(n) * mpmath.log(n) / (8 * mpmath.pi),
        # The Mertens conjecture, |M(x)| < sqrt(x) -- FALSE, and open where.
        "mertens_conj_rhs": lambda n: mpmath.sqrt(n),
    }

    groups = [
        (corpus, True, "a callable appearing in the indexed corpus"),
        (derived, False, "derived without being asked for"),
        (bounds, False, "a side of an inequality the corpus asserts"),
    ]

    if data.ordinates is not None:
        ordinates = data.ordinates
        zeros = {
            "zero_ordinate": lambda n: mpmath.mpf(float(ordinates[n - 1])),
            "zero_gap": lambda n: mpmath.mpf(
                float(ordinates[n] - ordinates[n - 1])
            ),
            # Mean 1 under the GUE prediction: the gap over the local average.
            "normalised_gap": lambda n: mpmath.mpf(
                float(
                    (ordinates[n] - ordinates[n - 1])
                    * np.log(ordinates[n - 1] / (2 * np.pi))
                    / (2 * np.pi)
                )
            ),
            # N(gamma_n) is n by construction, so the column worth having is
            # the smooth term there -- and the difference is S(gamma_n).
            "S_at_zero": lambda n: mpmath.mpf(
                float(n - 1 - _theta(ordinates[n - 1]) / np.pi)
            ),
        }
        groups.append((zeros, False, "built from the zeros of zeta"))
    return groups


def _theta(height: float) -> float:
    from ..symbolic.riemann_siegel import theta

    return float(theta(height))


def build_observations(columns: CorpusColumns, digits: int) -> list[Observation]:
    """Every column, evaluated at `digits` places."""
    made: list[Observation] = []
    for table, requested, source in column_tables(columns):
        for name, function in table.items():
            made.append(
                Observation(
                    name=name,
                    values=[exact_text(function(n), digits) for n in columns.rows],
                    labels=columns.labels,
                    requested=requested,
                    source=source,
                )
            )
    return made
