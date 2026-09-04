"""The number-theoretic functions these formulas are written in.

Most of them are SymPy's already. `zeta` is the Riemann zeta, `primepi` is
pi(x), `divisor_sigma` is sigma(n), `harmonic` is H_n, and `li` is the
logarithmic integral -- all symbolic under a symbolic argument and all exact
on a concrete one. This parser used to replace every one of them with an
undefined `Function` of the same name, which stored the right shape attached
to no meaning: `divisor_sigma(12)` is 28, but `Function('sigma')(12)` is
nothing at all, so Robin's inequality could be indexed and never checked.

The three defined here are the ones SymPy does not carry. They follow the
same contract: exact on a concrete integer, unevaluated otherwise, so a
formula written with them can be tested against actual values.
"""

from __future__ import annotations

from fractions import Fraction

import sympy as sp

#: Above this, a summatory function stays unevaluated.
#:
#: `Mertens(10**9)` is a well-defined integer and computing it by summing
#: mobius would hang the parse that mentioned it. Refusing to expand is not a
#: loss of meaning -- the term still denotes the same number -- where a wrong
#: answer or a hang would be.
_SUMMATORY_EVALUATION_LIMIT = 10**5


class NthPrime(sp.Function):
    """The k-th prime, one-based. `NthPrime(1)` is 2 and `NthPrime(3)` is 5.

    SymPy's `prime` raises on a symbolic index, so it cannot appear as the
    index expression of a `Product` running to infinity. This can, which is
    what lets the Euler product be written as a product over position.
    """

    is_integer = True
    is_positive = True

    @classmethod
    def eval(cls, index: sp.Expr) -> sp.Expr | None:
        if index.is_Integer:
            if index.is_positive:
                return sp.Integer(sp.prime(int(index)))
            raise ValueError(f"there is no prime at position {index}")
        return None

    def _eval_is_prime(self) -> bool:
        return True


class VonMangoldt(sp.Function):
    """`Lambda(n)`: `log p` when n is a prime power, and zero otherwise."""

    is_real = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n: sp.Expr) -> sp.Expr | None:
        if not n.is_Integer:
            return None
        if n < 2:
            return sp.Integer(0)
        factors = sp.factorint(int(n))
        if len(factors) == 1:
            (base,) = factors
            return sp.log(base)
        return sp.Integer(0)


class Mertens(sp.Function):
    """`M(x)`: the summatory Mobius function, `sum_{n <= x} mu(n)`.

    The Mertens conjecture said `|M(x)| < sqrt(x)` and is FALSE; RH says
    `M(x) = O(x^{1/2+eps})` and is open. The two differ by an epsilon, which
    is the reason this is worth being able to evaluate rather than store.
    """

    is_integer = True

    @classmethod
    def eval(cls, x: sp.Expr) -> sp.Expr | None:
        if not x.is_Integer:
            return None
        bound = int(x)
        if bound < 1:
            return sp.Integer(0)
        if bound > _SUMMATORY_EVALUATION_LIMIT:
            return None
        return sp.Integer(sum(sp.mobius(n) for n in range(1, bound + 1)))


class ChebyshevPsi(sp.Function):
    """`psi(x)`: the second Chebyshev function, `sum_{n <= x} Lambda(n)`."""

    is_real = True
    is_nonnegative = True

    @classmethod
    def eval(cls, x: sp.Expr) -> sp.Expr | None:
        if not x.is_Integer:
            return None
        bound = int(x)
        if bound < 2:
            return sp.Integer(0)
        if bound > _SUMMATORY_EVALUATION_LIMIT:
            return None
        return sp.Add(*[VonMangoldt(sp.Integer(n)) for n in range(2, bound + 1)])


class RiemannXi(sp.Function):
    """`xi(s) = (1/2) s (s-1) pi^{-s/2} Gamma(s/2) zeta(s)`.

    Kept unevaluated so the formula that DEFINES it does not collapse to a
    tautology, but it evaluates numerically, so `xi(s) = xi(1-s)` can actually
    be tested rather than merely recorded -- and it holds: xi(2) and xi(-1)
    agree to full precision.
    """

    @classmethod
    def eval(cls, s: sp.Expr) -> sp.Expr | None:
        # xi(0) = xi(1) = 1/2. Both are points where the FACTORED form has a
        # cancelling pole -- Gamma's at 0, zeta's at 1 -- and xi itself is
        # entire. Evaluating the factors separately raises there, which is a
        # fact about the factorisation and not about xi.
        if s in (sp.Integer(0), sp.Integer(1)):
            return sp.Rational(1, 2)
        return None

    def _eval_rewrite_as_zeta(self, s: sp.Expr, **kwargs: object) -> sp.Expr:
        return sp.Rational(1, 2) * s * (s - 1) * sp.pi ** (-s / 2) * sp.gamma(s / 2) * sp.zeta(s)

    def _eval_evalf(self, prec: int) -> sp.Expr | None:
        import mpmath

        argument = self.args[0]
        if not argument.is_number:
            return None
        digits = sp.core.evalf.prec_to_dps(prec)
        with mpmath.workdps(digits + 10):
            # Through mpmath at working precision, NOT through `complex`:
            # a double has 16 digits, so converting first and asking for 30
            # returns fourteen digits of noise with no sign that it did.
            value = argument.evalf(digits + 10)
            point = mpmath.mpc(
                mpmath.mpf(str(sp.re(value))), mpmath.mpf(str(sp.im(value)))
            )
            # The pole-free grouping. `s*Gamma(s/2)` is `2*Gamma(s/2 + 1)`,
            # which removes Gamma's pole at s = 0 outright; zeta's pole at
            # s = 1 is cancelled by `(s-1)` and filled by hand, since the
            # product tends to 1 there.
            if point == 1:
                shifted_zeta = mpmath.mpf(1)
            else:
                shifted_zeta = (point - 1) * mpmath.zeta(point)
            result = (
                mpmath.pi ** (-point / 2)
                * mpmath.gamma(point / 2 + 1)
                * shifted_zeta
            )
        return sp.sympify(result)


#: Float64 seeds from the bulk scan, the zeros refined from them, and the
#: working precision they were refined at.
#:
#: `mpmath.zetazero` is a ROOT-FIND, not a lookup, and the zeros do not move
#: between calls: `NthZetaZero(k)` over a column of indices re-found every one
#: of them, at about 160 ms each.
#:
#: Two caches rather than one, because the two halves have different costs and
#: different lifetimes. Seeds come a whole scan at a time and do not depend on
#: the working precision; refinements are per-index and do. Refining a block
#: eagerly was worse than the problem it solved -- at a block of 1024,
#: `NthZetaZero(1)` refined a thousand zeros nobody had asked for and the test
#: suite went from two minutes to twelve.
_SEED_ORDINATES: list[float] = []
_REFINED: dict[int, object] = {}
_LOCATED_AT_DPS: int | None = None

#: Seeds fetched when the seed cache is short, as a multiple of what was asked.
#: Geometric, so a column running to index n costs a logarithmic number of
#: scans rather than one per index.
_SEED_GROWTH = 2
_SEED_FLOOR = 128


def _seed_ordinate(index: int) -> float:
    """The index-th ordinate to float64 accuracy, from the vectorised scan."""
    if len(_SEED_ORDINATES) < index:
        from .riemann_siegel import first_zero_ordinates

        wanted = max(index * _SEED_GROWTH, _SEED_FLOOR)
        _SEED_ORDINATES[:] = [float(x) for x in first_zero_ordinates(wanted)]
    return _SEED_ORDINATES[index - 1]


def _locate_zero(index: int):
    """The index-th nontrivial zero, located once and remembered.

    Seeded rather than searched blind. `riemann_siegel` produces the ordinate to
    about 2e-12 in a fraction of a millisecond, and the exact root-find takes it
    from there in one or two steps -- returning, bit for bit, what `zetazero`
    returns.

    The seed is CHECKED, not trusted. A root-find handed a starting point can
    converge to the neighbouring zero, and a list of ordinates with one
    duplicated and one missing would still be the right length. Anything that
    moves further than a fifth of the local gap falls back to `zetazero`.

    The refinements are dropped when mpmath's working precision changes. A zero
    found at fifteen digits is not the same value as one found at fifty, and
    handing back the coarser one would make raising the precision have no
    effect -- silently wrong, which is worse than slow. The float64 seeds do not
    depend on the precision, so they survive.

    THAT GUARD ONLY PROTECTS DIRECT CALLERS. `NthZetaZero(k)` builds a SymPy
    expression, and SymPy memoises expression construction globally: after
    raising `mp.dps`, asking for the same index returns the identical object
    from before, and nothing here is consulted. Reaching the higher precision
    through the symbolic layer needs `sympy.core.cache.clear_cache()`. This is
    not new -- the same was true of the previous implementation -- but it was
    not written down, and it is exactly the sort of thing that reads as a bug in
    the refinement when it is not.
    """
    global _LOCATED_AT_DPS
    import math

    import mpmath

    if _LOCATED_AT_DPS != mpmath.mp.dps:
        _REFINED.clear()
        _LOCATED_AT_DPS = mpmath.mp.dps
    if index in _REFINED:
        return _REFINED[index]

    seed = _seed_ordinate(index)
    gap = 2 * math.pi / math.log(max(seed, 12.0) / (2 * math.pi))
    try:
        # The tolerance has to be tied to the WORKING precision, not left to
        # the default. With a float64 seed and a default tolerance the secant
        # iteration declares victory as soon as it matches the seed: at
        # dps = 40 that returned a value agreeing with `zetazero` in sixteen
        # digits and disagreeing in the next twenty-four, while the cache
        # invalidation above was faithfully recording that it had been computed
        # at forty. Precisely the silent kind of wrong.
        tolerance = mpmath.mpf(10) ** (-(mpmath.mp.dps + 5))
        ordinate = mpmath.findroot(
            mpmath.siegelz,
            (mpmath.mpf(seed), mpmath.mpf(seed) + mpmath.mpf(gap) / 1000),
            tol=tolerance,
            maxsteps=200,
        )
        if abs(float(ordinate) - seed) > gap / 5:
            raise ValueError("converged to a different zero")
        if abs(mpmath.siegelz(ordinate)) > mpmath.mpf(10) ** (-(mpmath.mp.dps - 3)):
            raise ValueError("Z does not vanish there to working precision")
        zero = mpmath.mpc(mpmath.mpf(1) / 2, ordinate)
    except (ValueError, ZeroDivisionError, TypeError):
        zero = mpmath.zetazero(index)
    _REFINED[index] = zero
    return zero


#: Above this height the zero count stays symbolic.
#:
#: 10^7, where the count is 21136125 -- not 200, which is where it sat while
#: N(T) was computed by locating every zero below T with a root-find each.
#: `mpmath.nzeros` walks Gram and Rosser blocks instead, in constant time, and
#: `test_the_zero_count_agrees_with_locating_every_zero` compares the two at
#: every integer and half-integer height to 300 -- 900 comparisons against 139
#: actual zeros, agreeing at all of them.
#:
#: Note what this is NOT. `nzeros` separates the zeros inside a block by the
#: sign changes of Z, so it is not independent of counting critical-line zeros;
#: it is a different, much cheaper route to the same quantity. An independent
#: count of the zeros in the STRIP is `argument_principle.strip_zero_count`.
#:
#: A limit remains only so that a formula mentioning N(10^100) cannot hang a
#: parse. It is no longer a statement about what can be checked.
_ZERO_COUNT_HEIGHT_LIMIT = 10**7


class ZeroCount(sp.Function):
    """`N(T)`: how many zeros of zeta have imaginary part in `(0, T]`.

    Exact. The Riemann-von Mangoldt formula ESTIMATES this quantity, so keeping
    the two distinct is the whole point: a formula that approximates N(T)
    cannot be checked against a definition that is the same approximation.

    Counted by Turing's method rather than by locating each zero. Locating them
    was quadratic across a column of heights -- one root-find per zero below T,
    repeated for every T -- and that cost, not the mathematics, is what had the
    reach pinned at T = 200. The two agree at every height they have been
    compared at; see `_ZERO_COUNT_HEIGHT_LIMIT`.
    """

    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, height: sp.Expr) -> sp.Expr | None:
        if not height.is_number or not height.is_real:
            return None
        if height <= 0:
            return sp.Integer(0)
        if height > _ZERO_COUNT_HEIGHT_LIMIT:
            return None

        import mpmath

        # Turing's method. Still exact, and still a computation about where the
        # zeros ARE rather than an application of the counting formula -- which
        # matters because the formula that ESTIMATES N(T) is one of the things
        # this is used to check, and a definition that was the same estimate
        # would check nothing.
        return sp.Integer(int(mpmath.nzeros(float(height))))


class BigO(sp.Function):
    """An asymptotic bound, and the point it is a bound at.

    NOT SymPy's `Order`, which is a germ at zero by default and absorbs the
    terms it dominates -- it would have deleted `li(x)` from von Koch's
    theorem outright. This one asserts nothing and absorbs nothing; it records
    the bounding term and the limit point so that two statements about
    different regimes cannot collide in the index.
    """

    @classmethod
    def eval(cls, bound: sp.Expr, point: sp.Expr | None = None) -> sp.Expr | None:
        if point is None:
            # Every asymptotic statement in this literature is at infinity.
            # Storing it explicitly is what keeps it from being ASSUMED.
            return cls(bound, sp.oo)
        return None


#: Beyond this index a zero stays symbolic. Locating one is a computation, not
#: a lookup, and a formula mentioning a large index must not hang the parse.
_ZERO_INDEX_LIMIT = 2000


class NthZetaZero(sp.Function):
    """`rho_k`: the k-th nontrivial zero of zeta in the upper half-plane.

    Ordered by imaginary part, which is the ordering the explicit formula
    REQUIRES -- its sum over zeros is conditionally convergent and means
    nothing without it. `NthZetaZero(1)` is 1/2 + 14.134725...i.

    Located, not assumed: mpmath finds the zero rather than placing it on the
    critical line, so a formula written with this is checked against where the
    zeros actually are.
    """

    @classmethod
    def eval(cls, index: sp.Expr) -> sp.Expr | None:
        if not index.is_Integer:
            return None
        if index < 1:
            raise ValueError(f"there is no zero at position {index}")
        if index > _ZERO_INDEX_LIMIT:
            return None
        import mpmath

        zero = _locate_zero(int(index))
        return sp.sympify(mpmath.re(zero)) + sp.I * sp.sympify(mpmath.im(zero))


#: Every name that, WRITTEN APPLIED, denotes a real function.
#:
#: Applied is the whole distinction. `\pi(x)` is the prime-counting function
#: and `2\pi` is the number; `\sigma(n)` is the divisor sum and `\sigma` alone
#: is the real part of s. Deciding by the syntax the source used is what keeps
#: one name from having to mean both.
#:
#: Both spellings are here on purpose. A formula is read once from LaTeX and
#: again from the form it printed as, and `\pi(x)` prints as `primepi(x)`. If
#: only the LaTeX spelling resolved, the second reading would rebuild an
#: undefined stub and the two readings would describe different objects.

#: Above this the Redheffer determinant is refused rather than computed.
#:
#: Measured, not guessed: exact integer elimination on the n x n matrix is
#: cubic, and it runs in 0.23 s at n = 200, 1.9 s at 400 and 15.9 s at 800.
#: The cap is where a guard check stays cheap enough to run in a pre-commit
#: hook. It bounds the CHECK, and says nothing about the identity, which is a
#: theorem for every n.
_REDHEFFER_LIMIT = 400

#: Above this the Farey quantities are refused rather than computed.
#:
#: `|F_n|` is asymptotically `3n^2/pi^2`, so the fractions themselves are the
#: cost: 304192 of them at n = 1000, three seconds to build and sort. Same
#: reasoning as the cap above -- a limit on what is checked here, not on what
#: is true.
_FAREY_LIMIT = 600


def _redheffer_determinant(size: int) -> int:
    """`det R_n`, by exact fraction-free elimination.

    Computed AS A DETERMINANT, deliberately and expensively. Returning
    `Mertens(n)` would be much faster and would make `det R_n = M(n)` a
    tautology -- the same mistake `cum_mu` was written to avoid, where the
    rediscovery of `Mertens = sum mu` was vacuous until the second column was
    accumulated by a different route. The whole content of this identity is
    that a linear-algebra computation and a number-theoretic one agree.

    Bareiss rather than rational Gaussian elimination: every intermediate is
    an integer and an exact divisor is known at each step, so nothing grows a
    denominator and nothing rounds.
    """
    matrix = [
        [1 if (column == 0 or (column + 1) % (row + 1) == 0) else 0
         for column in range(size)]
        for row in range(size)
    ]
    sign, previous = 1, 1
    for k in range(size - 1):
        if matrix[k][k] == 0:
            swap = next((i for i in range(k + 1, size) if matrix[i][k]), None)
            if swap is None:
                return 0
            matrix[k], matrix[swap] = matrix[swap], matrix[k]
            sign = -sign
        pivot = matrix[k][k]
        for i in range(k + 1, size):
            row_i, row_k = matrix[i], matrix[k]
            head = row_i[k]
            for j in range(k + 1, size):
                row_i[j] = (row_i[j] * pivot - head * row_k[j]) // previous
        previous = pivot
    return sign * matrix[-1][-1]


class RedhefferDet(sp.Function):
    """`det R_n`, the determinant of the n x n Redheffer matrix.

    `R_n` has a 1 wherever the column is 1 or the row divides the column, and
    0 elsewhere -- a matrix of nothing but divisibility. Its determinant is
    `M(n)`, so RH becomes a statement about the growth of a determinant of a
    0/1 matrix, which is why this is worth having: every other criterion in
    this corpus is analysis, and this one is linear algebra.
    """

    is_integer = True

    @classmethod
    def eval(cls, x: sp.Expr) -> sp.Expr | None:
        if not x.is_Integer:
            return None
        size = int(x)
        if size < 1:
            return sp.Integer(1)  # the empty determinant
        if size > _REDHEFFER_LIMIT:
            return None
        return sp.Integer(_redheffer_determinant(size))


def _farey(order: int) -> list[Fraction]:
    """The Farey fractions in `(0, 1]` of the given order, increasing.

    Enumerated and deduplicated rather than counted from `sum phi(k)`, for the
    same reason the determinant above is not read off `Mertens`: `|F_n| = sum
    phi(k)` is one of the statements being checked, and computing the left
    side from the right would check nothing.
    """
    return sorted(
        {Fraction(a, b) for b in range(1, order + 1) for a in range(1, b + 1)}
    )


class FareyCount(sp.Function):
    """`|F_n|`: how many Farey fractions of order n lie in `(0, 1]`."""

    is_integer = True
    is_positive = True

    @classmethod
    def eval(cls, x: sp.Expr) -> sp.Expr | None:
        if not x.is_Integer:
            return None
        order = int(x)
        if order < 1:
            return sp.Integer(0)
        if order > _FAREY_LIMIT:
            return None
        return sp.Integer(len(_farey(order)))


class FareyDeviation(sp.Function):
    """`sum_i |F_i - i/|F_n||`: how far the Farey fractions sit from uniform.

    Franel and Landau: this being `O(n^{1/2+eps})` is EQUIVALENT to RH. The
    Farey fractions are a purely combinatorial object -- every reduced
    fraction with a bounded denominator, in order -- so the criterion says
    that RH is a statement about how evenly they are spread, with no zeta in
    sight.

    Exact: a `Rational`, summed over `Fraction`s. Through floats the
    deviations, each around `1e-5` by n = 1000 and alternating in sign, would
    lose their leading digits to cancellation.
    """

    is_real = True
    is_nonnegative = True

    @classmethod
    def eval(cls, x: sp.Expr) -> sp.Expr | None:
        if not x.is_Integer:
            return None
        order = int(x)
        if order < 1:
            return sp.Integer(0)
        if order > _FAREY_LIMIT:
            return None
        fractions = _farey(order)
        count = len(fractions)
        total = sum(
            abs(value - Fraction(index, count))
            for index, value in enumerate(fractions, start=1)
        )
        return sp.Rational(total.numerator, total.denominator)


APPLIED_FUNCTIONS: dict[str, object] = {
    # LaTeX spelling
    "zeta": sp.zeta,
    "Gamma": sp.gamma,
    "beta": sp.beta,
    "arg": sp.arg,
    "li": sp.li,
    "Li": sp.Li,
    "pi": sp.primepi,
    "sigma": sp.divisor_sigma,
    "mu": sp.mobius,
    # Euler's totient. Without this `\phi(n)` parsed to `Function("phi")(n)`,
    # which evaluates to nothing -- a formula written with it could not be
    # caught being wrong. The corpus uses `\phi` for the totient and for
    # nothing else; where a source means something else by it, the corpus
    # spells that out, as it does for `\EulerGamma`.
    "phi": sp.totient,
    "H": sp.harmonic,
    "Lambda": VonMangoldt,
    "Lambda_": VonMangoldt,
    "xi": RiemannXi,
    "rho": NthZetaZero,
    "N": ZeroCount,
    "O": BigO,
    "psi": ChebyshevPsi,
    "M": Mertens,
    "RedhefferDet": RedhefferDet,
    "FareyCount": FareyCount,
    "FareyDeviation": FareyDeviation,
    # the spelling each prints as
    "gamma": sp.gamma,
    "primepi": sp.primepi,
    "divisor_sigma": sp.divisor_sigma,
    "mobius": sp.mobius,
    "totient": sp.totient,
    "harmonic": sp.harmonic,
    "VonMangoldt": VonMangoldt,
    "ChebyshevPsi": ChebyshevPsi,
    "Mertens": Mertens,
    "NthPrime": NthPrime,
    "NthZetaZero": NthZetaZero,
    "RiemannXi": RiemannXi,
    "ZeroCount": ZeroCount,
    "BigO": BigO,
}

#: Constants a bare name denotes, as opposed to a variable of that name.
#:
#: `\gamma` is NOT here, and that is the point. It is Euler's constant in
#: Robin's criterion and the ordinate of a zero in `\rho = 1/2 + i\gamma`,
#: both standing alone, so nothing in the syntax separates them -- reading it
#: as the constant turned a zero on the critical line into a fixed number.
#: Sources disambiguate by context this parser cannot see, so the corpus
#: writes `\EulerGamma` where it means the constant.
STANDALONE_CONSTANTS: dict[str, object] = {
    "EulerGamma": sp.EulerGamma,
    # `1/2 + it` is a point on the critical line. Read as a free symbol, `i`
    # made it a product of two unknowns -- and every formula about the
    # critical line was written that way. A binder that BINDS `i` still wins;
    # see `_bound_indices`.
    "i": sp.I,
    "I": sp.I,
}

#: The classes defined here, for reading an srepr back.
#:
#: Only these. `sympify` already knows SymPy's own names, and merging the rest
#: over a parse would undo the applied/standalone decision: `\gamma` standing
#: alone is Euler's constant, and a blanket mapping of `gamma` to the Gamma
#: FUNCTION silently replaced it.
SREPR_NAMESPACE: dict[str, object] = {
    "NthPrime": NthPrime,
    "VonMangoldt": VonMangoldt,
    "Mertens": Mertens,
    "ChebyshevPsi": ChebyshevPsi,
    "NthZetaZero": NthZetaZero,
    "RiemannXi": RiemannXi,
    "ZeroCount": ZeroCount,
    "BigO": BigO,
    "RedhefferDet": RedhefferDet,
    "FareyCount": FareyCount,
    "FareyDeviation": FareyDeviation,
}

#: Every function this engine defines, which is what the two tables above have
#: to agree about.
#:
#: They are separate registries for good reasons -- one resolves a LaTeX name
#: and the other an `srepr` name, and the LaTeX table also carries SymPy's own
#: functions, which `sympify` already knows. But a class in one and not the
#: other reads correctly and then loses its identity on the round trip:
#: `RedhefferDet` was added to the applied table alone and came back from its
#: own printed form as `Function('RedhefferDet')`, a stub that evaluates to
#: nothing. The guard caught it, but only because a formula happened to use it;
#: `test_every_engine_function_resolves_under_both_policies` catches it when
#: the function is added.
ENGINE_FUNCTIONS: tuple[type[sp.Function], ...] = (
    NthPrime,
    VonMangoldt,
    Mertens,
    ChebyshevPsi,
    NthZetaZero,
    RiemannXi,
    ZeroCount,
    BigO,
    RedhefferDet,
    FareyCount,
    FareyDeviation,
)
