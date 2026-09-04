"""Zeta functions where the Riemann hypothesis is a theorem.

WHY THIS EXISTS. Every criterion in this corpus is about one object, and about
that object the question is open. There is no example anywhere in the engine of
a zeta function whose Riemann-hypothesis-analogue is TRUE and PROVED -- so
nothing here has ever been pointed at a case where the right answer is known.

`experiments/synthetic_adversary` supplies the other half: zeros put off the
line on purpose, to see whether a criterion produces a false positive. Its
complement was missing. A curve over a finite field is the complement: its
zeta function is a rational function with integer coefficients, its zeros are
algebraic numbers, and Weil proved every one of them lies on the critical line
`Re s = 1/2`.

WHAT IS EXACT HERE, WHICH IS ALL OF IT. For a smooth projective curve `C` over
`F_q`,

    Z_C(T) = P(T) / ((1 - T)(1 - qT))

with `P` of degree `2g` over the integers. Writing `P(T) = prod (1 - a_i T)`,
Weil's theorem is `|a_i| = sqrt(q)` for every `i`, and substituting `T =
q^{-s}` turns that into `Re s = 1/2`. Nothing in the computation is
approximate: point counts are integers, `P` has integer coefficients, and for
genus 1 the whole of the Riemann hypothesis for `C` is the integer inequality

    a_q^2 <= 4q

which is Hasse's bound. That is the first zeta-like object in this engine
whose critical-line statement can be settled by exact arithmetic rather than
sampled -- `verify-line` is floating point and `certify-line` needs interval
arithmetic, and this needs neither.

WHAT IT IS NOT, AND THIS IS THE POINT OF THE FILE. It is not evidence for the
Riemann hypothesis. Not weak evidence, not suggestive evidence: none. The
function field case was proved by methods that have no counterpart over `Q` --
`C x C` is a surface with an intersection theory, and there is no such surface
here -- and the analogy has stood, complete on one side and open on the other,
since 1948. `CurveZeta` therefore refuses to be filed against `zeta`, and
`RIEMANN_HYPOTHESIS_IS_UNAFFECTED` says so where a reader will meet it.

What the analogy is good for is instrumentation: a checker that cannot confirm
the critical line where it provably holds is a broken checker, and until now
nothing could tell.
"""

from __future__ import annotations

import sympy as sp
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..contracts.epistemic import Confidence

#: Said once, here, so that no reader meets a `verified=True` record without
#: it. Repeated into every record's `caveat` rather than left in a docstring.
RIEMANN_HYPOTHESIS_IS_UNAFFECTED = (
    "the Riemann hypothesis for zeta is untouched by this: the function field "
    "case was proved by intersection theory on C x C, which has no counterpart "
    "over Q, and the analogy has stood complete on one side and open on the "
    "other since 1948"
)


class SingularCurve(ValueError):
    """`4a^3 + 27b^2 = 0`: the cubic has a repeated root, so this is not a curve.

    Raised rather than returned as a failed verification. A singular Weierstrass
    equation has no genus and no Weil theorem, so reporting it as "the Riemann
    hypothesis does not hold here" would be false in the most misleading
    available direction.
    """


def legendre_symbol(value: int, prime: int) -> int:
    """`(value | prime)` by Euler's criterion, in integers.

    `0` when `prime` divides `value`, which is the convention that makes the
    point count come out right: `y^2 = 0` has the one solution `y = 0`, not
    two and not none.
    """
    residue = value % prime
    if residue == 0:
        return 0
    return 1 if pow(residue, (prime - 1) // 2, prime) == 1 else -1


def point_count(a: int, b: int, prime: int) -> int:
    """`#E(F_p)` for `y^2 = x^3 + ax + b`, counting the point at infinity.

    `p + 1 + sum_x (x^3+ax+b | p)`: each `x` contributes two points when the
    right-hand side is a non-zero square, none when it is a non-residue, and
    one when it is zero -- which is `1 + legendre` in each case.
    """
    if prime <= 3:
        raise ValueError(
            f"p = {prime}: this Weierstrass form needs p > 3, since completing "
            "the square and the cube divides by 2 and 3"
        )
    if (4 * a**3 + 27 * b**2) % prime == 0:
        raise SingularCurve(
            f"4a^3 + 27b^2 = 0 mod {prime} for a = {a}, b = {b}: the cubic has "
            "a repeated root, so this is a singular Weierstrass equation and "
            "not an elliptic curve"
        )
    return prime + 1 + sum(
        legendre_symbol(x**3 + a * x + b, prime) for x in range(prime)
    )


def frobenius_trace(a: int, b: int, prime: int) -> int:
    """`a_p = p + 1 - #E(F_p)`, in integers and nothing else."""
    return prime + 1 - point_count(a, b, prime)


def satisfies_the_analogue(trace: int, prime: int) -> bool:
    """`a_p^2 <= 4p`: the whole Riemann hypothesis for a genus-1 curve.

    Split out from the full record because sweeping every curve over every
    small field is thousands of them, and building the symbolic zeta function
    for each costs about ten milliseconds against this one's microseconds.
    The saving is real and so is the risk: a fast path that is never held
    against the slow one is a fast path nobody has checked. `weil-control`
    verifies it against `curve_zeta` at sampled curves, the same discipline
    `corpus_sweep.verify_shortcuts` applies to its own shortcuts -- which once
    caught the reference rather than the column.
    """
    return trace * trace <= 4 * prime


class CurveZeta(BaseModel):
    """The zeta function of an elliptic curve over `F_p`, and its critical line.

    Every field is exact. `trace` and the coefficients of `P` are integers;
    `on_the_critical_line` is decided by an integer inequality, not by
    computing a modulus in floating point and comparing it to a tolerance.
    """

    model_config = ConfigDict(extra="forbid")

    #: `y^2 = x^3 + ax + b` over `F_p`.
    a: int
    b: int
    prime: int
    #: `#E(F_p)`, the point at infinity included.
    points: int
    #: `a_p = p + 1 - #E(F_p)`, the trace of Frobenius.
    trace: int
    #: `P(T) = 1 - a_p T + p T^2`, as a printable expression.
    numerator: str
    #: `Z(T) = P(T) / ((1 - T)(1 - pT))`.
    zeta: str
    #: Whether every reciprocal root of `P` has modulus exactly `sqrt(p)`.
    #: For genus 1 this is `a_p^2 <= 4p`, settled in integers.
    on_the_critical_line: bool
    #: `Z(1/(pT)) = Z(T)`, checked symbolically rather than asserted.
    functional_equation_holds: bool
    #: Exact algebra from stated definitions, which is what this value means.
    #: Not `rigorous_numerical`: nothing here is an enclosure, because nothing
    #: here is approximate.
    confidence: Confidence = Confidence.CERTIFIED
    caveat: str = RIEMANN_HYPOTHESIS_IS_UNAFFECTED
    evidence: dict[str, object] = Field(default_factory=dict)

    @field_validator("caveat")
    @classmethod
    def _keep_the_caveat(cls, value: str) -> str:
        """The caveat travels with the record or the record is not built.

        A `verified=True` about a critical line, in a repository about the
        Riemann hypothesis, is precisely the record that gets quoted without
        its qualifiers. It cannot be blanked, shortened away, or replaced with
        something friendlier.
        """
        if value != RIEMANN_HYPOTHESIS_IS_UNAFFECTED:
            raise ValueError(
                "the caveat is fixed: a curve satisfying its own Riemann "
                "hypothesis is not evidence about zeta, and a record that can "
                "drop that sentence will eventually be read as though it were"
            )
        return value

    @property
    def reciprocal_roots(self) -> list[sp.Expr]:
        """The `alpha_i` with `P(T) = prod (1 - alpha_i T)`, exactly.

        Roots of `A^2 - a_p A + p`, so they are algebraic integers written in
        radicals -- and `Abs(alpha)^2` simplifies to exactly `p`, an integer,
        rather than to a float near it.
        """
        symbol = sp.Symbol("A")
        return sp.solve(sp.Eq(symbol**2 - self.trace * symbol + self.prime, 0), symbol)


def curve_zeta(a: int, b: int, prime: int) -> CurveZeta:
    """Build the zeta function of `y^2 = x^3 + ax + b` over `F_p`, and check it.

    The critical-line test is `a_p^2 <= 4p` and NOT a comparison of
    `abs(alpha)` against `sqrt(p)`. They are the same statement, and only one
    of them is decidable: the second asks whether two irrational numbers are
    equal, which in floating point is a question about a tolerance. This
    engine has spent enough of its life on the difference between a bound that
    holds and a bound that holds to 1e-12.
    """
    points = point_count(a, b, prime)
    trace = prime + 1 - points

    variable = sp.Symbol("T")
    numerator = 1 - trace * variable + prime * variable**2
    zeta = numerator / ((1 - variable) * (1 - prime * variable))

    # Genus 1: |alpha| = sqrt(p) for both roots exactly when the roots are a
    # conjugate pair or a repeated real one, which is the discriminant test.
    on_line = trace * trace <= 4 * prime

    swapped = zeta.subs(variable, 1 / (prime * variable))
    functional = sp.simplify(swapped - zeta) == 0

    return CurveZeta(
        a=a,
        b=b,
        prime=prime,
        points=points,
        trace=trace,
        numerator=sp.srepr(numerator),
        zeta=sp.srepr(sp.together(zeta)),
        on_the_critical_line=on_line,
        functional_equation_holds=functional,
        evidence={
            "hasse_slack": 4 * prime - trace * trace,
            "genus": 1,
            "equation": f"y^2 = x^3 + {a}x + {b} over F_{prime}",
        },
    )


def roots_are_on_the_circle(numerator: sp.Expr, prime: int) -> bool:
    """Does every reciprocal root of `P` have modulus exactly `sqrt(p)`?

    The general test, used to check that the integer shortcut above agrees
    with it -- and to be able to REFUSE a polynomial whose roots are off the
    circle, which is what makes the shortcut falsifiable. A control that
    cannot fail on a bad input is not a control.

    Exact throughout: `Abs(alpha)**2` is simplified symbolically and compared
    against the integer `p`, never against a float.
    """
    variable = sp.Symbol("T")
    poly = sp.Poly(sp.expand(numerator), variable)
    degree = poly.degree()
    if degree == 0:
        return True
    # P(T) = prod(1 - alpha_i T), so the alpha_i are the roots of the reversed
    # polynomial: reverse the coefficient list and solve.
    reversed_poly = sp.Poly(list(reversed(poly.all_coeffs())), variable)
    for root in sp.roots(reversed_poly, variable, multiple=True):
        if sp.simplify(sp.Abs(root) ** 2 - prime) != 0:
            return False
    return True
