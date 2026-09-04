"""The positive control, and the guards that keep it from being read as more.

`experiments/synthetic_adversary` puts zeros off the line to test for false
positives. This is its complement: an object whose critical-line statement is a
theorem, so a checker that cannot confirm it is broken. The tests come in two
halves -- that the mathematics is right, and that a `verified=True` about a
critical line in a repository about the Riemann hypothesis cannot travel
without its qualifier.
"""

from __future__ import annotations

import pytest
import sympy as sp
from pydantic import ValidationError

from rh_research_engine.contracts.epistemic import RIGOROUS, Confidence
from rh_research_engine.symbolic.finite_field_zeta import (
    RIEMANN_HYPOTHESIS_IS_UNAFFECTED,
    CurveZeta,
    SingularCurve,
    curve_zeta,
    frobenius_trace,
    legendre_symbol,
    point_count,
    roots_are_on_the_circle,
    satisfies_the_analogue,
)

#: Non-singular curves with point counts confirmed by brute-force enumeration
#: of every `(x, y)` before they were written here.
KNOWN = [
    (1, 1, 5, 9, -3),
    (0, 1, 7, 12, -4),
    (1, 6, 13, 13, 1),
    (3, 5, 17, 23, -5),
    (0, 2, 19, 13, 7),
    (4, 3, 41, 42, 0),
]


# --- the mathematics ------------------------------------------------------


@pytest.mark.parametrize(("a", "b", "prime", "points", "trace"), KNOWN)
def test_point_counts_and_traces_are_what_they_should_be(a, b, prime, points, trace):
    """Held to literals, not to whatever comes back.

    A point count off by one -- the point at infinity forgotten, or `y^2 = 0`
    counted twice -- shifts every trace by one and leaves Hasse's bound still
    comfortably satisfied. It would look exactly like a working control.
    """
    assert point_count(a, b, prime) == points
    assert frobenius_trace(a, b, prime) == trace


def test_the_point_count_agrees_with_brute_force():
    """The formula against an enumeration that shares nothing with it.

    `p + 1 + sum (f(x) | p)` is a Legendre-symbol identity; counting `(x, y)`
    pairs is not. If the convention for `(0 | p)` were wrong -- and zero is
    exactly where the two natural conventions differ -- this is what notices.
    """
    for a, b, prime in ((1, 1, 11), (2, 3, 13), (5, 2, 17), (1, 4, 23)):
        by_symbol = point_count(a, b, prime)
        by_hand = 1 + sum(
            1
            for x in range(prime)
            for y in range(prime)
            if (y * y - (x**3 + a * x + b)) % prime == 0
        )
        assert by_symbol == by_hand, f"disagree on y^2 = x^3 + {a}x + {b} mod {prime}"


def test_the_legendre_symbol_is_zero_on_multiples():
    """`(0 | p) = 0`, which is what makes `y^2 = 0` contribute one point."""
    assert legendre_symbol(0, 7) == 0
    assert legendre_symbol(14, 7) == 0
    assert legendre_symbol(1, 7) == 1
    assert legendre_symbol(3, 7) == -1  # 3 is a non-residue mod 7


@pytest.mark.parametrize(("a", "b", "prime", "points", "trace"), KNOWN)
def test_every_reciprocal_root_has_modulus_exactly_root_p(a, b, prime, points, trace):
    """Weil's theorem, checked as an EXACT equality and not a proximity.

    `Abs(alpha)**2` simplifies to the integer `p`. Comparing a float modulus
    against `sqrt(p)` within a tolerance would be a different and much weaker
    statement -- and the one this engine keeps having to distinguish.
    """
    result = curve_zeta(a, b, prime)
    assert result.trace == trace
    assert result.on_the_critical_line
    for root in result.reciprocal_roots:
        assert sp.simplify(sp.Abs(root) ** 2 - prime) == 0


def test_the_functional_equation_holds_rather_than_being_asserted():
    """`Z(1/(pT)) = Z(T)` for genus 1, simplified symbolically."""
    for a, b, prime, _, _ in KNOWN:
        assert curve_zeta(a, b, prime).functional_equation_holds


def test_the_circle_test_rejects_roots_that_are_off_it():
    """A control that cannot fail on a bad input is not a control.

    Both fabricated polynomials have the right shape and integer coefficients;
    only their roots are wrong. `1 - 6T + 5T^2` factors as `(1-T)(1-5T)`, so
    its reciprocal roots are 1 and 5 -- product `p`, as a Weil polynomial's
    must be, and neither of modulus `sqrt(5)`.
    """
    variable = sp.Symbol("T")
    genuine = sp.sympify(curve_zeta(1, 1, 5).numerator)
    assert roots_are_on_the_circle(genuine, 5)

    assert not roots_are_on_the_circle(1 - 6 * variable + 5 * variable**2, 5)
    assert not roots_are_on_the_circle(1 - 5 * variable + 5 * variable**2, 5)


def test_the_integer_shortcut_agrees_with_the_root_computation():
    """`a_p^2 <= 4p` and "every root has modulus sqrt(p)" are one statement.

    The sweep uses the first because it is microseconds against milliseconds.
    A fast path never held against the slow one is a fast path nobody has
    checked -- and here the slow one is the definition.
    """
    variable = sp.Symbol("T")
    for prime in (5, 7, 11, 13, 17):
        for trace in range(-2 * prime, 2 * prime + 1):
            polynomial = 1 - trace * variable + prime * variable**2
            assert satisfies_the_analogue(trace, prime) == roots_are_on_the_circle(
                polynomial, prime
            ), f"disagree at p = {prime}, a_p = {trace}"


def test_a_singular_equation_is_refused_rather_than_failed():
    """No genus, no Weil theorem, and no verdict about a critical line.

    Reporting `y^2 = x^3 + 2x + 3` over `F_11` as "the Riemann hypothesis does
    not hold here" would be false in the most misleading available direction:
    it is not a curve.
    """
    with pytest.raises(SingularCurve) as caught:
        curve_zeta(2, 3, 11)
    assert "repeated root" in str(caught.value)

    with pytest.raises(ValueError):
        point_count(1, 1, 3)  # the Weierstrass form needs p > 3


# --- what the record refuses ----------------------------------------------


def _record(**overrides) -> dict:
    payload = {
        "a": 1,
        "b": 1,
        "prime": 5,
        "points": 9,
        "trace": -3,
        "numerator": "x",
        "zeta": "x",
        "on_the_critical_line": True,
        "functional_equation_holds": True,
    }
    payload.update(overrides)
    return payload


def test_the_caveat_cannot_be_dropped_or_softened():
    """The sentence travels with the record or the record is not built.

    A `verified=True` about a critical line, in this repository, is precisely
    the record that gets quoted without its qualifiers. Editable prose would
    be edited.
    """
    for attempt in ("", "verified", "strong evidence for RH"):
        with pytest.raises(ValidationError) as caught:
            CurveZeta(**_record(caveat=attempt))
        assert "not evidence about zeta" in str(caught.value)

    assert CurveZeta(**_record()).caveat == RIEMANN_HYPOTHESIS_IS_UNAFFECTED


def test_the_record_is_certified_and_not_something_stronger():
    """Exact algebra from stated definitions is what `certified` means.

    Not `proved` or `known`: those are about the mathematics, and this record
    is the result of running a computation. Not `rigorous_numerical` either --
    nothing here is an enclosure, because nothing here is approximate.
    """
    record = CurveZeta(**_record())
    assert record.confidence is Confidence.CERTIFIED
    assert Confidence.CERTIFIED in RIGOROUS, (
        "exact integer arithmetic over a finite computation is rigorous, and "
        "this test exists to notice if that ever stops being true"
    )
