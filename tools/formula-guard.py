#!/usr/bin/env python3
"""Refuse to let a formula be left broken, or quietly patched around.

WHY THIS EXISTS. The formulas are the product. Every other artifact in this
repository -- fingerprints, citations, proof-queue verdicts, Lean exports --
describes a formula, and describes nothing at all if the formula was read
wrong. A broken formula is not a broken feature; it is a research record that
says something its source did not.

The failures this gate exists to catch all shipped at some point, all passed
the test suite, and none of them looked like a failure:

  * `O(...)` resolved to SymPy's `Order`, a germ at ZERO that absorbs the
    terms it dominates. It deleted `li(x)` from von Koch's theorem and
    reported a clean parse.
  * `M(x)` came back as `Mul(Symbol('M'), Symbol('x'))` -- M stopped being a
    function -- because the fingerprint path re-read the extractor's own
    output under different name resolution.
  * `\\sigma(n)` was stored as an undefined `Function('sigma')`, so Robin's
    inequality could be indexed, cited and hashed, and never evaluated. A
    formula that cannot be evaluated cannot be caught being wrong.
  * the functional equation sat in the index twice, with and without its
    `sin(pi*s/2)` factor, because a correction files a NEW record rather than
    replacing the one it corrects.
  * a whole table of notation was "out of reach" -- refused rather than read.
    A refusal is a staging post, not a fix.

WHAT IT CHECKS, and what it deliberately does not. It checks that every
formula in the corpus can be read, indexed, and read back identically by every
consumer, and that nothing is stored as a placeholder for a function that
exists. It does NOT check that any formula is TRUE: that is what the proof
queue is for, and its refusals are honest ones -- they are claims about what
can be discharged, not about what could be parsed.

    python tools/formula-guard.py            # check
    python tools/formula-guard.py --quiet    # only failures, for hooks
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "docs" / "research" / "rh-ingestible-algebra.md"
SYMBOLIC = REPO / "src" / "rh_research_engine" / "symbolic"
#: Floor on the module enumeration in "One name-resolution policy". There are 34
#: today; the floor is well below that so a legitimate deletion does not trip it,
#: and well above zero so a glob that matches nothing cannot report PASS.
MINIMUM_SYMBOLIC_MODULES = 10

sys.path.insert(0, str(REPO / "src"))

import sympy as sp  # noqa: E402  (needs the path above)


@dataclass
class Check:
    title: str
    failures: list[str] = field(default_factory=list)
    #: Formulas this check actually evaluated, as opposed to skipped.
    #:
    #: Reported, because a check that quietly stops covering something looks
    #: exactly like a check that passes. Two of these were skipping every
    #: formula at one point and saying "ok".
    covered: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def _display(path: Path) -> str:
    """A path to show in a failure.

    Never raises. A check that crashes while REPORTING a failure reports
    nothing, which reads exactly like a pass.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _corpus_equations():
    from rh_research_engine.symbolic import extract_equations

    return extract_equations(CORPUS.read_text(encoding="utf-8"))


def every_formula_parses() -> Check:
    """A formula that does not parse is a formula this engine cannot hold."""
    check = Check("Every corpus formula parses")
    for item in _corpus_equations():
        if item.parse_error is not None:
            check.failures.append(f"{item.source}\n    {item.parse_error}")
    return check


def every_formula_indexes() -> Check:
    """Parsing is not enough: it has to survive being fingerprinted.

    `li` parsed fine and then raised inside canonicalization, which is how the
    `Order` absorption bug surfaced at all.
    """
    from rh_research_engine.symbolic import fingerprint

    check = Check("Every corpus formula fingerprints")
    for item in _corpus_equations():
        if item.parse_error is not None:
            continue
        try:
            fingerprint(item.normalized)
        except Exception as exc:
            check.failures.append(
                f"{item.source}\n    {type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
            )
    return check


def both_readings_agree() -> Check:
    """The printed form and the srepr must describe the same object.

    They are two ways of reading one extraction, and when they disagree the
    stored hash depends on which path the caller happened to take. Every
    name-resolution defect found so far shows up here first.
    """
    import sympy as sp

    from rh_research_engine.symbolic import fingerprint
    from rh_research_engine.symbolic.functions import SREPR_NAMESPACE

    check = Check("Printed form and srepr agree")
    for item in _corpus_equations():
        if item.parse_error is not None or not item.sympy_srepr:
            continue
        try:
            from_text = fingerprint(item.normalized).canonical
            from_srepr = fingerprint(
                sp.sympify(item.sympy_srepr, locals=dict(SREPR_NAMESPACE))
            ).canonical
        except Exception as exc:
            check.failures.append(f"{item.source}\n    {type(exc).__name__}: {exc}")
            continue
        if from_text != from_srepr:
            check.failures.append(
                f"{item.source}\n    text : {from_text[:110]}\n    srepr: {from_srepr[:110]}"
            )
    return check


#: Names this corpus writes that denote a REAL function, never a placeholder.
#:
#: Stated explicitly rather than discovered, because the mapping is the thing
#: being protected. `\sigma` is SymPy's `divisor_sigma` -- the names do not
#: match, so a check that looked for a SymPy attribute called `sigma` would
#: find nothing and pass a corpus where Robin's inequality had silently gone
#: back to being an opaque shape.
MUST_DENOTE_A_FUNCTION = frozenset(
    {
        "Gamma", "gamma", "zeta", "beta", "li", "Li", "mu", "mobius",
        "pi", "primepi", "sigma", "divisor_sigma", "H", "harmonic",
        "psi", "ChebyshevPsi", "Lambda", "Lambda_", "VonMangoldt",
        "M", "Mertens", "xi", "RiemannXi", "N", "ZeroCount",
        "NthPrime",
    }
)


def no_stub_for_a_real_function() -> Check:
    """An undefined `Function('sigma')` where `divisor_sigma` exists.

    The shape without the meaning. It indexes, it cites, it hashes -- and it
    can never be evaluated, so the formula can never be shown wrong.
    """
    from rh_research_engine.symbolic import fingerprint

    check = Check("No stub stands in for a real function")
    for item in _corpus_equations():
        if item.parse_error is not None:
            continue
        try:
            canonical = fingerprint(item.normalized).canonical
        except Exception:
            continue  # already reported by the fingerprint check
        for name in sorted(set(re.findall(r"Function\('([^']+)'\)", canonical))):
            if name in MUST_DENOTE_A_FUNCTION:
                check.failures.append(
                    f"{item.source}\n    `{name}` is stored as an undefined function. "
                    "It denotes a real one -- see symbolic/functions.py"
                )
    return check


#: Every free symbol the corpus is allowed to contain, and what it names.
#:
#: An undeclared symbol is the signature of a name that failed to resolve. It
#: is how `i` sat in the index as a free variable -- so `\zeta(1/2 + it)` was
#: a function of an unknown called "it" rather than a point on the critical
#: line -- and how `4xy` became an equation about a variable named "xy".
#: Neither looked wrong; both printed exactly as written.
#:
#: Adding a name here is a decision that it really is a variable. That is the
#: point: the list forces the question rather than letting a failed resolution
#: pass as a new unknown.
DECLARED_SYMBOLS = {
    "s": "the complex variable of zeta",
    "t": "the ordinate of a point on the critical line",
    "x": "a real variable, usually a counting bound",
    "y": "a second real variable",
    "n": "a positive integer",
    "u": "a normalised gap between zeros",
    "T": "a height in the critical strip",
    "Theta": "the supremum of the real parts of the zeros",
    "theta": "the exponent in the Theta/theta transfer",
    "epsilon": "an arbitrarily small positive quantity",
    "Lambda_": "the de Bruijn-Newman constant",
    "lambda__n": "the n-th Li coefficient",
    "c_k": "the k-th Baez-Duarte coefficient",
    "k": "a non-negative integer index",
}

#: Deterministic sample points. Not random: this repository hashes results,
#: and a gate that fails on a different point each run cannot be replayed.
_SAMPLE_POINTS = (
    sp.Rational(5, 2),
    sp.Rational(-13, 10),
    sp.Rational(7, 10) + sp.Rational(6, 5) * sp.I,
)

#: Heads whose value is not reachable by substitution alone.
_OPAQUE_HEADS = frozenset(
    {"Sum", "Product", "Integral", "Limit", "Derivative", "Subs", "BigO"}
)


def every_symbol_is_declared() -> Check:
    """A free symbol nobody declared is a name that failed to resolve."""
    import sympy as sp

    from rh_research_engine.symbolic.functions import SREPR_NAMESPACE

    check = Check("Every free symbol is declared")
    for item in _corpus_equations():
        if item.parse_error is not None or not item.sympy_srepr:
            continue
        expression = sp.sympify(item.sympy_srepr, locals=dict(SREPR_NAMESPACE))
        for symbol in sorted(expression.free_symbols, key=str):
            if symbol.name not in DECLARED_SYMBOLS:
                check.failures.append(
                    f"{item.source}\n    `{symbol.name}` is a free symbol and is not "
                    "declared. Either it is a variable -- say so in "
                    "DECLARED_SYMBOLS -- or a name failed to resolve"
                )
    return check


def _has_infinite_binder(expression) -> bool:
    import sympy as sp

    return any(
        isinstance(node, (sp.Sum, sp.Product))
        and bool(node.limits)
        and node.limits[0][2] == sp.oo
        for node in sp.preorder_traversal(expression)
    )


def _has_free_function(expression) -> bool:
    """An undefined function, or a derivative of one: nothing to evaluate."""
    import sympy as sp

    return any(
        isinstance(node, (sp.core.function.AppliedUndef, sp.Derivative, sp.Subs))
        or type(node).__name__ == "BigO"
        for node in sp.preorder_traversal(expression)
    )


def an_identity_holds_numerically() -> Check:
    """An identity has to be true, and that is checkable.

    Only equations whose two sides involve the SAME variables: those are
    claims about every value. `Theta = 1/2 + theta/2` introduces a symbol on
    one side and is a definition, not an identity, so substituting into it
    would manufacture a failure.

    This is the check that catches a formula transcribed wrong. The functional
    equation was missing its `sin(pi*s/2)` factor for weeks -- it parsed, it
    indexed, it fingerprinted, and at s = -1 it gives +1/12 where zeta(-1) is
    -1/12.
    """
    import sympy as sp

    from rh_research_engine.symbolic.functions import SREPR_NAMESPACE

    check = Check("Every identity holds numerically")
    for item in _corpus_equations():
        if item.parse_error is not None or not item.sympy_srepr:
            continue
        expression = sp.sympify(item.sympy_srepr, locals=dict(SREPR_NAMESPACE))
        condition, claim = _statement(expression)
        if not isinstance(claim, sp.Equality) or _has_infinite_binder(claim):
            continue  # an infinite series is the other check's job
        expression = claim
        variables = expression.lhs.free_symbols
        if not variables or variables != expression.rhs.free_symbols:
            continue  # a definition or a constraint, not an identity
        agreed = 0
        for point in _SAMPLE_POINTS + _TRIAL_POINTS:
            assignment = {symbol: point for symbol in sorted(variables, key=str)}
            if not _holds_at(condition, assignment):
                continue
            left = _at(expression.lhs, assignment)
            right = _at(expression.rhs, assignment)
            if left is None or right is None:
                continue
            if not agreed:
                check.covered.append(item.source)
            agreed += 1
            scale = max(abs(left), abs(right), 1.0)
            if abs(left - right) > 1e-12 * scale:
                check.failures.append(
                    f"{item.source}\n    at {assignment}: "
                    f"left = {left:.10g}, right = {right:.10g}"
                )
        if not agreed and not _has_free_function(expression):
            # Nothing here is opaque, so it SHOULD have evaluated. Silence
            # would mean the formula is stored with nothing testing it.
            check.failures.append(
                f"{item.source}\n    no sample point evaluated on both sides, so "
                "this identity is stored unchecked"
            )
    return check


def _finite(value: complex) -> bool:
    import math

    return math.isfinite(value.real) and math.isfinite(value.imag)


#: Points to try for a series or a relation, in order. The first at which both
#: sides evaluate finitely is used.
#:
#: Fixed, and ordered: `s = 4` is deep in the half-plane where the Dirichlet
#: series converge quickly, and 20 reaches the counting functions.
_TRIAL_POINTS = (20, 4, 3, 2)

#: Truncation levels for an infinite binder. Two, because the check is that
#: the error SHRINKS -- a series converging to the wrong value would sit at a
#: fixed distance from it, which a single truncation cannot tell from slow
#: convergence.
_TRUNCATIONS = (250, 1000)

#: And for a sum over the ZEROS, which costs a root-find per term.
#:
#: Locating the first 200 zeros takes about 19 seconds and the cost grows
#: superlinearly, so the ordinary levels would put this gate into the minutes
#: -- and it runs in pre-commit. Twenty and sixty cost about two seconds
#: together.
#:
#: The tolerance is correspondingly loose, and that is honest about what the
#: check is for: the explicit formula converges slowly and oscillates, so
#: sixty terms cannot certify precision. It CAN catch a formula transcribed
#: wrong, which would miss by an O(1) fraction rather than by a few percent.
_ZERO_TRUNCATIONS = (20, 60)
_ZERO_TOLERANCE = 0.2


def _sums_over_zeros(expression) -> bool:
    import sympy as sp

    return any(
        type(node).__name__ == "NthZetaZero"
        for node in sp.preorder_traversal(expression)
    )


def _substitute_bounds(expression, bound: int):
    """Replace every infinite upper limit with a finite one.

    Via `replace` with matchers rather than `subs`: handing `subs` a `Sum` as
    the pattern rebuilds the node it is walking and recurses until the stack
    gives out.
    """
    import sympy as sp

    def unbounded(node) -> bool:
        return (
            isinstance(node, (sp.Sum, sp.Product))
            and bool(node.limits)
            and node.limits[0][2] == sp.oo
        )

    def truncate(node):
        variable, lower, _ = node.limits[0]
        return node.func(node.function, (variable, lower, bound))

    try:
        return expression.replace(unbounded, truncate)
    except Exception:
        return expression


def _probe(check: Check, item, work) -> None:
    """Run one formula's numeric probe, converting a crash into a finding.

    SymPy raises from deep inside its own cache on some arguments -- an
    `arg(gamma(...))` at a complex point comes back as an AttributeError from
    the memoiser rather than anything meaningful. A gate that dies there
    reports nothing about the other thirty-four formulas, which reads exactly
    like a pass.
    """
    try:
        work()
    except Exception as exc:  # noqa: BLE001 - a probe, not the thing being probed
        check.failures.append(
            f"{item.source}\n    could not be probed "
            f"({type(exc).__name__}: {str(exc).splitlines()[0][:90]})"
        )


def _statement(expression):
    """Split `Implies(condition, claim)` into `(condition, claim)`.

    A formula valid only on part of the plane is not a formula plus a
    footnote: the condition IS part of it. `Gamma(s) = integral(...)` holds
    for Re s > 0 and diverges to the left of it, so a checker that ignores the
    condition reports the definition of Gamma as false.
    """
    import sympy as sp

    if isinstance(expression, sp.Implies) and len(expression.args) == 2:
        return expression.args[0], expression.args[1]
    return None, expression


def _holds_at(condition, assignment) -> bool:
    """Whether a condition is satisfied at a point. Unknown counts as no."""
    import sympy as sp

    if condition is None:
        return True
    try:
        return condition.subs(assignment) is sp.true
    except Exception:
        return False


#: SymPy name -> the mpmath callable to differentiate through.
#:
#: Only analytic functions belong here. A derivative of `primepi` is not a
#: thing, and asking for one should fail rather than return a plausible finite
#: difference of a step function.
def _mpmath_namespace() -> dict[str, object]:
    import mpmath

    from rh_research_engine.symbolic.functions import RiemannXi  # noqa: F401

    def xi(s):
        shifted = mpmath.mpf(1) if s == 1 else (s - 1) * mpmath.zeta(s)
        return mpmath.pi ** (-s / 2) * mpmath.gamma(s / 2 + 1) * shifted

    return {
        "zeta": mpmath.zeta,
        "gamma": mpmath.gamma,
        "loggamma": mpmath.loggamma,
        "li": mpmath.li,
        "Li": mpmath.li,
        "RiemannXi": xi,
        "log": mpmath.log,
        "exp": mpmath.exp,
        "sqrt": mpmath.sqrt,
        "sin": mpmath.sin,
        "cos": mpmath.cos,
        "Abs": abs,
        "re": mpmath.re,
        "im": mpmath.im,
        "arg": mpmath.arg,
        "pi": mpmath.pi,
    }


#: Working precision for a numeric derivative.
#:
#: Generous, because an n-th derivative is where precision goes to die if you
#: are stingy -- and because it is cheap. mpmath's `diff` is accurate to full
#: working precision, so the answer is limited by this number and nothing else.
_DERIVATIVE_DPS = 40


def _evaluate_derivatives(expression):
    """Replace `Subs(Derivative(f, (x, n)), x, a)` with its numeric value.

    SymPy leaves these unevaluated for functions it has no closed-form
    derivative of -- zeta among them -- so an expression containing one never
    reaches a number, and every check skips the formula. mpmath differentiates
    it directly.
    """
    import mpmath
    import sympy as sp

    def unevaluated(node) -> bool:
        return isinstance(node, sp.Subs) and isinstance(node.expr, sp.Derivative)

    def evaluate(node):
        derivative = node.expr
        variable, order = derivative.variable_count[0]
        if node.variables != (variable,):
            return node  # differentiating one variable, substituting another
        point = node.point[0]
        if not point.is_number:
            return node
        function = sp.lambdify(
            variable, derivative.expr, modules=[_mpmath_namespace(), "mpmath"]
        )
        with mpmath.workdps(_DERIVATIVE_DPS):
            value = mpmath.diff(function, mpmath.mpmathify(complex(point)), int(order))
        return sp.sympify(value)

    return expression.replace(unevaluated, evaluate)


def _at(expression, assignment) -> complex | None:
    """Value at a point, or None. Substitution itself can raise: an integer
    function refuses a complex argument before evalf ever sees it."""
    try:
        return _numeric(expression.subs(assignment))
    except Exception:
        return None


def _numeric(expression) -> complex | None:
    import math

    import sympy as sp

    try:
        # Unevaluated derivatives first. SymPy has no closed form for zeta's
        # derivative, so it leaves `Subs(Derivative(...))` in place and the
        # expression never reaches a number -- every check then silently skips
        # the formula rather than failing it. This is the one choke point that
        # all of them go through.
        if expression.has(sp.Subs) or expression.has(sp.Derivative):
            expression = _evaluate_derivatives(expression)
    except Exception:
        return None

    try:
        # 25 digits requested, then narrowed to a double for comparison. The
        # tolerances here are 1e-12 relative, so a double's ~1e-16 leaves four
        # digits of headroom -- enough for the transcription errors this
        # catches, which miss by orders of magnitude rather than by ULPs. Do
        # not raise a tolerance past 1e-14 without moving this to mpmath.
        value = complex(expression.evalf(25))
    except Exception:
        # Anything at all: this is a probe, and a probe that cannot get a
        # number reports that rather than taking the gate down with it.
        return None
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        return None
    return value


def a_series_converges_to_its_closed_form() -> Check:
    """A truncation has to approach the value the formula claims.

    Equality is the wrong test for an infinite series -- any truncation is
    wrong by something. Convergence is the right one: truncate twice, and the
    error must SHRINK. A series that converges to a different value sits at a
    fixed distance from the claim, and no single truncation can tell that from
    slow convergence.
    """
    import sympy as sp

    from rh_research_engine.symbolic.functions import SREPR_NAMESPACE

    check = Check("Every series converges to its closed form")
    for item in _corpus_equations():
        if item.parse_error is not None or not item.sympy_srepr:
            continue
        expression = sp.sympify(item.sympy_srepr, locals=dict(SREPR_NAMESPACE))
        condition, expression = _statement(expression)
        if not isinstance(expression, sp.Equality):
            continue
        variables = expression.lhs.free_symbols
        if not variables or variables != expression.rhs.free_symbols:
            continue
        if not any(
            isinstance(node, (sp.Sum, sp.Product)) and node.limits
            and node.limits[0][2] == sp.oo
            for node in sp.preorder_traversal(expression)
        ):
            continue

        for point in _TRIAL_POINTS:
            assignment = {symbol: point for symbol in variables}
            if not _holds_at(condition, assignment):
                continue
            over_zeros = _sums_over_zeros(expression)
            levels = _ZERO_TRUNCATIONS if over_zeros else _TRUNCATIONS
            tolerance = _ZERO_TOLERANCE if over_zeros else 1e-2
            errors = []
            for bound in levels:
                left = _numeric(_substitute_bounds(expression.lhs.subs(assignment), bound))
                right = _numeric(_substitute_bounds(expression.rhs.subs(assignment), bound))
                if left is None or right is None:
                    break
                errors.append(abs(left - right) / max(abs(left), abs(right), 1.0))
            if len(errors) != len(levels):
                continue
            check.covered.append(item.source)
            # Monotone decrease is a property of an absolutely convergent
            # series, and the sum over zeros is only CONDITIONALLY convergent
            # -- it oscillates, and demanding a shrinking error reported the
            # explicit formula as broken for moving 0.130 to 0.136. For those
            # the tolerance carries the check alone.
            if not over_zeros and errors[-1] > errors[0]:
                check.failures.append(
                    f"{item.source}\n    at {assignment}: the two sides move APART as "
                    f"the sum grows ({errors[0]:.3g} then {errors[-1]:.3g})"
                )
            elif errors[-1] > tolerance:
                check.failures.append(
                    f"{item.source}\n    at {assignment}: still {errors[-1]:.3g} apart "
                    f"at {levels[-1]} terms"
                )
            break
    return check


def _decide(relation, assignment) -> bool | None:
    """Whether a relation holds at a point, decided NUMERICALLY.

    `bool(relation)` asks SymPy to decide it exactly. For Schoenfeld's psi
    bound that means comparing a sum of twelve hundred logarithms against an
    algebraic expression, symbolically, and it does not come back. Two floats
    settle the same question.
    """
    import sympy as sp

    if relation is sp.true:
        return True
    if relation is sp.false:
        return False
    if not isinstance(relation, sp.Rel):
        return None
    # Each side separately. Substituting into the RELATION makes SymPy try to
    # decide it there and then -- for Schoenfeld that is an exact comparison
    # of twelve hundred logarithms and it does not come back.
    left = _at(relation.lhs, assignment)
    right = _at(relation.rhs, assignment)
    if left is None or right is None:
        return None
    a, b = left.real, right.real
    operator = type(relation).__name__
    if operator in ("StrictLessThan",):
        return a < b
    if operator in ("LessThan",):
        return a <= b
    if operator in ("StrictGreaterThan",):
        return a > b
    if operator in ("GreaterThan",):
        return a >= b
    if operator in ("Equality",):
        return abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1.0)
    if operator in ("Unequality",):
        return abs(a - b) > 1e-9 * max(abs(a), abs(b), 1.0)
    return None


def a_stated_relation_holds() -> Check:
    """An inequality is a claim, and a claim with a threshold can be tested.

    Robin's inequality is FALSE at n = 5040 and the criterion is about n above
    it, so the hypothesis is not decoration -- a criterion indexed without its
    threshold is a different, false statement. Carrying it makes the whole
    thing checkable: below the threshold the implication is vacuous, above it
    the inequality has to hold.
    """
    import sympy as sp

    from rh_research_engine.symbolic.functions import SREPR_NAMESPACE

    check = Check("Every stated relation holds where it is claimed")
    for item in _corpus_equations():
        if item.parse_error is not None or not item.sympy_srepr:
            continue
        try:
            expression = sp.sympify(item.sympy_srepr, locals=dict(SREPR_NAMESPACE))
        except Exception:
            continue
        if not isinstance(expression, sp.Implies):
            continue
        condition, claim = _statement(expression)
        import sympy as _sp

        if isinstance(claim, _sp.Equality):
            # A conditioned IDENTITY, not a criterion. The identity check
            # honours the condition too, and probes at points that suit an
            # identity -- `_RELATION_POINTS` exists to straddle criterion
            # THRESHOLDS, and Gamma(100) is 9.3e155.
            continue
        if _has_infinite_binder(claim):
            # A conditioned SERIES, not a criterion. Adding the half-plane to
            # the Dirichlet series turned each of them into an Implies, which
            # landed them here -- and evaluating an infinite sum numerically
            # does not come back. The series check owns them.
            continue
        variables = expression.free_symbols
        if len(variables) != 1:
            continue
        (variable,) = variables
        tested = 0
        for point in _RELATION_POINTS:
            assignment = {variable: point}
            if not _holds_at(condition, assignment):
                continue  # vacuous here; the claim says nothing
            verdict = _decide(claim, assignment)
            if verdict is None:
                continue
            if not tested:
                check.covered.append(item.source)
            tested += 1
            if not verdict:
                check.failures.append(
                    f"{item.source}\n    fails at {variable} = {point}"
                )
        if not tested:
            check.failures.append(
                f"{item.source}\n    no point evaluated, so this claim is stored "
                "unchecked"
            )
    return check


#: Where a criterion is probed. Spans the thresholds the corpus states, so
#: both the vacuous side and the claimed side of each are exercised.
_RELATION_POINTS = (1, 2, 12, 74, 100, 2657, 5040, 5041, 10080)


#: Points an asymptotic claim is sampled at, spanning two decades.
#:
#: Two decades is the point: a bound with the WRONG EXPONENT makes the ratio
#: grow like a power, so it moves by orders of magnitude across this range,
#: while a correct bound leaves it wandering within a constant factor.
_ASYMPTOTIC_POINTS = (100, 10000, 1000000)

#: A narrower fallback, for residuals whose functions cannot reach that far.
#:
#: Only `M(x)` needs it now: the summatory Mobius function is computed by
#: summing, so 10^6 is a million terms. N(T) used to fall through to a
#: narrower tier still and no longer does.
_ASYMPTOTIC_MEDIUM = (100, 1000, 10000)

#: Narrower still, for residuals that are quadratic in the sample point.
#:
#: `FareyDeviation(n)` needs every reduced fraction of order n, and there are
#: about `3n^2/pi^2` of them -- 109500 already at n = 600, where M(x) at the
#: same point is 600 terms. A tier exists rather than the formula going
#: unchecked: the coverage tally is the count of formulas that CAN be caught
#: being wrong, and a bound nothing samples is a bound nothing can catch.
#:
#: Three points over 100..600 is a weaker test than three over two decades,
#: and it is the same KIND of test: a wrong exponent still shows as a ratio
#: climbing at every step. It is not, and the check above does not claim to
#: be, a proof of anything asymptotic.
_ASYMPTOTIC_NARROW = (100, 300, 600)

#: The instance of an "for every epsilon > 0" claim that gets tested.
#:
#: A concrete epsilon makes one true instance of the family testable. Small
#: enough to be a real constraint: at epsilon = 1/2, `M(x) = O(x^{1/2+eps})`
#: reads `M(x) = O(x)`, which is true of anything and tests nothing.
_EPSILON_INSTANCE = sp.Rational(1, 10)

#: The largest power the ratio may be seen to grow by.
#:
#: Judged as a SLOPE in log-log, not as an endpoint factor, because the
#: formulas are still sampled over different ranges -- M(x) over two decades,
#: since the summatory Mobius function is computed by summing, and the rest
#: over four. An endpoint factor means different things over different ranges
#: and let `pi(x) = li(x) + O(log x)` through: over two decades that bound is
#: only about three times worse than the true one, which is inside any
#: threshold loose enough to allow the honest wandering of O().
#:
#: N(T) used to be the narrowest of these, sampled at (20, 60, 180) because
#: counting the zeros meant locating every one of them. It is counted by
#: Turing's method now and samples the full four decades with the rest.
#:
#: A slope is scale-free. A bound with the wrong exponent has a POSITIVE
#: slope; a correct one flattens or falls. 0.15 leaves room for noise in an
#: erratic residual like M(x) without admitting a missing power.
_RATIO_SLOPE = 0.15


def _split_asymptotic(equation):
    """`f = g + O(h)` -> `(f - g, h)`, or None if it is not that shape."""
    import sympy as sp

    terms = [
        node
        for node in sp.preorder_traversal(equation)
        if type(node).__name__ == "BigO"
    ]
    if len(terms) != 1:
        return None
    term = terms[0]
    bound = term.args[0]
    cleaned = equation.replace(term, sp.Integer(0))
    return cleaned.lhs - cleaned.rhs, bound


def an_asymptotic_bound_stays_bounded() -> Check:
    """`f = g + O(h)` says `|f - g| / h` does not run away.

    That is a claim about numbers and it can be sampled. It is not a proof of
    the asymptotic statement -- no finite sample is -- but a bound written
    with the wrong exponent, or against the wrong main term, shows up as a
    ratio climbing by orders of magnitude across two decades. That is the
    transcription error this catches, and it is the one that actually happens.
    """
    import sympy as sp

    from rh_research_engine.symbolic.functions import SREPR_NAMESPACE

    check = Check("Every asymptotic bound stays bounded")
    for item in _corpus_equations():
        if item.parse_error is not None or not item.sympy_srepr:
            continue
        try:
            expression = sp.sympify(item.sympy_srepr, locals=dict(SREPR_NAMESPACE))
        except Exception:
            continue
        _, claim = _statement(expression)
        if not isinstance(claim, sp.Equality):
            continue
        split = _split_asymptotic(claim)
        if split is None:
            continue
        residual, bound = split

        free = (residual.free_symbols | bound.free_symbols) - {sp.Symbol("epsilon")}
        if len(free) != 1:
            continue
        (variable,) = free
        residual = residual.subs(sp.Symbol("epsilon"), _EPSILON_INSTANCE)
        bound = bound.subs(sp.Symbol("epsilon"), _EPSILON_INSTANCE)

        for points in (_ASYMPTOTIC_POINTS, _ASYMPTOTIC_MEDIUM, _ASYMPTOTIC_NARROW):
            ratios = []
            for point in points:
                top = _at(sp.Abs(residual), {variable: point})
                low = _at(sp.Abs(bound), {variable: point})
                if top is None or low is None or low.real == 0:
                    break
                ratios.append(abs(top) / abs(low.real))
            if len(ratios) != len(points):
                continue
            check.covered.append(item.source)
            # Only GROWTH is a failure. A ratio that falls means the bound
            # is generous, which is what a bound is allowed to be --
            # Lindelof's drops from 1.7 to 0.14 across two decades.
            import math

            # Growth that is sustained, not merely present at the endpoints.
            #
            # A residual may pass near zero, and where it does the ratio is
            # tiny for a reason that has nothing to do with the exponent. An
            # endpoint slope reads that as steep growth: sampled over four
            # decades, N(T)'s ratio runs 0.0005, 0.1048, 0.0356 -- rising and
            # then FALLING, plainly bounded -- and its endpoint slope of 0.46
            # called it a missing power, because S(100) happens to be 0.002.
            #
            # So the ratio must also never fall back. A missing power keeps
            # growing: `pi(x) = li(x) + O(log x)` runs 1.1, 1.85, 9.4, rising
            # at every step, and is still caught. Requiring each STEP to clear
            # the threshold instead would NOT have caught it -- its first step
            # is 0.11, because li(x) - pi(x) is erratic at small x.
            #
            # None of this was visible while N(T) could only be sampled at
            # (20, 60, 180), which is how it survived.
            if all(ratio > 0 for ratio in ratios):
                overall = math.log(ratios[-1] / ratios[0]) / math.log(
                    points[-1] / points[0]
                )
                never_falls = all(
                    later >= earlier
                    for earlier, later in zip(ratios[:-1], ratios[1:], strict=True)
                )
                if never_falls and overall > _RATIO_SLOPE:
                    check.failures.append(
                        f"{item.source}\n    |f-g|/h grows like x^{overall:.3g} "
                        f"across {points} without falling back: "
                        f"{[round(r, 4) for r in ratios]}"
                    )
            break
    return check


def _definitions() -> dict:
    """Subjects the corpus defines, and what it defines them as.

    A definition is an equation whose left side is a bare symbol, or an
    undefined function applied to plain variables, and whose right side does
    not mention the subject. `theta(t) = arg(Gamma(...)) - ...` is one;
    `zeta(s) = prod_p ...` is not, because zeta is a real function rather than
    a subject this corpus introduces.

    Keyed by the subject, holding EVERY equation for it. More than one is not
    a conflict to resolve here -- `Theta` has three, and whether they agree is
    exactly what `definitions_are_consistent` asks.
    """
    import sympy as sp

    from rh_research_engine.symbolic.functions import SREPR_NAMESPACE

    found: dict = {}
    for item in _corpus_equations():
        if item.parse_error is not None or not item.sympy_srepr:
            continue
        try:
            expression = sp.sympify(item.sympy_srepr, locals=dict(SREPR_NAMESPACE))
        except Exception:
            continue
        _, claim = _statement(expression)
        if not isinstance(claim, sp.Equality):
            continue
        left = claim.lhs
        if isinstance(left, sp.Symbol):
            key = left
        elif isinstance(left, sp.core.function.AppliedUndef) and all(
            argument.is_Symbol for argument in left.args
        ):
            key = left.func
        else:
            continue
        if claim.rhs.has(key):
            continue
        found.setdefault(key, []).append((left, claim.rhs, item.source))
    return found


def _apply_definitions(expression, definitions, skip=None):
    """Substitute every UNAMBIGUOUS definition into an expression.

    Subjects with more than one equation are left alone: substituting one of
    three readings of `Theta` would be picking a winner, and picking is what
    the consistency check exists to avoid.
    """
    import sympy as sp

    for key, entries in definitions.items():
        if len(entries) != 1 or key == skip:
            continue
        left, right, _ = entries[0]
        if isinstance(key, sp.Symbol):
            expression = expression.subs(key, right)
        else:
            variable = left.args[0]
            expression = expression.replace(
                key,
                lambda argument, right=right, variable=variable: right.subs(
                    variable, argument
                ),
            )
    return expression


def a_definition_is_computable() -> Check:
    """A definition has to produce a value, or it defines nothing.

    Li's coefficients sat in the corpus as a derivative with no evaluation
    point: the right side was a function of s while the left was a number.
    That is not a weaker definition, it is not one -- and the only way to
    notice is to try to get a number out of it.
    """
    import sympy as sp

    definitions = _definitions()
    check = Check("Every definition computes a value")
    for key, entries in definitions.items():
        if len(entries) != 1:
            continue  # several statements about one subject; consistency's job
        left, right, source = entries[0]
        resolved = _apply_definitions(right, definitions, skip=key)

        # A derivative whose variable is STILL FREE was never evaluated
        # anywhere. Li's coefficients sat in the corpus in exactly that state:
        # `d^n/ds^n (...)` with no `|_{s=1}`, so the right side was a function
        # of s while the left was a number. `Subs` binds the variable, so a
        # properly written one does not trip this.
        # `variable_count`, not `variables`: the latter expands the order, and
        # Li's derivative has a SYMBOLIC order n, which raises rather than
        # answering.
        dangling = [
            node.variable_count[0][0]
            for node in sp.preorder_traversal(resolved)
            if isinstance(node, sp.Derivative)
            and node.variable_count[0][0] in resolved.free_symbols
        ]
        if dangling:
            check.failures.append(
                f"{source}\n    defines {left} with a derivative in "
                f"{dangling[0]}, which is never evaluated at a point -- so the "
                "right side produces no value"
            )
            continue

        variables = sorted(resolved.free_symbols, key=str)
        if len(variables) > 1:
            continue
        got = None
        for point in (2, 3, 4, 20):
            assignment = {variables[0]: point} if variables else {}
            value = _at(resolved, assignment)
            if value is not None:
                got = value
                break
        if got is None:
            check.failures.append(
                f"{source}\n    defines {left} but produces no value at any "
                "sample point"
            )
        else:
            check.covered.append(source)
    return check


def definitions_are_consistent() -> Check:
    """Every statement about a subject must survive substituting a definition.

    `Theta` carries three: the transfer from a remainder exponent, the RH
    endpoint, and the unconditional floor. `Lambda` carries two. Substituting
    one into another must not produce a falsehood.

    This is what makes the RH statements checkable at all. `Theta = 1/2` and
    `Lambda = 0` cannot be verified -- they ARE the open question, and no
    amount of computation settles them. They can still be held against the
    proved bounds `Theta >= 1/2` and `Lambda >= 0` and against the transfer
    relation, and a corpus whose own statements disagree is broken whatever
    the answer turns out to be. Consistency is not proof, and it is not
    nothing.
    """
    import sympy as sp

    from rh_research_engine.symbolic.functions import SREPR_NAMESPACE

    statements = []
    for item in _corpus_equations():
        if item.parse_error is not None or not item.sympy_srepr:
            continue
        try:
            expression = sp.sympify(item.sympy_srepr, locals=dict(SREPR_NAMESPACE))
        except Exception:
            continue
        _, claim = _statement(expression)
        if isinstance(claim, sp.Rel):
            statements.append((claim, item.source))

    check = Check("Statements about one subject agree")
    for key, entries in _definitions().items():
        if not isinstance(key, sp.Symbol):
            continue
        for left, right, source in entries:
            for claim, other in statements:
                if other == source or not claim.has(left):
                    continue
                try:
                    verdict = sp.simplify(claim.subs(left, right))
                except Exception:
                    continue
                check.covered.append(source)
                check.covered.append(other)
                if verdict is sp.false:
                    check.failures.append(
                        f"{other}\n    contradicts {source}: substituting gives "
                        f"{claim.subs(left, right)}, which is false"
                    )
    return check



def a_claim_about_a_defined_term_holds() -> Check:
    """A criterion written in a defined term is checkable through it.

    `lambda_n >= 0` says nothing on its own -- lambda_n is a free symbol. With
    the corpus's own definition substituted it becomes a statement about
    numbers, and Li's criterion can actually be tested rather than filed.
    """
    import sympy as sp

    from rh_research_engine.symbolic.functions import SREPR_NAMESPACE

    definitions = _definitions()
    single = {k for k, v in definitions.items() if len(v) == 1}
    check = Check("Claims written in defined terms hold")
    for item in _corpus_equations():
        if item.parse_error is not None or not item.sympy_srepr:
            continue
        try:
            expression = sp.sympify(item.sympy_srepr, locals=dict(SREPR_NAMESPACE))
        except Exception:
            continue
        condition, claim = _statement(expression)
        if isinstance(claim, sp.Equality) or not isinstance(claim, sp.Rel):
            continue  # an identity or a definition, handled elsewhere
        if not any(claim.has(key) for key in single):
            continue
        resolved = _apply_definitions(claim, definitions)
        variables = sorted(resolved.free_symbols, key=str)
        if len(variables) != 1:
            continue
        (variable,) = variables
        tested = 0
        for point in (1, 2, 3, 4, 5):
            assignment = {variable: point}
            if not _holds_at(condition, assignment):
                continue
            verdict = _decide(resolved, assignment)
            if verdict is None:
                continue
            if not tested:
                check.covered.append(item.source)
            tested += 1
            if not verdict:
                check.failures.append(
                    f"{item.source}\n    fails at {variable} = {point} once its "
                    "own definition is substituted"
                )
        if not tested:
            check.failures.append(
                f"{item.source}\n    references a defined term but never "
                "evaluated, so the claim is stored unchecked"
            )
    return check


#: Increasing points a limit at infinity is watched along.
_APPROACH_POINTS = (1000, 10000, 100000, 1000000)


def a_limit_approaches_its_value() -> Check:
    """`lim f = L` says f gets closer to L, and that is watchable.

    Not a proof of the limit -- no finite sample is -- but a limit written
    against the wrong value, or against the wrong normalisation, moves AWAY
    from it, and that shows in three points. The prime number theorem is the
    case here: `pi(x) log x / x` falls 1.132, 1.104, 1.084 toward 1.
    """
    import sympy as sp

    from rh_research_engine.symbolic.functions import SREPR_NAMESPACE

    check = Check("Every limit approaches its stated value")
    for item in _corpus_equations():
        if item.parse_error is not None or not item.sympy_srepr:
            continue
        try:
            expression = sp.sympify(item.sympy_srepr, locals=dict(SREPR_NAMESPACE))
        except Exception:
            continue
        _, claim = _statement(expression)
        if not isinstance(claim, sp.Equality):
            continue
        limits = [
            node for node in sp.preorder_traversal(claim) if isinstance(node, sp.Limit)
        ]
        if len(limits) != 1:
            continue
        limit = limits[0]
        body, variable, target = limit.args[0], limit.args[1], limit.args[2]
        if target != sp.oo:
            continue
        stated = _numeric(claim.rhs if claim.lhs.has(limit) else claim.lhs)
        if stated is None:
            continue

        distances = []
        for point in _APPROACH_POINTS:
            value = _at(body, {variable: point})
            if value is None:
                break
            distances.append(abs(value - stated))
        if len(distances) < 3:
            continue
        check.covered.append(item.source)
        if distances[-1] >= distances[0]:
            check.failures.append(
                f"{item.source}\n    moves AWAY from {stated.real:.6g}: "
                f"distances {[round(d, 6) for d in distances]}"
            )
    return check


#: Zeros the guard sums over for the explicit formula.
#:
#: Small on purpose: the guard runs in the pre-commit hook. The truncation error
#: at five thousand zeros is about 0.024, and the error a WRONG RECORD would
#: produce is the size of the term it dropped -- log(2 pi) is 1.84, seventy-five
#: times larger. The check does not need the sum to converge, only to be far
#: closer than the mistake it is looking for.
_EXPLICIT_FORMULA_ZEROS = 5000

#: How far the reconstruction may sit from psi(x) before it is a failure.
#: Ten times the truncation error and a tenth of the smallest constant that
#: could plausibly go missing.
_EXPLICIT_FORMULA_TOLERANCE = 0.25

#: Height for the pair-correlation sample, and how far the measured curve may
#: sit from the recorded one. Montgomery's is asymptotic, so agreement at this
#: height is loose by nature -- and a wrong curve misses by far more than this.
_PAIR_CORRELATION_HEIGHT = 5000.0
_PAIR_CORRELATION_TOLERANCE = 0.15


def a_statement_about_the_zeros_holds() -> Check:
    """The two corpus formulas that need the zeros, evaluated.

    Both had sat in the index since ingestion with nothing able to check them,
    because each wants thousands of zeros and a zero cost 160 ms. Neither is a
    test OF the mathematics -- von Mangoldt's formula is a theorem and
    Montgomery's correlation a well-studied conjecture. They test whether the
    corpus RECORDS them correctly, which is the question this repository keeps
    getting wrong: a missing constant parses, indexes, fingerprints and exports
    exactly as well as the true statement, and only a value tells them apart.
    """
    check = Check("Every statement about the zeros survives a value")

    from rh_research_engine.symbolic.explicit_formula import (
        check_explicit_formula,
        recorded_shape,
    )
    from rh_research_engine.symbolic.pair_correlation import check_pair_correlation

    # Coverage names the CORPUS ITEM, not a description of it. Reported as a
    # description, the tally read "40 of 38 formulas checked against values" --
    # two of them being sentences this function had written about the corpus
    # rather than anything in it.
    def _source_containing(*needles: str) -> str | None:
        for item in _corpus_equations():
            if all(needle in item.source for needle in needles):
                return item.source
        return None

    explicit_source = _source_containing(r"\rho(k)", r"\psi(x)")
    correlation_source = _source_containing(r"\sin(\pi u)")

    try:
        shape = recorded_shape()
    except (FileNotFoundError, LookupError) as exc:
        check.failures.append(f"explicit formula\n    {exc}")
        shape = None

    if shape is not None:
        if explicit_source:
            check.covered.append(explicit_source)
        missing = [name for name, present in shape.items() if not present]
        if missing:
            check.failures.append(
                "psi(x) explicit formula\n    the record is missing "
                f"{missing}; a dropped constant leaves psi and the "
                "reconstruction apart by exactly the constant, at every point"
            )
        else:
            result = check_explicit_formula(zeros=_EXPLICIT_FORMULA_ZEROS)
            if abs(result.worst_residual) > _EXPLICIT_FORMULA_TOLERANCE:
                check.failures.append(
                    "psi(x) explicit formula\n    rebuilt from "
                    f"{result.zeros_used} zeros it misses psi by "
                    f"{result.worst_residual:+.4f} at x = {result.worst_at}, "
                    f"against a truncation error of about 0.024"
                )

    try:
        correlation = check_pair_correlation(_PAIR_CORRELATION_HEIGHT)
    except (FileNotFoundError, LookupError, ValueError) as exc:
        check.failures.append(f"pair correlation\n    {exc}")
    else:
        if correlation_source:
            check.covered.append(correlation_source)
        if correlation.mean_deviation > _PAIR_CORRELATION_TOLERANCE:
            check.failures.append(
                "pair correlation\n    the spacing of "
                f"{correlation.zeros} zeros departs from the recorded curve by "
                f"{correlation.mean_deviation:.4f} on average, worst "
                f"{correlation.worst_deviation:.4f} at u = "
                f"{correlation.worst_at:.2f}"
            )
    return check


def an_expression_is_defined_where_it_is_used() -> Check:
    """A standalone expression has to BE a function.

    Montgomery's pair-correlation density is written `1 - (sin(pi u)/(pi u))^2`
    and is `0/0` at u = 0 -- where its whole content lives, since the value
    there is the level repulsion the conjecture is about. An expression whose
    removable singularity does not close is not the density.
    """
    import sympy as sp

    from rh_research_engine.symbolic.functions import SREPR_NAMESPACE

    check = Check("Every expression is defined where it is used")
    for item in _corpus_equations():
        if item.parse_error is not None or not item.sympy_srepr:
            continue
        try:
            expression = sp.sympify(item.sympy_srepr, locals=dict(SREPR_NAMESPACE))
        except Exception:
            continue
        if isinstance(expression, (sp.Rel, sp.Implies)):
            continue
        variables = sorted(expression.free_symbols, key=str)
        if len(variables) != 1:
            continue
        (variable,) = variables

        sampled = [
            _at(expression, {variable: point})
            for point in (sp.Rational(1, 2), 1, 2, 10)
        ]
        if any(value is None for value in sampled):
            check.failures.append(
                f"{item.source}\n    does not evaluate at every sample point"
            )
            continue
        check.covered.append(item.source)

        # And where it is 0/0, the removable singularity must close.
        at_zero = _at(expression, {variable: 0})
        if at_zero is None:
            try:
                limit = sp.limit(expression, variable, 0)
            except Exception:
                limit = None
            if limit is None or not limit.is_finite:
                check.failures.append(
                    f"{item.source}\n    is undefined at {variable} = 0 and the "
                    "limit there does not close"
                )
    return check


def one_name_resolution_policy(minimum: int = MINIMUM_SYMBOLIC_MODULES) -> Check:
    """Every reader of a formula resolves names the same way.

    A bare `parse_expr` with no `local_dict` reads the extractor's own output
    under different rules than produced it. That is not a style question: it
    is how `M(x)` became `M*x` in the index, and how the Lean exporter came to
    report a well-formed formula as unparseable.

    Checked on the syntax tree, per CALL. A line-window version of this looked
    right and passed a tree where one of two adjacent calls had lost its
    `local_dict` -- it was reading the neighbour's.
    """
    import ast

    check = Check("One name-resolution policy")
    # THE ENUMERATION NEEDS A FLOOR, and this check is the reason to state why.
    # It was already fooled once -- a five-line window read the NEIGHBOURING
    # call's `local_dict` -- and the repair hardened what it does with each
    # file while leaving "did it see any files" unasked. Point `SYMBOLIC` at an
    # empty directory and the loop body never runs, `failures` stays empty, and
    # this reports that one name-resolution policy holds across the package
    # having examined nothing. Verified by construction before this floor
    # existed: PASS on zero files.
    #
    # Breaking the checked thing does not find that. An injected missing
    # `local_dict` still fails. The hole is not that the check cannot fail
    # today; it is that it can succeed at scanning nothing after a rename or a
    # moved directory.
    # A PARAMETER, not a global the tests must fight. Two tests legitimately
    # point `SYMBOLIC` at a one-file directory to exercise the per-CALL logic,
    # and a hard floor broke both. They pass `minimum=1` to say out loud that
    # they are testing the file logic and not the enumeration; the production
    # call keeps the floor.
    sources = sorted(SYMBOLIC.glob("*.py"))
    if len(sources) < minimum:
        check.failures.append(
            f"{_display(SYMBOLIC)} yielded {len(sources)} modules, below the floor "
            f"of {minimum}. An empty enumeration passes every check below by "
            "having nothing to check, which is not the same as the policy holding."
        )
    for path in sources:
        if path.name == "parser.py":
            continue  # the policy itself lives here
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name != "parse_expr":
                continue
            supplied = {keyword.arg for keyword in node.keywords}
            if "local_dict" not in supplied:
                check.failures.append(
                    f"{_display(path)}:{node.lineno}\n"
                    "    parse_expr without local_dict: this reader has its own "
                    "name resolution"
                )
    return check


def no_consumer_calls_a_formula_unreadable() -> Check:
    """A consumer may refuse to PROVE a formula. It may not fail to read one.

    `unparseable` is a claim about the formula; every other verdict is a claim
    about what can be discharged. Only the second kind is honest when the
    formula is fine and the reader was not.
    """
    from rh_research_engine.symbolic.lean import export_polynomial_identity

    check = Check("No consumer reports a formula unreadable")
    for item in _corpus_equations():
        if item.parse_error is not None or item.lhs is None or item.rhs is None:
            continue
        reason = export_polynomial_identity(item.lhs, item.rhs).reason or ""
        if reason.startswith("parse failed"):
            check.failures.append(f"{item.source}\n    {reason[:120]}")
    return check


def the_index_holds_no_superseded_record() -> Check:
    """A corrected formula must not leave its error behind.

    Record ids are content hashes, so a correction files a SECOND record
    beside the one it corrects. The functional equation was in the index
    twice, once with its `sin(pi*s/2)` factor and once without, and a
    structural search would have returned the false one as indexed knowledge.
    """
    import json

    check = Check("Index holds nothing the corpus dropped")
    index_path = REPO / "research_state" / "formula_index.json"
    # A MISSING INDEX USED TO RETURN A PASSING CHECK. `check` has no failures at
    # this point, so `return check` reported that the index holds nothing the
    # corpus dropped, on the strength of never having opened it. Verified by
    # construction: delete the file and this passed.
    #
    # It is regenerable by `rhre symbolic ingest`, so absent is plausible here
    # rather than exotic -- which makes a silent pass more likely to be met, not
    # less.
    if not index_path.exists():
        check.failures.append(
            f"{_display(index_path)} does not exist, so nothing was compared. "
            "Regenerate it with `rhre symbolic ingest`; a check that read no "
            "index has not confirmed the index."
        )
        return check
    records = json.loads(index_path.read_text(encoding="utf-8"))
    records = records["records"] if isinstance(records, dict) else records
    corpus_name = _display(CORPUS)
    live = {
        item.source
        for item in _corpus_equations()
        if item.parse_error is None
    }
    # The ingest prunes by source; this confirms the count did not drift.
    indexed = sum(
        1
        for record in records
        if (record.get("source") or "").replace("\\", "/")
        == corpus_name.replace("\\", "/")
    )
    # AND TWO NOTHINGS MUST NOT AGREE. `indexed != len(live)` is satisfied by
    # 0 == 0, so an empty corpus and an empty index confirm each other and pass
    # -- "the index matches the corpus" and "there is no corpus" sharing one
    # verdict, which is the rule this repository states about refuted and
    # not-tested, in arithmetic.
    if not live:
        check.failures.append(
            f"{corpus_name} yielded no parseable equations, so the count "
            "comparison below would be 0 == 0 and pass. Two empty sets agreeing "
            "is not the index matching the corpus."
        )
    if indexed != len(live):
        check.failures.append(
            f"{indexed} record(s) indexed from the corpus, {len(live)} formula(s) in it. "
            "Re-run `rhre symbolic ingest` so the index matches the document."
        )
    return check


#: The checks that evaluate mathematics. They cost about ten of the gate's
#: twelve seconds -- locating zeros and summing series is real work -- so
#: pre-commit runs without them and pre-push and CI run them.
#:
#: This is a latency split, NOT a strength split: nothing here is optional,
#: and a push or a CI run executes every one.
NUMERIC_CHECKS = frozenset(
    {
        "an_identity_holds_numerically",
        "a_series_converges_to_its_closed_form",
        "a_stated_relation_holds",
        "an_asymptotic_bound_stays_bounded",
        "a_definition_is_computable",
        "definitions_are_consistent",
        "a_claim_about_a_defined_term_holds",
        "a_limit_approaches_its_value",
        "an_expression_is_defined_where_it_is_used",
        "a_statement_about_the_zeros_holds",
    }
)

CHECKS = (
    every_formula_parses,
    every_symbol_is_declared,
    an_identity_holds_numerically,
    a_series_converges_to_its_closed_form,
    a_stated_relation_holds,
    an_asymptotic_bound_stays_bounded,
    a_definition_is_computable,
    definitions_are_consistent,
    a_claim_about_a_defined_term_holds,
    a_limit_approaches_its_value,
    an_expression_is_defined_where_it_is_used,
    a_statement_about_the_zeros_holds,
    every_formula_indexes,
    both_readings_agree,
    no_stub_for_a_real_function,
    one_name_resolution_policy,
    no_consumer_calls_a_formula_unreadable,
    the_index_holds_no_superseded_record,
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only failures. For git hooks, where silence means pass.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip the checks that evaluate mathematics. For pre-commit only; "
        "pre-push and CI run everything.",
    )
    args = parser.parse_args(argv)

    selected = [
        check
        for check in CHECKS
        if not (args.fast and check.__name__ in NUMERIC_CHECKS)
    ]

    results = []
    for check in selected:
        try:
            results.append(check())
        except Exception as exc:  # noqa: BLE001
            # A gate that dies mid-run reports nothing about everything after
            # it, which reads exactly like a pass. SymPy raises from inside
            # its own memoiser on some arguments; that is a fact about the
            # probe, and it belongs in the output rather than in a traceback.
            failed = Check(check.__name__.replace("_", " ").capitalize())
            failed.failures.append(
                f"the check itself raised: {type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:120]}"
            )
            results.append(failed)
    width = max(len(result.title) for result in results)
    for result in results:
        if not args.quiet:
            print(f"{result.title:<{width}}  ... {'ok' if result.passed else 'FAIL'}")
        if not result.passed:
            if args.quiet:
                print(f"formula-guard: {result.title}: FAIL", file=sys.stderr)
            for failure in result.failures:
                print(f"  {failure}", file=sys.stderr)

    checked = sorted({source for result in results for source in result.covered})
    if not args.quiet:
        total = len(_corpus_equations())
        print(f"\n{len(checked)} of {total} formulas checked against values")

    failed = [result for result in results if not result.passed]
    if failed:
        print(
            "\nformula-guard: the formulas are the product. Fix the formula rather\n"
            "than the check -- a refusal, a stub, or a narrowed corpus is a staging\n"
            "post, not a fix.",
            file=sys.stderr,
        )
        return 1
    if not args.quiet:
        skipped = len(CHECKS) - len(selected)
        note = f" ({skipped} numeric check(s) skipped by --fast)" if skipped else ""
        print(f"\nverdict: EVERY FORMULA READS{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
