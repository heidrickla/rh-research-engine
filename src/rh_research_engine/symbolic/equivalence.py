from __future__ import annotations

import hashlib

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .models import EquivalenceResult, Fingerprint
from .parser import prepare_for_parsing

_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)


def _parse(expr: str) -> sp.Expr:
    """Parse under the same name resolution the extractor used.

    `sympy_locals` is not an optimisation here, it is the correctness of the
    fingerprint. Without it this function re-read an already-parsed equation
    with different rules than produced it, and quietly hashed a different
    object -- see the note on that function.
    """
    text, locals_ = prepare_for_parsing(expr)
    return parse_expr(text, transformations=_TRANSFORMS, local_dict=locals_)


def _parse_structural(expr: str) -> sp.Expr:
    """Parse without evaluating, so removable singularities survive.

    SymPy collapses `x/x` to `1` during parsing. Domain analysis has to see the
    division that was written, not the value it happens to equal wherever it is
    defined.
    """
    text, locals_ = prepare_for_parsing(expr)
    return parse_expr(
        text, transformations=_TRANSFORMS, local_dict=locals_, evaluate=False
    )


def _as_expr(expression: str | sp.Basic) -> sp.Basic:
    """Accept a parsed expression as readily as a string.

    A caller that already holds the object the extractor built should hand it
    over rather than round-trip it through text: printing and re-reading is
    where meaning was being lost.
    """
    return _parse(expression) if isinstance(expression, str) else expression


def _condition_text(base: sp.Expr) -> str:
    """Render a nonzero-condition, collapsing unevaluated `1*1` style noise."""
    try:
        base = sp.expand(base)
    except Exception:
        pass
    return f"{sp.sstr(base)} != 0"


def domain_conditions(expression: str | sp.Basic) -> list[str]:
    """Conditions under which an expression is defined.

    Cancellation removes removable singularities, so these must be recorded
    alongside any canonical form. Without them `(x**2-1)/(x-1)` and `x+1` are
    indistinguishable, and a certificate computed for one is accepted for the
    other -- including at the pole the certificate never covered.
    """
    if isinstance(expression, str):
        try:
            expr = _parse_structural(expression)
        except Exception:
            expr = _parse(expression)
    else:
        expr = expression
    conditions: set[str] = set()
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow):
            exp = node.exp
            is_negative = exp.is_negative
            if is_negative is None and exp.is_number:
                is_negative = bool(sp.expand(exp) < 0)
            if is_negative:
                conditions.add(_condition_text(node.base))
    try:
        if not isinstance(expr, sp.Expr):
            # A statement has no denominator. `together` on one builds a Mul
            # out of a Boolean, which SymPy deprecates and which means nothing:
            # the poles live in the expressions INSIDE it, and the Pow scan
            # above has already walked them.
            raise TypeError("not an expression")
        denominator = sp.denom(sp.together(expr))
        if denominator != 1:
            conditions.add(_condition_text(denominator))
    except Exception:
        pass
    return sorted(conditions)


def _canonical_form(expr: sp.Basic) -> sp.Basic:
    """Normalize the algebra inside a statement of any shape.

    `expand` is a method on expressions, not on statements: an `Implies` has
    no `.expand()` and raised straight out of the fingerprint. A criterion
    carrying its hypothesis -- `n > 5040 -> sigma(n) < ...` -- is a statement,
    and dropping the hypothesis to make it fingerprint would index a claim the
    source does not make. So the algebra is normalized where the algebra is,
    and the structure around it is rebuilt unchanged.
    """
    if isinstance(expr, sp.Expr):
        return sp.cancel(sp.together(sp.expand(expr)))
    if expr.args:
        return expr.func(*[_canonical_form(argument) for argument in expr.args])
    return expr


def canonicalize(expression: str | sp.Basic) -> str:
    """Canonical algebraic form, ignoring domain. Pair with `domain_conditions`."""
    return sp.srepr(_canonical_form(_as_expr(expression)))


def fingerprint(expression: str | sp.Basic) -> Fingerprint:
    """Domain-aware structural fingerprint.

    The digest covers the canonical algebra *and* the conditions under which it
    is valid, so two expressions that agree only off a pole do not collide.
    """
    canonical = canonicalize(expression)
    conditions = domain_conditions(expression)
    payload = canonical + "|domain:" + ";".join(conditions)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return Fingerprint(
        canonical=canonical,
        sha256=digest,
        metadata={
            "normalizer": "expand+together+cancel",
            "domain_conditions": conditions,
            "algebra_only_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        },
    )


def equivalent(left: str, right: str) -> EquivalenceResult:
    """Compare two expressions, reporting any domain gap rather than hiding it."""
    left_expr = _parse(left)
    right_expr = _parse(right)
    left_canon = canonicalize(left)
    right_canon = canonicalize(right)

    left_domain = set(domain_conditions(left))
    right_domain = set(domain_conditions(right))
    # Conditions required by one side but not the other are exactly the points
    # where an "equivalence" would be asserting something it has not checked.
    gap = sorted(left_domain ^ right_domain)

    def _finish(is_equal: bool | None, method: str, detail: str | None = None) -> EquivalenceResult:
        assumptions = list(gap)
        if is_equal is True and gap:
            method = f"{method}_conditional"
            detail = (
                (detail + "; " if detail else "")
                + "equality holds only where the differing domain conditions are satisfied"
            )
        return EquivalenceResult(
            equivalent=is_equal,
            method=method,
            left_canonical=left_canon,
            right_canonical=right_canon,
            assumptions=assumptions,
            detail=detail,
        )

    if left_canon == right_canon:
        return _finish(True, "canonical_fingerprint")
    try:
        diff = sp.simplify(left_expr - right_expr)
    except Exception as exc:
        return _finish(None, "inconclusive", f"simplify failed: {type(exc).__name__}: {exc}")
    if diff == 0:
        return _finish(True, "symbolic_difference")
    return _finish(False, "symbolic_difference_nonzero", f"simplified difference: {diff}")
