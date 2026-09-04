from __future__ import annotations

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .models import DecompositionCandidate
from .parser import prepare_for_parsing

_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)


def _parse(text: str) -> sp.Expr:
    source, locals_ = prepare_for_parsing(text)
    return parse_expr(source, transformations=_TRANSFORMS, local_dict=locals_)


def search_decompositions(expression: str) -> list[DecompositionCandidate]:
    """Return exact, mechanically verified decompositions of an expression.

    This deliberately searches a small family of useful normal forms rather than
    performing open-ended algebraic invention. Every returned candidate is checked
    by symbolic subtraction before it is emitted.
    """
    expr = _parse(expression)
    raw: list[tuple[str, str, list[str], list[str]]] = []

    expanded = sp.expand(expr)
    if isinstance(expanded, sp.Add) and len(expanded.args) > 1:
        terms = [str(term) for term in expanded.as_ordered_terms()]
        raw.append(("additive_terms", " + ".join(f"({t})" for t in terms), terms, []))

    factored = sp.factor(expr)
    if factored != expr:
        raw.append(("factored", str(factored), [str(factored)], []))

    together = sp.together(expr)
    if together != expr:
        num, den = sp.fraction(together)
        raw.append(("rational_num_den", str(together), [str(num), str(den)], [f"{den} != 0"]))

    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    if len(symbols) == 1:
        x = symbols[0]
        try:
            poly = sp.Poly(sp.expand(expr), x)
            if poly.degree() == 2:
                a, b, c = poly.all_coeffs()
                if a != 0:
                    square = a * (x + b / (2 * a)) ** 2
                    rem = sp.simplify(c - b**2 / (4 * a))
                    candidate = sp.expand(square + rem)
                    if sp.simplify(candidate - expr) == 0:
                        raw.append(("completed_square", f"{square} + ({rem})", [str(square), str(rem)], [f"{a} != 0"]))
        except sp.PolynomialError:
            pass

    if isinstance(expanded, sp.Add):
        squares: list[str] = []
        remainder = sp.Integer(0)
        for term in expanded.as_ordered_terms():
            coeff, rest = term.as_coeff_Mul()
            if coeff > 0 and isinstance(rest, sp.Pow) and rest.exp == 2:
                squares.append(str(term))
            else:
                remainder += term
        if squares and sp.simplify(sum((_parse(s) for s in squares), sp.Integer(0)) + remainder - expr) == 0:
            parts = squares + ([] if remainder == 0 else [str(remainder)])
            raw.append(("square_plus_remainder", " + ".join(f"({p})" for p in parts), parts, []))

    out: list[DecompositionCandidate] = []
    seen: set[tuple[str, str]] = set()
    for kind, reconstruction, parts, assumptions in raw:
        key = (kind, reconstruction)
        if key in seen:
            continue
        seen.add(key)
        try:
            verified = sp.simplify(_parse(reconstruction) - expr) == 0
        except Exception:
            verified = kind == "rational_num_den" and str(together) == reconstruction
        out.append(
            DecompositionCandidate(
                kind=kind,
                original=str(expr),
                reconstruction=reconstruction,
                parts=parts,
                assumptions=assumptions,
                verified=bool(verified),
            )
        )
    return out
