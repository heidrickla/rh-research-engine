from __future__ import annotations

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .models import AsymptoticResult, ResidueResult

_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)


def residue(expression: str, variable: str, pole: str) -> ResidueResult:
    var = sp.Symbol(variable)
    expr = parse_expr(expression, local_dict={variable: var}, transformations=_TRANSFORMS)
    pole_expr = parse_expr(pole, local_dict={variable: var}, transformations=_TRANSFORMS)
    value = sp.residue(expr, var, pole_expr)
    return ResidueResult(expression=expression, variable=variable, pole=pole, residue=str(value))


def asymptotic(expression: str, variable: str, point: str = "oo") -> AsymptoticResult:
    var = sp.Symbol(variable, positive=True)
    local = {variable: var, "oo": sp.oo}
    expr = parse_expr(expression, local_dict=local, transformations=_TRANSFORMS)
    p = sp.oo if point == "oo" else parse_expr(point, local_dict=local, transformations=_TRANSFORMS)
    try:
        limit = sp.limit(expr, var, p)
    except Exception:
        limit = None
    try:
        leading = sp.series(expr, var, p, 2).removeO()
        leading_text = str(leading)
    except Exception as exc:
        return AsymptoticResult(expression=expression, variable=variable, point=point, limit=None if limit is None else str(limit), error=f"leading-term analysis failed: {type(exc).__name__}: {exc}")
    return AsymptoticResult(expression=expression, variable=variable, point=point, leading=leading_text, limit=None if limit is None else str(limit))
