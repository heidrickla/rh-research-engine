from __future__ import annotations

import sympy as sp
from pydantic import BaseModel
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)


class AsymptoticCheck(BaseModel):
    expression: str
    reference: str
    variable: str
    point: str
    ratio_limit: str | None = None
    verdict: str
    detail: str | None = None


class GrowthExponent(BaseModel):
    expression: str
    variable: str
    exponent: str | None = None
    verdict: str
    detail: str | None = None


def check_asymptotic(expression: str, reference: str, variable: str = "X", point: str = "oo") -> AsymptoticCheck:
    var = sp.Symbol(variable, positive=True)
    local = {variable: var, "oo": sp.oo}
    try:
        expr = parse_expr(expression, local_dict=local, transformations=_TRANSFORMS)
        ref = parse_expr(reference, local_dict=local, transformations=_TRANSFORMS)
        target = sp.oo if point == "oo" else parse_expr(point, local_dict=local, transformations=_TRANSFORMS)
        ratio = sp.limit(sp.cancel(expr / ref), var, target)
    except Exception as exc:
        return AsymptoticCheck(expression=expression, reference=reference, variable=variable, point=point, verdict="unknown", detail=f"{type(exc).__name__}: {exc}")
    if ratio == 1:
        verdict = "confirmed_ratio_limit"
    elif ratio in (0, sp.oo, -sp.oo):
        verdict = "incompatible_scale"
    else:
        verdict = "nonunit_ratio_or_inconclusive"
    return AsymptoticCheck(expression=expression, reference=reference, variable=variable, point=point, ratio_limit=str(ratio), verdict=verdict)


def growth_exponent(expression: str, variable: str = "X") -> GrowthExponent:
    var = sp.Symbol(variable, positive=True)
    try:
        expr = parse_expr(expression, local_dict={variable: var}, transformations=_TRANSFORMS)
        exponent = sp.limit(sp.log(sp.Abs(expr)) / sp.log(var), var, sp.oo)
    except Exception as exc:
        return GrowthExponent(expression=expression, variable=variable, verdict="unknown", detail=f"{type(exc).__name__}: {exc}")
    if exponent.has(sp.oo) or exponent.has(sp.zoo) or exponent.has(sp.nan):
        return GrowthExponent(expression=expression, variable=variable, exponent=str(exponent), verdict="non_power_or_inconclusive")
    return GrowthExponent(expression=expression, variable=variable, exponent=str(exponent), verdict="power_exponent_resolved")
