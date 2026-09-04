from __future__ import annotations

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .models import Assumption
from .parser import prepare_for_parsing

_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)


def extract_assumptions(expression: str) -> list[Assumption]:
    text, locals_ = prepare_for_parsing(expression)
    expr = parse_expr(text, transformations=_TRANSFORMS, local_dict=locals_)
    found: dict[tuple[str, str], Assumption] = {}
    denom = sp.denom(sp.together(expr))
    if denom != 1:
        condition = f"{denom} != 0"
        found[(str(denom), condition)] = Assumption(expression=str(denom), condition=condition, reason="denominator must be nonzero")
    for node in sp.preorder_traversal(expr):
        if node.func == sp.log and node.args:
            arg = node.args[0]
            condition = f"{arg} > 0 (for real-valued log)"
            found[(str(arg), condition)] = Assumption(expression=str(arg), condition=condition, reason="real logarithm domain")
        if node.func == sp.gamma and node.args:
            arg = node.args[0]
            condition = f"{arg} not in {{0,-1,-2,...}}"
            found[(str(arg), condition)] = Assumption(expression=str(arg), condition=condition, reason="Gamma pole exclusion")
    return list(found.values())
