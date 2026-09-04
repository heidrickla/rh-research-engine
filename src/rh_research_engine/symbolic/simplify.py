from __future__ import annotations

from collections.abc import Callable

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .models import RewriteStep, SimplificationResult
from .parser import prepare_for_parsing

_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)


def _parse(expr: str) -> sp.Expr:
    text, locals_ = prepare_for_parsing(expr)
    return parse_expr(
        text, transformations=_TRANSFORMS, local_dict=locals_, evaluate=False
    )


def _gamma_recurrence(expr: sp.Expr) -> sp.Expr:
    z = sp.Wild("z")
    return expr.replace(sp.gamma(z + 1), z * sp.gamma(z))


def _reflection(expr: sp.Expr) -> sp.Expr:
    z = sp.Wild("z")
    return expr.replace(sp.gamma(z) * sp.gamma(1 - z), sp.pi / sp.sin(sp.pi * z))


def _finite_sum(expr: sp.Expr) -> sp.Expr:
    return expr.doit() if expr.has(sp.Sum) else expr


def _lost_domain_conditions(before: sp.Expr, after: sp.Expr) -> list[str]:
    """Conditions ``before`` required that ``after`` no longer records.

    Cancelling `(x**2-1)/(x-1)` to `x+1` is only valid away from x = 1. The
    rewrite itself is correct; dropping the condition is what turns a
    conditional identity into an unconditional one.
    """
    from .equivalence import domain_conditions

    try:
        return sorted(set(domain_conditions(before)) - set(domain_conditions(after)))
    except Exception:
        return []


_RULES: list[tuple[str, str, Callable[[sp.Expr], sp.Expr], list[str]]] = [
    ("ALG-CANCEL", "Cancel exact rational factors", sp.cancel, []),
    ("ALG-TOGETHER", "Combine rational terms over a common denominator", sp.together, []),
    ("GAMMA-RECURRENCE", "Apply Gamma(z+1)=z Gamma(z)", _gamma_recurrence, []),
    ("GAMMA-REFLECTION", "Apply Gamma(z)Gamma(1-z)=pi/sin(pi z)", _reflection, ["z not an integer pole"]),
    ("FINITE-SUM-DOIT", "Evaluate exact finite symbolic sums when SymPy can certify them", _finite_sum, []),
    ("ALG-FACTOR", "Factor exact polynomial/rational structure", sp.factor, []),
]


def simplify_with_trace(expression: str, *, max_passes: int = 3) -> SimplificationResult:
    current = _parse(expression)
    steps: list[RewriteStep] = []
    assumptions: list[str] = []
    for _ in range(max_passes):
        changed = False
        for rule_id, description, func, rule_assumptions in _RULES:
            before = current
            try:
                after = func(before)
            except Exception:
                continue
            if after != before:
                # Any domain condition the rewrite discarded becomes an
                # explicit assumption on the result.
                step_assumptions = list(rule_assumptions) + _lost_domain_conditions(before, after)
                steps.append(RewriteStep(rule_id=rule_id, description=description, before=str(before), after=str(after), assumptions=step_assumptions))
                current = after
                assumptions.extend(a for a in step_assumptions if a not in assumptions)
                changed = True
        if not changed:
            break
    return SimplificationResult(original=expression, simplified=str(current), steps=steps, assumptions=assumptions)
