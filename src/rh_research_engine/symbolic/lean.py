from __future__ import annotations

import re

import sympy as sp
from pydantic import BaseModel, Field
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .parser import prepare_for_parsing

_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)


class LeanExport(BaseModel):
    supported: bool
    theorem_name: str
    lean: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    reason: str | None = None


def _lean_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = "thm_" + cleaned
    return cleaned


def _to_lean(expr: sp.Expr) -> str:
    if isinstance(expr, sp.Symbol):
        return expr.name
    if isinstance(expr, sp.Integer):
        return str(int(expr))
    if isinstance(expr, sp.Float):
        # `Poly(..., domain=QQ)` accepts a Float by converting it exactly, so the
        # polynomial guard passes and the Float survives into the printer. Emit
        # the exact rational it denotes rather than raising.
        exact = sp.Rational(expr)
        return _to_lean(exact)
    if isinstance(expr, sp.Rational):
        if expr.q == 1:
            return str(int(expr.p))
        return f"(({expr.p} : ℚ) / {expr.q})"
    if isinstance(expr, sp.Add):
        return "(" + " + ".join(_to_lean(arg) for arg in expr.args) + ")"
    if isinstance(expr, sp.Mul):
        return "(" + " * ".join(_to_lean(arg) for arg in expr.args) + ")"
    if isinstance(expr, sp.Pow) and expr.exp.is_Integer and int(expr.exp) >= 0:
        return f"({_to_lean(expr.base)} ^ {int(expr.exp)})"
    raise ValueError(f"unsupported Lean expression node: {type(expr).__name__}")


def export_polynomial_identity(left: str, right: str, theorem_name: str = "generated_identity") -> LeanExport:
    try:
        left_text, left_locals = prepare_for_parsing(left)
        right_text, right_locals = prepare_for_parsing(right)
        lhs = parse_expr(
            left_text, transformations=_TRANSFORMS, local_dict=left_locals
        )
        rhs = parse_expr(
            right_text, transformations=_TRANSFORMS, local_dict=right_locals
        )
    except Exception as exc:
        return LeanExport(supported=False, theorem_name=_lean_name(theorem_name), reason=f"parse failed: {type(exc).__name__}: {exc}")
    symbols = sorted(lhs.free_symbols | rhs.free_symbols, key=lambda s: s.name)
    try:
        sp.Poly(lhs, *symbols, domain=sp.QQ)
        sp.Poly(rhs, *symbols, domain=sp.QQ)
    except Exception:
        return LeanExport(supported=False, theorem_name=_lean_name(theorem_name), reason="only polynomial identities over rationals are currently exported")
    if sp.expand(lhs - rhs) != 0:
        return LeanExport(supported=False, theorem_name=_lean_name(theorem_name), reason="identity is not symbolically verified")
    binders = " ".join(f"({sym.name} : ℚ)" for sym in symbols)
    try:
        lean_lhs, lean_rhs = _to_lean(lhs), _to_lean(rhs)
    except ValueError as exc:
        # Fail closed rather than propagating: an exporter that raises on a
        # node it cannot print is still an exporter that refused to print it.
        return LeanExport(
            supported=False,
            theorem_name=_lean_name(theorem_name),
            reason=f"cannot render to Lean: {exc}",
        )
    lean = f"theorem {_lean_name(theorem_name)} {binders} : {lean_lhs} = {lean_rhs} := by\n  ring\n"
    return LeanExport(supported=True, theorem_name=_lean_name(theorem_name), lean=lean)
