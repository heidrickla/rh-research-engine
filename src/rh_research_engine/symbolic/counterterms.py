from __future__ import annotations

import sympy as sp

from .models import CountertermCandidate


def generate_counterterm_basis(variable: str = "X", *, include_arithmetic: bool = True) -> list[CountertermCandidate]:
    """Generate an interpretable symbolic basis for lower-order counterterms."""
    X = sp.Symbol(variable, positive=True)
    basis: list[tuple[str, sp.Expr, str]] = [
        ("constant", sp.Integer(1), "scale-independent lower-order term"),
        ("inv_log", 1 / sp.log(X), "first logarithmic correction"),
        ("inv_log2", 1 / sp.log(X) ** 2, "second logarithmic correction"),
        ("log_over_X", sp.log(X) / X, "finite-scale arithmetic correction"),
        ("inv_X", 1 / X, "first power correction"),
    ]
    if include_arithmetic:
        q = sp.Symbol("q", positive=True)
        basis.extend(
            [
                ("log_q", sp.log(q), "resolution-dependent spectral count term"),
                ("q", q, "linear resolution counterterm"),
                ("q_log_q", q * sp.log(q), "leading high-resolution bandwidth law"),
                ("euler_gamma", sp.EulerGamma, "universal Euler constant candidate"),
                ("log_2pi", sp.log(2 * sp.pi), "Riemann-von Mangoldt normalization constant"),
            ]
        )
    return [CountertermCandidate(name=name, expression=str(expr), rationale=rationale, status="candidate") for name, expr, rationale in basis]


def build_counterterm_ansatz(names: list[str], variable: str = "X") -> str:
    candidates = {item.name: item for item in generate_counterterm_basis(variable)}
    missing = [name for name in names if name not in candidates]
    if missing:
        raise ValueError(f"unknown counterterm basis names: {', '.join(missing)}")
    terms = [f"c{i}*({candidates[name].expression})" for i, name in enumerate(names)]
    return " + ".join(terms) if terms else "0"
