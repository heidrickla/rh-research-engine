"""Rebuild the prime staircase from the zeros, and see whether it matches.

THE CORPUS RECORDS VON MANGOLDT'S EXPLICIT FORMULA:

    psi(x) = x - sum_k 2 Re(x^rho_k / rho_k) - log(2 pi) - (1/2) log(1 - x^-2)

It is the statement that the zeros and the primes determine each other: the
oscillating terms, summed, are exactly the jumps of `psi` at the prime powers.
Nothing had evaluated it, because the sum needs thousands of zeros and a zero
cost 160 ms.

WHAT THE CHECK IS FOR. Not to discover the formula is wrong -- it is a theorem.
It is to discover whether the corpus RECORDS it correctly, which is a different
question and the one this repository keeps getting wrong. A missing constant or
a sign would parse, index, fingerprint and export exactly as well as the true
statement.

    x = 10.5   residual  +0.0009
    x = 100.5  residual  +0.0033
    x = 500.5  residual  -0.0036

against 20000 zeros. The formula as recorded is right.

THE SUM IS CONDITIONALLY CONVERGENT and means nothing out of order, so the
truncation is by INDEX -- the first K zeros by ordinate -- and never by any
other criterion. The residual falls like the tail: about 1.0 at K = 100, 0.12 at
K = 1000, 0.024 at K = 5000, so a residual is only evidence about the formula
once it is well below the truncation error at that K.

AVOID HALF-INTEGERS' OPPOSITE. `psi` jumps at every prime power, and the
explicit formula converges to the MIDPOINT of the jump there. Evaluated at a
prime power the two sides differ by half of `log p` and the formula looks
broken; every x here is offset from an integer for that reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..contracts.epistemic import RIGOROUS, Confidence

#: Where the indexed formulas live.
INDEX_PATH = Path("research_state/formula_index.json")


class ExplicitFormulaCheck(BaseModel):
    """`psi(x)` summed directly, against the corpus's formula over the zeros."""

    model_config = ConfigDict(extra="forbid")

    #: Points the formula was evaluated at. Offset from integers on purpose.
    points: list[float] = Field(default_factory=list)
    #: `psi(x)`, summed over prime powers.
    direct: list[float] = Field(default_factory=list)
    #: The corpus's right-hand side, with the sum truncated.
    reconstructed: list[float] = Field(default_factory=list)
    residuals: list[float] = Field(default_factory=list)
    zeros_used: int
    worst_residual: float
    worst_at: float
    #: Never rigorous. A truncated conditionally convergent sum in floating
    #: point says how well the record matches at these points, and no more.
    confidence: Confidence = Confidence.NUMERICAL

    @field_validator("confidence")
    @classmethod
    def _reject_rigorous_confidence(cls, value: Confidence) -> Confidence:
        if value in RIGOROUS:
            raise ValueError(
                f"an explicit-formula residual may not claim {value.value!r}: "
                "it is a truncated conditionally convergent sum evaluated in "
                "floating point at finitely many points"
            )
        return value


def chebyshev_psi(limit: float) -> float:
    """`psi(x) = sum_{p^k <= x} log p`, summed directly from the factorisation."""
    import sympy as sp

    total = 0.0
    for n in range(2, int(limit) + 1):
        factors = sp.factorint(n)
        if len(factors) == 1:
            total += float(np.log(min(factors)))
    return total


def psi_from_zeros(x: float, ordinates: np.ndarray) -> float:
    """The corpus's right-hand side, with the sum truncated by index."""
    rho = 0.5 + 1j * np.asarray(ordinates, dtype=float)
    oscillating = 2.0 * np.real(np.power(complex(x), rho) / rho).sum()
    return float(
        x - oscillating - np.log(2 * np.pi) - 0.5 * np.log(1.0 - x**-2.0)
    )


def recorded_equation():
    """The corpus's explicit formula, as the index stores it.

    From the `canonical` srepr rather than the printed `expression`: the latter
    is `lhs = rhs` with one equals sign, which is not an expression at all, and
    re-parsing a printed form is the second reading this repository keeps being
    bitten by. The srepr IS the parsed object.

    Raises rather than falling back on a copy: this check exists to test what
    the corpus says, so a corpus that no longer says it has nothing to test.
    """
    import sympy as sp

    from .functions import SREPR_NAMESPACE

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{INDEX_PATH} is missing; run `rhre symbolic ingest` first"
        )
    for record in json.loads(INDEX_PATH.read_text(encoding="utf-8")):
        canonical = record.get("canonical", "")
        if "ChebyshevPsi" in canonical and "NthZetaZero" in canonical:
            return sp.sympify(canonical, locals=dict(SREPR_NAMESPACE))
    raise LookupError(
        "the corpus no longer records an explicit formula relating "
        "ChebyshevPsi to the zeros; there is nothing here to check"
    )


def recorded_shape() -> dict[str, bool]:
    """Which terms the recorded right-hand side actually contains.

    `psi_from_zeros` implements a specific formula. This reports what the corpus
    holds, so a test can insist the two agree instead of trusting they do: a
    missing `log(2 pi)` would leave the residual a flat 1.8379 at every point,
    which reads as a bug in the summation rather than as a wrong record.

    CHECKED AS MATHEMATICS, NOT AS TEXT. The first version searched the printed
    form for "log(2*pi)" and reported it missing -- because SymPy canonicalises
    that to `log(pi) + log(2)`, so the string was absent while the term was
    there. A check on the notation is the exact failure this repository is
    built against, in a file written to catch it.
    """
    import sympy as sp

    equation = recorded_equation()
    x, k = sp.Symbol("x"), sp.Symbol("k")
    right = equation.rhs

    sums = [node for node in sp.preorder_traversal(right) if isinstance(node, sp.Sum)]
    over_zeros = any(
        any(
            type(inner).__name__ == "NthZetaZero"
            for inner in sp.preorder_traversal(term)
        )
        for term in sums
    )
    to_infinity = any(
        term.limits and term.limits[0][1] == 1 and term.limits[0][2] is sp.oo
        for term in sums
    )

    # Partition the additive terms three ways. The sum appears as -2*Sum(...),
    # so collecting the Sum NODES and subtracting them leaves the factor of -2
    # behind -- which is what made the first attempt report the half-log term
    # missing. The terms CONTAINING a sum are what has to come out.
    additive = sp.Add.make_args(right.doit(deep=False))
    with_sum = [term for term in additive if term.atoms(sp.Sum)]
    constant = sp.Add(
        *[
            term
            for term in additive
            if not term.has(x) and not term.has(k) and not term.atoms(sp.Sum)
        ]
    )
    rest = sp.Add(*[term for term in additive if term not in with_sum]) - constant
    half_log = -sp.log(1 - x**-2) / 2
    remainder = sp.simplify(rest - half_log)

    return {
        "constant_is_minus_log_two_pi": sp.simplify(
            constant + sp.log(2 * sp.pi)
        ) == 0,
        "has_zero_sum": over_zeros,
        "has_half_log_term": remainder == x,
        "sums_from_one_to_infinity": to_infinity,
    }


def check_explicit_formula(
    points: list[float] | None = None, *, zeros: int = 20000
) -> ExplicitFormulaCheck:
    """Compare `psi(x)` against the corpus's reconstruction from the zeros."""
    from .riemann_siegel import first_zero_ordinates

    shape = recorded_shape()
    missing = [name for name, present in shape.items() if not present]
    if missing:
        raise LookupError(
            f"the recorded explicit formula is missing {missing}; the "
            "reconstruction here implements the full one, so comparing them "
            "would test this file rather than the corpus"
        )

    # Offset from integers: psi jumps at every prime power and the formula
    # converges to the midpoint of the jump there.
    chosen = points or [10.5, 20.5, 50.5, 100.5, 200.5, 500.5]
    ordinates = first_zero_ordinates(zeros)

    direct, reconstructed, residuals = [], [], []
    for x in chosen:
        left = chebyshev_psi(x)
        right = psi_from_zeros(x, ordinates)
        direct.append(left)
        reconstructed.append(right)
        residuals.append(left - right)

    worst = max(range(len(residuals)), key=lambda i: abs(residuals[i]))
    return ExplicitFormulaCheck(
        points=list(chosen),
        direct=direct,
        reconstructed=reconstructed,
        residuals=residuals,
        zeros_used=len(ordinates),
        worst_residual=float(residuals[worst]),
        worst_at=float(chosen[worst]),
    )
