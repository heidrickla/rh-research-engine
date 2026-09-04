from __future__ import annotations

from ..core.models import ExperimentResult
from ..math.filters import localized_gamma_filter


def run(x: float = 1000.0, q: float = 2.0) -> ExperimentResult:
    value = localized_gamma_filter(x=x, q=q)
    normalized = value / (x**0.5)
    return ExperimentResult(
        name="gamma-filter",
        parameters={"x": x, "q": q},
        metrics={"value": value, "x_minus_half_normalized": normalized},
        observations=[
            "RH predicts O_q(X^(1/2)) for fixed q.",
            "Single-X values are diagnostic only; exponent fitting needs a scale sweep.",
        ],
    )
