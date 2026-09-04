from __future__ import annotations

from pydantic import BaseModel

from ..core.bounds import THETA_FLOOR


class ImpossibleBoundError(ValueError):
    """Raised when an exponent maps to a Theta bound below 1/2.

    Theta >= 1/2 holds unconditionally, so a smaller value is not a stronger
    result -- it is a statement that is provably false, and it means the input
    exponent was outside the domain where the map is meaningful.
    """


class ExponentImplication(BaseModel):
    input_exponent: float
    theta_upper: float
    derivation: str


def _guard(theta_upper: float, *, source: str, value: float) -> float:
    if theta_upper != theta_upper:
        raise ImpossibleBoundError(f"{source}({value}) produced NaN")
    if theta_upper < THETA_FLOOR:
        raise ImpossibleBoundError(
            f"{source}({value}) implies Theta <= {theta_upper:g}, which is below 1/2 and "
            "therefore impossible: zeta has zeros on the critical line. The input exponent "
            "is out of domain."
        )
    return theta_upper


def screening_remainder_to_theta(theta: float) -> ExponentImplication:
    if theta < 0:
        raise ImpossibleBoundError(
            f"screening remainder exponent {theta} is negative and out of domain"
        )
    upper = _guard(0.5 + theta / 2.0, source="screening_remainder_to_theta", value=theta)
    return ExponentImplication(
        input_exponent=theta, theta_upper=upper, derivation="2*eta <= theta; Theta=1/2+eta"
    )


def safe_binomial_decay_to_theta(alpha: float) -> ExponentImplication:
    upper = _guard(2.0 - 2.0 * alpha, source="safe_binomial_decay_to_theta", value=alpha)
    return ExponentImplication(
        input_exponent=alpha,
        theta_upper=upper,
        derivation="rho-mode exponent is Re(rho)/2-1 <= -alpha",
    )
