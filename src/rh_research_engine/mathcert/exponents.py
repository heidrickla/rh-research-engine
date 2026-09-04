from __future__ import annotations

from .models import BigRational, RealInterval
from .predicates import interval_add, interval_scale

HALF = BigRational(numerator=1, denominator=2)
ZERO = BigRational(numerator=0)


class ImpossibleIntervalError(ValueError):
    """Raised when a certified exponent maps to a Theta interval below 1/2."""


def screening_exponent_to_theta(exponent: RealInterval) -> RealInterval:
    """Map a certified R_q(X)=O(X^theta) exponent interval to Theta <= 1/2 + theta/2.

    A negative exponent is out of domain, not a stronger result: Theta >= 1/2
    unconditionally because zeta has zeros on the critical line, so a certified
    interval implying anything smaller means the enclosure being propagated is
    not a screening exponent.
    """
    if exponent.lower < ZERO:
        raise ImpossibleIntervalError(
            f"screening exponent interval [{exponent.lower.as_fraction()}, "
            f"{exponent.upper.as_fraction()}] extends below 0 and is out of domain; "
            "it would imply Theta < 1/2, which is impossible"
        )
    half_exp = interval_scale(exponent, HALF)
    half_point = RealInterval(lower=HALF, upper=HALF)
    result = interval_add(half_point, half_exp)
    if result.lower < HALF:
        raise ImpossibleIntervalError(
            f"propagated Theta interval lower endpoint {result.lower.as_fraction()} is below 1/2"
        )
    return result
