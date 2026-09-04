from __future__ import annotations

from enum import StrEnum
from fractions import Fraction

from .models import BigRational, RealInterval


class TruthValue(StrEnum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


ZERO = BigRational(numerator=0)


def definitely_positive(interval: RealInterval) -> TruthValue:
    if interval.lower > ZERO:
        return TruthValue.TRUE
    if interval.upper <= ZERO:
        return TruthValue.FALSE
    return TruthValue.UNKNOWN


def definitely_negative(interval: RealInterval) -> TruthValue:
    if interval.upper < ZERO:
        return TruthValue.TRUE
    if interval.lower >= ZERO:
        return TruthValue.FALSE
    return TruthValue.UNKNOWN


def contains_zero(interval: RealInterval) -> TruthValue:
    if interval.lower <= ZERO and interval.upper >= ZERO:
        return TruthValue.TRUE
    return TruthValue.FALSE


def strictly_less_than(left: RealInterval, right: RealInterval) -> TruthValue:
    if left.upper < right.lower:
        return TruthValue.TRUE
    if left.lower >= right.upper:
        return TruthValue.FALSE
    return TruthValue.UNKNOWN


def interval_add(left: RealInterval, right: RealInterval) -> RealInterval:
    return RealInterval(
        lower=_from_fraction(left.lower.as_fraction() + right.lower.as_fraction()),
        upper=_from_fraction(left.upper.as_fraction() + right.upper.as_fraction()),
    )


def interval_scale(interval: RealInterval, factor: BigRational) -> RealInterval:
    f = factor.as_fraction()
    a = interval.lower.as_fraction() * f
    b = interval.upper.as_fraction() * f
    low, high = (a, b) if a <= b else (b, a)
    return RealInterval(lower=_from_fraction(low), upper=_from_fraction(high))


def _from_fraction(value: Fraction) -> BigRational:
    return BigRational(numerator=value.numerator, denominator=value.denominator)
