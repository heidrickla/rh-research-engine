from __future__ import annotations

from dataclasses import dataclass

#: Theta = sup Re(rho) over nontrivial zeros. Zeta provably has zeros on the
#: critical line, so Theta >= 1/2 holds unconditionally. Any computed bound
#: below 1/2 is not a strong result -- it is proof that an input was invalid.
THETA_FLOOR = 0.5


@dataclass(frozen=True)
class CorrelationBoundImplication:
    remainder_exponent: float
    wave_energy_exponent: float
    theta_upper: float
    rh_endpoint: bool
    rigorous: bool


def correlation_remainder_to_theta(
    remainder_exponent: float, *, rigorous: bool = False
) -> CorrelationBoundImplication:
    """Translate a screening-remainder exponent into a zero-edge bound.

    Assumes the deterministic Hardy--Littlewood model contribution is O_q(1)
    for one fixed q and the actual localized wave energy is therefore
    O(X^theta). Since an off-line zero with eta = Re(rho)-1/2 creates energy
    X^(2 eta), theta implies eta <= theta/2.

    This function is only algebraic bookkeeping; it does not certify that the
    input bound has been proved. ``rigorous`` records whether the caller is
    supplying a proved estimate: only then can the result be described as
    reaching the RH endpoint. A fitted log-log slope is not a proved estimate.

    A negative exponent is rejected rather than clamped. Clamping silently
    rewrote noise-driven fits -- which routinely go negative on finite ranges --
    into exactly the RH endpoint.
    """
    if remainder_exponent != remainder_exponent:  # NaN
        raise ValueError("remainder exponent must be a real number, got NaN")
    if remainder_exponent < 0:
        raise ValueError(
            f"remainder exponent {remainder_exponent} is negative and therefore out of domain. "
            "A negative screening exponent is not a stronger bound; it means the fit or "
            "derivation that produced it is invalid."
        )
    theta_upper = min(1.0, THETA_FLOOR + remainder_exponent / 2.0)
    return CorrelationBoundImplication(
        remainder_exponent=remainder_exponent,
        wave_energy_exponent=remainder_exponent,
        theta_upper=theta_upper,
        rh_endpoint=rigorous and remainder_exponent == 0.0,
        rigorous=rigorous,
    )
