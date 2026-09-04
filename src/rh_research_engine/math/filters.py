from __future__ import annotations

import math

from .arithmetic import von_mangoldt_sieve


def localized_gamma_filter(x: float, q: float, n_max: int | None = None) -> float:
    """S_q(X) using a finite cutoff chosen from the exponential tail."""
    if x <= 0 or q <= 0:
        raise ValueError("x and q must be positive")
    if n_max is None:
        # exp(-(n/x)^q) is negligible when (n/x)^q ~ 35.
        n_max = max(10, int(math.ceil(x * (35.0 ** (1.0 / q)))))
    lam = von_mangoldt_sieve(n_max)
    total = 0.0
    for n in range(2, n_max + 1):
        u = n / x
        total += (lam[n] - 1.0) * q * (u**q) * math.exp(-(u**q))
    return total


def normalized_wave(t: float, q: float, n_max: int | None = None) -> float:
    x = math.exp(t)
    return math.exp(-t / 2.0) * localized_gamma_filter(x=x, q=q, n_max=n_max)


def exact_pair_kernel(m: float, n: float, q: float, delta: float = 0.0) -> float:
    from math import cosh, gamma, log

    power = 2.0 + (1.0 + delta) / q
    c = q * gamma(power) / (2.0**power)
    return (
        c
        * (m * n) ** (-(1.0 + delta) / 2.0)
        * cosh((q / 2.0) * log(m / n)) ** (-power)
    )
