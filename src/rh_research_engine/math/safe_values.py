from __future__ import annotations

import mpmath as mp


def centered_log_derivative(s: mp.mpf | mp.mpc) -> mp.mpf | mp.mpc:
    """B(s) = -zeta'/zeta(s) - zeta(s) + 1."""
    z = mp.zeta(s)
    zp = mp.diff(mp.zeta, s)
    return -zp / z - z + 1


def safe_c(j: int, dps: int = 50) -> mp.mpf:
    mp.mp.dps = dps
    return mp.re(centered_log_derivative(2 * j + 2))


def safe_binomial_d(k: int, dps: int = 80) -> mp.mpf:
    mp.mp.dps = dps
    return mp.fsum(
        [(-1) ** j * mp.binomial(k, j) * safe_c(j, dps=dps) for j in range(k + 1)]
    )


def corrected_d(k: int, dps: int = 80) -> mp.mpf:
    if k <= 0:
        raise ValueError("k must be >= 1")
    return safe_binomial_d(k, dps=dps) + mp.mpf(1) / (2 * k * (k + 1))


def implied_theta_from_decay(alpha: float) -> float:
    """If corrected d_k = O(k^{-alpha}), then Theta <= 2 - 2 alpha."""
    return 2.0 - 2.0 * alpha
