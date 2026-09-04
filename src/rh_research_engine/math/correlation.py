from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .arithmetic import von_mangoldt_sieve

# Twin-prime constant C2 = prod_{p>2} (1 - 1/(p-1)^2)
# Stored to enough precision for numerical research experiments.
TWIN_PRIME_CONSTANT = 0.6601618158468695739278121100145557784326233602847334


def gamma_shell(u: np.ndarray | float, q: float) -> np.ndarray | float:
    """Localized shell kernel g_q(u) = q u^q exp(-u^q)."""
    if q <= 0:
        raise ValueError("q must be positive")
    arr = np.asarray(u, dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        uq = np.power(arr, q)
        out = q * uq * np.exp(-uq)
    if np.ndim(u) == 0:
        return float(out)
    return out


def singular_series_sieve(h_max: int) -> np.ndarray:
    """Hardy--Littlewood prime-pair singular series S(h), h=0..h_max.

    For h != 0:
      S(h) = 0 for odd h,
      S(h) = 2 C2 prod_{p|h,p>2} (p-1)/(p-2) for even h.

    S(0) is deliberately NaN: the diagonal is handled separately.
    """
    if h_max < 0:
        raise ValueError("h_max must be nonnegative")
    out = np.zeros(h_max + 1, dtype=float)
    if h_max >= 0:
        out[0] = np.nan
    if h_max < 2:
        return out

    # Multiplicative correction initialized on even shifts.
    corr = np.ones(h_max + 1, dtype=float)
    # Basic prime sieve.
    is_prime = np.ones(h_max + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, h_max + 1):
        if not is_prime[p]:
            continue
        if p * p <= h_max:
            is_prime[p * p : h_max + 1 : p] = False
        if p > 2:
            factor = (p - 1.0) / (p - 2.0)
            corr[p : h_max + 1 : p] *= factor

    even = np.arange(2, h_max + 1, 2)
    out[even] = 2.0 * TWIN_PRIME_CONSTANT * corr[even]
    return out


def autocorrelation_fft(values: np.ndarray) -> np.ndarray:
    """Return r[h] = sum_i values[i] values[i+h], h >= 0."""
    x = np.asarray(values, dtype=float)
    n = x.size
    if n == 0:
        return np.array([], dtype=float)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, size)
    corr = np.fft.irfft(f * np.conjugate(f), size)[:n]
    # Numerical FFT noise can leave ~1e-15 errors.
    return corr


@dataclass(frozen=True)
class CorrelationBreakdown:
    X: int
    q: float
    n_max: int
    h_max: int
    wave: float
    total_energy: float
    diagonal_energy: float
    actual_offdiag: float
    model_offdiag: float
    hl_model_energy: float
    screening_remainder: float
    diagonal_coefficient: float
    asymptotic_diagonal_coefficient: float


def shell_correlation_breakdown(
    X: int,
    q: float,
    *,
    tail_power: float = 30.0,
    h_max: int | None = None,
) -> CorrelationBreakdown:
    """Compute actual vs Hardy--Littlewood shell screening at scale X.

    W_q(log X) = X^{-1/2} sum_n b_n g_q(n/X), b_n = Lambda(n)-1.

    The squared wave is decomposed into a diagonal term and ordered
    off-diagonal shift correlations.  The HL model replaces b_n b_{n+h}
    by S(h)-1.  This is a diagnostic experiment, not a rigorous error bound.
    """
    if X < 2:
        raise ValueError("X must be at least 2")
    if q <= 0:
        raise ValueError("q must be positive")

    # Choose n_max so exp(-(n/X)^q) is roughly exp(-tail_power).
    support_factor = tail_power ** (1.0 / q)
    n_max = max(X + 16, int(math.ceil(X * support_factor)))
    lam = np.asarray(von_mangoldt_sieve(n_max), dtype=float)
    n = np.arange(0, n_max + 1, dtype=float)
    valid = n >= 2
    u = np.zeros_like(n)
    u[valid] = n[valid] / float(X)
    g = np.zeros_like(n)
    g[valid] = gamma_shell(u[valid], q)
    b = np.zeros_like(n)
    b[valid] = lam[valid] - 1.0

    a = b * g
    # Ignore n=0,1 naturally through zero weights.
    #
    # `math.fsum`, not `np.sum`. These reductions sum thousands of mixed-sign
    # terms to a result far smaller than the terms, so the rounding of the
    # summation ORDER survives into the metric -- and numpy's pairwise order
    # depends on SIMD width, which differs between machines. Two CPUs then
    # produced different metrics for the same computation, and every stored
    # hash of them disagreed. fsum is correctly rounded and order-independent,
    # so the result is the same everywhere.
    wave_numer = math.fsum(a.tolist())
    wave = wave_numer / math.sqrt(X)
    total_energy = wave * wave
    diagonal_energy = math.fsum((a * a).tolist()) / X

    actual_corr = autocorrelation_fft(a)
    weight_corr = autocorrelation_fft(g)

    if h_max is None:
        # The multiplicative window has additive scale X/q for q >= 1;
        # for small q it is broad, so use full effective support.
        if q >= 1.0:
            h_max = min(n_max - 1, max(32, int(math.ceil(12.0 * X / q))))
        else:
            h_max = n_max - 1
    h_max = min(max(1, h_max), n_max - 1)

    actual_offdiag = (2.0 / X) * math.fsum(actual_corr[1 : h_max + 1].tolist())

    singular = singular_series_sieve(h_max)
    model_cov = singular[1:] - 1.0
    model_offdiag = (2.0 / X) * math.fsum(
        (model_cov * weight_corr[1 : h_max + 1]).tolist()
    )

    # If truncated h_max misses some actual off-diagonal mass, report the
    # total energy independently.  screening_remainder is on the same h range.
    screening_remainder = actual_offdiag - model_offdiag
    hl_model_energy = diagonal_energy + model_offdiag

    power = 2.0 + 1.0 / q
    diagonal_coefficient = q * math.gamma(power) / (2.0**power)

    return CorrelationBreakdown(
        X=X,
        q=q,
        n_max=n_max,
        h_max=h_max,
        wave=wave,
        total_energy=total_energy,
        diagonal_energy=diagonal_energy,
        actual_offdiag=actual_offdiag,
        model_offdiag=model_offdiag,
        hl_model_energy=hl_model_energy,
        screening_remainder=screening_remainder,
        diagonal_coefficient=diagonal_coefficient,
        asymptotic_diagonal_coefficient=q / 4.0,
    )
