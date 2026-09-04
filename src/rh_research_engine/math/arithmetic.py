from __future__ import annotations

import math


def von_mangoldt_sieve(n_max: int) -> list[float]:
    """Return Lambda(n) for n=0..n_max using prime powers."""
    lam = [0.0] * (n_max + 1)
    is_prime = bytearray(b"\x01") * (n_max + 1)
    if n_max >= 0:
        is_prime[0] = 0
    if n_max >= 1:
        is_prime[1] = 0
    for p in range(2, n_max + 1):
        if is_prime[p]:
            logp = math.log(p)
            power = p
            while power <= n_max:
                lam[power] = logp
                if power > n_max // p:
                    break
                power *= p
            step_start = p * p
            if step_start <= n_max:
                is_prime[step_start : n_max + 1 : p] = b"\x00" * (((n_max - step_start) // p) + 1)
    return lam


def psi_from_lambda(lam: list[float]) -> list[float]:
    out = [0.0] * len(lam)
    total = 0.0
    for n in range(1, len(lam)):
        total += lam[n]
        out[n] = total
    return out
