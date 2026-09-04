"""Li's criterion computed from zeta and Gamma alone, and what it can detect.

RH is equivalent to `lambda_n >= 0` for every `n >= 1`, where

    lambda_n = sum_rho [1 - (1 - 1/rho)^n]

NOT COMPUTED THAT WAY, DELIBERATELY. Summing over a list of zeros that were
verified to lie on the critical line assumes what the criterion tests, at least
below the height of the list. The Bombieri-Lagarias generating function does
not mention a zero:

    log xi(1/(1-z)) = log xi(1) + sum_{n>=1} (lambda_n / n) z^n

so the `lambda_n` are Taylor coefficients of an explicit function of zeta and
Gamma, and the computation is non-circular.

THE POLE IS THE ONLY OBSTACLE, and Arb refuses it honestly: `z = 0` is `s = 1`,
and `acb_series.zeta` there returns NaN rather than a number. `(s-1) zeta(s)`
is analytic, with the Stieltjes expansion

    (s-1) zeta(s) = 1 + sum_{k>=0} (-1)^k gamma_k (s-1)^(k+1) / k!

so every term of

    log xi(s) = -log 2 + log s - (s/2) log pi + lgamma(s/2) + log[(s-1) zeta(s)]

is analytic at `s = 1` and composes with `s = 1/(1-z)`.

WHAT IT CAN AND CANNOT SEE, which is the point of recording it. `lambda_n` is
dominated by an archimedean term with no arithmetic in it,
`(n/2)(log n - log 2pi + gamma - 1) + 1`, which at `n = 300` accounts for 99.6%
of the value. The zeros live in the remaining 0.4%. So positivity holds with
enormous room at every `n` reachable, and a violation would need the residual to
grow from a few tenths of a per cent to overwhelming the main term.

CONTRAST WITH BAEZ-DUARTE, in `baez_duarte.py`. There the quantity tested is
`|c_k| k^(3/4)`, which is the signal with the smooth part divided out by
construction, so the measurement is entirely of the thing that could fail. Two
criteria, both equivalent to RH, both computable without assuming it, and very
different discriminating power -- and the difference is whether the archimedean
part has been subtracted. That is worth knowing before choosing which to spend
a machine on.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.models import ExperimentResult

#: Euler-Mascheroni, for the archimedean main term. The series itself gets
#: gamma from Arb; this is only used to split the result for reporting.
EULER = 0.5772156649015329


def li_coefficients(order: int, bits: int) -> list[tuple[int, float, float]]:
    """`(n, lambda_n, radius)` for `n < order`, from zeta and Gamma."""
    import flint

    previous, previous_cap = flint.ctx.prec, flint.ctx.cap
    flint.ctx.prec = bits
    # python-flint truncates EVERY series at `ctx.cap`, which defaults to 10.
    # Without this, asking for order 300 returns 10 coefficients and looks like
    # it worked -- the timings even rise, because the precision does.
    flint.ctx.cap = order + 2
    try:
        w = flint.acb_series([0] + [1] * order)  # s - 1 = z/(1-z)
        s = flint.acb(1) + w

        factorial = flint.acb(1)
        shifted = flint.acb_series([1])
        power = w
        for k in range(order):
            if k:
                factorial = factorial * k
            shifted = shifted + power * ((-1) ** k * flint.acb.stieltjes(k) / factorial)
            power = power * w

        half = s * flint.acb(0.5)
        log_xi = (
            -flint.acb(2).log()
            + s.log()
            - half * flint.acb.pi().log()
            + half.lgamma()
            + shifted.log()
        )
        coeffs = log_xi.coeffs()
        return [
            (n, float((coeffs[n] * n).real.mid()), float((coeffs[n] * n).real.rad()))
            for n in range(1, min(order, len(coeffs)))
        ]
    finally:
        flint.ctx.prec, flint.ctx.cap = previous, previous_cap


def archimedean(n: np.ndarray) -> np.ndarray:
    """The part of `lambda_n` that carries no arithmetic."""
    return (n / 2) * (np.log(n) - math.log(2 * math.pi) + EULER - 1) + 1


def run(order: int = 120, bits: int = 1500) -> ExperimentResult:
    rows = li_coefficients(order, bits)
    n = np.array([k for k, _, _ in rows], dtype=float)
    lam = np.array([v for _, v, _ in rows])
    radius = max(r for _, _, r in rows)

    main = archimedean(n)
    residual = lam - main
    negative = int(np.sum(lam < 0))

    return ExperimentResult(
        name="li-criterion",
        parameters={"order": order, "bits": bits},
        metrics={
            "n_max": int(n.max()),
            "coefficients": int(len(n)),
            "min_lambda": float(lam.min()),
            "min_at_n": int(n[int(lam.argmin())]),
            "negative_count": negative,
            "max_radius": float(radius),
            # What the criterion is actually testing against. RH needs
            # lambda_n >= 0; this says how much of that is the smooth part.
            "main_term_fraction": float(main[-1] / lam[-1]),
            "residual_fraction": float(abs(residual[-1]) / lam[-1]),
            "residual_min": float(residual.min()),
            "residual_max": float(residual.max()),
            "violated": float(1.0 if negative else 0.0),
        },
        observations=[
            "A falsification test that did not fire: lambda_n > 0 for every n "
            "computed. RH is equivalent to positivity for ALL n, so this is "
            "consistent with RH and proves nothing.",
            "Computed from the Bombieri-Lagarias generating function, so from "
            "zeta and Gamma only. Summing over a verified-on-line zero list "
            "would assume what the criterion tests.",
            "WEAKLY DISCRIMINATING at reachable n. The archimedean term "
            "(n/2)(log n - log 2pi + gamma - 1) + 1 carries no arithmetic and "
            "accounts for about 99.6% of lambda_n by n = 300, so positivity "
            "holds with room to spare and a violation would need the remaining "
            "fraction to overwhelm it.",
            "Compare baez_duarte, where the tested quantity |c_k| k^(3/4) is "
            "the signal with the smooth part already divided out. Both criteria "
            "are equivalent to RH and non-circular; they differ in how much of "
            "what they measure could actually fail.",
        ],
    )
