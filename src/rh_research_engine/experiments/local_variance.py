from __future__ import annotations

import math

import numpy as np

from ..core.models import ExperimentResult
from ..math.arithmetic import von_mangoldt_sieve
from ..math.correlation import gamma_shell


def _wave_from_lambda(lam: np.ndarray, X: float, q: float) -> float:
    n = np.arange(lam.size, dtype=float)
    valid = n >= 2
    u = np.zeros_like(n)
    u[valid] = n[valid] / X
    g = np.zeros_like(n)
    g[valid] = gamma_shell(u[valid], q)
    b = np.zeros_like(n)
    b[valid] = lam[valid] - 1.0
    return float(np.dot(b, g) / math.sqrt(X))


def run(X: int = 5_000, q: float = 4.0, width: float = 1.0, samples: int = 33) -> ExperimentResult:
    if width <= 0 or samples < 3:
        raise ValueError("width must be positive and samples >= 3")
    ts = np.linspace(math.log(X), math.log(X) + width, samples)
    xs = np.exp(ts)
    tail_power = 30.0
    support = tail_power ** (1.0 / q)
    n_max = int(math.ceil(float(xs[-1]) * support))
    lam = np.asarray(von_mangoldt_sieve(n_max), dtype=float)
    waves = np.array([_wave_from_lambda(lam, float(x), q) for x in xs])
    energy = float(np.trapezoid(waves * waves, ts))
    mean_energy = energy / width
    max_abs = float(np.max(np.abs(waves)))

    return ExperimentResult(
        name="local-window-variance",
        parameters={"X": X, "q": q, "width": width, "samples": samples},
        metrics={
            "window_energy": energy,
            "mean_energy": mean_energy,
            "max_abs_wave": max_abs,
            "wave_start": float(waves[0]),
            "wave_end": float(waves[-1]),
        },
        observations=[
            "Numerically approximates integral_T^{T+width} |W_q(t)|^2 dt.",
            "For fixed q, subexponential growth of this quantity is RH-equivalent at exponent level.",
            "Finite-scale boundedness is diagnostic only.",
        ],
    )
