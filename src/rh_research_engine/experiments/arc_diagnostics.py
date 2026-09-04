from __future__ import annotations

import math

import numpy as np

from ..core.models import ExperimentResult
from ..math.arithmetic import von_mangoldt_sieve
from ..math.correlation import gamma_shell


def run(
    X: int = 8_000,
    q: float = 4.0,
    fft_size: int = 65_536,
    major_width: float = 0.02,
) -> ExperimentResult:
    if not (0 < major_width < 0.5):
        raise ValueError("major_width must lie in (0, 0.5)")
    tail_power = 30.0
    n_max = int(math.ceil(X * tail_power ** (1.0 / q)))
    lam = np.asarray(von_mangoldt_sieve(n_max), dtype=float)
    n = np.arange(n_max + 1, dtype=float)
    valid = n >= 2
    g = np.zeros_like(n)
    g[valid] = gamma_shell(n[valid] / X, q)
    a = np.zeros_like(n)
    a[valid] = (lam[valid] - 1.0) * g[valid] / math.sqrt(X)

    size = max(fft_size, 1 << (a.size - 1).bit_length())
    spec = np.fft.fft(a, size)
    power = np.abs(spec) ** 2 / size
    freqs = np.fft.fftfreq(size)

    # This is an exploratory major/minor split around low additive frequencies.
    # It is intentionally not called the circle-method major arcs theorem.
    major = np.abs(freqs) <= major_width
    total = float(np.sum(power))
    major_power = float(np.sum(power[major]))
    minor_power = total - major_power

    return ExperimentResult(
        name="arc-diagnostics",
        parameters={"X": X, "q": q, "fft_size": size, "major_width": major_width},
        metrics={
            "total_fourier_power": total,
            "low_frequency_power": major_power,
            "high_frequency_power": minor_power,
            "low_frequency_fraction": major_power / total if total else 0.0,
            "high_frequency_fraction": minor_power / total if total else 0.0,
        },
        observations=[
            "Exploratory Fourier decomposition of the centered, smoothed prime shell.",
            "The low-frequency band is a diagnostic proxy only; rigorous circle-method major arcs require rational neighborhoods and analytic estimates.",
            "Use this experiment to identify where residual power lives before attempting a rigorous major/minor-arc proof decomposition.",
        ],
    )
