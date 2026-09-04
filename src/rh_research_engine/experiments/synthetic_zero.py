from __future__ import annotations

import math

import mpmath as mp
import numpy as np

from ..core.models import ExperimentResult


def run(
    eta: float = 0.02,
    gamma: float = 14.134725141734693,
    q: float = 20.0,
    T_min: float = 0.0,
    T_max: float = 120.0,
    points: int = 1000,
) -> ExperimentResult:
    if eta < 0 or q <= 0 or points < 10:
        raise ValueError("eta >= 0, q > 0, points >= 10 required")
    rho = 0.5 + eta + 1j * gamma
    coeff = complex(mp.gamma(1 + rho / q))
    ts = np.linspace(T_min, T_max, points)
    wave = 2.0 * np.real(coeff * np.exp((eta + 1j * gamma) * ts))
    envelope_energy = np.maximum(wave * wave, 1e-300)

    # Recover the exponential energy slope robustly from block RMS values.
    blocks = 20
    edges = np.linspace(0, points, blocks + 1, dtype=int)
    mids: list[float] = []
    rms2: list[float] = []
    for i in range(blocks):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        mids.append(float(np.mean(ts[lo:hi])))
        rms2.append(float(np.mean(envelope_energy[lo:hi])))
    slope = float(np.polyfit(np.asarray(mids), np.log(np.asarray(rms2)), 1)[0])
    recovered_eta = slope / 2.0

    detect_threshold = 1.0
    amp0 = max(abs(coeff), 1e-300)
    predicted_detect_T = 0.0 if eta == 0 else max(0.0, math.log(detect_threshold / amp0) / eta)

    return ExperimentResult(
        name="synthetic-zero-injection",
        parameters={
            "eta": eta,
            "gamma": gamma,
            "q": q,
            "T_min": T_min,
            "T_max": T_max,
            "points": points,
        },
        metrics={
            "gamma_filter_amplitude": float(abs(coeff)),
            "predicted_energy_slope": 2.0 * eta,
            "recovered_energy_slope": slope,
            "recovered_eta": recovered_eta,
            "predicted_unit_amplitude_T": predicted_detect_T,
        },
        observations=[
            "Injects a hypothetical off-line zero into the exact Gamma-filter response.",
            "Used to validate detection/scoring code and to calibrate finite-window exponent recovery.",
            "Synthetic injections are counterfactual tests and are never evidence about actual zeta zeros.",
        ],
    )
