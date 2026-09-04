from __future__ import annotations

import math

from ..core.models import ExperimentResult
from ..math.safe_values import corrected_d


def run(k_max: int = 40, dps: int = 80) -> ExperimentResult:
    vals: list[tuple[int, float]] = []
    roots: list[float] = []
    for k in range(1, k_max + 1):
        v = float(corrected_d(k, dps=dps))
        vals.append((k, v))
        if v != 0.0:
            roots.append(abs(v) ** (1.0 / math.log(max(k, 2))))
    tail = vals[max(0, len(vals) - 10) :]
    scaled = [abs(v) * (k ** 0.75) for k, v in tail]
    return ExperimentResult(
        name="safe-binomial",
        parameters={"k_max": k_max, "dps": dps},
        metrics={
            "last_k": vals[-1][0],
            "last_corrected_d": vals[-1][1],
            "max_tail_k34_abs": max(scaled) if scaled else 0.0,
        },
        observations=[
            "RH endpoint predicts bounded k^(3/4) * corrected_d_k.",
            "This experiment is numerical only and does not certify asymptotic decay.",
        ],
    )
