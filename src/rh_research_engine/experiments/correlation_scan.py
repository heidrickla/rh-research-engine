"""Log-log slopes of the shell correlation, and the Theta they were mapped to.

TWO IMPLEMENTATIONS OF ONE MAP, AND THE SILENT ONE WROTE THE RECORD. This
computed `theta = 0.5 + max(0.0, energy_slope) / 2.0` inline. The same map lives
in `symbolic.exponents.screening_remainder_to_theta`, which REFUSES a negative
exponent -- its docstring says such a value "is not a stronger result, it is a
statement that is provably false". The inline copy replaced that refusal with a
clamp, so every negative slope came out at exactly 0.5, the RH endpoint.

Both recorded runs had negative slopes, -0.3216 and -0.2422, and both recorded
`heuristic_theta_from_energy: 0.5`. A reader sees Theta = 0.5 beside a measured
slope and reads it as the data landing on RH. It is the clamp. Every run ever
recorded took the branch the shared function exists to refuse.

That is this repository's one-name-resolution rule in numeric form: a quantity
computed under two policies is two different quantities, and nothing downstream
can tell which one it is reading.

AND THE SLOPE ITSELF WAS A FACT ABOUT THE GRID. `total_energy` is a cancellation
residual -- diagonal 6.5874541 against off-diagonal -6.5871135, about 4.3 digits
gone -- and it dips to 3.8e-10 against a typical 1e-3 at arbitrary X, WITHOUT
CHANGING SIGN. One point seven decades low owns a four-point log-log fit. Across
twelve grids differing only in phase inside the same range, `energy_loglog_slope`
ran -4.319 to +7.885 against a recorded -0.3216 that carried no error bar.

`remainder_loglog_slope` survives this treatment: -0.0093 +/- 0.0014 over seven
phases. It is the one real number of the three, and the contrast is why the
others are reported as unreadable rather than quietly dropped.

The cancellation-minimum diagnosis, and the dynamic-range predicate that detects
it where a sign-change count cannot, came from the rh-research-engine-da session.
"""

from __future__ import annotations

import math

from ..core.models import ExperimentResult
from ..math.correlation import shell_correlation_breakdown
from ..math.loglog import LogLogSummary, geometric_phases, summarize
from ..symbolic.exponents import ImpossibleBoundError, screening_remainder_to_theta

#: Grids differing only in phase across the same range.
GRID_PHASES = 6
#: Theta <= 1 holds unconditionally, and the inline map had no ceiling: phase
#: noise pushing the slope positive produced 2.789 and 4.442 in testing.
THETA_CEILING = 1.0


def _describe(name: str, fit: LogLogSummary) -> list[str]:
    reasons = fit.unreadable_because
    if not reasons:
        return [
            f"{name} = {fit.slope:.4f}, grid-phase spread {fit.spread:.4f} over "
            f"{fit.phases} grids ({fit.lowest:.4f} to {fit.highest:.4f}). Readable: "
            "no sample sits far below its neighbours and the phase barely moves it."
        ]
    return [
        f"{name} = {fit.slope:.4f} is UNREADABLE and must not be quoted: "
        + "; ".join(reasons)
    ]


def _theta_from(slope: float) -> tuple[float | None, str]:
    """The shared map, with its refusal intact. Never a clamp.

    Returning the refusal as text rather than a number is the whole point: a
    negative exponent has no Theta, and 0.5 is not the nearest available answer
    to "no answer" -- it is the answer that looks most like a result.
    """
    try:
        implication = screening_remainder_to_theta(slope)
    except ImpossibleBoundError as error:
        return None, f"REFUSED by screening_remainder_to_theta: {error}"
    if implication.theta_upper > THETA_CEILING:
        return None, (
            f"REFUSED: exponent {slope:.6g} implies Theta <= "
            f"{implication.theta_upper:.6g}, above the unconditional ceiling of 1"
        )
    return implication.theta_upper, implication.derivation


def run(
    X_min: int = 2_000,
    X_max: int = 50_000,
    points: int = 7,
    q: float = 4.0,
    h_factor: float = 12.0,
    phases: int = GRID_PHASES,
) -> ExperimentResult:
    if points < 3:
        raise ValueError("points must be at least 3")

    grids = geometric_phases(float(X_min), float(X_max), points, phases, integral=True)
    remainder_grids, energy_grids, model_grids = [], [], []
    observations: list[str] = []
    for index, xs in enumerate(grids):
        remainders, energies, models = [], [], []
        for x in xs:
            scale = int(x)
            h_max = max(32, int(math.ceil(h_factor * scale / q))) if q >= 1 else None
            breakdown = shell_correlation_breakdown(X=scale, q=q, h_max=h_max)
            remainders.append(breakdown.screening_remainder)
            energies.append(breakdown.total_energy)
            models.append(breakdown.hl_model_energy)
        remainder_grids.append((xs, remainders))
        energy_grids.append((xs, energies))
        model_grids.append((xs, models))
        if index == 0:
            observations.extend(
                f"X={int(x)}: energy={e:.6g}, HL-model={m:.6g}, remainder={r:.6g}"
                for x, e, m, r in zip(xs, energies, models, remainders, strict=True)
            )

    remainder = summarize(remainder_grids)
    energy = summarize(energy_grids)
    model = summarize(model_grids)

    # AN UNREADABLE SLOPE HAS NO THETA. The first repair here let the map run
    # anyway and printed the result beside a note saying it was not a bound --
    # which puts a number in `metrics`, and `metrics` is what anything
    # downstream reads. Refusing on the readability of the input is the same
    # decision the shared map makes about its domain, one step earlier.
    energy_readable = not energy.unreadable_because
    if energy_readable:
        theta, derivation = _theta_from(energy.slope)
    else:
        theta, derivation = None, (
            "REFUSED: the energy slope is unreadable, so it is not an exponent to map. "
            + "; ".join(energy.unreadable_because)
        )

    observations.extend(_describe("remainder_loglog_slope", remainder))
    observations.extend(_describe("energy_loglog_slope", energy))
    observations.extend(_describe("hl_model_loglog_slope", model))
    observations.append(
        f"Theta from the energy slope: {derivation}"
        if theta is None
        else f"Theta <= {theta:.6f} from the energy slope, via {derivation}"
    )
    observations.extend(
        [
            "Log-log slopes are finite-range diagnostics, not proof exponents.",
            "The heuristic Theta value must never be promoted to a claim without a "
            "rigorous uniform bound.",
        ]
    )

    metrics: dict[str, float | int | str | bool] = {
        "q": q,
        "scales": len(grids[0]),
        "phases": phases,
        "remainder_loglog_slope": remainder.slope,
        "remainder_slope_spread": remainder.spread,
        "remainder_dynamic_range": remainder.dynamic_range,
        "energy_loglog_slope": energy.slope,
        "energy_slope_spread": energy.spread,
        "energy_dynamic_range": energy.dynamic_range,
        "hl_model_loglog_slope": model.slope,
        "hl_model_slope_spread": model.spread,
        "unreadable_slopes": float(
            sum(
                1
                for fit in (remainder, energy, model)
                if fit.unreadable_because
            )
        ),
        "points_dropped": float(max(remainder.dropped, energy.dropped, model.dropped)),
        "theta_refused": float(1.0 if theta is None else 0.0),
    }
    # Omitted rather than written as NaN: NaN serialises to JSON null and then
    # fails validation on the way back in, so the record could not be read.
    if theta is not None:
        metrics["heuristic_theta_from_energy"] = theta

    return ExperimentResult(
        name="correlation-scan",
        parameters={
            "X_min": X_min,
            "X_max": X_max,
            "points": points,
            "q": q,
            "h_factor": h_factor,
            "phases": phases,
        },
        metrics=metrics,
        observations=observations,
    )
