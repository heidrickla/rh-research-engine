"""The fitted exponent moved by 0.85 when the grid moved, and said so nowhere.

`S_q(X)` oscillates and changes sign. The fit is ordinary least squares of
`log|S|` against `log x`, and near a sign change `log|S|` dives towards minus
infinity -- so a grid point that happens to land close to one dominates the
slope. Over ten grids differing only in phase, inside the SAME range, this
returned exponents from 0.214 to 1.061. The recorded value was one of those
draws, carrying no error bar, beside an observation reading "a rigorous bound
S_q(X) << X^theta would imply Re(rho) <= theta".

That juxtaposition is the danger, not the imprecision. Every draw below 1/2 --
and the default was 0.316 -- reads as evidence for something known to be false,
since the zeros sit on Re = 1/2. The number was not merely uncertain; it was a
fact about where the sample points fell, presented as a fact about zeta.

SO THE GRID PHASE IS THE ERROR BAR. The range and the point count are already
arguments; the phase was the one nobody chose, and it turned out to matter most.
Averaging over phases and reporting the spread is not smoothing -- it is
measuring the thing that made the single fit meaningless.

AND SIGN CHANGES ARE NOT THE ONLY WAY THIS DIES. Where they occur an envelope
exponent cannot be read off `log|S|` at all -- there are two or three among ten
points at the defaults, which is the whole reason the phase matters. But a
sign-change count reports clean on the worse case: a CANCELLATION MINIMUM, where
|y| dips decades below its neighbours without ever crossing zero, and one sample
owns the fit. `correlation_scan`'s energy has exactly that and no sign changes
at all. The predicate that catches both is `min|y| / median|y|`, and it now
lives in `math/loglog.py` so all three scans ask the question one way.

The signal is injectable, which is what lets the controls exist: a pure power
must come back with its own exponent and near-zero spread, and an oscillation
with no trend must come back with a spread that covers zero. An estimator that
has not been shown to return nothing on nothing is not an estimator.
"""

from __future__ import annotations

from collections.abc import Callable

from ..core.models import ExperimentResult
from ..math.filters import localized_gamma_filter
from ..math.loglog import geometric_phases, summarize

#: Grids differing only in phase, across the same range. Eight is enough to see
#: a spread of 0.25 stand out against a mean of 0.53; it is not a sample whose
#: standard error means anything, and the spread is reported as a spread.
GRID_PHASES = 8


def run(
    x_min: float = 100.0,
    x_max: float = 3000.0,
    points: int = 10,
    q: float = 2.0,
    phases: int = GRID_PHASES,
    signal: Callable[[float], float] | None = None,
) -> ExperimentResult:
    if points < 3:
        raise ValueError("points must be >= 3")
    if signal is None:

        def signal(x: float) -> float:
            return localized_gamma_filter(x=x, q=q)

    grids = geometric_phases(x_min, x_max, points, phases)
    fit = summarize([(grid, [signal(x) for x in grid]) for grid in grids])
    reasons = fit.unreadable_because

    observations = [
        f"theta = {fit.slope:.4f} with a grid-phase spread of {fit.spread:.4f}, ranging "
        f"{fit.lowest:.4f} to {fit.highest:.4f} across {phases} grids that differ ONLY "
        "in phase within the same range. The spread is the error bar; a single grid "
        "reports one draw from it and looks precise.",
        f"{fit.sign_changes} sign changes among {points} points, and a dynamic range "
        f"min|S|/median|S| of {fit.dynamic_range:.3g}. Both are ways a log fit dies: "
        "at a crossing log|S| dives, and at a cancellation minimum one sample sits "
        "decades below its neighbours and owns the fit without any crossing at all.",
        "A rigorous bound S_q(X) << X^theta would imply Re(rho) <= theta. This is not "
        "that bound and must not be read as evidence toward it: draws below 1/2 occur "
        "routinely here, and would 'show' what the zeros on the critical line refute.",
    ]
    if reasons:
        observations.append("UNREADABLE, so this run constrains nothing: " + "; ".join(reasons))
    if fit.dropped:
        observations.append(
            f"{fit.dropped} points dropped for S_q = 0 exactly. Dropping them silently "
            "would report a fit over a narrower set than the parameters name."
        )

    return ExperimentResult(
        name="exponent-scan",
        parameters={
            "x_min": x_min,
            "x_max": x_max,
            "points": points,
            "q": q,
            "phases": phases,
        },
        metrics={
            "fitted_theta": fit.slope,
            "theta_grid_spread": fit.spread,
            "theta_min": fit.lowest,
            "theta_max": fit.highest,
            "intercept": fit.intercept,
            "sign_changes": float(fit.sign_changes),
            "dynamic_range": fit.dynamic_range,
            "points_dropped": float(fit.dropped),
            "unreadable": float(1.0 if reasons else 0.0),
        },
        observations=observations,
    )
