"""A counterterm basis fitted to the screening remainder, and the slope that was not there.

THE RESIDUAL WAS FORCED TO CHANGE SIGN, SO ITS LOG-LOG SLOPE COULD NEVER MEAN
ANYTHING. The basis `[1, 1/log X, 1/(log X)^2]` has an intercept column, so
least-squares residuals are orthogonal to it and SUM TO ZERO by construction.
They must therefore change sign, `log|residual|` must dive towards minus
infinity somewhere in every grid, and the fitted `corrected_loglog_slope` is
decided by whichever sample landed nearest the crossing. Measured: residual sum
1e-15 and three to six sign changes in every grid tried, with the slope ranging
-0.88 to +1.32 over ten grid phases against a recorded +0.6245 carrying no error
bar. Not incidental to this data -- structural, and true of any intercepted fit.

AND THERE WERE NO DEGREES OF FREEDOM LEFT. The recorded run used five points
against a three-function basis: two degrees of freedom, and then a two-parameter
log-log fit applied to them. The number was determined by the fit, not by zeta,
and its own observation invited the reader to "seek an analytic derivation of the
fitted terms" -- a derivation of noise.

So the slope is still computed, and reported with the reason it cannot be read.
Deleting it would lose the finding; quoting it would repeat the error. The
counterterm coefficients themselves are a different matter: they are a fit to
data, they come with an RMSE, and they are honest as a conjecture generator so
long as the degrees of freedom are stated.

The forced-sign-change diagnosis came from the rh-research-engine-da session.
"""

from __future__ import annotations

import numpy as np

from ..core.models import ExperimentResult
from ..math.correlation import shell_correlation_breakdown
from ..math.loglog import geometric_phases, summarize

#: Grids differing only in phase across the same range.
GRID_PHASES = 6
#: Columns in the counterterm basis: 1, 1/log X, 1/(log X)^2.
BASIS_SIZE = 3
#: Below this many points the residual has too little freedom left to say
#: anything at all. Four is already thin against a three-column basis; the
#: recorded run used five and had two.
MINIMUM_POINTS = BASIS_SIZE + 3


def _fit_basis(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, float]:
    # Deliberately small/interpretable basis: constant + 1/log X + 1/(log X)^2.
    lx = np.log(xs)
    design = np.column_stack([np.ones_like(lx), 1.0 / lx, 1.0 / (lx * lx)])
    coeff, *_ = np.linalg.lstsq(design, ys, rcond=None)
    pred = design @ coeff
    rmse = float(np.sqrt(np.mean((ys - pred) ** 2)))
    return coeff, rmse


def _residual(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    lx = np.log(xs)
    design = np.column_stack([np.ones_like(lx), 1.0 / lx, 1.0 / (lx * lx)])
    coeff, *_ = np.linalg.lstsq(design, ys, rcond=None)
    return ys - design @ coeff


def run(
    X_min: int = 2_000,
    X_max: int = 40_000,
    points: int = 8,
    q: float = 4.0,
    phases: int = GRID_PHASES,
) -> ExperimentResult:
    if points < MINIMUM_POINTS:
        raise ValueError(
            f"points must be at least {MINIMUM_POINTS}: a {BASIS_SIZE}-column basis "
            f"leaves {points - BASIS_SIZE} degrees of freedom at {points}, and a "
            "two-parameter slope was then fitted to them"
        )

    grids = geometric_phases(float(X_min), float(X_max), points, phases, integral=True)
    residual_grids = []
    coefficients, rmses, sums = [], [], []
    for xs in grids:
        scales = np.array([int(x) for x in xs], dtype=float)
        remainders = np.array(
            [shell_correlation_breakdown(int(x), q).screening_remainder for x in scales]
        )
        coeff, rmse = _fit_basis(scales, remainders)
        residual = _residual(scales, remainders)
        coefficients.append(coeff)
        rmses.append(rmse)
        sums.append(float(np.abs(residual.sum())))
        residual_grids.append((scales.tolist(), residual.tolist()))

    corrected = summarize(residual_grids)
    coeff = np.mean(np.array(coefficients), axis=0)
    freedom = len(grids[0]) - BASIS_SIZE

    observations = [
        f"corrected_loglog_slope = {corrected.slope:.4f} is UNREADABLE and must not be "
        "quoted: " + "; ".join(corrected.unreadable_because)
        if corrected.unreadable_because
        else f"corrected_loglog_slope = {corrected.slope:.4f}, spread {corrected.spread:.4f}",
        "The sign changes are FORCED, not observed. The basis carries an intercept "
        "column, so the least-squares residuals are orthogonal to it and sum to zero "
        f"-- measured at {max(sums):.2g} across the grids. A quantity that must change "
        "sign has no envelope exponent to fit, at any range or sample count.",
        f"{freedom} degrees of freedom remain after a {BASIS_SIZE}-column basis, and a "
        "two-parameter log-log fit was applied to them. The earlier default left two.",
        "Fits only an interpretable lower-order basis: 1, 1/log X, 1/(log X)^2.",
        "The fitted counterterm is a conjecture generator, not a theorem or proof input.",
        "A lower corrected slope is NOT a signal to seek an analytic derivation. That "
        "observation stood here while the slope was structural noise, and it invited "
        "the reader to go looking for a derivation of the fit's own residual.",
    ]

    return ExperimentResult(
        name="counterterm-discovery",
        parameters={
            "X_min": X_min,
            "X_max": X_max,
            "points": points,
            "q": q,
            "phases": phases,
        },
        metrics={
            "samples": len(grids[0]),
            "phases": phases,
            "degrees_of_freedom": float(freedom),
            "counterterm_constant": float(coeff[0]),
            "counterterm_inv_log": float(coeff[1]),
            "counterterm_inv_log2": float(coeff[2]),
            "fit_rmse": float(np.mean(rmses)),
            "corrected_loglog_slope": corrected.slope,
            "corrected_slope_spread": corrected.spread,
            "corrected_sign_changes": float(corrected.sign_changes),
            "corrected_dynamic_range": corrected.dynamic_range,
            "residual_sum_magnitude": float(max(sums)),
            "slope_unreadable": float(1.0 if corrected.unreadable_because else 0.0),
        },
        observations=observations,
    )
