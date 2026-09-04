"""When a log-log slope is a fact about zeta, and when it is a fact about the grid.

Three experiments here fitted `log|y|` against `log X` on a geometric grid and
recorded the slope as a number. All three were reporting the grid. The range and
the point count were arguments; the grid PHASE was the one nobody chose, and
shifting it inside the same range moved `exponent_scan` across 0.214..1.061 and
`counterterm_discovery` across -0.88..+1.32.

TWO WAYS A LOG FIT DIES, AND ONLY ONE OF THEM IS A SIGN CHANGE. The first
version of this check counted sign changes, because `S_q` oscillates and
`log|y|` dives towards minus infinity at a crossing. That misses the worse case:
`correlation_scan`'s `total_energy` is a cancellation residual -- diagonal
6.5874541 against off-diagonal -6.5871135, about 4.3 digits gone -- and it dips
to 3.8e-10 against a typical 1e-3 WITHOUT EVER CHANGING SIGN. A single point
seven decades below its neighbours owns a four-point log-log fit outright, and a
sign-change count reports clean.

So the predicate that catches both is the dynamic range, `min|y| / median|y|`.
A genuine crossing drives it small too, which is why it subsumes the earlier
check rather than sitting beside it. Sign changes are still counted, because
"this oscillates" and "this nearly cancels" are different things to tell a
reader, but the ratio is what decides readability.

Credit where due: the cancellation-minimum case, and this predicate, came from
the rh-research-engine-da session, against a check this module's author had
already written and believed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

#: A slope whose grid-phase spread exceeds this constrains nothing. Chosen
#: against the one fit that survives: `remainder_loglog_slope` measures
#: -0.0093 +/- 0.0014 over seven phases, two orders below this.
SPREAD_LIMIT = 0.05

#: `min|y| / median|y|` below this and one point owns the fit.
#:
#: MEASURED, not chosen. This was first written as 1e-3 and would have missed
#: the case it exists for: `total_energy`'s cancellation minimum comes in at
#: 9.4e-6 against a median 1.1e-3, a ratio of 8.6e-3. The separation from the
#: one fit that survives is enormous either way -- `screening_remainder` sits
#: at 0.99 -- so the limit is set where the failing case actually falls, and a
#: guard tuned tighter than its own motivating example is decoration.
DYNAMIC_RANGE_LIMIT = 1e-2


@dataclass(frozen=True)
class LogLogSummary:
    """A slope, and everything needed to decide whether to believe it."""

    slope: float
    intercept: float
    spread: float
    lowest: float
    highest: float
    sign_changes: int
    dynamic_range: float
    dropped: int
    phases: int

    @property
    def sign_determined(self) -> bool:
        """Do all the grids agree on the SIGN? A different question from readable.

        `unreadable_because` asks whether a magnitude is worth quoting, and it
        answers with a limit on the spread. A falsification test does not want a
        magnitude -- it wants a sign, and the two come apart:
        `baez_duarte`'s envelope slope has a binning spread of 0.019, well inside
        `SPREAD_LIMIT`, so it is READABLE, while its range runs -0.0574 to
        +0.0282 and is positive on 5 of 18 binnings. A caller that checked
        readability and then tested `slope > 0` would announce a refutation of RH
        on 5 of 18 equally defensible histograms -- reaching the trap through the
        guard rather than around it.

        So a verdict on the sign requires every grid to agree, which is
        `lowest` and `highest` sharing one. No new constant: a sign is
        determined or it is not.
        """
        return self.lowest > 0.0 or self.highest < 0.0

    @property
    def unreadable_because(self) -> list[str]:
        """Empty when the slope is worth reading. Never a bare boolean.

        A reason is what the record needs; "unreadable" alone would be another
        verdict nobody can act on.
        """
        reasons = []
        if self.spread > SPREAD_LIMIT:
            reasons.append(
                f"grid-phase spread {self.spread:.4g} exceeds {SPREAD_LIMIT}: the slope "
                f"ranges {self.lowest:.4g} to {self.highest:.4g} across {self.phases} "
                "grids differing only in phase within the same range"
            )
        if self.dynamic_range < DYNAMIC_RANGE_LIMIT:
            reasons.append(
                f"dynamic range min|y|/median|y| = {self.dynamic_range:.3g} is below "
                f"{DYNAMIC_RANGE_LIMIT}: at least one sample sits far enough below its "
                "neighbours to own the fit by itself, which a sign-change count does "
                "not detect"
            )
        if self.sign_changes:
            reasons.append(
                f"{self.sign_changes} sign changes: log|y| dives towards minus infinity "
                "at a crossing, so no envelope exponent can be read off it there"
            )
        return reasons


def dynamic_range(values: Sequence[float]) -> tuple[float, int]:
    """`min|y| / median|y|`, and WHERE the floor sits.

    The predicate that catches a cancellation minimum. Median rather than mean,
    because the outlier this exists to detect would otherwise drag the scale it
    is being compared against.

    The index is returned because the dip is itself grid-dependent: shifting the
    phase inside the same range puts the minimum somewhere else entirely, so
    "how far off, and at which sample" is the reportable thing and a boolean is
    not.
    """
    magnitudes = [abs(float(v)) for v in values]
    if not magnitudes:
        return 0.0, -1
    floor_index = min(range(len(magnitudes)), key=magnitudes.__getitem__)
    ordered = sorted(magnitudes)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
    if median == 0.0:
        return 0.0, floor_index
    return ordered[0] / median, floor_index


def _ordinary_least_squares(xs: list[float], ys: list[float]) -> tuple[float, float]:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        raise ValueError("every abscissa is identical; there is no slope to fit")
    slope = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)
    ) / denominator
    return slope, mean_y - slope * mean_x


def summarize(grids: Sequence[tuple[Sequence[float], Sequence[float]]]) -> LogLogSummary:
    """Fit `log|y|` against `log x` on each grid, and report what they disagree on.

    `grids` is one entry per phase: the SAME range and point count, sampled at
    shifted offsets. Two is the minimum that can show a grid dependence at all,
    which is why one is refused rather than averaged.
    """
    if len(grids) < 2:
        raise ValueError("at least two grids are needed; one cannot show grid dependence")

    slopes, intercepts, changes, ranges, dropped = [], [], [], [], []
    for xs, ys in grids:
        xs = [float(x) for x in xs]
        ys = [float(y) for y in ys]
        usable_x = [math.log(x) for x, y in zip(xs, ys, strict=True) if x > 0 and y != 0.0]
        usable_y = [math.log(abs(y)) for x, y in zip(xs, ys, strict=True) if x > 0 and y != 0.0]
        if len(usable_x) < 3:
            raise ValueError(f"only {len(usable_x)} usable points on one grid")
        slope, intercept = _ordinary_least_squares(usable_x, usable_y)
        slopes.append(slope)
        intercepts.append(intercept)
        changes.append(sum(1 for a, b in zip(ys, ys[1:], strict=False) if a * b < 0))
        ranges.append(dynamic_range(ys)[0])
        dropped.append(len(xs) - len(usable_x))

    mean = sum(slopes) / len(slopes)
    spread = math.sqrt(sum((s - mean) ** 2 for s in slopes) / (len(slopes) - 1))
    return LogLogSummary(
        slope=mean,
        intercept=sum(intercepts) / len(intercepts),
        spread=spread,
        lowest=min(slopes),
        highest=max(slopes),
        sign_changes=max(changes),
        dynamic_range=min(ranges),
        dropped=max(dropped),
        phases=len(grids),
    )


def geometric_phases(
    x_min: float, x_max: float, points: int, phases: int, *, integral: bool = False
) -> list[list[float]]:
    """The same geometric range and point count, sampled at `phases` offsets.

    `integral` rounds to integers, which several scans need because the
    underlying quantity is defined on integer X. Rounding can collide two
    offsets onto one abscissa at small X; duplicates are dropped here so the
    fit never sees a repeated point, and the caller sees a shorter grid.
    """
    if points < 3:
        raise ValueError("points must be >= 3")
    if phases < 2:
        raise ValueError("phases must be >= 2; one grid cannot show grid dependence")
    step = math.log(x_max / x_min) / (points - 1)
    grids = []
    for index in range(phases):
        offset = index / phases
        raw = [math.exp(math.log(x_min) + (i + offset) * step) for i in range(points)]
        if integral:
            grids.append(sorted({max(2, int(round(x))) for x in raw}))
        else:
            grids.append(raw)
    return [[float(x) for x in grid] for grid in grids]
