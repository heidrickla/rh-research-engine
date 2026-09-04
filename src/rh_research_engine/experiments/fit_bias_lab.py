"""b(N): how wrong `fit_ell` is at a known height, as a function of band size.

WHY THIS EXISTS. `height_recovery.MINIMUM_BAND` is the constant that decides
which bands enter the headline regression, and until this was measured it stood
on a proxy -- "lowering it drags the slope" -- with three candidate mechanisms
asserted and withdrawn. The comment above it said the question could not be
settled, because resolving a bias of a few tenths needs of order a hundred
slices of 20,000 zeros and that is 1.4M zeros IN ONE BAND.

`ladder-full.npz` holds 1.70M, 2.91M and 4.98M zeros in one band each, at three
separate heights. So it is settleable, and measuring at three heights is what
separates the count axis from the height axis -- which is the distinction the
two withdrawn explanations turned on.

WHAT IS MEASURED, AND WHY EACH CHOICE IS THE WAY IT IS.

  * DISJOINT CONTIGUOUS slices. Never every-k-th: pairs are between nearby
    zeros, and thinning would destroy the correlation the statistic measures.
  * Each slice against ITS OWN mean `log(t/2 pi)`, never the rung's.
  * The ragged remainder is DROPPED. A short final slice holds fewer than `N`
    zeros and so carries the very bias being measured, and it would enter the
    average as a point at nominal `N`. Dropped counts are recorded.
  * THE GAIN, at the same N and the same height, from `fit_ell`'s companion
    `fit_gain`. A residual measured on an estimator that does not respond is
    not a bias; it is a statement that nothing was measured. Cells whose gain
    is far from one are excluded from b, and the exclusion is recorded rather
    than silent.
  * FITS CENSORED AT THE GRID EDGE are counted. `fit_ell` returns a minimum at
    either end of `ELL_GRID_LOW..ELL_GRID_HIGH` as-is, so such a fit is
    censored rather than measured, and averaging censored draws pulls a mean
    inward. At 2,500 zeros a fifth of fits land there.

THE ANCHOR GATE, WHICH DECIDES WHAT THIS RESULT IS. If b does not go to zero at
large N, then the estimator is biased in the regime the headline slope is
measured in, and that is a bigger and different result from a small-band effect.
The anchor is the largest N with at least `MIN_SLICES` slices, per rung, and the
gate is `|b| <= 2 * SEM` there. Deciding that after seeing the curve is where a
measurement becomes a story, so it is a parameter and not a judgement.

WHAT THIS IS NOT. It says nothing about the Riemann hypothesis. It is a property
of an estimator applied to a finite sample of ordinates, and its only bearing on
anything else is which bands are fit to be regressed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.models import ExperimentResult

#: Band sizes swept. Reaches well below `MINIMUM_BAND` on purpose: the bands
#: where that constant actually bites hold 2k-33k zeros, so a sweep starting at
#: 20,000 would measure the wrong regime.
COUNTS = (2_500, 5_000, 10_000, 20_000, 40_000, 80_000, 160_000, 320_000)

#: A cell whose gain falls outside this is not reporting a bias.
#:
#: AND A CELL WHOSE GAIN IS NOT DETERMINED IS A THIRD STATE, not a failure. This window
#: is 0.5 wide, and at N = 2,500 the gain's own standard error runs 0.17 to 1.7 -- wider
#: than the window it is tested against. A window narrower than the uncertainty is not a
#: filter, it is a coin, and "cut for gain" would then mean "the gain could not be
#: measured" while reading as "the estimator did not respond". Those are the ledger's
#: not-tested and refuted, one layer further down, and they must not share a verdict.
#:
#: So a cell is usable only when its gain is INSIDE the window AND resolved well enough to
#: place there -- `gain_sem <= GAIN_RESOLUTION`. Cells failing that are counted separately
#: as undetermined rather than silently dropped among the unresponsive ones.
GAIN_LOW, GAIN_HIGH = 0.8, 1.3

#: How well the gain must be pinned before the window above means anything. Half the
#: window: a measurement whose error bar exceeds this cannot be placed inside or outside
#: a band of width 0.5 with any confidence. Measured rather than chosen -- at N = 2,500 it
#: excludes the l = 9.4, 10.4 and 10.9 cells whose SEMs are 0.92, 0.32 and 0.22 against
#: point estimates of 1.53, 1.34 and 1.38, none of which can be told from 1.0.
GAIN_RESOLUTION = 0.25

#: Above this fraction of fits censored at the grid edge, the mean is being
#: pulled by the boundary rather than by the zeros.
MAX_RAILED = 0.05

#: Fewest slices a cell needs before its SEM means anything. An error bar from
#: two draws is not an error bar, and the largest-N cells have very few.
MIN_SLICES = 8

#: Slices per cell that also carry the gain control.
GAIN_SLICES = 12

EDGES = np.linspace(0.0, 3.0, 31)


def _refusal(ladder: str, reason: str) -> ExperimentResult:
    """Absent input is its own verdict, never an empty measurement.

    "b(N) was measured and found flat" and "the input was not there" must never
    print the same, which is `patterns/ledger.py`'s rule about not-tested versus
    refuted, one layer up.
    """
    return ExperimentResult(
        name="fit-bias-lab",
        parameters={"ladder": ladder},
        metrics={"refused": 1.0, "cells": 0.0},
        observations=[
            f"REFUSED, not measured: {reason}",
            # THE TRAP BELONGS ON THE REFUSAL PATH ABOVE ALL. A refusal is exactly the
            # moment somebody goes looking for a builder -- and the wrong one is the copy
            # sitting next to the data. Carrying this warning only on the measured path
            # put it everywhere except where it is needed; a test caught that.
            "READ THIS BEFORE REBUILDING THE INPUT: THERE ARE TWO BUILDERS AND THE WRONG "
            "ONE SITS NEXT TO THE DATA. `~/rh-data/build_ladder.py` is an older builder "
            "that sizes rungs to a target COUNT; the committed `tools/build-ladder.py` "
            "sizes them to a WIDTH. Their rungs differ by a factor of 25 -- 0.020-0.060 "
            "wide holding 140k-180k zeros, against 0.5000 holding 1.70M-4.98M -- and "
            "rebuilding with the copy beside the data reproduces the starved rungs that "
            "cost a session to diagnose.",
            "The invocation that produced this record's input: "
            "`python tools/build-ladder.py <out.npz> 12.5 13.0 13.5`, which is also the "
            "bare default. VERIFIED AGAINST THE ARTIFACT, not read off a script.",
            "This experiment needs the prebuilt ladder rungs. THE ARTIFACT IS NOT IN "
            "THIS REPOSITORY AND IS NOT BACKED UP: `ladder-full.npz` exists only in "
            "a local data directory on the two machines it was built on, so nobody else can "
            "regenerate this record without rebuilding it with tools/build-ladder.py. "
            "That is a limitation of the record and it belongs on the record.",
            "A refusal says nothing about b(N). It says the measurement did not run.",
        ],
    )


def run(
    ladder: str | None = None,
    zeros: str | None = None,
    counts: tuple[int, ...] = COUNTS,
    max_slices: int | None = None,
    gain_low: float = GAIN_LOW,
    gain_high: float = GAIN_HIGH,
    max_railed: float = MAX_RAILED,
    min_slices: int = MIN_SLICES,
    prime_limit: int | None = 10_000,
) -> ExperimentResult:
    """Measure b(N) and the gain across the ladder rungs."""
    if ladder is None and zeros is None:
        return _refusal("", "neither a ladder nor a zeros path was given")
    for label, given in (("ladder", ladder), ("zeros", zeros)):
        if given is not None and not Path(given).exists():
            return _refusal(given, f"the {label} path {given} does not exist")

    from ..symbolic import pair_correlation as pc
    from ..symbolic.height_recovery import (
        curve_bank,
        ell_grid,
        fit_ell,
        fit_gain,
        narrow_ell_bands,
    )

    grid = ell_grid()
    bank = curve_bank(EDGES, grid, prime_limit=prime_limit)
    # ONE SOURCE OF BANDS, so both paths are measured by the same routine. Two copies of
    # a measurement drift apart and then disagree, which is the defect this module exists
    # to record. The rungs arrive prebuilt; the low bands are cut with `minimum=1` --
    # deliberately NOT `MINIMUM_BAND`, since the point is to characterise the bands that
    # constant excludes.
    pieces: list[tuple[str, np.ndarray]] = []
    if ladder is not None:
        loaded = np.load(Path(ladder))
        pieces += [
            ("ladder", np.sort(np.asarray(loaded[key], dtype=float)))
            for key in sorted(loaded.files, key=float)
        ]
    if zeros is not None:
        ordinates = np.asarray(np.load(Path(zeros)), dtype=float)
        pieces += [("band", np.sort(band)) for band in narrow_ell_bands(ordinates, minimum=1)]

    cells: list[dict[str, float]] = []
    for source, piece in pieces:
        ell_all = np.log(piece / (2 * np.pi))
        unfolded = pc.unfold(piece)
        for size in counts:
            if size > piece.size:
                continue
            slices = piece.size // size
            if max_slices is not None:
                slices = min(slices, max_slices)
            residuals, gains = [], []
            railed = 0
            for index in range(slices):
                low, high = index * size, (index + 1) * size
                density = pc.measured_density(
                    unfolded[low:high], window=3.0, bins=30
                ).density
                fitted = fit_ell(density, grid=grid, bank=bank)
                if fitted <= grid[0] + 1e-9 or fitted >= grid[-1] - 1e-9:
                    railed += 1
                true_ell = float(ell_all[low:high].mean())
                residuals.append(fitted - true_ell)
                if index < GAIN_SLICES:
                    # Anchored at the TRUE l, not at the fit: this asks
                    # d(fitted)/d(true), which is what decides whether the
                    # residual above is a bias. Anchoring at the fit measures
                    # responsiveness at a possibly-wrong answer, and reports
                    # 0.653 where this reports 0.349.
                    gains.append(fit_gain(density, near=true_ell, grid=grid, bank=bank))
            values = np.asarray(residuals)
            spread = float(values.std(ddof=1)) if values.size > 1 else float("nan")
            cells.append(
                {
                    # The band's own mean l whatever it was cut from, so a pooled
                    # value can always name the range it covers.
                    "rung": float(ell_all.mean()),
                    "source": source,
                    "count": float(size),
                    "slices": float(slices),
                    "dropped": float(piece.size - slices * size),
                    "bias": float(values.mean()),
                    "median": float(np.median(values)),
                    "sd": spread,
                    "sem": spread / np.sqrt(values.size) if values.size > 1 else float("nan"),
                    "railed": float(railed) / max(1, slices),
                    "gain": float(np.mean(gains)) if gains else float("nan"),
                    # The gain gets an error bar for the same reason the bias
                    # does: a pooled value without one cannot be pooled the
                    # same way, and two poolings of one quantity is how this
                    # file reported 0.4300 where its own docstring said 0.3486.
                    "gain_sem": (
                        float(np.std(gains, ddof=1) / np.sqrt(len(gains)))
                        if len(gains) > 1
                        else float("nan")
                    ),
                }
            )

    def _resolved(cell: dict) -> bool:
        sem = cell["gain_sem"]
        return sem == sem and sem <= GAIN_RESOLUTION          # NaN is not resolved

    usable = [
        cell
        for cell in cells
        if _resolved(cell)
        and gain_low <= cell["gain"] <= gain_high
        and cell["railed"] <= max_railed
        and cell["slices"] >= min_slices
    ]
    # COUNTED, NEVER SILENT. A cell excluded because its gain could not be measured is
    # a different fact from one excluded because the estimator did not respond, and a
    # reader who cannot tell them apart will read the second where only the first holds.
    undetermined = [
        cell for cell in cells
        if not _resolved(cell) and cell["railed"] <= max_railed and cell["slices"] >= min_slices
    ]

    metrics: dict[str, float] = {
        "refused": 0.0,
        "cells": float(len(cells)),
        "usable_cells": float(len(usable)),
        # Excluded because the gain could not be resolved, NOT because it was bad.
        "gain_undetermined_cells": float(len(undetermined)),
        "rungs": float(len({cell["rung"] for cell in cells})),
        "dropped_zeros": float(sum(cell["dropped"] for cell in cells)),
    }

    # Pooled b at each size, inverse-variance weighted across the bands.
    #
    # EVERY REQUESTED COUNT REPORTS ITS OWN BACKING, present or not. `bias_at_N` is
    # emitted only when some cell survives, so its ABSENCE would otherwise be
    # indistinguishable from a value nobody recorded -- an absent metric and a false one
    # are as easy to confuse as the two states `gain_undetermined_cells` separates, which
    # is the whole reason that key exists. So `usable_at_N` is always written, including
    # as 0.0, and a reader can tell "no cell could support this" from "nobody looked".
    for size in counts:
        chosen = [cell for cell in usable if cell["count"] == size]
        metrics[f"usable_at_{size}"] = float(len(chosen))
        metrics[f"undetermined_at_{size}"] = float(
            len([cell for cell in undetermined if cell["count"] == size])
        )
        if not chosen:
            continue
        weights = np.array([1.0 / cell["sem"] ** 2 for cell in chosen])
        biases = np.array([cell["bias"] for cell in chosen])
        pooled = float((weights * biases).sum() / weights.sum())
        error = float(1.0 / np.sqrt(weights.sum()))
        metrics[f"bias_at_{size}"] = pooled
        metrics[f"bias_error_at_{size}"] = error

    # Pooled gain, over ALL cells -- the gain is what decides usability, so
    # filtering by it first and then reporting it would be circular.
    #
    # POOLED THE SAME WAY AS THE BIAS, inverse-variance weighted, and NOT by a
    # simple mean. Two pooling policies in one file is two quantities wearing
    # one name: the simple mean reads 0.4300 at N = 2,500 where this reads
    # 0.3486 +/- 0.0643, because rung 13.5's gain draws scatter most and a
    # simple mean does not know that.
    for size in counts:
        chosen = [
            cell
            for cell in cells
            if cell["count"] == size and cell["gain_sem"] == cell["gain_sem"]
        ]
        if chosen:
            weights = np.array([1.0 / cell["gain_sem"] ** 2 for cell in chosen])
            values = np.array([cell["gain"] for cell in chosen])
            metrics[f"gain_at_{size}"] = float((weights * values).sum() / weights.sum())
            metrics[f"gain_error_at_{size}"] = float(1.0 / np.sqrt(weights.sum()))
            metrics[f"railed_at_{size}"] = float(np.mean([cell["railed"] for cell in chosen]))

    # The pre-registered gate, per rung.
    passed = 0
    checked = 0
    for rung in sorted({cell["rung"] for cell in cells}):
        candidates = [
            cell for cell in cells if cell["rung"] == rung and cell["slices"] >= min_slices
        ]
        if not candidates:
            continue
        anchor = max(candidates, key=lambda cell: cell["count"])
        checked += 1
        if abs(anchor["bias"]) <= 2 * anchor["sem"]:
            passed += 1
    metrics["anchor_gate_checked"] = float(checked)
    metrics["anchor_gate_passed"] = float(passed)

    observations = [
        "READ THIS BEFORE REBUILDING THE INPUT: THERE ARE TWO BUILDERS AND THE WRONG "
        "ONE SITS NEXT TO THE DATA. `~/rh-data/build_ladder.py` is an older builder "
        "that sizes rungs to a target COUNT; the committed `tools/build-ladder.py` "
        "sizes them to a WIDTH. Their rungs differ by a factor of 25 -- 0.020-0.060 "
        "wide holding 140k-180k zeros, against 0.5000 holding 1.70M-4.98M -- and "
        "rebuilding with the copy beside the data reproduces the starved rungs that "
        "cost a session to diagnose. An absent command makes someone ask; a "
        "plausible-looking wrong one makes them proceed.",
        "The invocation that produced this record's input: "
        "`python tools/build-ladder.py <out.npz> 12.5 13.0 13.5`, which is also the "
        "bare default. VERIFIED AGAINST THE ARTIFACT, not read off a script: the "
        "three rungs in the npz are centred at 12.5000, 13.0000 and 13.5000 and each "
        "spans exactly 0.5000 in l, matching WIDTH = 0.5 and DEFAULT_RUNGS. Checking "
        "it that way is what found the two-builder trap above.",
        "THE ARTIFACT IS NOT IN THIS REPOSITORY AND IS NOT BACKED UP. "
        "`ladder-full.npz` exists only in a local data directory on the two machines "
        "it was built on. A file with no invocation is not regenerable even with infinite "
        "disk, which is why the command above is recorded here rather than left to a "
        "reader to reconstruct.",
        "b(N) is the mean of (fitted l - the slice's own true l) over disjoint "
        "contiguous slices of N zeros inside ONE ladder rung, so the height is "
        "held fixed and only the count varies. Measured at three heights, which "
        "is what separates the count axis from the height axis.",
        "A CELL WHOSE GAIN IS NOT DETERMINED IS NOT A CELL WHOSE GAIN IS BAD. The usable "
        "window is 0.5 wide and at N = 2,500 the gain's own standard error reaches 1.7, so "
        "a cell can fail the window purely for want of resolution. Those are counted as "
        "gain_undetermined_cells and excluded from b, separately from cells that were "
        "measured and found unresponsive.",
        "THE GAIN IS WHAT MAKES A NULL READABLE. A residual measured where the "
        "estimator does not respond to its own input is not a bias, it is a "
        "statement that nothing was measured. Cells outside the gain window, or "
        "with too many fits censored at the grid edge, are excluded from b and "
        "the exclusion is recorded.",
        "The gain is anchored at the TRUE l, so it asks d(fitted)/d(true) rather "
        "than how responsive the fit is at its own answer. The two agree where "
        "the fit is good and diverge where it is not -- 0.349 against 0.653 at "
        "N = 2,500 -- which is the only regime the gain exists for.",
        f"The anchor gate passed on {passed} of {checked} rungs. It is "
        "pre-registered: the anchor is the largest N with at least "
        f"{min_slices} slices and the test is |b| <= 2 SEM there. Passing means "
        "this is a small-band effect; failing would mean the estimator is biased "
        "in the regime the headline slope is measured in, which is a different "
        "and larger result.",
        "The distribution behind b is strongly right-skewed, so the mean and the "
        "median differ: the typical small band fits slightly LOW and a minority "
        "fit far high. 'Small bands fit l too high' is a statement about the mean "
        "and about a tail, not about a typical band.",
        "This is a property of an estimator on a finite sample of ordinates. It "
        "asserts nothing about the Riemann hypothesis.",
    ]

    return ExperimentResult(
        name="fit-bias-lab",
        parameters={
            "ladder": ladder or "",
            "zeros": zeros or "",
            "counts": list(counts),
            "max_slices": max_slices if max_slices is not None else 0,
            "gain_low": gain_low,
            "gain_high": gain_high,
            "max_railed": max_railed,
            "min_slices": min_slices,
            "prime_limit": prime_limit,
        },
        metrics=metrics,
        observations=observations,
    )
