"""The `l`-recovery slope, as a recorded experiment rather than a docstring.

WHY THIS EXISTS. `recover_height` measures the strongest empirical result in
this repository -- the pair correlation of zeta's zeros carries the height they
were measured at, and only the curve that carries primes can be asked for it --
and until now that number lived in a module docstring and a markdown table. A
number in prose has no parameters, no provenance and no fingerprint: it cannot
be replayed, it cannot be compared against a later run, and nothing downstream
can tell whether the version it read is the version that was measured. Every
other finding here is an `ExperimentResult`. This one was not.

WHAT IT DOES NOT CLAIM. That the slope being one PROVES anything about the
Riemann hypothesis. It is a numerical measurement over a finite range of
heights, and the observations below say so on the record rather than leaving a
reader to infer it.

THE UNCERTAINTIES ARE TWO, AND BOTH TRAVEL. `slope_error` is statistical,
from the bands' own measured scatter. `slope_anchor_error` is the systematic
from the band grid's arbitrary phase -- `narrow_ell_bands` anchors its edges at
the lowest zero, which is an accident, and shifting within one width moves the
slope. Reporting the first alone understates the uncertainty by about half.
"""

from __future__ import annotations

import numpy as np

from ..core.models import ExperimentResult
from ..symbolic.height_recovery import narrow_ell_bands, recover_height


def run(
    height: float = 300_000.0,
    band_width: float = 0.5,
    anchors: int = 3,
    prime_limit: int | None = 10_000,
    ladder: str | None = None,
    zeros: str | None = None,
) -> ExperimentResult:
    """Fit `l` band by band and regress it on the height the zeros are at.

    `ladder` is an `.npz` of prebuilt rungs from `tools/build-ladder.py`, and
    `zeros` an `.npy` of ordinates to use instead of computing them. WITHOUT
    THEM THIS RECORDS A WEAKER RESULT THAN THE ONE IN THE DOCSTRINGS, which is
    a trap worth naming: `zero_ordinates` cannot reach `l = 12.5` and above, so
    a run with no ladder spans about 2.5 in `l` and reaches a few sigma from
    zero, where the twelve-band fit reaches twenty-two. A corpus record that
    quietly holds the weaker number is worse than no record.
    """
    from ..symbolic.riemann_siegel import zero_ordinates

    if zeros is not None:
        ordinates = np.asarray(np.load(zeros), dtype=float)
        height = float(ordinates.max())
    else:
        ordinates = np.asarray(zero_ordinates(height), dtype=float)

    rungs: list[np.ndarray] = []
    if ladder is not None:
        loaded = np.load(ladder)
        rungs = [np.sort(np.asarray(loaded[key], dtype=float)) for key in loaded.files]

    bands = list(narrow_ell_bands(ordinates, width=band_width)) + rungs
    fit = recover_height(
        ordinates,
        band_width=band_width,
        prime_limit=prime_limit,
        anchors=anchors,
        extra_bands=rungs or None,
    )

    ell = [np.log(band / (2 * np.pi)) for band in bands]
    return ExperimentResult(
        name="height-recovery-lab",
        parameters={
            "height": height,
            "band_width": band_width,
            "anchors": anchors,
            "prime_limit": prime_limit,
            "ladder": ladder or "",
            "zeros": zeros or "",
        },
        metrics={
            "zeros": int(len(ordinates)),
            "bands": int(len(fit.true_ell)),
            "ell_low": float(min(part.mean() for part in ell)) if ell else 0.0,
            "ell_high": float(max(part.mean() for part in ell)) if ell else 0.0,
            "slope": float(fit.slope),
            "slope_error": float(fit.slope_error),
            # None when a single anchor was run, which is not zero: see the
            # module docstring. `ExperimentResult` metrics are numbers, so an
            # unmeasured systematic is recorded as a NaN rather than as 0.0,
            # which would read as "measured, and it is nothing".
            "slope_anchor_error": float(
                fit.slope_anchor_error if fit.slope_anchor_error is not None else np.nan
            ),
            "slope_over_anchors": float(
                fit.slope_over_anchors if fit.slope_over_anchors is not None else np.nan
            ),
            "sigmas_from_one": float(abs(fit.slope - 1.0) / fit.slope_error),
            "sigmas_from_zero": float(abs(fit.slope) / fit.slope_error),
            "bias": float(fit.bias),
            "bias_error": float(fit.bias_error),
            "null_p": float(fit.null_p),
            "band_size": int(fit.band_size),
            "worst_band_error": float(max(fit.fitted_error)) if fit.fitted_error else 0.0,
            "best_band_error": float(min(fit.fitted_error)) if fit.fitted_error else 0.0,
        },
        observations=[
            "Regression of the l fitted from each band's pair correlation on the "
            "l the band's zeros are actually at. Slope one means the fit tracks "
            "the height; slope zero means it is insensitive to it and any "
            "agreement was a coincidence of one band.",
            "Unfolding removes the height by construction -- w_n = theta(gamma_n)/pi "
            "has mean spacing 1 at every height -- so the only route left for l "
            "into the measured density is the arithmetic. Montgomery's curve has "
            "no l at all.",
            "slope_error is statistical only. slope_anchor_error is the systematic "
            "from the band grid's arbitrary phase, and the two must be quoted "
            "together; it is NaN when a single anchor was run, meaning not "
            "measured rather than measured to be zero.",
            "This is a numerical measurement over a finite range of heights. It "
            "asserts nothing about the Riemann hypothesis and no rigorous bound "
            "is inferred from it.",
        ],
    )
