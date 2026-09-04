"""The spacing residual held against Odlyzko's zeros, at heights we cannot reach.

WHY AN EXTERNAL COMPARISON. `level_spacing` measures a residual of about 0.022
at T < 9 x 10^4 -- five times its noise floor, with the same shape in disjoint
bands and in two halves of one band. That establishes the residual is real. It
does NOT establish two things a reader would want:

  * that it is not an artefact of THIS engine's zero-finder, and
  * that it is a finite-height correction rather than a permanent departure
    from GUE.

Both are settled by data nobody here produced. Odlyzko published the first
100000 zeros, and 10000 more at each of the indices 10^12, 10^21 and 10^22 --
sixteen orders of magnitude past anything this engine computes.

WHAT THE COMPARISON FOUND, and it answers both.

  * The ordinates agree. Worst difference over 100000 zeros is 3.005e-9
    against his stated accuracy of 3e-9, so the two computations agree as
    closely as his file claims to be right. The residual computed from HIS
    numbers and from ours correlates at 1.00000 and has the same mean, 0.02242.
    The shape is in the zeros.

  * The shape is gone at height. At index 10^12, 10^21 and 10^22 the residual
    sits at the noise floor of a 10000-zero sample -- 0.0092, 0.0129, 0.0112
    against a floor of 0.0109 -- and projecting each onto the low-height shape
    gives +0.043 +/- 0.150, -0.176 +/- 0.150 and +0.166 +/- 0.150 OF THAT
    SHAPE. All consistent with nothing.

So the finite-height reading holds, and the honest form of it is a bound: at
index 10^12 and beyond, at most about a third of the low-height shape survives,
and the best estimate is none. Ten thousand zeros will not resolve better than
that, and saying "it is zero" would be claiming a precision the sample does
not carry.

THE DATA IS NOT VENDORED. It is somebody else's, published at
https://www-users.cse.umn.edu/~odlyzko/zeta_tables/ as `zeros1`, `zeros3`,
`zeros4` and `zeros5`. Fetch them into a directory and point this at it;
absent, the comparison raises rather than reporting a thinner check under the
same name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..contracts.epistemic import RIGOROUS, Confidence
from .level_spacing import exact_gue_bin_average, sampling_noise_floor, unfold

#: Where the tables came from, carried in the record.
#:
#: A comparison against an external dataset is only as good as knowing which
#: dataset, and "Odlyzko's zeros" names four files with different heights and
#: different stated accuracies.
SOURCE = "https://www-users.cse.umn.edu/~odlyzko/zeta_tables/"

#: File, the base its values are offsets from, and the zero index it starts at.
#:
#: The files give `gamma - base` because the heights run to 10^21 and a plain
#: decimal would lose every digit that matters. Spacings are differences of
#: those offsets, so the base cancels exactly and nothing is lost; it is needed
#: only for the mean density, which depends on `log(gamma/2pi)` and is
#: insensitive to the last digits.
HIGH_TABLES: dict[str, tuple[float, str]] = {
    "zeros3": (267653395647.0, "10^12"),
    "zeros4": (144176897509546973000.0, "10^21"),
    "zeros5": (1370919909931995300000.0, "10^22"),
}


class OdlyzkoComparison(BaseModel):
    """What an external computation of the same zeros says about the residual."""

    model_config = ConfigDict(extra="forbid")

    source: str = SOURCE
    #: Worst absolute difference between our ordinates and his, over the
    #: overlap, and the accuracy his file claims.
    ordinate_agreement: float | None = None
    stated_accuracy: float = 3e-9
    #: Correlation between the residual computed from his numbers and ours.
    #: Anything below 1 would mean the shape depends on whose zeros are used.
    residual_correlation: float | None = None
    #: Per table: the zero index, the residual, and the noise floor beside it.
    indices: list[str] = Field(default_factory=list)
    residuals: list[float] = Field(default_factory=list)
    noise_floors: list[float] = Field(default_factory=list)
    #: The low-height shape's amplitude surviving at each index, as a fraction,
    #: with the uncertainty the sample size allows. A BOUND, not a detection.
    surviving_fraction: list[float] = Field(default_factory=list)
    surviving_uncertainty: list[float] = Field(default_factory=list)
    confidence: Confidence = Confidence.NUMERICAL

    @field_validator("confidence")
    @classmethod
    def _reject_rigorous_confidence(cls, value: Confidence) -> Confidence:
        if value in RIGOROUS:
            raise ValueError(
                f"a comparison against published zeros may not claim "
                f"{value.value!r}: it is two finite samples agreeing, and the "
                "GUE law they are measured against is a conjecture about a "
                "limit neither reaches"
            )
        return value

    @property
    def shape_is_gone(self) -> bool:
        """Every high index is consistent with none of the shape surviving."""
        return bool(self.surviving_fraction) and all(
            abs(fraction) < 2 * uncertainty
            for fraction, uncertainty in zip(
                self.surviving_fraction, self.surviving_uncertainty, strict=True
            )
        )


def read_table(path: Path) -> np.ndarray:
    """Ordinates or offsets from one of Odlyzko's files.

    The files open with prose describing the base and the accuracy. Lines that
    are not numbers are skipped rather than parsed, and a file yielding too few
    is refused: a truncated download would otherwise be measured as though it
    were a small sample, which is a different and much weaker thing.
    """
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            values.append(float(text))
        except ValueError:
            continue
    if len(values) < 1000:
        raise ValueError(
            f"{path.name} yielded {len(values)} numbers, which is too few for "
            "any of the published tables -- a partial download measured as a "
            "small sample would read as a weaker result rather than a broken "
            f"input. Re-fetch from {SOURCE}"
        )
    return np.asarray(values, dtype=float)


def compare(directory: Path, *, bins: int = 25, upper: float = 3.0) -> OdlyzkoComparison:
    """Hold our residual against his zeros, and against his highest heights."""
    from .riemann_siegel import zero_ordinates

    directory = Path(directory)
    missing = [
        name
        for name in ("zeros1", *HIGH_TABLES)
        if not (directory / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(missing)} not in {directory}. The tables are not "
            f"vendored -- they are somebody else's data. Fetch them from {SOURCE}"
        )

    edges = np.linspace(0.0, upper, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    width = upper / bins
    exact = exact_gue_bin_average(edges)

    def residual_of(spacings: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(spacings, bins=edges, density=True)
        return counts - exact

    his = read_table(directory / "zeros1")
    ours = np.asarray(
        [float(value) for value in zero_ordinates(float(his[-1]) + 0.5)],
        dtype=float,
    )
    overlap = min(len(his), len(ours))
    agreement = float(np.max(np.abs(his[:overlap] - ours[:overlap])))

    his_residual = residual_of(np.diff(unfold(his[:overlap])))
    our_residual = residual_of(np.diff(unfold(ours[:overlap])))
    correlation = float(np.corrcoef(his_residual, our_residual)[0, 1])

    # The template is OUR low-height shape, which is what the high tables are
    # asked about. Its norm sets the scale the fractions are reported in.
    amplitude = float(np.linalg.norm(our_residual))
    direction = our_residual / amplitude

    indices, residuals, floors, fractions, uncertainties = [], [], [], [], []
    for name, (base, label) in HIGH_TABLES.items():
        offsets = read_table(directory / name)
        density = np.log((base + offsets[0]) / (2 * np.pi)) / (2 * np.pi)
        spacings = np.diff(offsets) * density
        residual = residual_of(spacings)

        errors = np.sqrt(np.maximum(exact, 0.0) / (len(spacings) * width))
        indices.append(label)
        residuals.append(float(np.mean(np.abs(residual))))
        floors.append(sampling_noise_floor(len(spacings), centres, width))
        fractions.append(float(residual @ direction) / amplitude)
        uncertainties.append(
            float(np.sqrt(np.sum((direction * errors) ** 2))) / amplitude
        )

    return OdlyzkoComparison(
        ordinate_agreement=agreement,
        residual_correlation=correlation,
        indices=indices,
        residuals=residuals,
        noise_floors=floors,
        surviving_fraction=fractions,
        surviving_uncertainty=uncertainties,
    )
