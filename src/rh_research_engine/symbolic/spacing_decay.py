"""How fast the spacing residual dies with height.

`level_spacing` establishes that the residual is real -- not the curve, not the
histogram -- and `odlyzko` establishes that it is gone by zero index 10^12.
Between those two facts is a rate, and this measures it.

THE ESTIMATOR HAS NO TEMPLATE, and it must not have one. Projecting each band's
residual onto a shape taken from some band biases every band overlapping that
one upward, which is exactly the direction that fakes a decay. Instead, for
bins `i` with residual `r_i`, systematic `s_i` and sampling variance
`sigma_i^2`:

    E[sum r_i^2] = sum s_i^2 + sum sigma_i^2

so `sum r_i^2 - sum sigma_i^2` estimates the squared systematic without a
template and without assuming the shape. `sigma_i^2` is the multinomial
variance, which `level_spacing.bootstrap_noise_floor` shows is right to a few
percent and slightly conservative.

Unbiasedness is the property the whole measurement rests on, so it is tested
rather than argued: samples drawn from the exact law contain no systematic and
must measure none. An estimator with a positive bias would report an amplitude
in every band and a decay from the band sizes alone -- and it did, until the
bin-average bug was found this way.

WHAT IT COMES OUT AT, over Odlyzko's 2001052 zeros in six disjoint bands from
T = 10^3 to 1.1 x 10^6:

    T ~   2045   amplitude 0.236 +/- 0.067
    T ~   6643             0.182 +/- 0.030
    T ~  20324             0.179 +/- 0.016
    T ~  66075             0.143 +/- 0.008
    T ~ 202522             0.122 +/- 0.004
    T ~ 710449             0.104 +/- 0.002

    alpha = 1.61 +/- 0.17          chi^2 = 1.1 on 4 degrees of freedom
    pure 1/log T                   chi^2 = 11.9 on 5

So the decay is steeper than `1/log T` over this range, at about three and a
half sigma.

AND THE EXPONENT IS EFFECTIVE, NOT ASYMPTOTIC. `log T` runs from 7.6 to 13.5
here -- a lever of 1.8. Over so short a lever a two-term expansion is
indistinguishable from a single power: fitting `c/log T + d/(log T)^2` does
just as well and returns `d/c` near 20, which means the "correction" exceeds
the leading term everywhere it was measured. The data does not separate them,
and `EFFECTIVE_NOT_ASYMPTOTIC` says so on every record rather than in a
docstring a reader may not reach.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..contracts.epistemic import RIGOROUS, Confidence
from .level_spacing import exact_gue_bin_average, unfold

#: Carried on every record, for the same reason `CurveZeta` carries its own.
#:
#: "The residual decays like (log T)^-1.6" is the sentence that will travel,
#: and detached from its range it reads as a statement about the limit. It is
#: not one: the measurement spans a factor of 1.8 in log T, over which a
#: two-term expansion and a single power cannot be told apart.
EFFECTIVE_NOT_ASYMPTOTIC = (
    "an effective exponent over the range measured, not an asymptotic one: "
    "log T spans a factor of 1.8 here, and over that a two-term expansion fits "
    "as well as a single power with d/c near 20 -- so the leading behaviour is "
    "not separated from its correction"
)


def amplitude(spacings: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    """`|systematic|` and its standard error, with the noise subtracted.

    Returns zero when the estimate of the squared systematic is negative,
    which happens when the band is small enough that the noise is everything.
    Zero is the honest answer there -- clamping to a small positive number
    would put a floor under the fit and bend it.
    """
    exact = exact_gue_bin_average(edges)
    width = float(edges[1] - edges[0])
    counts, _ = np.histogram(spacings, bins=edges, density=True)
    residual = counts - exact
    variance = np.maximum(exact, 0.0) / (len(spacings) * width)

    squared = float(np.sum(residual**2) - np.sum(variance))
    systematic = np.maximum(residual**2 - variance, 0.0)
    spread = float(np.sqrt(np.sum(2 * variance**2 + 4 * systematic * variance)))
    if squared <= 0:
        return 0.0, spread
    value = float(np.sqrt(squared))
    return value, spread / (2 * value)


class DecayFit(BaseModel):
    """The residual's amplitude across bands, and the power of `log T` it fits."""

    model_config = ConfigDict(extra="forbid")

    bands: list[tuple[float, float]] = Field(default_factory=list)
    heights: list[float] = Field(default_factory=list)
    spacings: list[int] = Field(default_factory=list)
    amplitudes: list[float] = Field(default_factory=list)
    uncertainties: list[float] = Field(default_factory=list)
    #: `amplitude = c (log T)^-exponent`.
    exponent: float = 0.0
    exponent_error: float = 0.0
    chi_squared: float = 0.0
    degrees_of_freedom: int = 0
    #: The same fit with the exponent held at 1. Reported because "steeper
    #: than 1/log T" is the claim, and a claim about a comparison needs the
    #: thing compared against.
    inverse_log_chi_squared: float = 0.0
    inverse_log_degrees_of_freedom: int = 0
    #: How far `log T` reaches, as a ratio. The exponent means less the smaller
    #: this is, and a reader deciding how much to believe it needs it.
    lever: float = 1.0
    caveat: str = EFFECTIVE_NOT_ASYMPTOTIC
    confidence: Confidence = Confidence.NUMERICAL

    @field_validator("confidence")
    @classmethod
    def _reject_rigorous_confidence(cls, value: Confidence) -> Confidence:
        if value in RIGOROUS:
            raise ValueError(
                f"a fitted decay rate may not claim {value.value!r}: it is a "
                "weighted least squares through six measured points, about a "
                "correction to a limit no computation reaches"
            )
        return value

    @field_validator("caveat")
    @classmethod
    def _keep_the_caveat(cls, value: str) -> str:
        """An exponent detached from its range reads as an asymptotic one."""
        if value != EFFECTIVE_NOT_ASYMPTOTIC:
            raise ValueError(
                "the caveat is fixed: an effective exponent over a lever of "
                "1.8 is not the asymptotic one, and a record that can drop "
                "that sentence will be read as though it were"
            )
        return value

    @property
    def steeper_than_inverse_log(self) -> bool:
        """Is `1/log T` excluded, over this range and by this data?"""
        return (
            self.exponent_error > 0
            and (self.exponent - 1.0) / self.exponent_error > 2.0
        )


def measure_decay(
    ordinates: np.ndarray,
    bands: Sequence[tuple[float, float]],
    *,
    bins: int = 25,
    upper: float = 3.0,
) -> DecayFit:
    """Fit `amplitude = c (log T)^-alpha` across disjoint bands of zeros.

    Bands must not overlap: a band contained in another shares its data, and
    the two amplitudes then agree partly by construction -- which is how a
    nested comparison earlier produced a correlation of 0.98 that meant
    nothing.
    """
    ordered = sorted(bands)
    for (_, high), (low, _) in zip(ordered, ordered[1:], strict=False):
        if low < high:
            raise ValueError(
                f"bands overlap at ({low:g}, {high:g}): a band sharing zeros "
                "with another agrees with it by construction, and a fit "
                "through such points measures the sharing"
            )

    ordinates = np.asarray(ordinates, dtype=float)
    edges = np.linspace(0.0, upper, bins + 1)

    heights, sizes, values, errors = [], [], [], []
    for low, high in ordered:
        band = ordinates[(ordinates > low) & (ordinates <= high)]
        spacings = np.diff(unfold(band))
        if len(spacings) < 10 * bins:
            raise ValueError(
                f"band ({low:g}, {high:g}] holds {len(spacings)} spacings, too "
                f"few for {bins} bins"
            )
        value, error = amplitude(spacings, edges)
        if value <= 0:
            raise ValueError(
                f"band ({low:g}, {high:g}] measured no systematic above its "
                "noise, so it carries no amplitude to fit -- widen it or drop "
                "it, rather than fitting a zero"
            )
        heights.append(float(np.median(band)))
        sizes.append(len(spacings))
        values.append(value)
        errors.append(error)

    heights_array = np.array(heights)
    values_array = np.array(values)
    errors_array = np.array(errors)
    logs = np.log(heights_array)

    # Weighted least squares on log(amplitude) against log(log T).
    x = np.log(logs)
    weights = (values_array / errors_array) ** 2
    design = np.vstack([np.ones_like(x), x]).T
    normal = design.T @ (design * weights[:, None])
    solution = np.linalg.solve(normal, design.T @ (weights * np.log(values_array)))
    covariance = np.linalg.inv(normal)
    fitted = np.exp(design @ solution)

    # And with the exponent pinned at 1, which is the hypothesis being tested.
    plain = 1 / errors_array**2
    scale = float(np.sum(plain * values_array / logs) / np.sum(plain / logs**2))

    return DecayFit(
        bands=[tuple(band) for band in ordered],
        heights=heights,
        spacings=sizes,
        amplitudes=values,
        uncertainties=errors,
        exponent=float(-solution[1]),
        exponent_error=float(np.sqrt(covariance[1, 1])),
        chi_squared=float(np.sum(((values_array - fitted) / errors_array) ** 2)),
        degrees_of_freedom=len(values) - 2,
        inverse_log_chi_squared=float(
            np.sum(((values_array - scale / logs) / errors_array) ** 2)
        ),
        inverse_log_degrees_of_freedom=len(values) - 1,
        lever=float(logs.max() / logs.min()),
    )
