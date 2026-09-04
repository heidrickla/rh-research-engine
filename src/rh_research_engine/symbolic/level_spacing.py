"""Nearest-neighbour spacing of the zeros, against the GUE law.

WHY A SECOND STATISTIC. `pair_correlation` already measures the zeros against
Montgomery's curve, and that is an average over ALL pairs at a given
separation. The spacing distribution is about CONSECUTIVE ones, and it is where
level repulsion lives: two independent random points can land arbitrarily close
together, and two eigenvalues of a random Hermitian matrix cannot. Pair
correlation sees that too, in the dip near `u = 0`, but the spacing
distribution is the statistic the physics is usually stated in and it is a
sharper test of the same claim.

THE CURVE IS THE WIGNER SURMISE, AND IT IS NOT THE EXACT LAW. For beta = 2,

    p(s) = (32/pi^2) s^2 exp(-4 s^2 / pi)

which comes from a 2x2 random matrix and approximates the true GUE
distribution -- a Fredholm determinant of the sine kernel, with no elementary
closed form. Naming the surmise rather than "the GUE law" is the difference
between a curve this engine can evaluate and one it cannot, and the deviation
reported below is against the surmise, so part of it is the surmise's own
error -- and how much turns out to be the interesting question.

WHAT THE RESIDUAL IS MADE OF. The deviation falls with height and flattens:
0.044 at T = 2000, 0.021 at T = 80000. This file first recorded that the
flattening was "consistent with the surmise's own error" and that nothing here
could separate the two. **The first half was wrong and the second is no longer
true.** `exact_gue_density` computes the exact law, and with it the residual
splits three ways:

    the CURVE   mean |exact - surmise|              0.0018
    NOISE       what a histogram of N would show    0.0045   at 59232 spacings
    the ZEROS   everything left                     ~0.014

So the surmise's error is about a twelfth of the residual, not the bulk of it.
Measuring against the exact law gives a slightly LARGER deviation than against
the surmise (0.0223 against 0.0210 at T = 8 x 10^4), because at these heights
the finite-height correction happens to push the zeros the way the surmise is
already wrong. That is a coincidence and not a virtue of the surmise.

WHY THE THIRD PART IS A FINDING AND NOT ARITHMETIC. Subtracting a noise floor
from a mean absolute deviation assumes something about how the two combine, and
a wrong assumption there manufactures a result. `residual_shape` asks a
question that needs no such assumption: noise is independent between disjoint
samples, so if the SIGNED residual has the same shape in bands of zeros that
share no data, the shape is not the noise. Over (0, 2x10^4], (2x10^4, 5x10^4]
and (5x10^4, 9x10^4] the pairwise correlations are 0.96, 0.96 and 0.98, where
200 samples drawn from the exact law at the same sizes gave a mean |r| of 0.24
and never exceeded 0.67.

The first version of that test used NESTED ranges -- the zeros below 30000
against those below 80000, one a third of the other -- and reported r = 0.98
that was partly guaranteed by construction. Same error as checking a shortcut
against itself.

TWO ASSUMPTIONS THAT HAD TO BE CHECKED RATHER THAN MADE. Both the noise floor
and the null treat the spacings as independent draws, and they are not:
consecutive gaps are correlated, which is what pair correlation measures.

  * The FLOOR. `bootstrap_noise_floor` resamples contiguous runs, so local
    correlation survives, and block lengths of 1, 10, 50 and 200 all give
    0.0040 to 0.0042 against the formula's 0.0045. The formula is slightly
    CONSERVATIVE, which is the right direction and not the obvious one: a
    rigid spectrum fluctuates LESS than independent points, so level repulsion
    suppresses exactly the statistic the formula overestimates.

  * The HEIGHT. Bands at different heights leave open that the residual varies
    with height rather than being a property of the zeros. Two halves of ONE
    band -- disjoint, same height -- correlate at 0.98, which closes that off.

AND THEN CHECKED AGAINST ZEROS THIS ENGINE DID NOT COMPUTE. Everything above
still leaves two readings open: that the shape is an artefact of this
zero-finder, and that it is permanent rather than a finite-height correction.
`odlyzko.py` settles both against published data. The ordinates agree with
Odlyzko's to 3.005e-9 against his stated 3e-9, and the residual computed from
his numbers correlates with ours at 1.00000 -- so the shape is in the zeros. At
his tables for indices 10^12, 10^21 and 10^22 the residual sits at the noise
floor of a 10000-zero sample, and projecting each onto the low-height shape
gives +0.04, -0.18 and +0.17 OF IT, all within the uncertainty of 0.15. The
correction is gone by 10^12, to the resolution ten thousand zeros allow.

LEVEL REPULSION IS THE ROBUST HALF. `P(s < 0.1)` is about 0.0007, where
independent points at the same density would give `1 - exp(-0.1) = 0.095`. A
factor of a hundred and thirty is not a fitted agreement that could hide a
normalisation error; it is a qualitative fact about the zeros, and it survives
whatever the residual turns out to be made of.

UNFOLDING IS THE SAME TRAP AS NEXT DOOR, so `unfold` is imported from
`pair_correlation` rather than rewritten. `theta(gamma)/pi`, never
`gamma log(gamma/2pi)/(2pi)`: the two differ by a term linear in gamma, which
is harmless to a shift and fatal to a spacing. That mistake produced a clean,
stable, entirely convincing ten-per-cent deficit once already.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..contracts.epistemic import RIGOROUS, Confidence
from .pair_correlation import unfold

#: The Poisson comparison, at unit mean density: `P(s < x) = 1 - exp(-x)`.
#:
#: Not decoration. "The spacings follow the surmise" is a fitted agreement and
#: a normalisation error can survive one; "almost no spacing is small, where
#: independent points would make a tenth of them small" cannot.
REPULSION_WINDOW = 0.1

#: Gauss-Legendre nodes for the Fredholm determinant behind the exact law.
#:
#: Measured rather than chosen: five nodes give seven digits of `E(1)`,
#: fifteen give sixteen, and past that it does not move. Set well beyond
#: the observed convergence because an unconverged determinant looks
#: exactly like a converged one.
GUE_NODES = 24

#: Step for the second difference giving `p = E''`.
#:
#: `E` is smooth and computed to about 1e-15, so the error is roughly
#: `1e-15/h^2 + h^2`, minimised near `h = 3e-4` and flat around it. At 1e-3
#: the density's moments come back as 1.000000005 and 1.000000000, which
#: is four orders better than the effect being measured.
GUE_STEP = 1e-3


def wigner_surmise(spacing: np.ndarray) -> np.ndarray:
    """`p(s) = (32/pi^2) s^2 exp(-4 s^2/pi)`, the GUE surmise.

    Written out rather than read from the formula index, because the corpus
    does not contain it -- and a check against a curve nothing recorded is a
    check on this function. `check_level_spacing` reports the curve by name in
    `model` so a reader is never left to guess which one was used.
    """
    values = np.asarray(spacing, dtype=float)
    return (32 / np.pi**2) * values**2 * np.exp(-4 * values**2 / np.pi)


class LevelSpacingCheck(BaseModel):
    """The measured spacing distribution against the Wigner surmise."""

    model_config = ConfigDict(extra="forbid")

    height: float
    zeros: int
    bins: int
    upper: float
    #: The curve compared against, by name. A deviation is meaningless without
    #: it, and "GUE" alone would not distinguish the surmise from the exact
    #: distribution -- which differ by more than the finite-height effect being
    #: looked for.
    model: str = "Wigner surmise (beta = 2)"
    #: Mean spacing after unfolding. Should be 1; anything else means the
    #: unfolding is wrong, and every other number here with it.
    mean_spacing: float
    #: Mean absolute deviation from `model` -- the Wigner surmise.
    mean_deviation: float
    #: And from the EXACT law. Reported beside it rather than instead of it,
    #: because the surmise is what the literature usually states and the two
    #: differ by less than either differs from the zeros.
    deviation_from_exact: float
    #: `mean |exact - surmise|` over the same bins: how much of a deviation
    #: from the surmise is the surmise. Measured at 0.0018, which is about a
    #: twelfth of what the zeros actually show -- so the answer is "almost
    #: none of it", and the earlier guess that the floor was "consistent with
    #: the surmise's own error" was wrong.
    curve_error: float
    #: What the deviation would be if the law held exactly and only the
    #: histogram fluctuated. A finding below this number is the histogram.
    noise_floor: float
    worst_deviation: float
    worst_at: float
    #: `P(s < REPULSION_WINDOW)` measured, and what independent points would
    #: give. The qualitative half of the finding, and the half a normalisation
    #: error cannot fake.
    repulsion: float
    poisson_repulsion: float
    centres: list[float] = Field(default_factory=list)
    measured: list[float] = Field(default_factory=list)
    predicted: list[float] = Field(default_factory=list)
    #: Never rigorous, for the same reason the pair correlation is not: a
    #: finite sample against an asymptotic law.
    confidence: Confidence = Confidence.NUMERICAL

    @field_validator("confidence")
    @classmethod
    def _reject_rigorous_confidence(cls, value: Confidence) -> Confidence:
        if value in RIGOROUS:
            raise ValueError(
                f"a measured spacing distribution may not claim {value.value!r}: "
                "the GUE law for the zeros is a conjecture and an asymptotic "
                "one, and the curve compared against is a surmise that is not "
                "even the exact law -- a finite sample agreeing with it is "
                "evidence and never a proof"
            )
        return value

    @property
    def repulsion_factor(self) -> float:
        """How many times rarer a close pair is than chance would make it."""
        return self.poisson_repulsion / self.repulsion if self.repulsion else float("inf")


def check_level_spacing(
    height: float, *, bins: int = 25, upper: float = 3.0
) -> LevelSpacingCheck:
    """Measure consecutive spacings of the zeros below `height`.

    Raises rather than returning a thin result when there are too few zeros:
    a histogram over 25 bins built from a hundred spacings is noise with a
    shape, and reporting its deviation as a small number would be worse than
    reporting nothing.
    """
    from .riemann_siegel import zero_ordinates

    ordinates = np.asarray(
        [float(value) for value in zero_ordinates(height)], dtype=float
    )
    if len(ordinates) < 10 * bins:
        raise ValueError(
            f"{len(ordinates)} zeros below T = {height:g} is too few for "
            f"{bins} bins: a histogram needs enough spacings that its shape is "
            "not noise, and a deviation measured from noise is a number "
            "without a meaning"
        )

    spacings = np.diff(unfold(ordinates))
    counts, edges = np.histogram(
        spacings, bins=bins, range=(0.0, upper), density=True
    )
    centres = (edges[:-1] + edges[1:]) / 2
    # Averaged over each bin, not evaluated at its centre: a histogram bin IS
    # an average, and the two differ by enough to floor every residual.
    predicted = curve_bin_average(edges, wigner_surmise)
    exact = exact_gue_bin_average(edges)

    deviations = np.abs(counts - predicted)
    worst = int(np.argmax(deviations))
    repulsion = float((spacings < REPULSION_WINDOW).mean())

    return LevelSpacingCheck(
        height=height,
        zeros=len(ordinates),
        bins=bins,
        upper=upper,
        mean_spacing=float(spacings.mean()),
        mean_deviation=float(deviations.mean()),
        deviation_from_exact=float(np.mean(np.abs(counts - exact))),
        curve_error=float(np.mean(np.abs(exact - predicted))),
        noise_floor=sampling_noise_floor(len(spacings), centres, upper / bins),
        worst_deviation=float(deviations[worst]),
        worst_at=float(centres[worst]),
        repulsion=repulsion,
        poisson_repulsion=float(1 - np.exp(-REPULSION_WINDOW)),
        centres=[float(value) for value in centres],
        measured=[float(value) for value in counts],
        predicted=[float(value) for value in predicted],
    )


# --- the exact law, and the two floors that stand between it and a finding --


def _sine_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """`K(x,y) = sin(pi(x-y))/(pi(x-y))`, with the removable point filled in."""
    difference = np.subtract.outer(x, y)
    out = np.ones_like(difference)
    away = difference != 0
    out[away] = np.sin(np.pi * difference[away]) / (np.pi * difference[away])
    return out


@lru_cache(maxsize=8192)
def gap_probability(s: float, *, nodes: int = GUE_NODES) -> float:
    """`E(s) = det(I - K_s)`: no eigenvalue in a gap of length `s`.

    A Fredholm determinant, evaluated by Nystrom discretisation on
    Gauss-Legendre nodes. Bornemann's point is that this converges
    super-exponentially for an analytic kernel, and the sine kernel is entire:
    measured here, five nodes give seven digits and fifteen give sixteen, after
    which it does not move. `GUE_NODES` is set well past that, because a
    determinant that has silently not converged looks exactly like one that
    has.

    Cached: a pure function of two numbers, and the same `s` recurs constantly
    -- every bin edge, at every band, on every call. Without it the density
    over a 20000-point grid is sixty thousand determinants, and the tests that
    sample from it took three minutes.
    """
    if s <= 0:
        return 1.0
    points, weights = np.polynomial.legendre.leggauss(nodes)
    points = 0.5 * s * (points + 1.0)
    weights = 0.5 * s * weights
    root = np.sqrt(weights)
    matrix = np.eye(nodes) - root[:, None] * _sine_kernel(points, points) * root[None, :]
    return float(np.linalg.det(matrix))


def exact_gue_density(
    s: np.ndarray, *, nodes: int = GUE_NODES, step: float = GUE_STEP
) -> np.ndarray:
    """`p(s) = E''(s)`: the exact GUE spacing density, not the surmise.

    THE STENCIL IS THE WHOLE DIFFICULTY. `E` is not defined for a gap of
    negative length, so below `s = step` the centred three-point formula
    reaches past the origin. Clamping it there -- the obvious fix -- leaves
    three points that are not equally spaced, and the formula returns about
    -499 near zero rather than a small positive number.

    That version integrated to 0.0005 instead of 1 and its FIRST moment still
    came out at 1.0000, because the bad values sit where `s` is nearly zero and
    are multiplied away. Checking only the mean would have passed it. Below the
    step, a forward difference of the same order is used instead.
    """
    values = np.atleast_1d(np.asarray(s, dtype=float))
    out = np.empty_like(values)
    for index, value in enumerate(values):
        if value >= step:
            out[index] = (
                gap_probability(value - step, nodes=nodes)
                - 2 * gap_probability(value, nodes=nodes)
                + gap_probability(value + step, nodes=nodes)
            ) / step**2
        else:
            out[index] = (
                gap_probability(value, nodes=nodes)
                - 2 * gap_probability(value + step, nodes=nodes)
                + gap_probability(value + 2 * step, nodes=nodes)
            ) / step**2
    return out


def gap_slope(s: float, *, step: float = GUE_STEP, nodes: int = GUE_NODES) -> float:
    """`E'(s)`, the derivative of the gap probability."""
    if s < step:
        return (
            gap_probability(s + step, nodes=nodes)
            - gap_probability(max(s - step, 0.0), nodes=nodes)
        ) / (step + min(s, step))
    return (
        gap_probability(s + step, nodes=nodes)
        - gap_probability(s - step, nodes=nodes)
    ) / (2 * step)


def exact_gue_bin_average(edges: np.ndarray) -> np.ndarray:
    """The exact density AVERAGED over each bin, which is what a histogram is.

    `np.histogram(..., density=True)` returns the mean density over a bin.
    Comparing that against the density AT THE BIN CENTRE compares two different
    quantities: for a curved density they differ by about `w^2 p''/24`, which
    at `w = 0.12` is of order 1e-3 per bin.

    It is small and it does not shrink with the sample, so it is a floor under
    every residual -- and it is the whole of what the estimator reports once
    the sample is large. The null test found it as a bias of 0.44 of the noise
    term at 500000 draws, refining the sampler did not remove it, and switching
    to bin averages took it to -0.035.

    No quadrature is needed: `p = E''`, so the mean of `p` over `[a, b]` is
    exactly `(E'(b) - E'(a))/(b - a)`.
    """
    slopes = np.array([gap_slope(float(edge)) for edge in np.asarray(edges)])
    return (slopes[1:] - slopes[:-1]) / np.diff(np.asarray(edges, dtype=float))


def curve_bin_average(edges: np.ndarray, curve, *, samples: int = 101) -> np.ndarray:
    """The same for a curve with no antiderivative to hand -- the surmise.

    Simpson-grade quadrature per bin. The surmise is smooth and the bins are
    narrow, so 101 points is far past what the comparison needs; the point is
    that the surmise and the exact law are averaged the SAME way, or the
    difference between them would carry a discretisation artefact.
    """
    edges = np.asarray(edges, dtype=float)
    return np.array(
        [
            np.trapezoid(curve(np.linspace(a, b, samples)), np.linspace(a, b, samples))
            / (b - a)
            for a, b in zip(edges[:-1], edges[1:], strict=True)
        ]
    )


def sampling_noise_floor(spacings: int, centres: np.ndarray, width: float) -> float:
    """What the mean absolute deviation would be if the law held exactly.

    Bin counts are multinomial, so the density estimate in a bin has standard
    deviation `sqrt(p/(N w))`, and the mean absolute deviation of a normal is
    `sqrt(2/pi)` times its standard deviation.

    A finding smaller than this number is the histogram. It matters most at
    the heights where the zeros are cheapest: at 1500 spacings the floor is
    0.028, which is most of the deviation that used to be reported as though it
    were about zeta.
    """
    edges = np.concatenate([[centres[0] - width / 2], centres + width / 2])
    density = np.maximum(exact_gue_bin_average(edges), 0.0)
    return float(np.mean(np.sqrt(2 / np.pi) * np.sqrt(density / (spacings * width))))


def bootstrap_noise_floor(
    spacings: np.ndarray,
    centres: np.ndarray,
    width: float,
    *,
    block: int = 50,
    replicates: int = 200,
    seed: int = 3,
) -> float:
    """The same floor, without assuming the spacings are independent.

    THEY ARE NOT INDEPENDENT, which is the whole reason this exists.
    `sampling_noise_floor` treats bin counts as multinomial -- exact for
    independent draws -- and consecutive gaps between zeros are correlated,
    since that is what pair correlation measures. So the formula rests on an
    assumption the data violates, and the number it produces decides whether a
    residual is a finding.

    A moving-block bootstrap resamples contiguous RUNS of the observed
    spacings, so whatever local correlation is present survives into the
    replicates. Measured on the band (5x10^4, 9x10^4], block lengths of 1, 10,
    50 and 200 all give 0.0040 to 0.0042 against the formula's 0.0045: the
    correlation does not inflate the fluctuation, and the formula is slightly
    CONSERVATIVE rather than optimistic.

    That is the right direction and not the obvious one. A rigid spectrum
    fluctuates LESS than independent points, not more -- level repulsion
    suppresses the counting statistics -- so the assumption errs toward calling
    a real residual noise, which is the error worth having.
    """
    generator = np.random.default_rng(seed)
    edges = np.concatenate(
        [[centres[0] - width / 2], centres + width / 2]
    )
    count = len(spacings)
    blocks = int(np.ceil(count / block))
    densities = []
    for _ in range(replicates):
        starts = generator.integers(0, count - block + 1, size=blocks)
        drawn = np.concatenate([spacings[s : s + block] for s in starts])[:count]
        density, _ = np.histogram(drawn, bins=edges, density=True)
        densities.append(density)
    errors = np.std(np.asarray(densities), axis=0)
    return float(np.mean(np.sqrt(2 / np.pi) * errors))


class ResidualShape(BaseModel):
    """Whether what is left over is a shape or is the histogram.

    Subtracting a noise floor from a mean absolute deviation assumes something
    about how the two combine, and a wrong assumption there manufactures a
    finding. This asks a question with no such assumption: noise is
    independent between disjoint samples, so if the SIGNED per-bin residual has
    the same shape in bands that share no zeros, the shape is not the noise.
    """

    model_config = ConfigDict(extra="forbid")

    bands: list[tuple[float, float]] = Field(default_factory=list)
    spacings: list[int] = Field(default_factory=list)
    deviations: list[float] = Field(default_factory=list)
    noise_floors: list[float] = Field(default_factory=list)
    #: Pearson correlation of the signed residual between each pair of bands.
    correlations: list[float] = Field(default_factory=list)
    #: The same statistic on samples drawn FROM the exact law at the same
    #: sizes -- what "no shape at all" looks like, rather than an assumed zero.
    null_mean: float = 0.0
    null_worst: float = 0.0
    confidence: Confidence = Confidence.NUMERICAL

    @field_validator("confidence")
    @classmethod
    def _reject_rigorous_confidence(cls, value: Confidence) -> Confidence:
        if value in RIGOROUS:
            raise ValueError(
                f"a residual shape may not claim {value.value!r}: it is a "
                "correlation measured on a finite sample, and the thing it "
                "detects -- a finite-height correction -- is a statement "
                "about a limit no computation reaches"
            )
        return value

    @property
    def is_a_shape(self) -> bool:
        """Every band pair correlates beyond anything the null produced."""
        return bool(self.correlations) and min(self.correlations) > self.null_worst


def residual_shape(
    bands: Sequence[tuple[float, float]],
    *,
    bins: int = 25,
    upper: float = 3.0,
    trials: int = 200,
    seed: int = 7,
) -> ResidualShape:
    """Compare the leftover shape across bands of zeros that share no data.

    BANDS, NOT NESTED RANGES. Comparing the zeros below 30000 against those
    below 80000 correlates two sets one of which is a third of the other, so
    they agree partly by construction -- the same error as checking a shortcut
    against itself. It gave r = 0.98 and would have been reported as a
    property of the zeros.
    """
    from .riemann_siegel import zero_ordinates

    ceiling = max(high for _, high in bands)
    ordinates = np.asarray(
        [float(value) for value in zero_ordinates(ceiling)], dtype=float
    )
    edges = np.linspace(0.0, upper, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    width = upper / bins
    exact = exact_gue_bin_average(edges)

    residuals, sizes, deviations, floors = [], [], [], []
    for low, high in bands:
        band = ordinates[(ordinates > low) & (ordinates <= high)]
        spacings = np.diff(unfold(band))
        if len(spacings) < 10 * bins:
            raise ValueError(
                f"band ({low:g}, {high:g}] holds {len(spacings)} spacings, too "
                f"few for {bins} bins"
            )
        counts, _ = np.histogram(spacings, bins=edges, density=True)
        residuals.append(counts - exact)
        sizes.append(len(spacings))
        deviations.append(float(np.mean(np.abs(counts - exact))))
        floors.append(sampling_noise_floor(len(spacings), centres, width))

    correlations = [
        float(np.corrcoef(residuals[i], residuals[j])[0, 1])
        for i in range(len(residuals))
        for j in range(i + 1, len(residuals))
    ]

    # The null, measured rather than assumed to be zero. With 25 bins a
    # correlation of two independent noise vectors has a standard deviation
    # near 0.2, so "greater than zero" is not the bar.
    generator = np.random.default_rng(seed)
    grid = np.linspace(0.0, 9.0, 4001)
    density = exact_gue_density(grid)
    cumulative = np.cumsum(density) * (grid[1] - grid[0])
    cumulative /= cumulative[-1]
    nulls = []
    for _ in range(trials):
        drawn = [
            np.histogram(
                np.interp(generator.random(size), cumulative, grid),
                bins=edges,
                density=True,
            )[0]
            - exact
            for size in sizes[:2]
        ]
        nulls.append(abs(float(np.corrcoef(drawn[0], drawn[1])[0, 1])))

    return ResidualShape(
        bands=list(bands),
        spacings=sizes,
        deviations=deviations,
        noise_floors=floors,
        correlations=correlations,
        null_mean=float(np.mean(nulls)),
        null_worst=float(np.max(nulls)),
    )
