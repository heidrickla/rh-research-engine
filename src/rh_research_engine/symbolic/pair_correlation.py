"""Measure the pair correlation of the zeros against what the corpus asserts.

THE FIRST FORMULA IN THE INDEX IS `1 - (sin(pi u)/(pi u))^2`. It is Montgomery's
pair correlation function, it is a statement about how the zeros are spaced, and
nothing in the engine had ever held it against anything -- because until the
zeros could be computed in bulk there was nothing to hold it against. Three
hundred thousand ordinates is a minute now.

This is the move the pattern package exists for, applied to the corpus itself:
measure the thing that was asserted and never asked to be measured.

THE EXPRESSION IS TAKEN FROM THE INDEX, NOT RETYPED. A check that compares the
zeros against a copy of the formula is a check on the copy. `montgomery_density`
locates the recorded expression and evaluates it, so if the corpus is corrected
the check follows it -- and if the corpus stops containing it, this raises
rather than quietly testing a hard-coded twin.

UNFOLDING IS THE WHOLE DIFFICULTY. The raw gaps between ordinates shrink like
`1/log gamma`, so a correlation in them would be swamped by the trend; the
statement is about `w_n = theta(gamma_n)/pi`, whose mean spacing is 1. The
obvious-looking `gamma log(gamma/2 pi)/(2 pi)` is NOT that. It differs by a term
linear in gamma -- irrelevant to a shift, fatal to a spacing, because the
derivative of the first is `log(t/2pi)/(2pi)` and of the second
`(log(t/2pi) + 1)/(2pi)`. Every gap comes out `(L+1)/L` too wide: 13% at
t = 5000, 8.8% at t = 200000. Measured with it, the correlation sat about ten
per cent below Montgomery's curve at every point past the first -- a clean,
stable, entirely convincing deficit that was the normalisation.

WHAT AGREEMENT IS WORTH. Montgomery's is a CONJECTURE, and an asymptotic one:
it describes the limit as the height goes to infinity, so imperfect agreement at
finite height is expected rather than evidence against. Data agreeing with it is
evidence for a conjecture and is not a proof of one, and the record refuses a
rigorous confidence at construction to keep that from being forgotten.

A HISTOGRAM BIN IS A MEAN, AND THIS COMPARED IT AGAINST A POINT. `measured` is
the average density over a bin; `montgomery_density(centres)` is the curve at
the bin's midpoint. For a curved density those differ by about `w^2 f''/24` --
here 2.7e-3 at the worst bin and 5.9e-4 on average, at the default thirty bins.
It does not shrink with the sample, so it was a floor under every deviation
this module has ever reported. `level_spacing.exact_gue_bin_average` had
already been written for exactly this, one file away, and its docstring already
said so. Both curves are now averaged over the bin, and the same way.

The size of that floor is why it had to be fixed before the next paragraph:
the thing it was hiding is the same size.

LOWER-ORDER TERMS, WHICH ARE WHERE THE ARITHMETIC IS. Montgomery's curve is
universal -- random matrices and quantum billiards obey it too -- so following
it says the zeros are a spectrum and says nothing about primes. `conrey_snaith`
carries the full form with every lower-order term, and `check_pair_correlation`
will compare against both when asked. The departure between the two curves is
of order 1e-2 falling to 1e-3, which is the scale the bin-centre bias sat at.

THE POOLED SAMPLE SPANS A RANGE OF `l`, AND THE CURVE DEPENDS ON IT. The
lower-order form is a statement at a single height, through `l = log(t/2 pi)`;
the measurement pools every pair below `height`, whose `l` runs from about 4 at
the bottom to `log(height/2 pi)` at the top. Comparing a pooled measurement
against the curve at one `l` would be a result set by how the sample was
arranged. So the pairs' own `l` distribution is measured in the same loop that
builds the histogram, and the curve is averaged over it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..contracts.epistemic import RIGOROUS, Confidence

#: Where the indexed formulas live.
INDEX_PATH = Path("research_state/formula_index.json")

#: How far apart, in mean spacings, a pair still counts.
#:
#: Montgomery's result is proved only for test functions supported in `(-1, 1)`;
#: past that it is conjecture. Three covers the interesting structure -- the
#: repulsion at zero and the first oscillation -- without the tail where the
#: statistic is thin.
DEFAULT_WINDOW = 3.0

#: Ordinates skipped at the bottom.
#:
#: The unfolding is asymptotic, so the lowest zeros are normalised worst, and
#: they are the ones a short run has proportionally most of.
SKIP_LOWEST = 200

#: Bands the pairs' `l = log(t/2 pi)` is split into when averaging the
#: lower-order curve over the pooled sample.
#:
#: `l` is logarithmic in the height, so it varies slowly: over the whole of
#: `t = 400..10^6` it runs 4.2 to 12.0.
#:
#: Measured by refining against sixteen bands, at `t = 30000`: two bands are
#: out by 1.9e-3, four by 4.6e-4, eight by 9.2e-5 -- a quarter per doubling,
#: which is the midpoint rule behaving. Eight is 40x below the deviation it
#: has to resolve at the heights reachable here.
#:
#: It is a discretisation of the average, so it does not shrink with the
#: sample: at a height where the residual itself fell below about 1e-3 this
#: would have to be refined with it. The test halves and doubles the count and
#: asserts the quartering, so the number to refine to is derivable rather than
#: guessed at.
ELL_BANDS = 8

#: Contiguous chunks the sample is split into to MEASURE the estimator's own
#: scatter. See `pair_sampling_noise_floor`.
#:
#: Eight and sixteen agree to within 2% on the real zeros, so the number is not
#: doing any work; eight keeps the chunks long enough that pairs lost at chunk
#: boundaries are a part in 10^4 of the count.
NOISE_CHUNKS = 8


class PairStatistics(NamedTuple):
    """What one pass over the pairs measures.

    `ell_nodes` and `ell_weights` are the RETAINED ZEROS' height distribution,
    one count per zero, because that is the average the density is an estimate
    of: dividing by `len(unfolded)` makes its expectation
    `(1/N) sum_n R_2(.; l_n)`, with every zero weighted once.

    This first counted one per PAIR, which is the same thing to about 1e-4 and
    is not the same statistic: it weights each height by a random quantity that
    fluctuates with the very spacings being measured. Caught in review, not by
    a test, and now by `test_the_height_weights_count_zeros_and_not_pairs`.
    """

    centres: np.ndarray
    density: np.ndarray
    ell_nodes: np.ndarray
    ell_weights: np.ndarray


class PairCorrelationCheck(BaseModel):
    """How closely the measured correlation follows the asserted one."""

    model_config = ConfigDict(extra="forbid")

    height: float
    #: Ordinates used, after skipping the lowest.
    zeros: int
    window: float
    bins: int
    #: Mean absolute deviation from the corpus's curve, over the window.
    mean_deviation: float
    worst_deviation: float
    worst_at: float
    #: Bin centres and the measured density, so a reader can see the shape
    #: rather than a single number standing in for it.
    centres: list[float] = Field(default_factory=list)
    measured: list[float] = Field(default_factory=list)
    predicted: list[float] = Field(default_factory=list)
    #: The Conrey-Snaith curve with its lower-order terms, averaged over the
    #: bins and over the pairs' own `l` distribution -- empty unless asked for,
    #: because it costs a product over a million primes per quadrature point.
    lower_order: list[float] = Field(default_factory=list)
    #: Mean absolute deviation from THAT curve, when it was computed. The pair
    #: worth reading is (`mean_deviation`, `lower_order_deviation`): the first
    #: is distance from a universal law that any spectrum obeys, the second
    #: from one that knows about primes.
    lower_order_deviation: float | None = None
    #: Mean absolute gap between the two curves, over the same bins. Without
    #: it the two deviations above cannot be told from a tie -- if the curves
    #: differ by less than the sampling noise, neither number is evidence.
    curve_separation: float | None = None
    #: The estimator's own scatter, measured from the sample rather than
    #: assumed. Every deviation above is meaningless without it, and it is NOT
    #: `level_spacing.sampling_noise_floor` -- see `pair_sampling_noise_floor`
    #: for why that one is a ruler from a different quantity, and 2.2x too
    #: small.
    noise_floor: float = 0.0

    @property
    def curves_are_distinguishable(self) -> bool:
        """Whether this sample can tell the two curves apart at all.

        False is a real answer, and at `T = 5000` it is the answer: the curves
        sit 0.030 apart where the noise is 0.036. Reporting which curve the
        data is "closer to" there would be reporting a coin.
        """
        return (
            self.curve_separation is not None
            and self.noise_floor > 0.0
            and self.curve_separation > self.noise_floor
        )

    #: Never rigorous. A finite sample, compared against a conjecture that is
    #: asymptotic; agreement is evidence for it and is not a proof of it.
    confidence: Confidence = Confidence.NUMERICAL

    @field_validator("confidence")
    @classmethod
    def _reject_rigorous_confidence(cls, value: Confidence) -> Confidence:
        if value in RIGOROUS:
            raise ValueError(
                f"a measured pair correlation may not claim {value.value!r}: "
                "Montgomery's pair correlation is a conjecture and an "
                "asymptotic one, so a finite sample agreeing with it is "
                "evidence for it and never a proof of it"
            )
        return value


def unfold(ordinates: np.ndarray) -> np.ndarray:
    """Normalise the ordinates so the mean spacing is 1.

    `w_n = theta(gamma_n) / pi`. See the module docstring for why this is not
    `gamma log(gamma / 2 pi) / (2 pi)`, which looks equivalent and stretches
    every gap by about a tenth.
    """
    from .riemann_siegel import theta

    return theta(np.asarray(ordinates, dtype=float)) / np.pi


def _indexed_expression() -> str:
    """The recorded pair-correlation expression, from the formula index."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{INDEX_PATH} is missing, so there is no recorded expression to "
            "check the zeros against; run `rhre symbolic ingest` first"
        )
    records = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    for record in records:
        expression = record.get("expression", "")
        if "sin(pi*u)" in expression and "**2" in expression:
            return expression
    raise LookupError(
        "the corpus no longer records a pair-correlation expression in u; "
        "this check compared the zeros against the corpus, so there is "
        "nothing left to compare them to"
    )


def montgomery_density(u: np.ndarray) -> np.ndarray:
    """Evaluate the corpus's own pair-correlation expression at `u`."""
    import sympy as sp

    from .parser import prepare_for_parsing

    recorded = _indexed_expression()
    symbol = sp.Symbol("u")
    # The one name-resolution policy. Reading the index's own output under a
    # bare parse would read it under different rules than produced it.
    text, resolution = prepare_for_parsing(recorded)
    expression = sp.parse_expr(text, local_dict=resolution, evaluate=True)
    free = expression.free_symbols
    if free != {symbol}:
        raise LookupError(
            f"the recorded expression {recorded!r} is in "
            f"{free or 'no variables'}, "
            "not in u alone; it cannot be a density in the spacing"
        )
    function = sp.lambdify(symbol, expression, "numpy")
    u = np.asarray(u, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.asarray(function(u), dtype=float)
    # The expression has a removable singularity at u = 0, where the density is
    # zero -- the level repulsion that is the whole point of the curve.
    return np.where(u == 0, 0.0, values)


def measured_density(
    unfolded: np.ndarray,
    *,
    ordinates: np.ndarray | None = None,
    window: float = DEFAULT_WINDOW,
    bins: int = 30,
    ell_bands: int = ELL_BANDS,
) -> PairStatistics:
    """Empirical density of differences `w_n - w_m` within `window`.

    When `ordinates` is given -- the unnormalised heights the unfolded values
    came from -- the pairs' own distribution of `l = log(t/2 pi)` is counted in
    the same pass, so the lower-order curve can be averaged over the sample
    that was actually taken rather than over an assumption about it.
    """
    edges = np.linspace(0.0, window, bins + 1)
    counts = np.zeros(bins)
    ell_counts = np.zeros(ell_bands)
    ell_edges = np.zeros(ell_bands + 1)
    if ordinates is not None:
        ell_of_zero = np.log(np.asarray(ordinates, dtype=float) / (2 * np.pi))
        ell_edges = np.linspace(ell_of_zero.min(), ell_of_zero.max(), ell_bands + 1)
        # A single band would put every pair at one l and silently undo the
        # averaging this exists for.
        band_of_zero = np.clip(
            np.searchsorted(ell_edges, ell_of_zero, side="right") - 1, 0, ell_bands - 1
        )
    start = 0
    for index in range(len(unfolded)):
        while unfolded[index] - unfolded[start] > window:
            start += 1
        differences = unfolded[index] - unfolded[start:index]
        if len(differences):
            counts += np.histogram(differences, bins=edges)[0]
        if ordinates is not None:
            # ONE PER ZERO, not one per pair, and unconditionally -- a zero
            # with no pair inside the window still divides the density.
            #
            # `density` is `counts / (len(unfolded) * width)`, so its
            # expectation is `(1/N) sum_n R_2(.; l_n)`: an average over the
            # RETAINED ZEROS, each weighted once. Weighting a band by the pairs
            # observed near it instead would weight each height by a random
            # quantity that fluctuates with the very spacings being measured,
            # so the curve compared against would be a different, data-dependent
            # statistic from the one returned here. Raised by review; it was
            # `+= len(differences)`, inside the branch above.
            ell_counts[band_of_zero[index]] += 1
    width = edges[1] - edges[0]
    # Forward differences only, so each pair is counted once; the statistic is
    # symmetric, and the density is per zero.
    density = counts / (len(unfolded) * width)
    total = ell_counts.sum()
    return PairStatistics(
        centres=0.5 * (edges[:-1] + edges[1:]),
        density=density,
        ell_nodes=0.5 * (ell_edges[:-1] + ell_edges[1:]),
        ell_weights=ell_counts / total if total else ell_counts,
    )


def montgomery_bin_average(edges: np.ndarray, *, samples: int) -> np.ndarray:
    """The corpus's curve averaged over each bin, which is what a histogram is.

    See the module docstring: comparing this against `montgomery_density` at
    the bin centres is the 2.7e-3 floor that used to sit under every reported
    deviation.
    """
    from .level_spacing import curve_bin_average

    return curve_bin_average(edges, montgomery_density, samples=samples)


def pair_sampling_noise_floor(
    unfolded: np.ndarray,
    *,
    window: float = DEFAULT_WINDOW,
    bins: int = 30,
    chunks: int = NOISE_CHUNKS,
) -> float:
    """What the mean absolute deviation would be if the curve held exactly.

    NOT `level_spacing.sampling_noise_floor`, and the difference is a factor of
    about 2.2. That one is a formula for a different statistic: it evaluates
    the GUE NEAREST-NEIGHBOUR SPACING density, and it treats bin counts as
    multinomial over `N` independent draws. Neither holds here. This histogram
    counts about `3N` pairs drawn from `N` zeros, every zero appears in several
    of them, and the zeros are correlated -- which is the thing the statistic
    exists to measure. Using it anyway is a ruler borrowed from another
    quantity, and it silently reports a residual as 2.2x more significant than
    it is.

    So the scatter is MEASURED. The sample is split into contiguous chunks, the
    density is estimated in each, and the standard error of the pooled estimate
    is read off the scatter between them; `sqrt(2/pi)` converts a standard
    deviation into the mean absolute deviation that the checks report. Nothing
    is assumed about independence, because the chunks carry whatever
    correlation the zeros have.

    Contiguous rather than interleaved: pairs are between NEARBY zeros, so
    taking every k-th zero would destroy the correlation this is trying to
    keep. The cost is that chunks sit at slightly different heights -- measured,
    that inflates the estimate by 6-12% at `T = 2x10^5` and not at all at
    `T = 3x10^4`, so it is a small conservative bias and not a correction.

    VALIDATED AGAINST A KNOWN ANSWER. A Poisson process has `R_2 = 1`
    identically, so its deviation from 1 is entirely the estimator. Over
    replicates this function predicts the observed mean absolute deviation to
    1.6% at `N = 35473` and 5.5% at `N = 298000`, where the multinomial formula
    is out by 2.24x at every `N` from 4320 to 298000.
    """
    unfolded = np.asarray(unfolded, dtype=float)
    if chunks < 2:
        raise ValueError(f"a scatter needs at least two chunks to be a scatter; got {chunks}")
    size = len(unfolded) // chunks
    if size < 50:
        raise ValueError(
            f"{len(unfolded)} values in {chunks} chunks is {size} each, which "
            "is too few for the density in a chunk to mean anything; the floor "
            "would be an estimate of its own error"
        )
    densities = np.array(
        [
            measured_density(
                unfolded[index * size : (index + 1) * size], window=window, bins=bins
            ).density
            for index in range(chunks)
        ]
    )
    standard_error = densities.std(axis=0, ddof=1) / np.sqrt(chunks)
    return float(np.mean(np.sqrt(2 / np.pi) * standard_error))


def lower_order_bin_average(
    edges: np.ndarray,
    ell_nodes: np.ndarray,
    ell_weights: np.ndarray,
    *,
    samples: int,
    prime_limit: int | None = None,
) -> np.ndarray:
    """The Conrey-Snaith curve, averaged over the bins AND over the sample's `l`.

    Averaged the SAME way as `montgomery_bin_average` -- same quadrature, same
    sample count -- because the whole point is the difference between the two
    curves, and a difference of two quadratures done differently carries the
    difference of the quadratures.
    """
    from . import conrey_snaith
    from .level_spacing import curve_bin_average

    if prime_limit is None:
        prime_limit = conrey_snaith.ARITHMETIC_PRIME_LIMIT
    total = np.zeros(len(edges) - 1)
    for node, weight in zip(ell_nodes, ell_weights, strict=True):
        if weight == 0.0:
            continue
        total += weight * curve_bin_average(
            edges,
            lambda u, node=node: conrey_snaith.pair_correlation(u, node, prime_limit=prime_limit),
            samples=samples,
        )
    return total


def check_pair_correlation(
    height: float,
    *,
    window: float = DEFAULT_WINDOW,
    bins: int = 30,
    lower_order: bool = False,
    prime_limit: int | None = None,
) -> PairCorrelationCheck:
    """Measure the correlation of the zeros below `height` against the corpus.

    With `lower_order`, also against the Conrey-Snaith form carrying every
    lower-order term. That is off by default because it costs a Euler product
    over a million primes per quadrature point, not because it is optional to
    the question: Montgomery's curve is universal, so the deviation from it
    alone cannot distinguish the zeros from any other spectrum.
    """
    from .conrey_snaith import BIN_SAMPLES
    from .riemann_siegel import zero_ordinates

    ordinates = zero_ordinates(height)
    if len(ordinates) <= SKIP_LOWEST + 50:
        raise ValueError(
            f"only {len(ordinates)} zeros below {height}: the unfolding is "
            f"asymptotic and the lowest {SKIP_LOWEST} are dropped, so this "
            "height cannot say anything about the correlation"
        )
    kept = np.asarray(ordinates, dtype=float)[SKIP_LOWEST:]
    unfolded = unfold(ordinates)[SKIP_LOWEST:]
    statistics = measured_density(unfolded, ordinates=kept, window=window, bins=bins)
    centres, density = statistics.centres, statistics.density
    edges = np.linspace(0.0, window, bins + 1)
    # Bin means on both sides. See the module docstring for the 2.7e-3 this
    # replaced, and why it mattered here specifically.
    predicted = montgomery_bin_average(edges, samples=BIN_SAMPLES)
    deviation = np.abs(density - predicted)

    curve = None
    curve_deviation = None
    separation = None
    if lower_order:
        curve = lower_order_bin_average(
            edges,
            statistics.ell_nodes,
            statistics.ell_weights,
            samples=BIN_SAMPLES,
            prime_limit=prime_limit,
        )
        curve_deviation = float(np.abs(density - curve).mean())
        separation = float(np.abs(curve - predicted).mean())

    return PairCorrelationCheck(
        height=float(height),
        zeros=len(unfolded),
        window=float(window),
        bins=bins,
        mean_deviation=float(deviation.mean()),
        worst_deviation=float(deviation.max()),
        worst_at=float(centres[deviation.argmax()]),
        centres=[float(x) for x in centres],
        measured=[float(x) for x in density],
        predicted=[float(x) for x in predicted],
        lower_order=[] if curve is None else [float(x) for x in curve],
        lower_order_deviation=curve_deviation,
        curve_separation=separation,
        noise_floor=pair_sampling_noise_floor(unfolded, window=window, bins=bins),
    )
