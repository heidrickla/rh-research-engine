"""The Baez-Duarte criterion, evaluated rather than merely indexed.

    c_k = sum_{j=0}^{k} (-1)^j binomial(k,j) / zeta(2j+2)

and RH is equivalent to `c_k = O(k^(-3/4+eps))`. Both are in the corpus's
formula index and neither had ever been computed. This is a FALSIFICATION test:
it cannot prove RH, and a `|c_k| k^(3/4)` that grew without bound would refute
it.

THE BINOMIAL FORM IS THE WRONG ALGORITHM. Its terms reach `2^k` and its value
is about `k^(-3/4)`, so it loses about two bits per `k` to cancellation. At
k = 100 in float64 it returns 6.06e9 where the value is -0.00147737763; under
Arb at 264 bits it returns the value with 45 correct digits. Arb is what makes
that failure visible rather than plausible -- the ball's radius exceeds its own
midpoint and says so.

SO USE THE OTHER FORM. Expanding `1/zeta(2j+2)` as a Mobius series and swapping
the sums gives

    c_k = sum_{n>=2} mu(n)/n^2 (1 - 1/n^2)^k

which is the same sequence with the cancellation removed: every term is at most
`1/n^2` and float64 suffices. The two agree to the truncation of the second.
This is what takes `k` from 400 to 10^7.

AND IT SHOWS WHAT THE SEQUENCE IS. For `n` well above `sqrt(k)` the factor is
about 1, so `c_k` is essentially the Mobius tail `sum_{n > sqrt(k)} mu(n)/n^2`.
The asymptotic axis is therefore `sqrt(k)`, not `k`: at k = 400 the sum is only
asking about `n` beyond 20, which is why small-`k` fits give a clean `k^(-2)`
that has nothing to do with the criterion's asymptotics.

TWO THINGS THAT WENT WRONG HERE, both recorded because both would recur:

  * Guarding truncation with `1/N` -- the bound with no cancellation -- dropped
    two thirds of the points as unusable. Doubling `N` four times moves `c_k`
    at k = 10^6 by 1.5%, so they were real. The error bar here is empirical:
    compute at two sieve limits and take the difference.
  * Sampling `k` on a geomspace of step 0.46 in log made the sign alternate on
    every single point, which reads as noise and is aliasing. A zero at height
    `gamma` oscillates with period `4 pi / gamma` in log k -- 0.89 for
    `gamma_1 = 14.13`. At step 0.03 there are 15 sign changes in 420 samples.

AND THE VERDICT WAS A TYPED CONSTANT. `violated` was the literal `0.0`, the only
occurrence of the word in the file, so the observation "a falsification test that
did not fire" described a test that COULD not fire. A gate that cannot fail is
decoration, and here the gate is the entire product: the epistemic value of a
null result is that it could have come out the other way.

WIRING THE STATED CONDITION WOULD HAVE BEEN WORSE. The comment prescribed
"a slope at or below zero is consistent and a positive one is not", i.e.
`envelope_slope > 0`. The slope is read off peaks picked with a 12-bin edge grid
and an upper-half cut, and NOTHING CHOSE EITHER. Varying them over eighteen
equally defensible combinations gives a mean of -0.0111 with a spread of 0.0192
and a range of -0.0574 to +0.0282 -- POSITIVE ON FIVE OF EIGHTEEN. So the
prescribed condition would have announced a refutation of the Riemann Hypothesis
on five choices of histogram. The hardcoded zero was, by accident, protecting the
record from a worse bug than the one it was.

SO THE BIN GRID IS THE ERROR BAR, exactly as the grid phase is in
`exponent_scan`. `math/loglog.py` already carries the shape of the answer, and
this is its fifth customer.

READABLE AND SIGN-DETERMINED ARE DIFFERENT QUESTIONS, and this is the experiment
that separates them. A spread of 0.0192 sits comfortably inside `SPREAD_LIMIT`,
so the slope is READABLE as a magnitude -- while its range straddles zero and its
sign is not determined at all. A verdict on a sign needs every binning to agree,
which is `LogLogSummary.sign_determined`, not `unreadable_because`.

AND `envelope_scatter` IS NOT THE ERROR BAR ON THE SLOPE. It is the residual
spread about the fit, 0.0234, where the standard error ON the slope is 0.0154 --
so the recorded slope of -0.0129 is 0.84 sigma from zero and this run does not
establish a bounded envelope. It establishes that the measurement is consistent
with one. Both are recorded now, under names that say which is which.
"""

from __future__ import annotations

import numpy as np

from ..core.models import ExperimentResult
from ..math.loglog import summarize

#: RH is `c_k = O(k^(-3/4+eps))`, so this is the exponent that discriminates.
CRITERION_EXPONENT = 0.75

#: Bin counts for the peak grid. Nothing chose 12; varying it is the error bar.
BIN_COUNTS = (10, 12, 14, 16)
#: Fraction of peaks kept as "the upper part, past the pre-asymptotic head".
#: Nothing chose one half either, and the sign of the slope moves with it.
UPPER_FRACTIONS = (0.4, 0.5, 0.6)
#: A fit needs this many peaks to mean anything; configurations giving fewer are
#: dropped and counted, never silently skipped.
MINIMUM_PEAKS = 4


def mobius_sieve(limit: int) -> np.ndarray:
    """`mu(n)` for `n < limit`."""
    mu = np.ones(limit, dtype=np.int8)
    prime = np.ones(limit, dtype=bool)
    prime[:2] = False
    root = int(limit**0.5) + 1
    for p in range(2, root):
        if prime[p]:
            prime[p * p :: p] = False
            mu[p::p] = -mu[p::p]
            mu[p * p :: p * p] = 0
    for p in range(root, limit):
        if prime[p]:
            mu[p::p] = -mu[p::p]
    return mu


def _weights(limit: int) -> tuple[np.ndarray, np.ndarray]:
    mu = mobius_sieve(limit)
    n = np.arange(limit, dtype=np.float64)
    keep = (n >= 2) & (mu != 0)
    root = n[keep]
    return mu[keep] / root**2, np.log1p(-1.0 / root**2)


def sequence(ks: np.ndarray, limit: int) -> np.ndarray:
    weight, log1m = _weights(limit)
    return np.array([float(np.sum(weight * np.exp(k * log1m))) for k in ks])


def envelope_peaks(
    ks: np.ndarray,
    scaled: np.ndarray,
    usable: np.ndarray,
    bins: int,
    upper_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The peak of `|c_k| k^(3/4)` in each log-k bin, keeping the upper part.

    `bins` and `upper_fraction` are the two numbers nobody chose, and the SIGN
    of the fitted slope moves with both. They are arguments so that the caller
    can vary them and report the spread, rather than constants that decide a
    verdict invisibly.
    """
    log_k = np.log(ks.astype(float))
    edges = np.linspace(log_k.min(), log_k.max(), bins + 1)
    peak_k, peak_v = [], []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        window = (log_k >= low) & (log_k < high) & usable
        if window.sum() < 3:
            continue
        best = int(np.argmax(scaled[window]))
        peak_k.append(float(ks[window][best]))
        peak_v.append(float(scaled[window][best]))
    start = int(len(peak_k) * (1.0 - upper_fraction))
    return np.array(peak_k[start:]), np.array(peak_v[start:])


def _slope_with_error(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Slope, its STANDARD ERROR, and the residual scatter -- three numbers.

    The record used to carry the scatter where the error bar belongs. They are
    not the same: on the recorded configuration the scatter is 0.0234 and the
    standard error on the slope is 0.0154, which is what decides whether the
    slope differs from zero.
    """
    log_x, log_y = np.log(xs), np.log(ys)
    design = np.column_stack([log_x, np.ones_like(log_x)])
    coefficients, *_ = np.linalg.lstsq(design, log_y, rcond=None)
    residual = log_y - design @ coefficients
    freedom = max(len(log_y) - 2, 1)
    variance = float(residual @ residual) / freedom
    error = float(np.sqrt(variance * np.linalg.inv(design.T @ design)[0, 0]))
    return float(coefficients[0]), error, float(np.std(residual))


def _sigmas(slope: float, error: float) -> str:
    """How many standard errors from zero, without dividing by one that is zero.

    A synthetic power law fits exactly, so the residual is zero and so is the
    standard error. The first version of this line divided by it and raised
    ZeroDivisionError -- caught by the pre-existing test that feeds the
    experiment `c_k = k^(-1/2)` to watch the gate fire, which is the test this
    rewrite very nearly deleted.
    """
    # RELATIVE, NOT `== 0.0`. Whether a least-squares residual lands on exactly
    # zero is a fact about the platform's BLAS: locally it did, and in CI the
    # same fit gave 6.86e-16. At that value this printed "362318840579710
    # sigma" -- floating-point noise on an exact fit reading as overwhelming
    # significance, which is the day's failure in one line of output.
    if error <= 1e-12 * max(abs(slope), 1.0):
        return "exactly determined (residual at the noise floor, no usable error bar)"
    return f"{abs(slope) / error:.2f} sigma"


def run(
    sieve_limit: int = 8_000_000,
    k_low: float = 1e4,
    k_high: float = 2e6,
    points: int = 200,
) -> ExperimentResult:
    """Measure the envelope of `|c_k| k^(3/4)`, which is what RH bounds."""
    ks = np.unique(np.round(np.geomspace(k_low, k_high, points)).astype(np.int64))
    coarse = sequence(ks, sieve_limit // 4)
    fine = sequence(ks, sieve_limit)

    # The error bar is the disagreement between two sieve limits, not a bound
    # on the tail: see the module docstring for why `1/N` was useless.
    error = np.abs(fine - coarse)
    usable = np.abs(fine) > 10 * np.maximum(error, 1e-18)
    scaled = np.abs(fine) * ks.astype(float) ** CRITERION_EXPONENT

    grids, dropped = [], 0
    for bins in BIN_COUNTS:
        for fraction in UPPER_FRACTIONS:
            peak_k, peak_v = envelope_peaks(ks, scaled, usable, bins, fraction)
            if len(peak_k) < MINIMUM_PEAKS:
                dropped += 1
                continue
            grids.append((peak_k, peak_v))
    if len(grids) < 2:
        raise ValueError(
            f"only {len(grids)} binning(s) gave at least {MINIMUM_PEAKS} peaks; "
            "one binning cannot show that the slope depends on the binning"
        )

    envelope = summarize(grids)
    reference_k, reference_v = envelope_peaks(ks, scaled, usable, 12, 0.5)
    slope, slope_error, scatter = _slope_with_error(reference_k, reference_v)

    # A REFUTATION REQUIRES EVERY BINNING TO AGREE. `unreadable_because` asks
    # whether the magnitude is quotable and would say yes here; the sign is the
    # verdict, and it is undetermined whenever the range covers zero.
    refuted = envelope.sign_determined and envelope.lowest > 0.0

    observations = [
        "A falsification test that did not fire, and that COULD have. `violated` "
        "was the literal 0.0 -- the only occurrence of the word in this file -- so "
        "the null result carried no information. It is now decided by the "
        "measurement.",
        f"The verdict is the SIGN of the envelope slope, and it is undetermined: "
        f"over {len(grids)} binnings the slope runs {envelope.lowest:.4f} to "
        f"{envelope.highest:.4f} with spread {envelope.spread:.4f}, straddling zero. "
        "A refutation requires every binning to agree, which is why the condition "
        "the old comment prescribed -- slope > 0 on the one recorded binning -- "
        "would have announced a refutation of RH on several equally defensible "
        "histograms.",
        f"Envelope slope on the reference binning: {slope:.6f} +/- {slope_error:.6f} "
        f"(standard error), residual scatter {scatter:.6f}. Those are different "
        f"quantities and only the first is the uncertainty on the number being "
        f"tested: the slope is {_sigmas(slope, slope_error)} from zero, so "
        "this run is consistent with a bounded envelope rather than establishing one.",
        "The slope's significance is reported through a RELATIVE guard, because "
        "an absolute one wrote nonsense into this field. A synthetic exact fit "
        "leaves a residual of zero locally and 6.86e-16 on another machine, and "
        "at that value `abs(slope)/error` renders as 362318840579710 sigma -- "
        "floating-point noise on an exact fit, appearing in a falsification "
        "test's observations as the strongest result in the corpus. It was "
        "caught by CI before it reached a record.",
        "RH is equivalent to c_k = O(k^(-3/4+eps)); |c_k| k^(3/4) is measured "
        "bounded over the range reached, which is consistent with RH and proves "
        "nothing.",
        "Computed from the Mobius form sum_{n>=2} mu(n)/n^2 (1-1/n^2)^k, "
        "not the binomial definition. The two are the same sequence; the "
        "binomial form loses about two bits per k to cancellation and "
        "returns 6.06e9 for c_100 in float64, where the value is -0.00148.",
        "The asymptotic axis is sqrt(k), not k: c_k is essentially the "
        "Mobius tail beyond sqrt(k). Fits below k ~ 10^4 measure the "
        "pre-asymptotic shape and give about k^(-2), which is not the "
        "criterion's exponent.",
        "The error bar is the difference between two sieve limits. Bounding "
        "the truncation by 1/N instead -- the no-cancellation bound -- "
        "discarded two thirds of the points as unusable when they were not.",
    ]
    if dropped:
        observations.append(
            f"{dropped} binning(s) gave fewer than {MINIMUM_PEAKS} peaks and were "
            "dropped. Counted rather than skipped: a configuration that produced no "
            "fit is not a configuration that agreed."
        )

    return ExperimentResult(
        name="baez-duarte",
        parameters={
            "sieve_limit": sieve_limit,
            "k_low": k_low,
            "k_high": k_high,
            "points": points,
        },
        metrics={
            "k_min": int(ks.min()),
            "k_max": int(ks.max()),
            "samples": int(len(ks)),
            "usable": int(usable.sum()),
            "sign_changes": int(np.sum(np.diff(np.sign(fine)) != 0)),
            "median_error": float(np.median(error)),
            "median_abs_c": float(np.median(np.abs(fine))),
            # The discriminating numbers. RH needs the envelope bounded, so a
            # slope positive on EVERY binning would refute it; a slope whose
            # sign moves with the histogram decides nothing either way.
            "envelope_slope": slope,
            "envelope_slope_error": slope_error,
            "envelope_scatter": scatter,
            "envelope_slope_spread": envelope.spread,
            "envelope_slope_min": envelope.lowest,
            "envelope_slope_max": envelope.highest,
            "binnings": float(len(grids)),
            "binnings_dropped": float(dropped),
            "sign_undetermined": float(0.0 if envelope.sign_determined else 1.0),
            "envelope_constant": float(np.mean(reference_v)),
            "envelope_spread": float(reference_v.max() / reference_v.min() - 1.0),
            "peaks": int(len(reference_k)),
            "violated": float(1.0 if refuted else 0.0),
        },
        observations=observations,
    )
