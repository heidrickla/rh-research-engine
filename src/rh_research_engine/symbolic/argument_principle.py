"""Counting the zeros in the STRIP, without looking at the critical line.

WHY A SECOND COUNTER. `riemann_siegel` finds critical-line zeros by the sign
changes of `Z`, and checks its list against `ZeroCount`. That check catches a
grid that stepped over a close pair, and it is worth having -- but it is not
independent: `mpmath.nzeros` separates the zeros inside a Gram block by the
sign changes of `Z` too. Two routes to the same quantity, one much cheaper.

So it cannot answer the question worth asking. The zeros counted by the
argument principle are the zeros in the STRIP, wherever they sit; the sign
changes of `Z` are zeros ON the line, of odd order. Those two numbers agreeing
below a height says every zero below it is on the critical line and simple --
which is what a verification of the Riemann hypothesis to that height IS.

    N(T) = theta(T)/pi + 1 + S(T),        S(T) = (1/pi) arg zeta(1/2 + iT)

with `arg` taken by continuous variation along `2 -> 2 + iT -> 1/2 + iT`. That
path never touches the critical line, so nothing here depends on where the
zeros are. The main term is the argument principle applied to the completed
`xi`, and `S(T)` is the whole content: it is the correction that says how far
the actual count sits from the smooth one.

WHAT THIS DOES AND DOES NOT ESTABLISH. It is a finite computation in floating
point. It is EVIDENCE, filed as such:

  * the counts agreeing to a height is not a proof of anything about zeros
    above that height, and RH is a statement about all of them;
  * `arg` is tracked on a finite sample, so a winding that happens entirely
    between two samples is missed. The sampling is refined until the tracked
    increments are all well below pi, which makes that unlikely and not
    impossible;
  * float64 evaluation of zeta carries no error bound at all, so this
    record is NUMERICAL and refuses anything stronger at construction.

Stated here because "verified to height T" is exactly the phrase that gets
repeated without its qualifiers.

WHAT IS NOW CERTIFIED, AND WHAT STILL IS NOT. `symbolic/certified_line.py`
does the same job through Arb's interval arithmetic and is filed
RIGOROUS_NUMERICAL, so the third bullet above is no longer the end of the
story -- but it does not retire this module, because the two halves of the
question have wildly different costs. The certified COUNT is free at any
height and now confirms every figure this file has produced, including
`N(10^6) = 1747146`; the certified POSITIONS cost milliseconds per zero and
reach `T = 10^4`. Above that height the sign changes of `Z` computed here are
still the only thing placing the zeros, and they are still floating point.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..contracts.epistemic import RIGOROUS, Confidence

#: Where the path starts, on the right of the critical strip.
#:
#: At `sigma = 2` the Euler product converges absolutely and `zeta` is within
#: `zeta(2) - 1` of 1, so `arg zeta(2 + it)` stays inside a small neighbourhood
#: of zero and never winds. That is what makes the vertical leg contribute
#: nothing and leaves `S(T)` to the horizontal one.
RIGHT_EDGE = 2.0

#: The largest increment in tracked `arg` that is accepted without refining.
#:
#: Continuous variation is being reconstructed from samples, so an increment
#: near `pi` is a sample too coarse to tell `+d` from `d - 2 pi`. Kept well
#: below `pi` rather than just under it: the reconstruction is only as good as
#: the assumption that nothing happened between two samples.
MAX_INCREMENT = 0.5


class StripCount(BaseModel):
    """How many zeros the argument principle puts below a height."""

    model_config = ConfigDict(extra="forbid")

    height: float
    #: `N(T)`, rounded to the integer it must be.
    count: int
    #: `theta(T)/pi + 1`, the smooth main term.
    smooth: float
    #: `S(T)`, the correction. Known to be `O(log T)` and small in practice; a
    #: value far outside that is a sign the tracking went wrong rather than a
    #: discovery.
    correction: float
    #: How far `count` sat from an integer before rounding. The honest error
    #: bar on the whole computation: a residual near 1/2 means the answer was
    #: a coin toss.
    distance_from_integer: float
    #: Samples used along the horizontal leg, after refinement.
    samples: int
    #: Never rigorous. Floating point, on a finite sample, with no enclosure.
    confidence: Confidence = Confidence.NUMERICAL

    @field_validator("confidence")
    @classmethod
    def _reject_rigorous_confidence(cls, value: Confidence) -> Confidence:
        return _refuse_rigorous(value, "a count of zeros from sampled arg zeta")


class LineVerification(BaseModel):
    """Whether every zero below a height is on the critical line, and simple."""

    model_config = ConfigDict(extra="forbid")

    height: float
    #: From the argument principle: zeros in the strip.
    strip: int
    #: Sign changes of `Z`: zeros on the line, of odd order.
    on_line: int
    #: True when they agree. See the module docstring for what that is worth.
    agrees: bool
    evidence: str
    #: Never rigorous, and never promotable. A finite floating-point
    #: computation about a finite range is not a statement about zeta.
    confidence: Confidence = Confidence.NUMERICAL
    notes: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _reject_rigorous_confidence(cls, value: Confidence) -> Confidence:
        return _refuse_rigorous(
            value, "agreement of two counts below a finite height"
        )


def _refuse_rigorous(value: Confidence, what: str) -> Confidence:
    """Refused at construction, not checked at export.

    "Verified to height T" is the phrase that travels without its qualifiers,
    and the record is where it would acquire an authority it has not got. This
    is a floating-point computation over a finite range: it cannot be filed as
    established mathematics however many zeros agree, for the same reason a
    pattern holding in thirteen of thirteen cases cannot.
    """
    if value in RIGOROUS:
        raise ValueError(
            f"{what} may not claim {value.value!r}: it is a finite "
            "floating-point computation with no enclosure, and the Riemann "
            "hypothesis is a statement about every zero"
        )
    return value


def _zeta_on(points: np.ndarray) -> np.ndarray:
    """`zeta` at complex points, via mpmath, as a complex array."""
    import mpmath

    return np.array(
        [complex(mpmath.zeta(complex(point))) for point in points], dtype=complex
    )


def _increments(values: np.ndarray) -> np.ndarray:
    """Step-by-step change in `arg` along a sampled path.

    Each step is taken to be the increment in `(-pi, pi]`, which is the right
    one exactly when the true increment is smaller than `pi`. Returned per step
    rather than summed, so a caller can see WHERE the reconstruction is
    unreliable and subdivide there instead of everywhere.
    """
    steps = np.diff(np.angle(values))
    return (steps + np.pi) % (2 * np.pi) - np.pi


#: Samples along the horizontal leg to begin with.
#:
#: Small, because `arg zeta` barely moves along that segment: `S(T)` is under
#: one in absolute value at every height reached here, so the total change is a
#: fraction of a radian and eight samples resolve it. The first version asked
#: for four hundred, which was four hundred `mpmath.zeta` evaluations to
#: reconstruct a curve with no features.
INITIAL_SAMPLES = 32

#: Refuse rather than refine past this.
#:
#: Doubling without a ceiling turned the one case that cannot converge -- a
#: height sitting on a zero, where `arg zeta(1/2 + iT)` is not defined at all --
#: into eight hundred thousand `zeta` evaluations before giving up. The
#: refinement exists to fix a coarse sample, not to grind through an
#: impossibility.
MAX_SAMPLES = 4096

#: Below this, `zeta(1/2 + iT)` is taken to be at a zero.
#:
#: `S(T)` is `arg zeta(1/2 + iT)`, which has no value when zeta vanishes there,
#: and `N(T)` itself is ambiguous at a zero -- the zero is either counted or
#: not. Refusing is the honest answer; nudging the height would silently
#: answer a different question.
AT_A_ZERO = 1e-9


def strip_zero_count(
    height: float, *, samples: int = INITIAL_SAMPLES, max_samples: int = MAX_SAMPLES
) -> StripCount:
    """`N(T)`: zeros of zeta in the critical strip with `0 < Im <= height`.

    By the argument principle, along a path that never touches the critical
    line -- so this is independent of anything computed from `Z`. Refuses at a
    height where zeta vanishes on the line, since the count is ambiguous there.
    """
    import mpmath

    from .riemann_siegel import theta

    if height <= 0:
        raise ValueError("the height must be positive")

    endpoint = abs(complex(mpmath.zeta(complex(0.5, height))))
    if endpoint < AT_A_ZERO:
        raise ValueError(
            f"zeta(1/2 + {height}i) is {endpoint:.2e}, so the height is at a "
            "zero: arg zeta has no value there and N(T) is ambiguous. Ask "
            "about a height between two zeros."
        )

    # ADAPTIVE, not uniform. Approaching the critical line at a height near a
    # zero, `arg zeta` swings sharply over the last stretch while the rest of
    # the segment is featureless -- so doubling the whole grid spends
    # everything in the wrong place and still fails. At a height 1e-4 from the
    # first zero, four thousand uniform samples left an increment of 1.3
    # radians; subdividing only the offending intervals resolves it in a few
    # dozen points.
    nodes = np.linspace(RIGHT_EDGE, 0.5, samples)
    values = _zeta_on(nodes + 1j * height)
    while True:
        steps = _increments(values)
        largest = float(np.abs(steps).max(initial=0.0))
        if largest <= MAX_INCREMENT:
            break
        if len(nodes) >= max_samples:
            raise RuntimeError(
                f"arg zeta still moved by {largest:.3f} between samples at "
                f"height {height} with {len(nodes)} of them; refusing to "
                "reconstruct a winding from a sample that coarse"
            )
        crowded = np.abs(steps) > MAX_INCREMENT
        midpoints = 0.5 * (nodes[:-1] + nodes[1:])[crowded]
        extra = _zeta_on(midpoints + 1j * height)
        nodes = np.concatenate([nodes, midpoints])
        values = np.concatenate([values, extra])
        # The path runs from RIGHT_EDGE down to 1/2, so descending order is
        # path order. Sorting by node keeps the increments consecutive.
        order = np.argsort(-nodes)
        nodes, values = nodes[order], values[order]

    used = len(nodes)
    change = float(_increments(values).sum())
    # `arg zeta` at the right edge, where the Euler product converges
    # absolutely and the vertical leg has contributed nothing.
    start = float(np.angle(values[0]))
    correction = (start + change) / np.pi
    smooth = float(theta(height)) / np.pi + 1.0
    total = smooth + correction
    nearest = round(total)
    return StripCount(
        height=float(height),
        count=int(nearest),
        smooth=smooth,
        correction=float(correction),
        distance_from_integer=abs(total - nearest),
        samples=used,
    )


#: A strip count further than this from an integer is not to be relied on.
#:
#: `N(T)` is an integer, so the distance from one is the error bar on the whole
#: computation. In practice it is around 1e-13; anything near 1/2 means the
#: rounding chose the answer.
UNCOMFORTABLE_ROUNDING = 0.1


def _notes_for(strip: StripCount, on_line: int) -> list[str]:
    """The reader-facing account of a comparison, including its caveats."""
    notes = [
        f"argument principle: N({strip.height:g}) = {strip.count} "
        f"(smooth {strip.smooth:.3f} + S = {strip.correction:.3f}, "
        f"{strip.distance_from_integer:.2e} from an integer, "
        f"{strip.samples} samples)",
        f"sign changes of Z below {strip.height:g}: {on_line}",
    ]
    if strip.count != on_line:
        notes.append(
            "a disagreement is a bug report before it is a mathematical claim"
        )
    if strip.distance_from_integer > UNCOMFORTABLE_ROUNDING:
        notes.append(
            f"the strip count was {strip.distance_from_integer:.3f} from an "
            "integer, which is too far to be comfortable: treat it as unsettled"
        )
    return notes


def verify_zeros_on_the_line(height: float) -> LineVerification:
    """Are all the zeros below `height` on the critical line, and simple?

    Compares the strip count from the argument principle against the number of
    sign changes of `Z`. Agreement means yes, for this height, numerically --
    read the module docstring before repeating that without its qualifiers.
    """
    from .riemann_siegel import zero_ordinates

    strip = strip_zero_count(height)
    ordinates = zero_ordinates(height)
    on_line = len(ordinates)
    agrees = strip.count == on_line

    notes = _notes_for(strip, on_line)
    if agrees:
        evidence = (
            f"every one of the {on_line} zeros with 0 < Im <= {height:g} is on "
            "the critical line and simple, as far as a finite floating-point "
            "computation can say so"
        )
    else:
        evidence = (
            f"the strip holds {strip.count} zeros below {height:g} and only "
            f"{on_line} sign changes of Z were found: either some zero is off "
            "the line or of even order, or one of the two computations is "
            "wrong -- and the second is far likelier"
        )
    return LineVerification(
        height=float(height),
        strip=strip.count,
        on_line=on_line,
        agrees=agrees,
        evidence=evidence,
        notes=notes,
    )
