"""The Z function and the zeros of zeta, computed in bulk.

WHY THIS EXISTS. `mpmath.zetazero` is a root-find per zero and costs about
160 ms of it, essentially independent of the requested precision -- the time is
Python-level overhead, not arithmetic, so asking for fewer digits does not help.
Anything that needs many zeros could not have them: a column of ordinates over
n = 2..999 took 137 seconds, and the explicit formula's sum over zeros was out
of reach entirely. Zeros here cost about 0.1 ms each, which is the difference
between sampling the first thousand and sampling the first million.

HOW. `Z(t)` is evaluated for a whole array at once, by two methods with a
crossover:

  * EULER-MACLAURIN below `CROSSOVER`, which is exact to machine precision
    there. Its correction series needs a main sum longer than `|t|` -- the k-th
    term carries a rising factorial of size `|t|^(2k-1)` against `N^(-2k+1)` --
    so N scales with the height, and the batch is chunked to keep that array
    from growing without bound.

  * RIEMANN-SIEGEL above it, whose expansion parameter `sqrt(t/2pi)` is only
    1.5 at the first zero and useless there, but which needs just
    `sqrt(t/2pi)` terms once the height is real.

THE CORRECTION COEFFICIENTS ARE POLYNOMIALS, AND THAT IS THE WHOLE TRICK.
Written as published -- `C_1 = -Psi'''/(96 pi^2)` and so on -- they evaluate to
nonsense in float64 near `p = 1/4` and `p = 3/4`: `Psi` has removable
singularities there, and a twelfth derivative of a removable singularity is a
difference of enormous numbers. Sampled with the symbolic derivatives the error
reached 3e12.

But `Psi` is ENTIRE. Every zero of `cos(2 pi p)` is cancelled: at
`p = (2m+1)/4` the numerator's argument is `pi(m^2 - m - 1)/2` with `m^2 - m`
even, so the numerator vanishes too. An entire function has a Taylor series
about `p = 1/2` that converges everywhere, so the series is taken once and
DIFFERENTIATED AS A POLYNOMIAL. Every `C_k` is then a polynomial, stable at
every `p`, including exactly at the old singularities.

HOW THE ZEROS ARE BRACKETED. Between consecutive GRAM POINTS: `theta` is
monotone, so `theta(g_n) = n pi` inverts by Newton for nothing, and Gram's law
puts exactly one zero in each interval. That is one evaluation of Z per zero,
where a uniform grid fine enough not to step over a close pair spends two
dozen -- 3.7 million points at height 10^5, which did not finish in ten
minutes. The law fails at a few per cent of the intervals, rising to 21% by
that height, and those blocks are rescanned; the count decides when to stop.

WHAT MAKES THE ZERO LIST TRUSTWORTHY, AND WHAT DOES NOT. `zero_ordinates`
refines until its count agrees with `ZeroCount` or raises, and that check earns
its keep: an earlier uniform grid missed four zeros in ten thousand, silently,
which is the only unacceptable outcome.

But it is a CROSS-IMPLEMENTATION check, not an independent one, and it became
WEAKER when the bracketing moved to Gram points. `ZeroCount` calls
`mpmath.nzeros`, which also walks Gram and Rosser blocks and also separates the
zeros inside a block by the sign changes of Z. The two now agree about method as
well as answer, so their agreeing says this implementation did not miss what
mpmath's finds, and no more.

For a count that does not pass through Z at all, see `strip_zero_count` in
`argument_principle.py`: `N(T) = theta(T)/pi + 1 + S(T)` with `S(T)` obtained by
tracking `arg zeta` along a path that never touches the critical line. That one
counts zeros in the STRIP, so comparing it against the number of sign changes
found here is a statement about whether the zeros are on the line.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# --- Psi and the correction coefficients, as polynomials ------------------


def _psi_taylor(degree: int) -> np.ndarray:
    """Taylor coefficients of `Psi(1/2 + u)` in `u`.

    Computed symbolically, so the cancellation that makes `Psi` entire happens
    exactly rather than in floating point.
    """
    import sympy as sp

    u, p = sp.Symbol("u"), sp.Symbol("p")
    psi = sp.cos(2 * sp.pi * (p**2 - p - sp.Rational(1, 16))) / sp.cos(2 * sp.pi * p)
    series = sp.series(psi.subs(p, sp.Rational(1, 2) + u), u, 0, degree).removeO()
    coefficients = sp.Poly(sp.expand(series), u).all_coeffs()[::-1]
    return np.array([float(c) for c in coefficients])


#: Enough terms that the tail is below machine precision over `|u| <= 1/2`,
#: with room to lose twelve orders to differentiation.
_PSI_DEGREE = 60


def _differentiate(coefficients: np.ndarray, order: int) -> np.ndarray:
    out = coefficients.copy()
    for _ in range(order):
        out = out[1:] * np.arange(1, len(out))
    return out


def _correction_polynomials() -> list[np.ndarray]:
    psi = _psi_taylor(_PSI_DEGREE)
    derivative = {k: _differentiate(psi, k) for k in range(13)}
    pi2, pi4, pi6, pi8 = np.pi**2, np.pi**4, np.pi**6, np.pi**8

    def combine(*pairs: tuple[float, int]) -> np.ndarray:
        out = np.zeros(max(len(derivative[order]) for _, order in pairs))
        for scale, order in pairs:
            out[: len(derivative[order])] += scale * derivative[order]
        return out

    return [
        combine((1.0, 0)),
        combine((-1.0 / (96 * pi2), 3)),
        combine((1.0 / (64 * pi2), 2), (1.0 / (18432 * pi4), 6)),
        combine(
            (-1.0 / (64 * pi2), 1),
            (-1.0 / (3840 * pi4), 5),
            (-1.0 / (5308416 * pi6), 9),
        ),
        combine(
            (1.0 / (128 * pi2), 0),
            (19.0 / (24576 * pi4), 4),
            (1.0 / (491520 * pi6), 8),
            (1.0 / (2038431744 * pi8), 12),
        ),
    ]


_CORRECTIONS: list[np.ndarray] | None = None


def _corrections() -> list[np.ndarray]:
    """Built on first use: the symbolic series costs a few seconds."""
    global _CORRECTIONS
    if _CORRECTIONS is None:
        _CORRECTIONS = _correction_polynomials()
    return _CORRECTIONS


# --- theta ----------------------------------------------------------------

#: `theta(t) = arg Gamma(1/4 + it/2) - (t/2) log pi`, by its asymptotic series.
#:
#: Agrees with `mpmath.siegeltheta` to within a float64 step from t = 20 upward,
#: and THAT is what bounds this module's accuracy at height. Theta enters Z
#: inside a cosine, so an absolute error in the phase is an absolute error in Z
#: -- and theta grows like `t log t`, so the spacing of a double at the phase is
#: about 3e-11 at t = 5e4 and 1e-9 at t = 1e6. The series is not the limit; the
#: width of a double is.
#:
#: MEASURED, because two sessions have now reasoned from this paragraph. Error
#: in units of `ulp(theta(t))`, computed per point, sixteen samples per decade:
#:
#:     range            median err    worst err   worst/ulp
#:     [1e4, 1e5]         1.71e-11     8.49e-11         1.5
#:     [1e6, 1e7]         2.68e-09     9.98e-09         1.3
#:     [1e8, 1e9]         4.35e-07     1.13e-06         1.2
#:     [1e10,1e11]        1.98e-05     1.68e-04         1.4
#:     [1e12,1e13]        4.35e-03     9.27e-03         1.1
#:
#: So "within a float64 step" is exact, over nine decades, and the series is
#: nowhere the limit. Sample a decade rather than a point: the error depends on
#: where the rounding falls, and taking `ulp` at the bottom of a range while
#: sampling across it makes this look ten times worse than it is -- which is
#: how an earlier draft of this note came to claim it.
#:
#: WHAT WAS WRONG HERE was the sentence that used to follow: "past roughly
#: t = 1e8 the phase would need to be carried in extended precision". That
#: rests on a phase budget nothing states, and it is the wrong quantity. What
#: matters is whether a zero POSITION moves enough to shift a statistic.
#: Measured by jittering the zeros and watching the pair correlation, the
#: density does not move until the jitter reaches about 1e-2 -- two per cent of
#: a mean gap -- while the actual position error at `T = 10^6` is 1.16e-10.
#: Eight orders of margin. See `docs/research/pair-correlation-lower-order.md`.
#:
#: THE SMOOTH CEILING IS REAL, IT IS JUST FAR AWAY, and an earlier draft of
#: this note said it was gone. Raised in review. A Gram point sits where
#: `theta = n pi` and they are spaced by the mean gap `2 pi / l`, so a phase
#: error `d` moves a zero by `d / theta'(t)` -- which as a FRACTION OF A MEAN
#: GAP is just `d / pi`. That is the quantity to compare against the 2%, and it
#: grows with `ulp(theta)`:
#:
#:     t          ulp(theta)    shift/gap
#:     1e+06        9.31e-10     2.96e-10
#:     1e+10        1.53e-05     4.86e-06
#:     1e+13        1.56e-02     4.97e-03
#:     1e+14        2.50e-01     7.96e-02   <- past 2%
#:
#: float64 crosses the tolerance at `t = 3.96e13`, around zero index 1.8e14.
#: Below that the smooth error is negligible and the finder's completeness is
#: what fails first; above it, the positions themselves go. Both ceilings
#: exist; the discrete one is simply nearer.
#:
#: For scale: at Odlyzko's index 1e12 (`t = 2.67e11`) the shift is 1.55e-4 of a
#: gap, so float64 covers every height this engine has any prospect of
#: reaching. At his index 1e22 it is 1.34e6 gaps, which is nothing at all.
#:
#: The discrete failure -- the finder missing a zero, or resolving a Rosser
#: block wrongly -- is what `zero_ordinates` and `zeros_in_band` refuse on,
#: against `ZeroCount`. Measured, that bites first: a band at `t = 2e6` already
#: comes back two zeros short and raises.
_THETA_TERMS = ((1, 48), (7, 5760), (31, 80640), (127, 430080))


#: Below this, `theta` is computed from its definition instead of its series.
#:
#: MEASURED, not chosen. The asymptotic expansion is wrong by 4.9e-2 at t = 0.5,
#: 2.1e-2 at t = 1 and 9.4e-4 at t = 2, reaching 4.4e-13 by t = 10 and machine
#: precision by 20. Nothing in this engine had ever noticed, because the lowest
#: zero is at 14.135 and everything that uses theta -- Gram points, unfolding,
#: the spacing statistics -- starts there or above.
#:
#: It surfaced from an integral that starts at t = 0. The corpus records
#: `theta(t) = -t log(pi)/2 + arg Gamma(it/2 + 1/4)`, which is exact
#: everywhere, and below this threshold the implementation was not computing
#: it. A formula and the code that claims to evaluate it disagreeing in a
#: region nothing had looked at is the failure this repository is about.
THETA_ASYMPTOTIC_FLOOR = 20.0


def _theta_exact(t: np.ndarray) -> np.ndarray:
    """`theta` from its definition: `-t log(pi)/2 + arg Gamma(1/4 + it/2)`.

    Through mpmath, one point at a time, because that is the accurate route
    and the region is bounded -- below `THETA_ASYMPTOTIC_FLOOR` there are
    twenty units of `t` and at most one zero. The asymptotic series above the
    threshold is the fast route, and this is what it is checked against, the
    same relationship `verify_shortcuts` keeps everywhere else.
    """
    import mpmath

    return np.array([float(mpmath.siegeltheta(float(value))) for value in t])


def theta(t: np.ndarray | float) -> np.ndarray:
    """The Riemann-Siegel theta function.

    Asymptotic above `THETA_ASYMPTOTIC_FLOOR`, where it is exact to the last
    bit, and from the definition below it, where the series is not. At `t = 1`
    the series is wrong by 2.1e-2 -- and it divides by `t`, so at `t = 0` it
    is not wrong but absent, returning NaN.

    Nothing here had noticed: the lowest zero is at 14.135 and every consumer
    -- Gram points, unfolding, the spacing statistics -- starts at or above
    it. An integral running from zero is what found it.
    """
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    values = np.atleast_1d(t)

    out = np.empty_like(values)
    low = values < THETA_ASYMPTOTIC_FLOOR
    if low.any():
        out[low] = _theta_exact(values[low])
    high = ~low
    if high.any():
        above = values[high]
        series = above / 2 * np.log(above / (2 * np.pi)) - above / 2 - np.pi / 8
        power = above
        for numerator, denominator in _THETA_TERMS:
            series = series + numerator / (denominator * power)
            power = power * above * above
        out[high] = series
    return out[0] if scalar else out


# --- Euler-Maclaurin, for the low-height regime ---------------------------

_EULER_MACLAURIN_CORRECTIONS = 10
_BERNOULLI: np.ndarray | None = None

#: Cap on `len(t) * N` per chunk, so the main sum's array stays about 100 MB
#: however many points are asked for at once.
_CHUNK_BUDGET = 6_000_000


def _bernoulli() -> np.ndarray:
    global _BERNOULLI
    if _BERNOULLI is None:
        import sympy as sp

        _BERNOULLI = np.array(
            [
                float(sp.bernoulli(2 * k) / sp.factorial(2 * k))
                for k in range(1, _EULER_MACLAURIN_CORRECTIONS + 1)
            ]
        )
    return _BERNOULLI


def _zeta_on_critical_line(t: np.ndarray) -> np.ndarray:
    """`zeta(1/2 + it)` by Euler-Maclaurin, for one chunk of heights."""
    s = 0.5 + 1j * t
    length = int(max(30.0, float(np.abs(t).max()) * 1.5)) + 20
    terms = np.arange(1, length, dtype=float)
    total = np.sum(terms[None, :] ** (-s[:, None]), axis=1)
    total += length ** (-s) / 2.0 + length ** (1.0 - s) / (s - 1.0)
    for index, bernoulli in enumerate(_bernoulli(), start=1):
        rising = np.ones_like(s)
        for offset in range(2 * index - 1):
            rising = rising * (s + offset)
        total += bernoulli * rising * length ** (-s - 2 * index + 1)
    return total


def _z_euler_maclaurin(t: np.ndarray) -> np.ndarray:
    length = int(max(30.0, float(np.abs(t).max()) * 1.5)) + 20
    per_chunk = max(1, _CHUNK_BUDGET // max(length, 1))
    out = np.empty_like(t)
    for start in range(0, len(t), per_chunk):
        block = t[start : start + per_chunk]
        out[start : start + per_chunk] = np.real(
            np.exp(1j * theta(block)) * _zeta_on_critical_line(block)
        )
    return out


# --- Riemann-Siegel, above the crossover ----------------------------------


def _z_riemann_siegel(t: np.ndarray) -> np.ndarray:
    a = np.sqrt(t / (2 * np.pi))
    count = np.floor(a).astype(np.int64)
    angle = theta(t)
    total = np.zeros_like(t)

    # The main sum runs to floor(sqrt(t/2pi)), which grows with t -- so for
    # each term only the points high enough contribute. Selecting them with a
    # boolean mask copies three arrays per term, and there are 126 terms at
    # height 10^5: evaluating Z at 138000 points took 12.7 seconds, for about
    # 1.7e7 pieces of arithmetic.
    #
    # `count` is non-decreasing wherever `t` is, so on sorted input that
    # selection is a CONTIGUOUS SUFFIX and `searchsorted` turns it into a
    # slice. Slices are views. Every caller here passes sorted heights, but
    # this checks rather than assuming: an unsorted array would silently get
    # the wrong terms, which is the sort of thing that reads as a bad zero
    # rather than as a bad index.
    terms = int(count.max()) if len(count) else 0
    if terms and np.all(np.diff(t) >= 0):
        for n in range(1, terms + 1):
            first = int(np.searchsorted(count, n, side="left"))
            if first >= len(t):
                continue
            block = slice(first, None)
            total[block] += np.cos(angle[block] - t[block] * np.log(n)) / np.sqrt(n)
    else:
        for n in range(1, terms + 1):
            take = count >= n
            total[take] += np.cos(angle[take] - t[take] * np.log(n)) / np.sqrt(n)

    offset = a - count - 0.5
    remainder = np.zeros_like(t)
    for power, polynomial in enumerate(_corrections()):
        remainder += np.polyval(polynomial[::-1], offset) / a**power
    remainder *= np.where(count % 2 == 0, -1.0, 1.0) / np.sqrt(a)
    return 2 * total + remainder


#: Where the two methods hand over when accuracy is what matters.
#:
#: Riemann-Siegel with five correction terms is worst just above its floor:
#: measured against `mpmath.siegelz` its error is 3e-8 near t = 500 and 4e-10
#: by t = 2000, while Euler-Maclaurin is 4e-13 everywhere it is used. So for a
#: value that will be reported, the handover goes where RS becomes as good as
#: the rest of the computation rather than where it becomes usable.
CROSSOVER = 2000.0

#: Where they hand over while SEARCHING.
#:
#: Euler-Maclaurin's main sum is longer than the height, so at t = 2000 it is
#: three thousand terms -- and a bisection evaluates Z sixty times over. Using
#: the accurate crossover inside the search made locating a thousand zeros
#: eleven times slower for no benefit: a bracket only has to be on the right
#: side of the root, and 3e-8 decides that. The located roots are then polished
#: with the accurate Z, which is a handful of evaluations rather than sixty.
SEARCH_CROSSOVER = 400.0


def z_function(t: np.ndarray | float, *, crossover: float = CROSSOVER) -> np.ndarray:
    """`Z(t)`, the Hardy function, for an array of heights at once.

    Real, and zero exactly at the ordinates of zeta's critical-line zeros.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty_like(t)
    low = t < crossover
    if low.any():
        out[low] = _z_euler_maclaurin(t[low])
    if (~low).any():
        out[~low] = _z_riemann_siegel(t[~low])
    return out


def _search_z(t: np.ndarray) -> np.ndarray:
    return z_function(t, crossover=SEARCH_CROSSOVER)


#: Newton steps used to polish a located root with the accurate Z.
_POLISH_STEPS = 3


def _polish(ordinates: np.ndarray) -> np.ndarray:
    """Newton on the accurate Z, from roots the cheap one already bracketed.

    A step larger than a small fraction of the local gap is refused: Newton can
    walk to a neighbouring root, and a list of ordinates with one of them
    duplicated and one missing would still pass a count check.
    """
    if len(ordinates) == 0:
        return ordinates
    current = ordinates.copy()
    for start in range(0, len(current), _SCAN_CHUNK):
        block = current[start : start + _SCAN_CHUNK]
        for _ in range(_POLISH_STEPS):
            step = np.maximum(block * 1e-11, 1e-9)
            centre = z_function(block)
            slope = (z_function(block + step) - z_function(block - step)) / (2 * step)
            with np.errstate(divide="ignore", invalid="ignore"):
                update = np.where(slope != 0, centre / slope, 0.0)
            limit = 0.05 * _average_gap(np.maximum(block, 12.0))
            update = np.clip(np.nan_to_num(update), -limit, limit)
            block = block - update
        current[start : start + _SCAN_CHUNK] = block
    return current


# --- locating the zeros ---------------------------------------------------

#: Two ordinates closer than this are the same zero found twice.
#:
#: Refinement rescans the interval BETWEEN two known zeros, and a grid over
#: that interval finds its endpoints again at values differing in the last bits.
#: That is how a scan for 10142 zeros returned 10206. Real zeta zeros are not
#: this close: the tightest pairs near t = 10^4 are about 1e-3 apart.
SAME_ZERO = 1e-7

#: Fraction of the local average gap for the first scan. Coarser is faster and
#: misses close pairs; this is about where the first pass usually suffices.
_FIRST_PASS_DIVISOR = 24

#: Bisection steps. Enough to bracket the root tightly enough for Newton on
#: the accurate Z to converge without being able to reach a neighbour; the
#: digits come from `_polish`, not from here.
_BISECTION_STEPS = 40


def _bisect(low: np.ndarray, high: np.ndarray, at_low: np.ndarray) -> np.ndarray:
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (low + high)
        at_mid = _search_z(mid)
        left = np.sign(at_mid) * np.sign(at_low) > 0
        low = np.where(left, mid, low)
        at_low = np.where(left, at_mid, at_low)
        high = np.where(left, high, mid)
    return 0.5 * (low + high)


#: Grid points evaluated at once.
#:
#: The scan's grid is a fixed fraction of the mean gap, so its length grows like
#: `T log T`: 29000 points at height 1420 and 3.7 MILLION at 10^5. Evaluated in
#: one array that is several 30-megabyte temporaries per term of the
#: Riemann-Siegel sum, of which there are 126 at that height -- the run did not
#: finish in ten minutes. Chunking bounds the memory and keeps the working set
#: in cache; the arithmetic is identical.
_SCAN_CHUNK = 250_000


def _scan(start: float, stop: float, step: float) -> np.ndarray:
    if stop <= start or step <= 0:
        return np.array([])
    total = int(np.ceil((stop - start) / step)) + 1
    if total < 2:
        return np.array([])

    found: list[np.ndarray] = []
    begun = 0
    while begun < total - 1:
        # One point of overlap, so a crossing that falls between two chunks is
        # still seen by one of them.
        end = min(begun + _SCAN_CHUNK, total)
        grid = start + step * np.arange(begun, end)
        values = _search_z(grid)
        crossings = np.nonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]
        if len(crossings):
            found.append(_bisect(grid[crossings], grid[crossings + 1], values[crossings]))
        begun = end - 1
    if not found:
        return np.array([])
    return np.concatenate(found)


def theta_derivative(t: np.ndarray | float) -> np.ndarray:
    """`theta'(t)`, for inverting theta by Newton."""
    t = np.asarray(t, dtype=float)
    return 0.5 * np.log(t / (2 * np.pi)) - 1.0 / (48 * t**2) - 7.0 / (1920 * t**4)


def gram_points(indices: np.ndarray) -> np.ndarray:
    """`g_n`, where `theta(g_n) = n pi`. Vectorised.

    theta is monotone above t = 6.29, so this inverts by Newton and costs
    nothing beside evaluating Z.
    """
    indices = np.asarray(indices, dtype=float)
    target = indices * np.pi
    current = np.maximum(2 * np.pi * (indices + 1), 8.0)
    for _ in range(80):
        step = (theta(current) - target) / theta_derivative(current)
        current = np.maximum(current - step, 7.0)
        if np.all(np.abs(step) < 1e-12):
            break
    return current


def _gram_scan(height: float, low: float = 9.0) -> np.ndarray:
    """Zeros bracketed between consecutive Gram points.

    ONE evaluation of Z per zero, where a uniform grid fine enough not to miss
    close pairs spends two dozen. At height 10^5 that is the difference between
    3.7 million evaluations and 140 thousand, and at 3 x 10^5 between a run that
    does not finish in ten minutes and one that does.

    Gram's law puts exactly one zero between consecutive Gram points and fails
    at a few per cent of them. The caller repairs those; the count is the
    authority either way.
    """
    indices, points, values = _gram_structure(height, low)
    if len(points) < 2:
        return np.array([])
    crossings = np.nonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]
    if len(crossings) == 0:
        return np.array([])
    return _bisect(points[crossings], points[crossings + 1], values[crossings])


def _gram_structure(height: float, low: float = 9.0) -> tuple[np.ndarray, ...]:
    """The Gram points over a range, their indices, and `Z` at each.

    The indices are kept because Gram BLOCKS are defined by them: `g_n` is a
    good Gram point when `(-1)**n Z(g_n) > 0`, and that parity is the whole
    definition. `_gram_scan` throws the structure away and only wants the sign
    changes; `_rosser_deficits` needs all three.
    """
    top = int(float(theta(height)) / np.pi) + 2
    # Start from the Gram index just below `low` rather than from the bottom.
    # `theta(g_n) = n pi` inverts, so entering the scan at height costs one
    # Newton solve instead of every zero underneath.
    first = -1 if low <= 9.0 else int(float(theta(low)) / np.pi) - 2
    indices = np.arange(first, top)
    points = gram_points(indices)
    keep = (points > max(low - _average_gap(low) * 4, 9.0)) & (
        points <= height + _average_gap(height) * 4
    )
    indices, points = indices[keep], points[keep]
    if len(points) < 2:
        return indices, points, np.array([])
    return indices, points, _search_z(points)


def _rosser_deficits(
    indices: np.ndarray,
    points: np.ndarray,
    values: np.ndarray,
    found: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Which Gram blocks are short, and by how many.

    WHY THIS BEATS THE GAP TEST. The refinement's other route rescans any gap
    that looks too WIDE, which is a guess: a cluster hides inside a normal gap,
    and at `t = 2002828.4` four zeros sat inside 1.194 mean gaps and were
    invisible to it. A Gram block does not need guessing. `g_n` is good when
    `(-1)**n Z(g_n) > 0`; between two consecutive good points spanning `k` Gram
    intervals, Rosser's rule puts exactly `k` zeros. So counting what has been
    found inside a block says precisely how many are missing and exactly where,
    which turns the search into an assignment.

    Rosser's rule has known exceptions, so this is a REPAIR HINT and never an
    authority: `ZeroCount` still decides, and a block reported complete here is
    not therefore complete. The callers keep the gap-based pass behind it.
    """
    if len(points) < 2 or len(values) != len(points):
        empty = np.array([])
        return empty, empty, empty.astype(int)
    good = np.nonzero(((-1.0) ** indices) * values > 0)[0]
    if len(good) < 2:
        empty = np.array([])
        return empty, empty, empty.astype(int)

    starts, stops = good[:-1], good[1:]
    expected = indices[stops] - indices[starts]
    lows, highs = points[starts], points[stops]
    inside = np.searchsorted(found, highs) - np.searchsorted(found, lows)
    short = np.nonzero(inside < expected)[0]
    return lows[short], highs[short], (expected - inside)[short]


def _rescan_intervals(lows: np.ndarray, highs: np.ndarray, per_interval: int) -> np.ndarray:
    """Scan many intervals at once.

    One concatenated grid and ONE call to Z, rather than a Python loop calling
    it per interval. Looping is what made Gram bracketing lose to a blind grid
    the first time it was measured: the law fails at 8% of the points by height
    10^4, and eight per cent of ten thousand is two thousand trips round a loop
    whose body is a numpy call on forty numbers.

    IN BATCHES, THOUGH, BECAUSE ONE ARRAY OVER EVERYTHING IS THE CEILING. The
    grid is `len(lows) * per_interval` float64s, and both factors grow: the
    refinement relaxes which gaps count as suspect AND raises the density, and
    the two multiply. Unbounded, that reached 2.5 GB at `T = 10^6` and 11.3 GB
    at 4x10^6 -- a run there held 16.9 GB, thrashed for seven hours at 24% duty
    and was killed. It did not fail; it swapped, on a 32 GB desktop. The same
    code later asked for a single 14.3 GiB array at `T = 10^6` and was refused
    outright. Neither number is a property of the algorithm alone: 16.9 GB is
    comfortable on a 125 GB box, so the first was the machine's ceiling and the
    second was one allocation too large for any of ours.

    It bit again in miniature and much faster. With the Gram-block pass
    recovering the structural misses first, what the width loop has left is a
    small hard residual, so it relaxes the threshold to 0.55 -- where nearly
    every interval qualifies, a mean gap being 1.0 -- and escalates density to
    8000. Twenty thousand intervals at 8000 points is 160 million, 1.28 GB, and
    one sub-band of a ladder rung ran 23 minutes on it.

    `_RESCAN_POINT_BUDGET` caps the points in flight. The work is identical --
    the same intervals at the same density -- and only the peak allocation
    changes.
    """
    if len(lows) == 0:
        return np.array([])
    per_batch = max(1, _RESCAN_POINT_BUDGET // max(1, per_interval))
    if len(lows) > per_batch:
        pieces = [
            _rescan_intervals(
                lows[start : start + per_batch], highs[start : start + per_batch], per_interval
            )
            for start in range(0, len(lows), per_batch)
        ]
        pieces = [piece for piece in pieces if len(piece)]
        return np.concatenate(pieces) if pieces else np.array([])
    offsets = np.linspace(0.0, 1.0, per_interval)[None, :]
    grids = lows[:, None] + (highs - lows)[:, None] * offsets
    values = _search_z(grids.ravel()).reshape(grids.shape)
    rows, columns = np.nonzero(np.sign(values[:, :-1]) * np.sign(values[:, 1:]) < 0)
    if len(rows) == 0:
        return np.array([])
    return _bisect(grids[rows, columns], grids[rows, columns + 1], values[rows, columns])


def _dedupe(ordinates: np.ndarray) -> np.ndarray:
    ordinates = np.sort(np.asarray(ordinates, dtype=float))
    if len(ordinates) < 2:
        return ordinates
    return ordinates[np.concatenate([[True], np.diff(ordinates) > SAME_ZERO])]


def _average_gap(height: np.ndarray | float) -> np.ndarray:
    return 2 * np.pi / np.log(np.maximum(height, 12.0) / (2 * np.pi))


def _sweep_edges(found: np.ndarray, low: float, high: float, density: int = 2000) -> np.ndarray:
    """Rescan from `low` to the first zero and from the last zero to `high`.

    NEITHER MECHANISM COVERS A PARTIAL INTERVAL, and a band's ends are two of
    them. Gram's law brackets a zero between consecutive Gram points, and
    Rosser's rule counts zeros between consecutive GOOD ones -- so a zero above
    the last good Gram point below `high`, or below the first above `low`, is in
    no bracket and no block. The width test does not reach it either: the last
    interval is truncated by the boundary, so it looks SHORT rather than wide.

    Measured: the sub-band `(812724.4, 823781.7]` of a ladder rung refused one
    zero short for exactly this. The zero was at 823781.638367, its right-hand
    neighbour was `high` itself at 823781.702954, and the gap containing it was
    0.513 mean gaps -- half a gap, because the band cut it. The last good Gram
    point was 823781.5809, below the zero, so no block reached it either.

    Two intervals, so this is nearly free, and it runs before the loop rather
    than as a fallback: a boundary miss is systematic, not unlucky.
    """
    if len(found) == 0:
        return _rescan_intervals(np.array([low]), np.array([high]), density)
    ordered = np.sort(found)
    lows = np.array([low, ordered[-1]])
    highs = np.array([ordered[0], high])
    inset = np.minimum(SAME_ZERO * 100, (highs - lows) / 100)
    keep = highs - lows > 2 * inset
    if not keep.any():
        return np.array([])
    return _rescan_intervals((lows + inset)[keep], (highs - inset)[keep], density)


def _repair_by_block(
    found: np.ndarray,
    indices: np.ndarray,
    points: np.ndarray,
    values: np.ndarray,
    *,
    density: int = 48,
) -> np.ndarray:
    """One pass of Gram-block repair, returning whatever it recovered.

    Cheap: the Gram points and their `Z` values are already computed by the
    scan, so this costs one rescan of only the blocks that are short, at a
    density set by how short they are.

    48, MEASURED. The density is the whole trade: too low and the block pass
    finds nothing and is pure overhead, too high and it costs more than the
    width-based loop it is saving. Measured on this machine, full runs:

        density   T = 10^5    T = 3x10^5
        (none)      20.6 s      245.8 s
        48          19.1 s       81.7 s
        128         18.3 s      100.2 s
        400         29.8 s      160.2 s

    So 48 is 3.0x at 3x10^5 and costs nothing at 10^5, where Gram's law rarely
    fails and there is little to repair. 400 was the first guess and it was
    SLOWER than no block pass at all at 10^5 -- a repair that finds its zeros
    at ten times the resolution it needed.
    """
    lows, highs, deficits = _rosser_deficits(indices, points, values, np.sort(found))
    if len(lows) == 0:
        return np.array([])
    # NO INSET HERE, AND THAT IS THE DIFFERENCE BETWEEN THE TWO REPAIRS. The
    # width-based rescan runs between two zeros ALREADY FOUND, so it steps
    # inside its endpoints -- re-finding them is what produced duplicates. A
    # block's endpoints are GRAM POINTS, which are not zeros, and a zero may
    # sit arbitrarily close to one.
    #
    # Measured: the block [846402.857118, 846403.921084] was correctly flagged
    # with a deficit of 2, and the second zero is at 846403.92108 -- 4e-6 below
    # its upper edge. An inset of SAME_ZERO * 100 = 1e-5 excluded it, so the
    # block was rescanned at density 8000 and came back one short anyway. The
    # sub-band then refused, honestly and for a reason that was ours.
    #
    # `_dedupe` already removes anything re-found, so scanning the closed block
    # costs a duplicate and no correctness.
    per = int(min(8000, max(density, density * int(deficits.max()))))
    return _rescan_intervals(lows, highs, per)


#: Most points the rescan will hold in one array, so the grid cannot grow
#: without bound as the refinement widens its net and sharpens it at once.
#:
#: 4 million float64s is 32 MB, which is nothing against the hours these runs
#: take, and it is the difference between a bounded rescan and the 11.3 GB one
#: measured at `T = 4x10^6`.
_RESCAN_POINT_BUDGET = 4_000_000

#: How wide a gap has to look before the refinement will rescan it, as a
#: multiple of the local mean gap.
#:
#: 1.3 catches the ordinary miss, which leaves an obviously wide hole. It
#: cannot catch a CLUSTER -- at `t = 2002828.4` four zeros sit inside 1.194
#: mean gaps -- and the first fix for that relaxed this number toward 0.55.
#: That works and is far too slow: at 0.55 nearly every interval qualifies, a
#: mean gap being 1.0, so a 20,000-zero sub-band rescans 20,000 intervals at
#: density 8000. Measured, 23 minutes and still going.
#:
#: What replaced it uses the count instead. `ZeroCount` says how many zeros are
#: missing, so the fallback takes that many of the WIDEST gaps and no more --
#: work proportional to the deficit rather than to the band.
_GAP_THRESHOLD = 1.3


def zeros_in_band(low: float, high: float, *, max_passes: int = 14) -> np.ndarray:
    """Every ordinate in `(low, high]`, without computing the ones underneath.

    WHY THIS EXISTS. `zero_ordinates` walks Gram points from the bottom, so
    asking for the zeros near `t = 4x10^6` means finding the seven million
    below them first. The statistics here are measured in BANDS -- the height
    recovery cuts them by `l`, the spacing decay compares disjoint ones -- and
    a band is a small fraction of everything under it. At `l = 14` the band is
    a twentieth of the run.

    `theta(g_n) = n pi` inverts, so entering the scan at height costs one
    Newton solve rather than every zero beneath it.

    THE COUNT CHECK STILL HOLDS, and that is the point. `ZeroCount` is
    monotone, so the zeros in a band number `N(high) - N(low)`, and a band that
    comes back short is refused exactly as a full run is. Counting is nearly
    free -- `N(10^5)` costs 0.05 s against 22 s to locate them -- so the band
    is verified for almost nothing. A cheaper way to get zeros that could not
    be checked would not be worth having: the failure at height is a MISSED
    zero, not an imprecise one, and this is what catches it.
    """
    import sympy as sp

    from .functions import ZeroCount

    if not 9.0 <= low < high:
        raise ValueError(
            f"a band runs from a height above 9 up to a greater one; got ({low}, {high}]"
        )
    expected = int(ZeroCount(sp.Float(high))) - int(ZeroCount(sp.Float(low)))
    gram_indices, gram_points_, gram_values = _gram_structure(high, low)
    found = _dedupe(_gram_scan(high, low=low))
    found = found[(found > low) & (found <= high)]

    # ROSSER FIRST. A short Gram block names its own deficit, so this repairs
    # what the law missed without guessing from gap widths. It is a hint, not
    # an authority -- the rule has exceptions and `ZeroCount` still decides --
    # so the width-based loop stays behind it.
    recovered = _repair_by_block(found, gram_indices, gram_points_, gram_values)
    if len(recovered):
        found = _dedupe(np.concatenate([found, recovered]))
        found = found[(found > low) & (found <= high)]

    # The two intervals no bracket and no block covers. See `_sweep_edges`.
    edge = _sweep_edges(found, low, high)
    if len(edge):
        found = _dedupe(np.concatenate([found, edge]))
        found = found[(found > low) & (found <= high)]

    density = 40
    block_density = 48
    threshold = _GAP_THRESHOLD
    for _ in range(max_passes):
        if len(found) == expected:
            return np.sort(_polish(found))
        edges = np.concatenate([[low], found, [high]])
        widths = np.diff(edges)
        suspect = np.nonzero(widths > threshold * _average_gap(edges[:-1]))[0]
        extra = np.array([])
        if len(suspect):
            lows, highs = edges[suspect], edges[suspect + 1]
            inset = np.minimum(SAME_ZERO * 100, (highs - lows) / 100)
            extra = _rescan_intervals(lows + inset, highs - inset, density)
        if len(extra):
            found = _dedupe(np.concatenate([found, extra]))
            found = found[(found > low) & (found <= high)]
            density = min(density * 4, 8000)
            continue
        # NOTHING NEW FROM THE WIDTH TEST, SO ASK THE BLOCKS AGAIN, HARDER.
        # A gap holding extra zeros need not be a WIDE gap -- at t = 2002828.4
        # four zeros sit inside 1.194 mean gaps, under the 1.3 that qualifies
        # -- so ranking gaps by width cannot find a cluster however many of
        # them are taken. Two ways to widen the search were tried and both are
        # recorded because both are wrong in an instructive way:
        #
        #   * Relax the threshold toward 0.55. Correct, and far too slow: at
        #     0.55 nearly every interval qualifies, a mean gap being 1.0, so a
        #     20,000-zero sub-band rescans 20,000 intervals at density 8000.
        #     Measured, 23 minutes and still climbing.
        #   * Take the widest `16 * deficit` gaps. Fast, and it misses exactly
        #     the case the relaxation existed for -- a cluster is not wide.
        #     Measured, two sub-bands of one ladder rung refused, each one zero
        #     short.
        #
        # A SHORT GRAM BLOCK IS NOT A GUESS. It names its own deficit and its
        # own interval, so escalating the density INSIDE the short blocks is a
        # search proportional to what is missing and aimed where it is. The
        # block pass before the loop runs at density 48; here it climbs.
        recovered = _repair_by_block(
            found, gram_indices, gram_points_, gram_values, density=block_density
        )
        # DID `found` GROW, not did the rescan RETURN anything. The block pass
        # scans its blocks closed -- see `_repair_by_block` -- so it re-finds
        # the zeros already in them and always returns something. Testing that
        # made the loop `continue` on every pass without ever escalating, and
        # sub-bands plateaued two zeros short.
        before = len(found)
        if len(recovered):
            found = _dedupe(np.concatenate([found, recovered]))
            found = found[(found > low) & (found <= high)]
        if len(found) > before:
            continue
        if block_density < 8000:
            block_density = min(block_density * 8, 8000)
            continue
        if density < 8000:
            density = min(density * 4, 8000)
            continue
        break

    raise RuntimeError(
        f"located {len(found)} zeros in ({low}, {high}], but ZeroCount says "
        f"{expected}; refine further rather than trusting the short list"
    )


def zero_ordinates(height: float, *, max_passes: int = 14) -> np.ndarray:
    """Every ordinate of a critical-line zero in `(0, height]`, in order.

    Raises if the list cannot be made to agree with `ZeroCount`. Returning a
    short list quietly is the one outcome that cannot be allowed: every
    downstream sum over zeros would be missing a term and would still look like
    an answer.

    `ZeroCount` is a SECOND IMPLEMENTATION, not an independent count -- it walks
    Gram and Rosser blocks and separates the zeros inside a block by the sign
    changes of Z, which is what this does too. The independent count is
    `argument_principle.strip_zero_count`, which never touches the critical
    line. This docstring claimed otherwise after the module docstring had been
    corrected, which is how a claim survives being retracted.
    """
    import sympy as sp

    from .functions import ZeroCount

    expected = int(ZeroCount(sp.Float(height)))
    found = _dedupe(_gram_scan(height))
    found = found[found <= height]

    # See `zeros_in_band`: repair by Gram block before falling back on how wide
    # a gap looks.
    gram_indices, gram_points_, gram_values = _gram_structure(height)
    recovered = _repair_by_block(found, gram_indices, gram_points_, gram_values)
    if len(recovered):
        found = _dedupe(np.concatenate([found, recovered]))
        found = found[found <= height]

    # See `_sweep_edges`. A full run starts at 9.0, so only the top edge can be
    # partial, but the sweep costs two intervals either way.
    edge = _sweep_edges(found, 9.0, height)
    if len(edge):
        found = _dedupe(np.concatenate([found, edge]))
        found = found[found <= height]

    density = 40
    block_density = 48
    threshold = _GAP_THRESHOLD
    for _ in range(max_passes):
        if len(found) == expected:
            return np.sort(_polish(found))
        edges = np.concatenate([[9.0], found, [height]])
        widths = np.diff(edges)
        suspect = np.nonzero(widths > threshold * _average_gap(edges[:-1]))[0]
        extra = np.array([])
        if len(suspect):
            # Strictly inside each gap: the endpoints are zeros already found,
            # and rescanning them is what produced duplicates.
            lows, highs = edges[suspect], edges[suspect + 1]
            inset = np.minimum(SAME_ZERO * 100, (highs - lows) / 100)
            extra = _rescan_intervals(lows + inset, highs - inset, density)
        if len(extra):
            found = _dedupe(np.concatenate([found, extra]))
            found = found[found <= height]
            density = min(density * 4, 8000)
            continue
        # See `zeros_in_band`: escalate the density inside the SHORT GRAM
        # BLOCKS, which name what is missing, rather than over the whole range.
        recovered = _repair_by_block(
            found, gram_indices, gram_points_, gram_values, density=block_density
        )
        # See `zeros_in_band`: the test is whether `found` grew.
        before = len(found)
        if len(recovered):
            found = _dedupe(np.concatenate([found, recovered]))
            found = found[found <= height]
        if len(found) > before:
            continue
        if block_density < 8000:
            block_density = min(block_density * 8, 8000)
            continue
        if density < 8000:
            density = min(density * 4, 8000)
            continue
        break

    raise RuntimeError(
        f"located {len(found)} zeros below {height}, but ZeroCount says "
        f"{expected}; refine further rather than trusting the short list"
    )


# --- verifying ordinates that were stored rather than just computed --------

#: How many ulps of displacement the check is willing to call "the same zero".
#:
#: SET BY WHAT COULD CHANGE AN ANSWER, NOT BY WHAT IS DETECTABLE. At t = 10^6 a
#: mean gap is 0.52, which is 4.5 BILLION ulps. Any corruption that could move a
#: spacing statistic has to move an ordinate by a real fraction of a gap, so it
#: is millions of ulps at least. 1024 is therefore about a million times below
#: anything that matters, while still leaving room for the arithmetic.
#:
#: THE ROOM IS NEEDED, AND 64 DID NOT LEAVE IT. Calibrated on the first 20,000
#: zeros the worst clean ordinate sat at 22.1 and 64 looked generous. Over the
#: real 1,747,146-zero set it is not: 48 ordinates land at 66-167 ulps. They are
#: not corrupt. They sit at CLOSE PAIRS -- median local gap 0.039 of the mean
#: against 0.723 for ordinary zeros -- where `|Z'|` is small, the root find is
#: ill-conditioned, and the stored value was never computed better than that.
#: A bound taken at the bottom of a range and read at the top is exactly the
#: mistake this repository keeps writing down.
#:
#: Flipping mantissa bit `k` moves an ordinate by `2**k` ulps and moves `|Z|` by
#: the same factor, so this catches every flip at bit 11 and above, about half
#: at bit 10, and none below. Not catching the low bits is correct and not a
#: weakness: a value a few ulps from the zero IS the zero in float64, and a gate
#: that called that corruption would report on the representation rather than on
#: the data.
ORDINATE_ULP_TOLERANCE = 1024.0


class OrdinateCheck(NamedTuple):
    """What one verification pass over stored ordinates found.

    `ratio` is per-ordinate and dimensionless: `|Z(gamma)|` divided by what one
    ulp of `gamma` alone would produce. A CONSTANT threshold cannot do this job
    -- `|Z'|` grows with height, so a bound that is generous at t = 100 is
    impossibly tight at t = 10^6, and the same fixed number would pass
    everything low and fail everything high.
    """

    count: int
    ratio: np.ndarray
    tolerance: float

    @property
    def failed(self) -> np.ndarray:
        """Indices whose residual exceeds what their own precision allows."""
        return np.flatnonzero(self.ratio > self.tolerance)

    @property
    def ok(self) -> bool:
        return self.failed.size == 0

    @property
    def worst(self) -> float:
        return float(self.ratio.max()) if self.count else 0.0


def verify_ordinates(
    ordinates: np.ndarray, *, tolerance: float = ORDINATE_ULP_TOLERANCE
) -> OrdinateCheck:
    """Evaluate `Z` at each stored ordinate and compare it to its own precision.

    WHY THIS EXISTS. Ordinates get computed once and read back many times, from
    a `.npy` on a desktop with NO ECC MEMORY. A flipped bit in a stored zero is
    silent: the array still loads, the statistics still run, and the answer is
    wrong by however much that bit was worth. This is the cheap pass that says
    so -- around 70,000 zeros a second, which is a rounding error against the
    hours that produced them.

    WHAT IT DOES NOT CHECK. That the set is COMPLETE. Every ordinate here can
    be a genuine zero while zeros are missing between them; `ZeroCount` is what
    answers that, and this says nothing about it.

    WHAT A HIT MEANS. The ratio is, near enough, the stored ordinate's distance
    from the true zero measured in ulps. It does NOT separate a flipped bit from
    a root find that stopped early, and at close pairs the second happens: see
    `ORDINATE_ULP_TOLERANCE`. What it does guarantee is that nothing has moved
    far enough to change a measurement.

    THE DERIVATIVE IS TAKEN OVER THE MEAN GAP, NOT OVER AN ULP. The obvious
    central difference at +/- one ulp is almost pure cancellation -- the two
    evaluations agree to nearly every bit -- so it returns a slope that is too
    small exactly where the slope is hardest to estimate. Measured that way the
    worst clean ratio was 11,291 against a 99th percentile of 15: a tail that
    described the estimator and not the data. Stepping a thousandth of the
    local mean gap `2 pi / log(t / 2 pi)` keeps the difference well clear of
    the noise and brought the worst clean ratio to 22.
    """
    ordinates = np.atleast_1d(np.asarray(ordinates, dtype=float))
    if ordinates.size == 0:
        return OrdinateCheck(0, np.empty(0), tolerance)

    gap = 2 * np.pi / np.log(ordinates / (2 * np.pi))
    step = gap / 1000.0
    slope = np.abs(z_function(ordinates + step) - z_function(ordinates - step))
    slope /= 2 * step

    allowance = slope * np.spacing(ordinates)
    ratio = np.abs(z_function(ordinates)) / allowance
    return OrdinateCheck(int(ordinates.size), ratio, float(tolerance))


def height_for(count: int) -> float:
    """Roughly the height below which there are `count` zeros.

    By inverting the Riemann-von Mangoldt formula `N(T) = x log x - x + 7/8`
    with `x = T / 2 pi`, rather than by assuming a gap.

    Assuming the gap was 2 pi put the thousandth zero at t = 6300 when it is at
    1419, so a request for a thousand ordinates scanned six thousand of them --
    the estimate cost more than everything it was feeding. The average gap is
    `2 pi / log(t / 2 pi)`, which is about 1.16 there, not 6.28.
    """
    if count < 1:
        return 25.0
    x = max(float(count), 2.0)
    for _ in range(60):
        value = x * np.log(x) - x + 0.875 - count
        slope = np.log(x)
        if slope <= 0:
            break
        step = value / slope
        x -= step
        if x < 1.5:
            x = 1.5
        if abs(step) < 1e-12:
            break
    return max(25.0, float(2 * np.pi * x))


def first_zero_ordinates(count: int) -> np.ndarray:
    """The first `count` ordinates.

    The height needed is estimated and then CHECKED, so an estimate that fell
    short widens rather than silently truncating.
    """
    if count < 1:
        return np.array([])
    # A little above the estimate, since N(T) wanders around its main term.
    height = height_for(count) + 12.0
    for _ in range(40):
        found = zero_ordinates(height)
        if len(found) >= count:
            return found[:count]
        height = height * 1.1 + 10.0
    raise RuntimeError(f"could not reach {count} zeros")
