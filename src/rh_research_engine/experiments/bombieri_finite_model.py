"""Bombieri's finite model: a published experiment to check this project against.

Every threshold this project measures is a number nobody else has computed, which
means nothing else has ever disagreed with one. That is not a strong position. A
control taken against a published result is what makes the rest readable, and
this is the only one available: Bombieri ran the off-line-zero experiment in
2000, printed the curves, and proved two exact theorems about the matrix.

[E. Bombieri, *Remarks on Weil's quadratic functional in the theory of prime
numbers I*, Rend. Lincei 11(3):183-233 (2000)], §7 eqs. (7.1)-(7.5), p. 203.

THE CONSTRUCTION IS ALREADY HIS. Footnote 11, p. 225: the experiments added a
fake zero off the critical line "together with its symmetric images" to the first
k ordinates, k up to 320. Theorem 8 was *discovered* by running it.

His quadruple and this project's are NOT the same perturbation, and the header
of this file said they were until the contradiction with `with_quadruple` below
was noticed. He ADDS four zeros and removes nothing; `weil_sensitivity` collides
two on-line pairs and moves them off, conserving the count as the functional
equation requires of zeta. Both are legitimate -- his is the published one, so it
is what this control runs -- but their thresholds are different numbers and must
not be quoted for one another. §13 then observes a critical
support `t_c` -- below which the negative eigenvalue decays to zero as the
truncation refines, above which it settles -- and calls locating it "the main
question here". It is still open; Suzuki (arXiv:2606.09096 Thm 1.3) later proved
the continuity that makes the crossing exist.

ASSEMBLE FROM (7.1), NOT FROM `K*`. Substituting (7.1) into (7.3) gives

    H(x,y,t) = 2t R(x,y,t) / [(1/4 + x^2)(1/4 + y^2)]
    R(x,y,t) = (1/4 + xy) sin(t(x-y))/(t(x-y))
               - [cosh(t) cos(t(x-y)) - cos(t(x+y))] / (2t sinh(t))

`R` is manifestly symmetric and so is the denominator, so H is symmetric -- and
for real ordinates entirely real. The `K*` form of (7.1) evaluates the kernel at
COMPLEX arguments `t(i/2 +/- x)` in every entry; this one never does, except in
the four rows the fake zero occupies. The only singularity is removable.

H IS INDEXED BY THE ZEROS, NOT BY A BASIS OF FUNCTIONS. It is `|Gamma| x |Gamma|`
and contains NO PRIMES: the arithmetic sits entirely in which ordinates are fed
to it. So this control has no prime sum, no quadrature and no special functions,
which makes it the one computation in this literature with no exposure to the
Hurwitz-Lerch defect that has produced published false positives elsewhere.

TWO EXACT GATES, BOTH UNCONDITIONAL IN `t`:

  `assert_lemma_10` -- with every ordinate real and none repeated, no eigenvalue
  is negative, at any `t`. This is the one that catches a sign or normalisation
  error quietly manufacturing detections out of an on-line-only zero set.

  `negative_count` against Theorem 8 -- the number of negative eigenvalues equals
  the number of distinct complex conjugate pairs. One fake quadruple gives
  exactly two, one per parity sector.

THE SECOND GATE REPRODUCED §13's OWN TRAP INSIDE ITSELF. At N = 20, t = 0.5 the
count came back zero, because the two negative eigenvalues are -1.6e-15 and
-5.5e-18 against a float64 floor near 1e-14. Theorem 8 is unconditional, so that
is precision running out exactly where Bombieri says the eigenvalue decays
exponentially -- reporting it as a refutation would be the whole not-tested-
versus-refuted rule failing one level down. Hence UNRESOLVED as a third verdict,
and hence the floor is an argument rather than something read off the answer.

THE FIGURES PLOT THE EIGENVALUE, NOT `Lambda = 1/lambda`, AND TREND SETTLES IT.
§13 calls the plotted quantity `lambda^{+/-}_N(t)` while (7.2) sets `Lambda =
1/lambda`, so the convention is genuinely ambiguous from the text. Fig. 1 grows
in magnitude with `t`; across `t = 1 -> 5.5` the eigenvalue grows (-5.3e-5 ->
-8.8e-3) and its reciprocal SHRINKS by two orders (-18763 -> -114). Only one of
those can be the figure. Settled without needing the axis scale, which is the
point -- the scale could not be settled, see below.

THE BELOW-`t_c` BRANCH IS NOW ENCLOSED, NOT OBSERVED. §13's dichotomy is that
below a critical support the negative eigenvalue decays to zero as the truncation
refines, and above it settles -- and deciding which happens is what he calls the
main question. In ball arithmetic at t = 0.5, with ordinates from Arb's own zero
isolation (`acb.zeta_zeros`, radius 1e-28, `Re = 1/2` returned exactly):

    N = 5      -1.404213926671850975459439689060...e-07   radius 4.3e-61
    N = 10     -2.848807426222857935303207199538...e-11   radius 8.3e-61
    N = 20     -1.434768839282985557014601673680...e-15   radius 4.0e-60

Four to five orders per doubling, rigorously. So for this `rho_0` at this support
the decay is proved rather than seen -- the branch Bombieri could only observe at
N <= 160 in floating point in 2000.

AND IT SHOWS WHERE FLOAT64 STOPS. The same values in double precision are exact
to six figures at N = 5 and 10, and 8% wrong at N = 20 (-1.54962e-15 against
-1.43477e-15). Perturbing each stored ordinate by its own ulp had predicted that
row was 14% ordinate-limited, before the certified run existed to check it. Two
estimates of one boundary, made by different means, agreeing.

BOTH CERTIFIED ROUTES ARE PRECISION-LIMITED, AND AN EARLIER VERSION OF THIS FILE
SAID OTHERWISE. `acb_mat.eig` with `nonstop=False` refuses to isolate at N = 40,
t = 0.5 -- and returns the correct count of 2 one rung up:

    size 84, t = 0.5    dps  60   eig REFUSED        descartes UNRESOLVED
    size 84, t = 0.5    dps 150   eig 2, in 17 s
    size 84, t = 0.5    dps 800                      descartes UNRESOLVED
    size 84, t = 0.5    dps 1600                     descartes 2, in 502 s

This paragraph previously read that the refusal was structural: that below
t ~ 0.75 the spectrum accumulates at zero, eigenvalues which are not separated
stay unseparated at any precision, and the route into the deep end was therefore
to change the object rather than the solver. It told the reader not to go looking
for a better eigensolver. **All of that was measured on dirty inputs** -- a
precision ladder that constructed the quadruple's `arb` literals once at 60
digits and reused them at 120, 200 and 300, so a 1e-60-wide input sat in four
rows and nothing downstream could improve. Same bug, same file, twice: the
Descartes route was diagnosed the same way an hour earlier.

WHAT SURVIVES, BECAUSE IT WAS MEASURED WITH CLEAN INPUTS. The cluster is real: at
t = 0.5 the eigenvalues within 1e-12 of zero number 0, 1, 11, 35, 90 as N runs 5,
10, 20, 40, 80. At t = 2.3 nothing comes within 1e-12 of zero at any N.

The neighbour gaps in that scan go 7.1e-07, 1.4e-10, and then 9.4e-20, 1.0e-21,
6.3e-23 -- AND THE LAST THREE ARE NOT MEASUREMENTS. They are differences of
float64 eigenvalues of a matrix whose largest is order one, so anything below
about 1e-16 relative is the subtraction's own noise, not a gap. Quoting them as
a boundary would be reading structure off a floor, which is the defect this
project is named for.

THE REAL GAPS ARE FORTY ORDERS BELOW WHAT FLOAT64 REPORTED, and the certified
spectrum gives them for nothing: each eigenvalue is an enclosure, so each
difference is one too. At size 84, t = 0.5, dps 150 all 84 eigenvalues resolve
away from zero, all 83 neighbour gaps are PROVED positive, and the smallest is

    8.360e-62 relative      (float64 had reported 1.0e-21 -- its own floor)

which needs 61.1 digits to represent at all. Ladders on the same cell:

    dps   62   66   75   90  120  |  135  150
    eig  ref  ref  ref  ref  ref  |   2    2

So `eig` crosses between 1.96x and 2.21x the bare gap requirement on this cell.
One point is not an exponent, so six, across two implementations sharing only the
paper (rows marked * were run by the other):

    size   t     min gap      digits g   crossing    factor
     84   2.3*   8.751e-09       8.1      (13,  15]   1.60-1.85
     44   2.3    8.224e-08       7.1      (  -, 20]      <= 2.82
     44   0.5    2.570e-25      24.6      (  -, 50]      <= 2.03
     84   0.75*  1.140e-31      30.9      (59,  64]   1.91-2.07   PREDICTED FIRST
     80   0.5    7.734e-55      54.1      (102,113]   1.88-2.09   HERMITIAN
     84   0.5    8.360e-62      61.1      (120,135]   1.96-2.21

Sizes 44-84, `t` from 0.5 to 2.3, gaps spanning 8 to 61 digits, Hermitian and
not. The `t = 0.75` row was the strong one: the gap was extracted first and the
crossing PREDICTED at 62 before any ladder ran, and it landed in (59, 64].

BUDGET `2 |log10(min relative gap)| + margin`, NOT "the crossing is 2g". The
shallow cell already contradicts the exact form by a digit -- 2g predicts 16 and
the crossing is at most 15 -- but across all six 2g never once UNDER-estimated,
which is the direction that matters for a budget.

AND THE OBVIOUS MECHANISM FOR THE 2 IS REFUTED. Near-degenerate pairs of a
non-normal matrix are nearly defective, and eigenvalues of nearly defective pairs
move as the SQUARE ROOT of a perturbation -- so resolving a gap `G` would need
backward error `G^2`, exactly twice the digits. That predicts 1x for a Hermitian
matrix, whose eigenvalues move linearly by Weyl. The on-line-only form is real
symmetric with the same cluster: predicted 54 digits, measured 113, factor 2.09.
Same 2x. The factor is a property of the certification path and is indifferent to
matrix structure.

THE END-TO-END RECIPE, WITH NO CERTIFIED PRE-RUN ANYWHERE. Take a float64 gap
scan. Above its own noise floor it is accurate -- 8.8e-9 against a certified
8.751e-9, 0.6% -- so use it as `g` and run once at `2g + margin`. Bottomed out,
it is fiction: across four deep cells the true gap was 6 to 40 orders below what
float64 printed, so the reading does not even bound the truth. There, take the
true gap from the first success and every rerun of that cell is priced.

Under that reading nine cells from two independent implementations agree and none
disagrees.

And the mechanism stands, because it is about the operator rather than a solver.
Up to the weight, `H` is the Gram matrix of `exp(-i gamma u)` on `[-t, t]`, and on
a short interval those functions are nearly parallel -- so the form is nearly
degenerate, more so with every ordinate added. The same fact is Yoshida's
small-support positivity seen from the other side: little room means positive,
and means degenerate. That is why the below-`t_c` regime is expensive.

WHAT DOES NOT SURVIVE IS "AND THEREFORE NO PRECISION CAN HELP". The cluster sets
the COST -- 150 digits where 60 will not do, 1600 for the other route -- and the
ceiling inferred from it did not exist. A radius that does not move as precision
rises is input-limited, not arithmetic-limited, and that signature was present in
the ladder the whole time.

The general lesson is not "check ladders". It is that WHEN A BUG IS FOUND IN ONE
CONCLUSION, EVERY OTHER CONCLUSION DRAWN FROM THE SAME FILE IS SUSPECT. The
Descartes ladder was corrected and the eigensolver conclusion beside it, reached
the same way in the same file, was left standing for another hour.

Three boundaries land near t = 0.75: where `t_c` sits, where float64 ordinates
stop carrying the computation, and where `eig` stops isolating. THEY ARE NOT
THREE INDEPENDENT CONFIRMATIONS. The last two are both driven by the
accumulation, so they were never going to disagree. Only the first is a
mathematical statement; the others are that statement observed through an
instrument.

FOUR TRAPS IN THE CERTIFIED PATH, ALL VERIFIED HERE RATHER THAN READ:

  * `sin(u)/u` on a ball straddling zero returns nan and one nan poisons the
    factorisation; `acb.sinc` is exact there. The argument between `gamma_0` and
    its conjugate is `-2 i eta t`, which goes to zero along the DISPLACEMENT
    axis -- so this is not a diagonal special case, it is the small-eta regime.
  * `acb.sinc_pi` is `sin(pi u)/(pi u)`, numpy's convention, and would build a
    clean finite entirely wrong matrix -- `sinc(1) = 0.8414...`, `sinc_pi(1) = 0`.
  * `acb_mat.eig` defaults to `vdhoeven_mourrain`, "faster and less accurate";
    `rump` is the one to pass.
  * The ordinates are the uncertified input, not the arithmetic. float64 carries
    ~1e-13 at gamma ~ 143 while the eigenvalues here reach 1e-17.

SHAPE VALIDATED, ABSOLUTE SCALE NOT DETERMINED. Ratios within the published
curve against ratios within this one: `t = 4.5 -> 5.5` gives 1.83 against 1.83,
`3.5 -> 5.5` gives 4.4 against 3.9, `2.0 -> 5.5` gives 18.3 against 21.1. The
agreement degrades toward small `t`, which is where reading a near-zero curve off
a scan is least reliable, and the residual scale factor drifts by 50% across the
range rather than sitting constant -- so it is not a missing unit. Recorded as
validated on `t >~ 3.5` and left there; chasing a factor that is not constant is
chasing scan noise.
"""

from __future__ import annotations

import numpy as np

from ..core.models import ExperimentResult

#: Bombieri's own fake zero, §13: `rho_0 = 0.52 + 3.14i`, so the ordinate is
#: `gamma_0 = (rho_0 - 1/2)/i = mu - i eta`. Quoted so the control is run at the
#: published point rather than at one chosen here.
FAKE_HEIGHT = 3.14
FAKE_DISPLACEMENT = 0.02
#: Truncations to sweep. The question `t_c` answers is about the limit, so a
#: single one of these cannot address it -- see `retention`.
TRUNCATIONS = (5, 10, 20, 40)


def _sinc(argument: np.ndarray) -> np.ndarray:
    """`sin(w)/w` with the removable singularity handled, for complex `w`."""
    argument = np.asarray(argument, dtype=complex)
    small = np.abs(argument) < 1e-12
    safe = np.where(small, 1.0, argument)
    return np.where(small, 1.0 - argument**2 / 6.0, np.sin(safe) / safe)


def kernel(gammas: np.ndarray, t: float) -> np.ndarray:
    """`H(Gamma;t)` from (7.1) and (7.3). Symmetric; real when every ordinate is."""
    x = np.asarray(gammas, dtype=complex)[:, None]
    y = np.asarray(gammas, dtype=complex)[None, :]
    remainder = (0.25 + x * y) * _sinc(t * (x - y)) - (
        np.cosh(t) * np.cos(t * (x - y)) - np.cos(t * (x + y))
    ) / (2 * t * np.sinh(t))
    return 2 * t * remainder / ((0.25 + x**2) * (0.25 + y**2))


def spectrum(gammas: np.ndarray, t: float) -> tuple[np.ndarray, float]:
    """Eigenvalues, and how far from real they were before the real part was taken.

    `Gamma` is closed under conjugation, so the spectrum is too and the
    eigenvalues are real. Taking `.real` without checking would hide a genuinely
    complex eigenvalue, which would be a fact about a broken assembly rather than
    a rounding artifact -- so the drift is returned and the caller must look.
    """
    values = np.linalg.eigvals(kernel(gammas, t))
    scale = max(float(np.abs(values.real).max()), 1e-300)
    return np.sort(values.real), float(np.abs(values.imag).max() / scale)


def on_line(ordinates: np.ndarray, count: int) -> np.ndarray:
    """`{+/- gamma_1, ..., +/- gamma_count}`, every one on the critical line."""
    return np.concatenate([ordinates[:count], -ordinates[:count]]).astype(complex)


def with_quadruple(
    ordinates: np.ndarray,
    count: int,
    height: float = FAKE_HEIGHT,
    displacement: float = FAKE_DISPLACEMENT,
) -> np.ndarray:
    """The same, plus `rho_0` and its three images -- Bombieri's own perturbation.

    `rho_0 = 1/2 + displacement + i height` has ordinate `gamma_0 = height - i
    displacement`, and the functional equation and conjugation supply
    `conj(gamma_0)`, `-gamma_0`, `-conj(gamma_0)`. Nothing is removed: this ADDS
    four zeros, which is his construction and not the count-conserving one in
    `weil_sensitivity`. The two are different perturbations and their thresholds
    are not interchangeable.
    """
    fake = height - 1j * displacement
    return np.concatenate(
        [on_line(ordinates, count), [fake, fake.conjugate(), -fake, -fake.conjugate()]]
    )


def negative_count(gammas: np.ndarray, t: float) -> tuple[int, int, float]:
    """Negatives beyond the floor, and how many sit inside it.

    Theorem 8 makes the count exact and unconditional in `t`, so a count that
    disagrees is either a broken assembly or a floor the eigenvalues have fallen
    below. Those are different findings and the second is the expected one at
    small `t`, so both are returned rather than collapsed into a verdict here.
    """
    values, _ = spectrum(gammas, t)
    floor = len(values) * float(np.finfo(float).eps) * float(np.abs(values).max())
    return int((values < -floor).sum()), int((np.abs(values) <= floor).sum()), floor


def assert_lemma_10(
    ordinates: np.ndarray, count: int, t: float, _assemble=None
) -> float:
    """Lemma 10: real, distinct ordinates give no negative eigenvalue, at any `t`.

    Returns the smallest eigenvalue, which must be non-negative beyond the floor.
    Raises otherwise -- a fail-closed refusal, because a violation here means the
    matrix is wrong and every threshold built on it is meaningless.

    `_assemble` EXISTS SO THE REFUSAL CAN BE WATCHED WORKING. Lemma 10 is a
    theorem, so no legitimate input reaches the raise: the only way in is a wrong
    matrix, and the only honest way to supply one is a seam here rather than by
    mutating numpy from a test. A fail-closed branch nobody has seen fire is a
    branch nobody has seen fire, and this one carries every other number in the
    module.
    """
    assemble = _assemble or kernel
    matrix = assemble(on_line(ordinates, count), t)
    raw = np.linalg.eigvals(matrix)
    scale = max(float(np.abs(raw.real).max()), 1e-300)
    values, drift = np.sort(raw.real), float(np.abs(raw.imag).max() / scale)
    floor = len(values) * float(np.finfo(float).eps) * float(np.abs(values).max())
    if values[0] < -floor:
        raise ValueError(
            f"Lemma 10 violated at count={count}, t={t}: smallest eigenvalue "
            f"{values[0]:.6e} against a floor of {floor:.3e}. Every ordinate is "
            "real and distinct, so the matrix is wrong."
        )
    if drift > 1e-9:
        raise ValueError(f"H is not real for real ordinates: drift {drift:.3e}")
    return float(values[0])


def retention(
    ordinates: np.ndarray,
    t: float,
    displacement: float = FAKE_DISPLACEMENT,
    truncations: tuple[int, ...] = TRUNCATIONS,
) -> tuple[float, list[float]]:
    """How much of the negative eigenvalue survives refining the truncation.

    Below `t_c` it tends to zero as the truncation grows; above, it settles. The
    fraction retained is therefore the only thing a finite sweep can report, and
    IT IS NOT AN ESTIMATOR OF `t_c`. Suzuki Thm 1.3 gives the eigenvalue
    continuous in `t`, so no finite curve has a corner at `t_c` -- what changes
    there is the character of the LIMIT, and interpolating a retention curve to
    find a crossing measures the convention, not the threshold. Report the
    bracket the trend supports and nothing narrower.
    """
    values = [
        float(spectrum(with_quadruple(ordinates, count, displacement=displacement), t)[0][0])
        for count in truncations
    ]
    return (values[-1] / values[0] if values[0] else float("nan")), values


def decay_ratios(
    ordinates: np.ndarray,
    t: float,
    displacement: float = FAKE_DISPLACEMENT,
    truncations: tuple[int, ...] = TRUNCATIONS,
) -> list[float]:
    """`lambda(N_{k+1}) / lambda(N_k)` -- the shape of §13's dichotomy, without a convention.

    `retention` above answers "how much survives", which is a single number whose
    crossing is a threshold somebody chose. THIS is the better instrument,
    because the SEQUENCE has structure that no choice is needed to read:

        t      N=10->20     N=20->40     reading
       0.50    5.036e-05    4.420e-08    collapsing, and accelerating
       0.60    7.433e-01    5.958e-01    falling
       0.70    6.859e-01    8.499e-01    rising toward 1
       0.80    8.269e-01    9.243e-01    rising
       0.90    9.529e-01    9.654e-01    rising
       1.00    9.776e-01    9.922e-01    settled
       1.50    9.937e-01    9.953e-01    settled

    Ratios rising toward 1 mean the sequence is converging to a non-zero limit;
    ratios falling mean it is not; and at t = 0.5 they do not merely fall but
    accelerate, losing five orders and then eight. That is Bombieri's two
    branches, visible in a finite computation, and the second difference is a
    fact about the sequence rather than a line drawn across it.

    Compare the retention fractions over the same range -- 0.00, 0.39, 0.46,
    0.73, 0.90, 0.95 -- which are a smooth crossover with nothing to read.

    AND AT FOUR POINTS THE TWO NEIGHBOURING VALUES OF `t` SEPARATE COMPLETELY,
    certified, each cell priced from its float64 gap and run once:

        t      10->20    20->40    40->80
       0.60     0.743     0.596     0.491     falling, and accelerating
       0.70     0.686     0.850     0.926     rising monotonically toward 1

    Monotone in opposite directions over three doublings. So `t_c` lies between
    them: `t_c` in (0.6, 0.7), against Bombieri's own (0.48, 2.3) read off Figs
    1-3 -- about eighteen times tighter, on his operator, for his `rho_0`. In
    cutoff terms `c = e^(2t)` in (3.3, 4.1).

    AND IT IS NOT SPECIAL TO HIS `rho_0`. Repeating at mu = 17.5784 -- the
    midpoint of gamma_1 = 14.135 and gamma_2 = 21.022, so 3.45 clear of both and
    properly AMONG the ordinates rather than in the empty region below the first
    zero:

        t      10->20    20->40    40->80
       0.60     0.0007    0.0000              collapsing
       0.70     0.0509    0.1782    0.6359    rising
       0.90     0.8597    0.8803              rising

    Same window. `t_c` does not move when the off-line zero sits among the
    on-line ones, over a 5.6-fold range in height. A float64 scan had suggested
    it jumps to about 0.9 there; that was the noise floor, at heights within one
    or two of a real ordinate.

    AND THE DISPLACEMENT AXIS IS FLAT WHERE HE PUT HIS ZERO, which is what makes
    the bracket a general number rather than a measurement of one point. On the
    ratio instrument in float64, on a +/-0.025 grid:

        eta     0.005   0.010   0.020   0.050   0.100   0.200
        t_c     0.625   0.625   0.625   0.575   0.525   0.525

    flat to within resolution for `eta <= 0.02`, decreasing only once the
    excursion gets large. `FAKE_DISPLACEMENT` is 0.02 -- Bombieri's own -- so his
    `rho_0` sits inside the plateau, and (0.6, 0.7) is the value for ANY small
    off-line excursion rather than a point on a surface.

    THE STRENGTH GOES AS `eta^2` IN THE LIMIT AND NOT IN THE MEASURED RANGE, and
    the difference is the whole content. `weil_sensitivity` derives
    `Delta = -2 eta^2 F''(mu) + O(eta^4)` for the count-conserving quadruple, and
    the same expansion governs the additive one here: at `eta = 0` Lemma 10
    forbids a negative eigenvalue, so the entire negative part is the `eta^2`
    term. Measured at mu = 3.14, t = 0.8, N = 20, as the exponent per doubling:

        eta range              log2(ratio)
        0.000625 -> 0.00125       2.014
        0.00125  -> 0.0025        2.032
        0.0025   -> 0.005         2.068
        0.005    -> 0.01          2.140
        0.01     -> 0.02          2.242
        0.02     -> 0.04          2.199

    Monotone toward 2 as `eta -> 0` -- slope 2.023 over the three smallest and
    2.221 over the three largest. So the derivation is confirmed by measurement
    AND the departure from it is the `O(eta^4)` term the same derivation
    predicts, which is a stronger result than agreement would have been.

    BUT DO NOT QUOTE `eta^2` FOR THE MEASURED REGIME. At eta = 0.005..0.02 the
    effective exponent is 2.14-2.24, up to twenty percent per doubling away from
    the leading term, and `FAKE_DISPLACEMENT` sits at the top of that range. A
    scan over only those three values returns ratios near 4.3-4.7 and reads as
    "eta^2 confirmed" if 4 is the only number being looked for. The limit law and
    the fitted exponent are different claims about different ranges.

    AN EARLIER SCAN OF THIS AXIS GAVE A STEADY DECLINE AND WAS WRONG, and the
    reason matters more than the numbers. It reported `t_c` = 0.741, 0.622,
    0.604, 0.578, 0.555 over the same range, i.e. monotone in `eta`. Those ran on
    THIS operator, not on the Gaussian proxy -- so re-measuring on the real
    operator does not fix them. They are wrong because they are RETENTION
    CROSSINGS, the convention `decay_ratios` above replaces, and a scan that
    switches instrument gets a different shape. Calling them "proxy-era" would
    send the next reader to re-run them with the same broken tool and read the
    identical answer back as a confirmation.

    AND THE PLATEAU IS CERTIFIED, not merely float64. Repeating on the ratio
    instrument in ball arithmetic, reading the direction of the second ratio:

        eta      t=0.5   t=0.6   t=0.7   t=0.8   t=0.9      flip
        0.005    fall    fall    rise    rise    rise     (0.6, 0.7)
        0.020    fall    fall    rise    rise    rise     (0.6, 0.7)
        0.100    fall    rise    rise    rise    rise     (0.5, 0.6)

    So the flat region and the drop at larger displacement are both real, and
    `FAKE_DISPLACEMENT = 0.02` -- Bombieri's own -- lands squarely in the flat
    part. **(0.6, 0.7) IS THE SMALL-DISPLACEMENT PLATEAU VALUE**, not a reading
    taken at one convenient displacement.

    The float64 grid gave the same shape at +/-0.025 and could not pin either the
    plateau or the onset; this pins the plateau to the same bracket as the
    headline and puts the onset between eta = 0.02 and 0.1.

    TWO DIFFERENT QUANTITIES, AND CONFLATING THEM IS THE TRAP. The SIGN threshold
    is height-independent -- detection is possible at the same support whatever
    the height. The STRENGTH above it is not: at t = 0.7 the N = 40 eigenvalue
    goes -2.217e-06 at mu = 3.14 to -3.483e-11 at mu = 17.58, a decay of 0.766
    per unit height. So "can this be detected" and "how loud is it" answer
    differently, and the height ceiling this project once claimed was a statement
    about the second read as though it were about the first.

    THE HEIGHT DECAY IS NOT A CLEAN EXPONENTIAL, and a single rate should not be
    quoted. Measured at t = 0.8, N = 20, with every height taken at the MIDPOINT
    of a gap between consecutive ordinates so the fake zero is never adjacent to
    a real one:

        mu       |lambda_min|   headroom over floor   local rate
         3.140     1.068e-05          3.0e+09
        17.578     1.306e-08          1.2e+08          -0.464
        27.718     2.493e-12          3.0e+04          -0.845
        35.261     9.341e-15          1.1e+02          -0.740

    Overall `d ln|lambda| / d mu = -0.657`, but the local rate runs -0.46 to
    -0.85 -- nearly a factor of two across the range. So `e^{-c mu}` describes the
    order of magnitude and not the curve, and `c` is a summary rather than a
    constant. Same status as the `b ~ 0.55` decay exponent: a bounded qualitative
    shape, not a rate law.

    IT WEAKENS WITH SUPPORT, AND THE RATE IS A SUMMARY AT EVERY `t`. Certified at
    dps 250 over five gap midpoints, all interior to `gamma_20 = 77.1`:

        t      rate     local range
       0.70   -0.825   -1.15 to -0.66
       0.80   -0.679   -0.84 to -0.46
       0.90   -0.547   -0.71 to -0.30
       1.00   -0.421   -0.66 to -0.29

    Monotone in `t`: more support buys back the height suppression. And the local
    variation is about a factor of two at EVERY support, so quoting a single `c`
    is wrong everywhere rather than only at `t = 0.8`.

    A FLOAT64 VERSION OF THIS TABLE GAVE 0.77 / 0.51 / 0.30 AND EVERY VALUE WAS
    TOO SMALL, for two reasons already named on this axis: one height sat 0.025
    from `gamma_4`, and the far points fell to the float64 floor -- at t = 0.7,
    0.8 and 0.9 the fifth height has headroom 0.0, 0.0 and 5.0, so those fits ran
    through numbers that were not measurements. Only `t = 1.0` had room (1368).
    Certified arithmetic removes the floor problem entirely, which is why the
    table above is computed rather than filtered.

    A NON-MONOTONICITY WAS FOUND AT N = 20 AND IS A TRUNCATION EDGE EFFECT.
    Extending the sweep to seven gap midpoints at N = 20, t = 0.8 gave

        mu       |lambda_min|      mu       |lambda_min|
        17.578     1.30573e-08     45.666     6.23390e-17
        23.016     2.74296e-11     62.972     4.52308e-18
        27.718     2.49272e-12     73.886     4.78513e-16   <-- 100x larger
        35.261     9.34128e-15

    and that was written up here as "there is no height-decay law", with three
    failed model fits and a refusal to try a fourth. The values are rigorous --
    stable to forty digits across dps 52, 150 and 400, radius reaching exactly
    zero -- and every height is a gap midpoint, so neither arithmetic nor the
    on-ordinate collision explains it.

    IT IS THE TRUNCATION. Repeating the top three heights:

        setting        mu=45.666    mu=62.972    mu=73.886    upturn
        N=20, t=0.8    6.2342e-17   4.5229e-18   4.7854e-16    yes
        N=20, t=0.9    3.5806e-14   1.6993e-15   4.7763e-14    yes
        N=40, t=0.8    3.1425e-25   5.0520e-32   3.6739e-34    NO

    It survives changing `t` and does not survive changing `N`. And
    `gamma_20 = 77.145`, so at N = 20 the height 73.886 sits **3.26 below the top
    of the retained set** while at N = 40 it is 49 below it. The fake zero was
    near the edge of the truncation, not near anything arithmetic.

    SO A FOURTH CONTAMINATION RULE FOR THIS AXIS: the height must be interior to
    the retained ordinates, not merely in a gap between them. Gap midpoints are
    not enough on their own. The `-0.657` fit above runs to mu = 35.261 against
    `gamma_20 = 77.145`, so it is comfortably interior and stands.

    THE MODEL SEARCH WAS THE WRONG RESPONSE, and that is the part worth keeping.
    Three forms were fitted to the anomaly and a fourth was refused on the
    grounds that three failures on seven points is fishing -- which was correct
    reasoning applied to the wrong question. The anomaly did not need a model. It
    needed the series recomputed at a different truncation, which took one run
    and settled it. Reaching for a functional form to describe an outlier is a
    way of accepting it.

    TWO WAYS THIS MEASUREMENT CAN GO WRONG, BOTH SEEN. A companion scan put the
    rate at 0.52 using a height of 30.4 -- which is 0.025 from `gamma_4 =
    30.425`, so the fake zero sat essentially on top of a real ordinate rather
    than in a gap. And the headroom above the float64 floor collapses from 3e9 to
    1.1e2 across these four points, so a fifth would not be measurable in double
    precision at all. Take the heights from gap midpoints, and check the floor at
    the far end before fitting anything through it.

    That also gives the retraction its positive form. The gap shrinks with
    height, so by the `2 |log10 gap|` budget above the DIGITS needed to certify a
    detection grow roughly linearly in `mu` -- detection stays possible at any
    height and gets linearly dearer to prove, which is where the ceiling actually
    lives.

    A TIGHTER BRACKET WAS ATTEMPTED AND FAILED ITS OWN PRE-REGISTERED TEST. The
    sign of the second ratio predicted the N = 80 behaviour correctly at both
    0.6 and 0.7, so it was swept across the interior at N <= 40 only:

        t      10->20    20->40
       0.60     0.7433    0.5958   falling
       0.62     0.9211    0.8953   falling
       0.64     0.9423    0.9650   RISING
       0.70     0.6859    0.8499   rising

    which would put `t_c` in (0.62, 0.64) -- fifty-eight times tighter. Before
    running the check, three conditions were fixed in writing: a monotone k-trend
    at BOTH t, the peak at 0.64 washing rather than sharpening, and both cells
    resolving. At N = 80:

        t = 0.62:   0.9211,  0.8953,  0.9064    NOT MONOTONE -- down, then up
        t = 0.64:   0.9423,  0.9650,  0.9706    monotone rising

    The first condition fails. The two-rung reading at 0.62 said "falling"
    because it had only two rungs; the third turns back up. So the refinement is
    NOT taken, and the interior does not resolve at N <= 80.

    THAT FAILURE IS PREDICTED, WHICH IS WHY IT IS RECORDED RATHER THAN RETRIED.
    The discriminant `r2 - r1` is 0.15-0.16 at 0.6 and 0.7 and only 0.02-0.03 at
    the attempted flip -- six times weaker exactly where it is being relied on.
    And the sinc backbone of this kernel is the time-band-limiting operator whose
    transition width grows as `log N` (Slepian; Karnik-Romberg-Davenport
    arXiv:2006.00427), so doubling the truncation sharpens the reading by
    `log 80 / log 40 = 1.19` -- NINETEEN PERCENT. Critical slowing-down with a
    scaling law behind it, not an effort problem. No affordable N closes the
    interior, and the route to a tighter number is a data collapse across
    `(t, N)` against the logistic form rather than more truncation at fixed `t`.

    THAT IS AN INFERENCE ABOUT `t_c` AND A PROVED STATEMENT ABOUT THE SEQUENCES.
    Suzuki Thm 1.3 makes the eigenvalue continuous in `t`, so no finite
    truncation settles which side of `t_c` a given `t` is on; what is settled is
    that these two sequences behave oppositely over N = 10..80. It is the same
    inference Bombieri drew from his figures, with certified arithmetic under it
    and three doublings instead of an eyeball.

    IT STILL DOES NOT PROVE WHICH SIDE OF `t_c` A GIVEN `t` IS ON. Suzuki Thm 1.3
    makes the eigenvalue continuous in `t`, so no finite truncation settles a
    statement about the limit. What this supports is "the sequences at t >= 0.7
    are converging and the one at t = 0.5 is not, over the range computed" --
    proved of those sequences, inferred about `t_c`. It is the inference Bombieri
    drew from his figures, with three doublings and certified arithmetic under it
    instead of an eyeball, and it is still an inference.

    THREE LAWS, THREE SIGNATURES, so "rising or falling" is a reading with named
    alternatives rather than a heuristic:

        lambda_N -> L > 0     r_k -> 1                  (non-zero limit)
        lambda_N ~ N^-p       r_k = 2^-p, CONSTANT      (power law, to zero)
        lambda_N ~ c^N        r_k squares each step     (geometric, to zero)

    AND THE DECAYING SIDE IS NOT WHAT THE SOURCE GUESSED. §13 calls the
    below-`t_c` decay "quite fast, suggesting an exponential rate", from an
    eyeball on N <= 160 in float precision. Certified at t = 0.5,
    `r1 = 5.0364e-05` and `r2 = 4.4201e-08` against `r1^2 = 2.5365e-09` -- so
    `r2` is SEVENTEEN TIMES larger than geometric would give. Faster than a power
    law, slower than geometric. Fitting `lambda ~ exp(-a N^b)` on
    `-ln|lambda| = 15.78, 24.28, 34.18, 51.11` at N = 5, 10, 20, 40 gives

        b = 0.558 over four points, 0.537 over the last three,
        and 0.622 / 0.493 / 0.581 pairwise -- NON-MONOTONE

    A three-parameter fit `-ln|lambda| = a N^b + c` over the same four points
    gives `b = 0.555, a = 6.58, c = -0.018` with residuals spread rather than
    concentrated -- so the data want NO prefactor, which is why the log-log fit
    and the three-parameter fit agree. An earlier estimate of 0.775 came from the
    ratio of two first differences over three points: algebraically a
    three-parameter fit to three points, which fits exactly and therefore
    constrains nothing. An estimator that leaves no residual has told you nothing.

    So `b ~ 0.55, unresolved at the 0.1 level`, and NOT a rate law -- four
    points, one of them pre-asymptotic, and pairwise estimates that do not even
    drift consistently.

    WHAT THIS CONTRADICTS IS HIS RATE; WHAT IT CONFIRMS IS HIS DICHOTOMY, and
    those are different claims about the same paragraph. It also contradicts it
    in the direction that matters: `b ~ 0.55` is a SLOWER collapse than the
    exponential `b = 1` he guessed, so more zeros are needed to see it, not
    fewer. The bracket rests only on the dichotomy and holds under any of the
    three laws above.

    IT REFUSES BELOW THE FLOAT64 FLOOR, and the first version did not. At
    t = 0.5 the N = 40 eigenvalue is about -1.6e-17 against a floor near 1e-14 --
    noise -- so the 20 -> 40 ratio it produced was a quotient of a real number by
    a rounding artifact, and it read as 1.1e-02 rather than the certified
    4.4e-08. A ratio whose numerator is noise is worse than no ratio, because it
    lands in the plausible range. `None` marks those rungs; the certified path
    supplies them if they are wanted.
    """
    resolved = []
    for count in truncations:
        values, _ = spectrum(with_quadruple(ordinates, count, displacement=displacement), t)
        floor = len(values) * float(np.finfo(float).eps) * float(np.abs(values).max())
        smallest = float(values[0])
        resolved.append(smallest if abs(smallest) > floor else None)
    return [
        (b / a if (a and b) else None)
        for a, b in zip(resolved, resolved[1:], strict=False)
    ]


def certified_ordinates(count: int, flint):
    """Arb's own zero isolation, and the on-line check that comes with it.

    `acb.zeta_zeros` certifies `Re = 1/2` as part of isolating each zero, so
    on-line-ness and the ordinate arrive together and there is no separate
    verification step. Radius is about 1e-28 at `gamma ~ 827` against float64's
    1e-13, and 512 of them cost under a second.

    THE ORDINATES ARE THE UNCERTIFIED INPUT, NOT THE ARITHMETIC. Certifying the
    matrix while feeding it float64 ordinates gives tight balls around the wrong
    number, which is the failure this project has already shipped once.
    """
    zeros = flint.acb.zeta_zeros(1, count)
    for z in zeros:
        if z.real != flint.arb("0.5"):
            raise ValueError(f"zeta_zero({count}) did not certify Re = 1/2: {z.real}")
    return [flint.acb(z.imag) for z in zeros]


def certified_kernel(
    count: int,
    t: float,
    flint,
    quadruples: int = 1,
    height: float = FAKE_HEIGHT,
    displacement: float = FAKE_DISPLACEMENT,
):
    """`H(Gamma;t)` in ball arithmetic.

    `sinc`, NOT `sin(u)/u`: on a ball straddling zero the division returns nan
    and one nan poisons everything downstream. And NOT `sinc_pi`, which is
    `sin(pi u)/(pi u)` -- numpy's convention, one letter away, and it would build
    a clean finite entirely wrong matrix. The argument between `gamma_0` and its
    conjugate is `-2 i eta t`, which goes to zero along the DISPLACEMENT axis, so
    this is not a diagonal special case but the whole small-`eta` regime.

    Quadruples are placed at DISTINCT heights. Theorem 8 counts distinct
    conjugate pairs, so stacking them at one height would make a passing test
    that asserts nothing.
    """
    gammas = certified_ordinates(count, flint)
    gammas = gammas + [-g for g in gammas]
    for index in range(quadruples):
        fake = flint.acb(flint.arb(str(height)) + index, -flint.arb(str(displacement)))
        gammas += [fake, fake.conjugate(), -fake, -(fake.conjugate())]

    scalar = flint.acb(t)
    quarter = flint.acb("0.25")
    denominators = [quarter + g * g for g in gammas]
    scale = 2 * scalar * scalar.sinh()
    size = len(gammas)
    matrix = flint.acb_mat(size, size)
    for j in range(size):
        for k in range(size):
            x, y = gammas[j], gammas[k]
            difference, total = scalar * (x - y), scalar * (x + y)
            remainder = (quarter + x * y) * difference.sinc() - (
                scalar.cosh() * difference.cos() - total.cos()
            ) / scale
            matrix[j, k] = 2 * scalar * remainder / (denominators[j] * denominators[k])
    return matrix


def negative_by_descartes(matrix, flint) -> tuple[int | None, int | None]:
    """Theorem 8's count without an eigensolver, and provably.

    Descartes' rule bounds the positive roots by the sign variations, with the
    deficiency an even number -- but FOR A POLYNOMIAL WHOSE ROOTS ARE ALL REAL
    the deficiency vanishes and the count is exact. Bombieri's Lemmas 8-9 prove
    every eigenvalue of `H` is real, so

        #negative eigenvalues = sign variations in the coefficients of p(-x)

    That is only sign tests, which ball arithmetic decides provably: a
    coefficient's enclosure either excludes zero, giving a sign, or straddles it,
    and then the count is genuinely unknown. Returns `(None, index)` in that
    case, naming the coefficient it could not sign -- a count of zero would be
    the exact confusion this module exists to avoid.

    IT SHARES NO CODE WITH `eig`. Different flint entry point, different
    algorithm, different failure mode -- so agreement between the two is evidence
    about the assembly rather than about one solver.

    ITS COST IS SET BY THE DETERMINANT, AND DIGITS ALWAYS BUY IT. `c_0` is the
    product of `size` eigenvalues, so the precision needed grows with
    `|log10 det|`: at size 44 the count is UNRESOLVED at 60 digits and PROVED at
    150 for t = 2.3, at 400 for t = 0.5, and a second implementation proves size
    84 at t = 0.5 with 1600. Unlike `eig`, more digits always work.

    THIS FILE FIRST RECORDED THE OPPOSITE, AND THE WAY IT WAS WRONG IS THE
    LESSON. A ladder at 60/150/400/1000 digits returned the same UNRESOLVED at
    every rung, which was written up as "flint loses the sign at that degree
    regardless". It was not: the ladder built the quadruple's `arb` literals ONCE
    at 60 digits and reused them, so a 1e-60-wide INPUT sat in four rows and no
    working precision can shrink an input.

    A RADIUS THAT DOES NOT MOVE AS PRECISION RISES IS INPUT-LIMITED, NOT
    ARITHMETIC-LIMITED, and that signature is worth more than this bug: it is the
    same tell as mpmath's `lerchphi` defect, whose error is stable across 300-500
    digits so that raising precision reports convergence on a wrong value. Before
    believing any "precision does not help" conclusion, check the radius moves.
    Which is why `certified_kernel` above constructs every literal from a string
    at call time rather than at import.

    SO THE TWO ROUTES ARE COMPLEMENTARY, NOT ORDERED. `eig`'s obstruction is
    separation -- below t ~ 0.75 the spectrum accumulates at zero and eigenvalues
    that are not separated stay unseparated at any precision. This route's
    obstruction is only the size of a sign, and digits buy signs. So THIS is the
    instrument for the deep end, and `eig` is the cheap one above t = 1.
    """
    signs = []
    for degree, coefficient in enumerate(list(matrix.charpoly())):
        # `P H P = conj(H)` for the conjugation permutation `P`, so the charpoly
        # is real. A coefficient whose imaginary part PROVABLY excludes zero
        # therefore indicts the matrix and never the precision, and must not be
        # returned as UNRESOLVED -- more digits would not help and reporting it
        # as a precision problem would send the next reader after bits.
        if hasattr(coefficient, "imag") and not coefficient.imag.contains(0):
            raise ValueError(
                f"charpoly coefficient {degree} has a provably non-zero imaginary "
                f"part {coefficient.imag}. `P H P = conj(H)` makes it real, so the "
                "assembly is wrong -- this is not a precision failure."
            )
        value = coefficient.real if hasattr(coefficient, "real") else coefficient
        # p(-x) flips the odd-degree coefficients.
        flipped = value if degree % 2 == 0 else -value
        if flipped.contains(0):
            return None, degree
        signs.append(1 if flipped > 0 else -1)
    return sum(1 for a, b in zip(signs, signs[1:], strict=False) if a != b), None


def certified_smallest(matrix, flint):
    """The least eigenvalue's enclosure, after the reality gate.

    `rump` rather than the default `vdhoeven_mourrain`, which flint documents as
    "faster and less accurate" -- two orders on its own example, free. And
    `nonstop=False`, so an eigenvalue it cannot isolate raises instead of being
    returned wide and unremarked.

    THE REALITY GATE IS FREE AND SEPARATES TWO FAILURES BY CONSTRUCTION. With a
    quadruple present `H` is complex symmetric but NOT Hermitian, and a
    non-Hermitian eigenproblem has no spectral theorem behind it -- yet Lemmas
    8-9 prove the spectrum is real anyway. An imaginary part EXCLUDING zero is
    Arb proving a non-real eigenvalue, contradicting a theorem, so the assembly
    is wrong. One that is merely wide is a precision statement. Refuse on the
    first only.
    """
    values = matrix.eig(algorithm="rump", nonstop=False)
    for value in values:
        if not value.imag.contains(0):
            raise ValueError(
                f"Arb proved a non-real eigenvalue {value}, contradicting Bombieri "
                "Lemmas 8-9. The assembly is wrong, not imprecise."
            )
    return min(values, key=lambda v: float(v.real.mid())).real


def run(
    t: float = 2.3,
    count: int = 20,
    ordinates: np.ndarray | None = None,
) -> ExperimentResult:
    if ordinates is None:
        from ..symbolic.riemann_siegel import first_zero_ordinates

        ordinates = first_zero_ordinates(64)

    margin = assert_lemma_10(ordinates, count, t)
    negatives, unresolved, floor = negative_count(with_quadruple(ordinates, count), t)
    kept, trend = retention(ordinates, t)
    ratios = [r for r in decay_ratios(ordinates, t) if r is not None]

    values, drift = spectrum(with_quadruple(ordinates, count), t)
    metrics = {
        "t": t,
        "prime_cutoff": float(np.exp(2 * t)),
        "count": float(count),
        "on_line_margin": margin,
        "negative_eigenvalues": float(negatives),
        "inside_floor": float(unresolved),
        "floor": floor,
        "imaginary_drift": drift,
        "smallest": float(values[0]),
        "retained_at_largest_truncation": kept,
        "resolved_decay_ratios": float(len(ratios)),
    }
    if ratios:
        metrics["decay_ratio_final"] = ratios[-1]
    if len(ratios) >= 2:
        # Rising toward 1 is convergence to a non-zero limit; falling is not.
        # THE LAST TWO, NOT FIRST-TO-LAST. Convergence is a property of the tail,
        # and the N = 5 rung is pre-asymptotic: at t = 0.9 the sequence is
        # 0.9748, 0.9529, 0.9654, so first-to-last reads "falling" while the tail
        # rises. Comparing across a pre-asymptotic rung inverts the reading.
        metrics["decay_ratios_rising"] = float(ratios[-1] > ratios[-2])

    verdict = (
        "as Theorem 8 requires"
        if negatives == 2
        else (
            "UNRESOLVED -- the missing negatives are inside the float64 floor, "
            "which is precision running out, not a counterexample"
            if negatives + unresolved >= 2
            else "DISAGREES with Theorem 8, which is unconditional: the matrix is wrong"
        )
    )
    observations = [
        f"One fake quadruple at rho_0 = 0.5 + {FAKE_DISPLACEMENT} + {FAKE_HEIGHT}i "
        f"gives {negatives} negative eigenvalues beyond the floor -- {verdict}.",
        f"Lemma 10 holds: with the quadruple removed the smallest eigenvalue is "
        f"{margin:.6e}, non-negative as it must be for real distinct ordinates. "
        "This is the gate that would catch a sign error inventing detections out "
        "of an on-line-only zero set, and it is the reason the number above can "
        "be read as detection rather than as noise.",
        f"Across truncations {TRUNCATIONS} the negative eigenvalue goes "
        + " -> ".join(f"{v:.3e}" for v in trend)
        + f", retaining {kept:.5f} of its first value. Below t_c that fraction "
        "goes to zero and above it settles, but a retention crossing is NOT a "
        "t_c estimator: Suzuki Thm 1.3 makes the eigenvalue continuous in t, so "
        "the corner being looked for does not exist at any truncation.",
        (
            "The ratios of successive truncations are "
            + ", ".join(f"{r:.4g}" for r in ratios)
            + (
                " -- rising toward 1, so this sequence is converging to a "
                "non-zero limit."
                if len(ratios) >= 2 and ratios[-1] > ratios[-2]
                else " -- not rising, so this sequence is not converging to a "
                "non-zero limit over the range computed."
            )
            + " That reading needs no threshold drawn across anything, unlike "
            "the retained fraction above, and it is still an inference about "
            "t_c rather than a proof: Suzuki Thm 1.3 makes the eigenvalue "
            "continuous in t, so no finite truncation settles the limit."
        )
        if ratios
        else "No decay ratio resolved above the float64 floor at this t.",
        "NO PRIMES ENTER. H is indexed by the ordinates and built from "
        "trigonometric functions of t(x +/- y); the arithmetic is entirely in "
        "which ordinates are supplied. So this control shares no machinery with "
        "the rest of the project, which is what makes agreement informative.",
    ]
    return ExperimentResult(
        name="bombieri-finite-model",
        parameters={"t": t, "count": count},
        metrics=metrics,
        observations=observations,
    )
