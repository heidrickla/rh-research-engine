"""How far off the critical line would a zero have to be for the form to notice?

`weil_positivity` finds no violation. That is unreadable on its own: the test
function is CHOSEN, so a null result is a statement about where one looked, and
the size of that statement is exactly the smallest displacement the chosen basis
would have caught. This measures it.

THE PERTURBATION IS COUNT-CONSERVING, which is the only kind zeta can have. The
functional equation sends an off-line zero to a quadruple, so a violation cannot
be made by nudging one zero: two conjugate pairs at `gamma_i, gamma_{i+1}` must
collide at `mu` and leave the line as `1/2 +/- eta + i mu`. The four
`gamma_rho = +/- mu +/- i eta` contribute

    2 exp(-s^2 (mu^2 - eta^2)) [ e^(-eta d) cos(mu(d - 2 s^2 eta))
                               + e^(+eta d) cos(mu(d + 2 s^2 eta)) ]

which is real, even in `d`, even in `eta` (swapping the sign of `eta` exchanges
the two zeros), and at `eta = 0` equals twice the on-line pair -- a double zero,
as it must. `eta*` is the smallest `eta` at which the form goes negative.

BUILT ON THE ORDINATES, NOT THE PRIMES. The two sides were shown equal to 5e-15
in `weil_positivity`, and the question here is a property of the functional, not
of the arithmetic that evaluates it. That buys a vectorised sum over a thousand
ordinates in place of a refining quadrature per entry, which is what makes a
sweep over pairs and sizes possible at all.

TWO CONTROLS, because "no violation" and "would not have seen one" print the
same. Displacing an ordinate ALONG the line by the same `eta` must never make
the form negative -- `h >= 0` for real gammas, whatever they are -- and the
collision at `eta = 0` must not either, since a double zero is still on the
line. If either fires, the test is detecting change rather than off-line-ness.

AND A CUT, because the bisection reads a sign. Once the unperturbed margin falls
to the float64 noise floor every perturbation reads negative and `eta*` collapses
towards zero for a reason that has nothing to do with zeta. Rows below the cut
are reported as cut, never as sensitive.

`size` IS A SUPPORT, AND THE THRESHOLD IS A FUNCTION OF IT. The basis is Dirac
masses at `u = log m`, m = 1..size, so the test function has support of length
`log(size)`, `G = F * Fbar` lives on `[-log size, log size]`, and the primes
that enter are `n <= size`. Matching the symmetric interval `[-t, t]` used in
this literature gives

    t = log(size) / 2,     prime cutoff c = size

which reconciles four notations for one number: `c = lambda^2` (Connes-Consani-
Moscovici), `a = log lambda` (Suzuki), `t` (Bombieri). At the default size 20
this is t = 1.498.

SO A THRESHOLD REPORTED WITHOUT ITS SUPPORT IS NOT A PROPERTY OF THE METHOD.
Measured, at sigma = 0.03:

    size      t      eta*(17.6)  eta*(48.9)   unperturbed margin
      10   1.151      0.30128      blind          7.01e-02
      14   1.320      0.17712      blind          6.38e-03
      20   1.498      0.04337     0.37288         3.00e-04
      28   1.666      0.00897     0.14637         5.48e-06
      40   1.844      0.00035     0.05746         1.56e-08
      56   2.013      0.00001     0.00441         2.10e-11
      80   2.191       ~0          ~0            -4.90e-15  UNRESOLVED

Four orders of magnitude between size 20 and size 56, every row of it far above
the noise floor. An earlier record quoted the size-20 value as the smallest
displacement the method would detect. It is the smallest displacement THIS BASIS
detects, and this basis was not chosen for the question.

AND SIZE 20 IS THE WORST AVAILABLE CHOICE, because it sits at the onset. Holding
eta = 0.05 and varying size, the form is positive through t = 1.320 and negative
from t = 1.498 onward, growing monotonically to -0.293 at t = 2.191 -- eleven
orders above the floor, so the detection is real. But a threshold measured AT the
onset reports how close the basis is to critical, not how sensitive it is.

THE HEIGHT CEILING IS RETRACTED. This module previously carried, and a research
note repeated, that `eta* ~ exp(sigma^2 mu^2 / 2)` puts a hard ceiling near
height 40 that no precision buys past. That was measured at fixed size and is
false as a statement about the method: `eta*` falls with size at every height.
The ceiling is real but sits elsewhere -- in certifying the sign of the
UNPERTURBED form, which goes unresolved in float64 at size 80 (margin -4.90e-15
against a floor of 1.77e-12). Detection gets cheaper and certification gets
dearer together, and it is the second that stops first.

THE UNPERTURBED MARGIN IS ONE NEAR-RADICAL DIRECTION, AND IS NOT THE QUANTITY
THIS LITERATURE CALLS `eps_lambda`. Measured: the ground eigenvalue is isolated,
with the next one 95, 275, 579 and 1416 times larger at sizes 14, 20, 28 and 40.
And the ground vector is near-radical in the exact sense -- its Dirichlet
polynomial has weighted mass 1.5e-4 at size 20 and 7.8e-9 at size 40 on the
ordinates the form samples, against about 1.0 off them. It is roughly the
constant function everywhere except where it is being looked at.

Connes-Consani-Moscovici QUOTIENT that direction out before recovering zeros
(arXiv:2511.22755 Thm 5.10(i), on `E_N / C xi`), because the Weil form has a
genuine radical. So their ground value is this module's SECOND eigenvalue, and
the two differ by the gap above -- two to three orders of magnitude, widening
with size. Any comparison of the margin below against a published `eps_lambda`
has to say which of the two it is, and this one is the un-quotiented one.

AND THE GAP NARROWS SMOOTHLY, WHICH A COARSE GRID MADE LOOK LIKE AN EVENT. The
sizes above step 14, 20, 28, 40, 56, and on that grid the gap appears to collapse
at 56 -- from 1416 to 83 in one step, which was written up here as the near-null
space ceasing to be one-dimensional at a particular size. Filling in the grid:

    size      40      44      48      52      56      60      64
    e[1]/e[0]  1416     901     339     131      83      62      25

Monotone and continuous, already falling at 40. Nothing happens at 56. The
near-null space thickens gradually as the basis grows and there is no size at
which it becomes multi-dimensional -- so `quotient one direction` degrades
smoothly rather than failing at a threshold, and no size can be quoted as the
place it stops working.

`e[1]` itself falls by a clean factor of about ten every four sizes (2.2e-5,
2.5e-6, 1.7e-7, 1.8e-8, 1.7e-9, 1.6e-10, 1.5e-11) and reaches 12 times the
float64 floor at size 64, so the SECOND eigenvalue runs out of precision there
much as the first does at 80.

A FINITE COMPUTATION HERE IS ROUTINELY SILENT ABOUT ITS OWN LIMIT. Three
published instances, in one literature, and in each the pre-limit artifact is
indistinguishable from the signal:

  * Bombieri (Rend. Lincei 11, 2000, §13): below a critical support the negative
    eigenvalue exists at EVERY finite truncation and decays exponentially to
    zero. `lambda_min < 0` and "an off-line zero is detected" come apart there.
  * Connes-van Suijlekom (arXiv:2511.23257, App. A): an operator whose
    truncations all have simple extreme eigenvalues, and whose limit does not.
  * Silva (Zenodo 20650146): "deep-spectrum values computed at a single T carry
    no internal evidence of their own validity."

So the size trend is part of the measurement, not a robustness check to add
afterwards. The table above is the measurement; a single row of it is not.

TWO ROUTES TO THE THRESHOLD, because one route cannot be checked. `threshold`
bisects on the sign. `secular_threshold` solves the second-order problem in
closed form. They share the ordinate data and nothing else, and they agree to
1.000 wherever `eta*` is small, drifting to 0.95-0.97 only where `eta*` reaches
0.3 and above -- which is the right sign and size for a second-order theory
meeting a large displacement, and is evidence rather than a discrepancy.

ORDINATES ARE MATCHED BY INDEX, NEVER BY VALUE. Hard-coded literals for
`gamma_1` and `gamma_2` differed from the stored values in the last two bits, so
the removal matched nothing and the perturbation added a quadruple WITHOUT
deleting the pairs it replaces -- silently, because a drop that matches nothing
builds the same matrix as no drop at all. It changed every threshold in a
scaling table by up to a factor of three and was invisible until the numbers
disagreed with an earlier run.
"""

from __future__ import annotations

import numpy as np

from ..core.models import ExperimentResult

#: `Re(rho) = 1/2 + eta`, and zeta has no zeros with `Re >= 1`, so a threshold at
#: or above this is not a displacement zeta could make: the basis detects nothing.
STRIP = 0.5
#: Ordinates past `WEIGHT_CUTOFF / sigma` carry less than `exp(-3600)` and are
#: dropped; the sum is over everything below it.
WEIGHT_CUTOFF = 60.0
#: The bisection is meaningless once the margin approaches the noise floor.
MARGIN_FLOOR_RATIO = 1e4
#: Pairs to attempt. The comment this replaces called 16 a SATURATION -- the
#: point past which "every higher pair's threshold leaves the critical strip".
#: That was the bisection's censoring read as the data's limit. The bisection
#: brackets inside `[0, STRIP]` and can say nothing above it, so a pair needing
#: eta = 1.026 comes back indistinguishable from one needing 0.6. The closed form
#: reports both, and at size 10 the pair at height 48.9 -- `blind` under the
#: bisection -- has eta* = 1.026, a displacement zeta cannot make. Nothing
#: saturates at 16; the count of pairs the BISECTION can speak about does.
PAIR_LIMIT = 16
#: Permutations of the gap column, for the control that carries the claim.
SHUFFLE_ROUNDS = 2000


def _log_ratios(size: int) -> np.ndarray:
    m = np.arange(1, size + 1, dtype=float)
    return np.log(np.outer(m, 1.0 / m))


def quadruple(mu: float, eta: float, d: np.ndarray, sigma: float) -> np.ndarray:
    """The four `+/- mu +/- i eta` zeros, as a contribution to the kernel."""
    return (
        2.0
        * np.exp(-(sigma**2) * (mu**2 - eta**2))
        * (
            np.exp(-eta * d) * np.cos(mu * (d - 2 * sigma**2 * eta))
            + np.exp(eta * d) * np.cos(mu * (d + 2 * sigma**2 * eta))
        )
    )


def perturbed_form(
    ordinates: np.ndarray,
    size: int,
    sigma: float,
    pair: tuple[int, int] | None = None,
    eta: float = 0.0,
    along: float | None = None,
) -> np.ndarray:
    """The Weil form, optionally with one pair collided and displaced by `eta`.

    `along` is the control: it moves the ordinate at index `pair[0]` to that
    value, keeping it on the critical line, and applies no quadruple.
    """
    d = _log_ratios(size)
    keep = ordinates[ordinates < WEIGHT_CUTOFF / sigma]
    mu = None
    if pair is not None:
        mask = np.ones(len(keep), dtype=bool)
        mask[list(pair)] = False
        removed = len(keep) - int(mask.sum())
        if removed != len(pair):
            raise ValueError(f"removal took {removed} ordinates, not {len(pair)}")
        mu = float(keep[list(pair)].mean())
        keep = keep[mask]
    if along is not None:
        keep = np.concatenate([keep, [along]])
        mu = None
    matrix = 2.0 * (
        np.exp(-(sigma**2) * keep**2)[:, None, None] * np.cos(keep[:, None, None] * d[None])
    ).sum(0)
    if mu is not None:
        matrix = matrix + quadruple(mu, eta, d, sigma)
    return matrix


def _smallest(matrix: np.ndarray, size: int) -> tuple[float, float]:
    from .weil_positivity import noise_floor

    eigenvalues = np.linalg.eigvalsh(matrix)
    return float(eigenvalues[0]), noise_floor(eigenvalues, size)


def is_negative(matrix: np.ndarray, size: int) -> bool:
    """Negative BEYOND the floor. Anything smaller is a fact about float64."""
    smallest, floor = _smallest(matrix, size)
    return smallest < -floor


def threshold(
    ordinates: np.ndarray, size: int, sigma: float, pair: tuple[int, int], steps: int = 60
) -> float | None:
    """`eta*`, or None when no displacement inside the critical strip is caught."""
    if not is_negative(perturbed_form(ordinates, size, sigma, pair, STRIP), size):
        return None
    low, high = 0.0, STRIP
    for _ in range(steps):
        middle = 0.5 * (low + high)
        if is_negative(perturbed_form(ordinates, size, sigma, pair, middle), size):
            high = middle
        else:
            low = middle
    return high


def secular_threshold(
    ordinates: np.ndarray, size: int, sigma: float, pair: tuple[int, int]
) -> float | None:
    """`eta*` from the second-order problem, in closed form rather than bisected.

    Removing the two on-line pairs and adding the quadruple changes the form by
    `-2 eta^2 F''(mu) + O(eta^4)` with `F = W Q_c` -- second order, because the
    quadruple at `eta = 0` is a double zero and the constant terms cancel. So the
    governing quantity is the CURVATURE `F''`, not the weight `F`.

    Writing `u_m = log m`, `v_m = m^(i mu)`, `U = diag(u)`, and using
    `exp(i mu d_jk) = (v v*)_jk` with `d_jk = u_j - u_k`,

        F'' = W''(v v*) + 2i W' [(Uv)v* - v(Uv)*]
              - W [(U^2 v)v* - 2 (Uv)(Uv)* + v(U^2 v)*]

    which has rank at most four and IS INDEFINITE. Every term above carries a
    bare `v` on one side except one, so on the hyperplane `Phi_c(mu) = <c,v> = 0`
    all of them vanish and

        c* F'' c  =  2 W(mu) |<c, Uv>|^2  =  2 W(mu) |Phi'_c(mu)|^2  >=  0

    rank ONE and positive semidefinite. The threshold is then the root of a
    monotone scalar function, which a bracket proves, rather than a minimum a
    solver claims -- and an earlier unconstrained attempt on the indefinite `F''`
    returned thresholds LARGER than a fixed-vector bisection, impossible for a
    minimum over every vector, which is what caught it.

    `c` IS REAL AND `v` IS NOT, so `<c,v> = 0` is TWO real constraints and
    `|Phi'|^2` is a rank-TWO real form -- which is why the null space is taken
    from `[Re v; Im v]` rather than from `v`.

    THE CONSTRAINT BINDS WEAKLY, AND LESS AS THE BASIS GROWS. Removing it
    entirely -- `null = I`, so the minimisation runs over every vector -- shifts
    the threshold by

        size          14        20        28        40        56
        rel. shift   7.3e-03   2.7e-04   1.5e-05   3.2e-08   1.1e-05

    Two predictions written here before this was measured were both wrong. The
    first said halving the codimension would return a visibly smaller threshold.
    The second, written after measuring only the half-codimension case, said
    dropping it changed nothing a tolerance could see -- and a test asserting
    agreement to 1e-6 duly failed in the full suite. Both were mechanisms
    reasoned about rather than numbers looked at.

    It binds weakly because `K` is assembled with the pair at `mu` already
    removed, so its smallest directions are close to orthogonal to `v` before
    anything is imposed, and closer as the basis grows. The derivation needs the
    hyperplane regardless: off it the response is the indefinite four-term `F''`
    above and the rank-one reduction is unavailable.

    The size-56 row breaking the trend is the near-degenerate regime rather than
    the constraint returning -- `eta*` is 9e-6 there against a ground eigenvalue
    of 2.1e-11 and a floor of 9.9e-13, so four figures is all that row carries.
    Left in rather than smoothed out.

    None when the constrained form has no positive direction: that basis cannot
    reach the height at any displacement, which is a different statement from a
    large threshold and must not be returned as one.
    """
    keep = ordinates[ordinates < WEIGHT_CUTOFF / sigma]
    mu = float(keep[list(pair)].mean())
    matrix = perturbed_form(ordinates, size, sigma, pair, 0.0)

    u = np.log(np.arange(1, size + 1, dtype=float))
    v = np.exp(1j * mu * u)
    weight = float(np.exp(-(sigma**2) * mu**2))

    _, _, right = np.linalg.svd(np.vstack([v.real, v.imag]))
    null = right[2:].T

    values, vectors = np.linalg.eigh(null.T @ matrix @ null)
    if values.min() <= 0:
        return None
    root_inverse = vectors @ np.diag(values**-0.5) @ vectors.T

    response = np.vstack([null.T @ (u * v).real, null.T @ (u * v).imag]) @ root_inverse
    largest = float(np.linalg.eigvalsh(response @ response.T).max())
    if largest <= 0:
        return None
    return float(np.sqrt(1.0 / (4.0 * weight * largest)))


def _fit_with_error(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares, and the standard error the earlier version did not report.

    Three coefficients were recorded bare. That is the same defect this session
    fixed in three other experiments, committed in this one.
    """
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    freedom = max(len(target) - design.shape[1], 1)
    variance = (residual @ residual) / freedom
    errors = np.sqrt(variance * np.diag(np.linalg.inv(design.T @ design)))
    return coefficients, errors


def _shuffle_control(
    design: np.ndarray, target: np.ndarray, predicted: float, rounds: int
) -> float:
    """How often a PERMUTED gap lands as close to the prediction as the real one.

    Adding any second regressor to a small fit moves the first coefficient, so
    "it moved towards the prediction" is not evidence on its own. Permuting the
    gap column keeps both marginals and destroys only its pairing with height,
    which is the thing being claimed.
    """
    real, _ = _fit_with_error(design, target)
    target_distance = abs(real[0] - predicted)
    generator = np.random.default_rng(20260828)
    permuted = design.copy()
    hits = 0
    for _ in range(rounds):
        permuted[:, 1] = generator.permutation(design[:, 1])
        coefficients, *_ = np.linalg.lstsq(permuted, target, rcond=None)
        if abs(coefficients[0] - predicted) <= target_distance:
            hits += 1
    return hits / rounds


def run(
    size: int = 20,
    sigma: float = 0.03,
    pairs: int = PAIR_LIMIT,
    ordinates: np.ndarray | None = None,
) -> ExperimentResult:
    if ordinates is None:
        from ..symbolic.riemann_siegel import first_zero_ordinates

        ordinates = first_zero_ordinates(3000)

    margin, floor = _smallest(perturbed_form(ordinates, size, sigma), size)
    ratio = margin / floor if floor else float("inf")
    if margin <= 0 or ratio < MARGIN_FLOOR_RATIO:
        return ExperimentResult(
            name="weil-sensitivity",
            parameters={"size": size, "sigma": sigma, "pairs": pairs},
            metrics={
                "size": size,
                "sigma": sigma,
                "margin": margin,
                "margin_over_floor": ratio,
                "measured_pairs": 0.0,
            },
            observations=[
                f"CUT, not measured: the unperturbed margin {margin:.4g} stands only "
                f"{ratio:.3g} times the float64 noise floor, and the bisection reads a "
                "sign. Below the cut every perturbation registers as negative and the "
                "threshold collapses towards zero for a reason that has nothing to do "
                "with zeta. This is a statement about precision, not sensitivity.",
            ],
        )

    heights, gaps, thresholds = [], [], []
    # Both routes on every pair, including the ones the bisection cannot reach:
    # `secular` is recorded there too, because "needs eta = 1.026, which is
    # outside the strip" and "the bisection had nothing to say" are different
    # findings that the old code returned as one.
    secular, agreements = [], []
    for index in range(pairs):
        found = threshold(ordinates, size, sigma, (index, index + 1))
        closed = secular_threshold(ordinates, size, sigma, (index, index + 1))
        if closed is not None:
            secular.append(closed)
        if found is None or found >= STRIP - 1e-6:
            continue
        if closed:
            agreements.append(abs(found - closed) / closed)
        heights.append(float(ordinates[[index, index + 1]].mean()))
        gaps.append(float(ordinates[index + 1] - ordinates[index]))
        thresholds.append(found)

    # Controls. Neither can legitimately go negative, so a single firing is a
    # bug in the test rather than a finding.
    # ONE index, not two: `(0, 1)` would delete a zero as well as move one, which
    # is a different control than the observation below claims to have run.
    along_line = [
        is_negative(
            perturbed_form(ordinates, size, sigma, (0,), along=float(ordinates[0]) + step),
            size,
        )
        for step in (0.05, 0.2, 0.5, 1.0, 2.0)
    ]
    collision = is_negative(perturbed_form(ordinates, size, sigma, (0, 1), 0.0), size)

    height = np.array(heights)
    gap = np.array(gaps)
    star = np.array(thresholds)
    metrics = {
        "size": size,
        "sigma": sigma,
        # The support the basis actually spans, and the prime cutoff it
        # corresponds to. Recorded because every threshold below is conditional
        # on them and a bare threshold has been mistaken for a property of the
        # method once already.
        "support": float(np.log(size) / 2.0),
        "prime_cutoff": float(size),
        "margin": margin,
        "margin_over_floor": ratio,
        "measured_pairs": float(len(star)),
        "controls_fired": float(sum(along_line) + collision),
        "secular_pairs": float(len(secular)),
    }
    if secular:
        metrics["secular_smallest_threshold"] = float(min(secular))
    if agreements:
        metrics["secular_agreement_worst"] = float(max(agreements))
    # Omitted rather than written as NaN, which serialises to JSON null and then
    # fails to load back -- a recorded run that cannot be read is not a record.
    if len(star):
        metrics["smallest_threshold"] = float(star.min())
        metrics["smallest_threshold_height"] = float(height[star.argmin()])
        metrics["largest_measured_height"] = float(height.max())

    observations = [
        "The number that makes the null result readable. A violation at height "
        f"{metrics.get('smallest_threshold_height', float('nan')):.2f} would have been "
        "caught by this basis once Re(rho) reached "
        f"{0.5 + metrics.get('smallest_threshold', float('nan')):.6f}; below that it "
        "would have passed unseen. Nothing here is evidence for RH.",
        "The perturbation conserves the zero count, because the functional "
        "equation gives no other option: an off-line zero comes as a quadruple, "
        "so two on-line pairs must collide and leave the line together.",
        f"CONDITIONAL ON THE SUPPORT, which is the whole content of the number "
        f"above. This basis spans t = log(size)/2 = {np.log(size) / 2.0:.3f}, "
        f"equivalently a prime cutoff of {size}. The threshold falls by four "
        "orders of magnitude between size 20 and size 56 at this sigma, every "
        "row of it far above the noise floor, so a threshold quoted without its "
        "support says nothing about the method. Sizes near 20 are the worst "
        "available: they sit at the onset, where what is measured is distance "
        "to critical rather than sensitivity.",
        f"Controls: {sum(along_line)} of 5 along-the-line displacements and "
        f"{int(collision)} of 1 on-line collisions produced a negative eigenvalue. "
        "Both must be zero -- h >= 0 holds for any real ordinates however they are "
        "arranged, so a firing control means the form is detecting change rather "
        "than off-line-ness.",
    ]

    if len(star) >= 6:
        # exp(s^2 mu^2 / 2) was DERIVED, then refuted at one variable, and it
        # returns once the gap is in the design. Every coefficient now carries a
        # standard error, and the claim that the gap matters rests on a shuffle
        # control rather than on the two point estimates, which differ by about
        # one sigma and whose intervals both cover the prediction.
        design = np.column_stack([height**2, np.log(gap), np.ones_like(height)])
        joint, joint_error = _fit_with_error(design, np.log(star))
        alone_design = np.column_stack([height**2, np.ones_like(height)])
        alone, alone_error = _fit_with_error(alone_design, np.log(star))
        predicted = sigma**2 / 2
        closeness = _shuffle_control(design, np.log(star), predicted, SHUFFLE_ROUNDS)

        metrics["height_coefficient"] = float(joint[0])
        metrics["height_coefficient_error"] = float(joint_error[0])
        metrics["height_coefficient_ungapped"] = float(alone[0])
        metrics["height_coefficient_ungapped_error"] = float(alone_error[0])
        metrics["gap_coefficient"] = float(joint[1])
        metrics["gap_coefficient_error"] = float(joint_error[1])
        metrics["predicted_height_coefficient"] = predicted
        metrics["shuffled_gap_p_value"] = closeness
        metrics["gap_separation_sigma"] = float(
            abs(joint[0] - alone[0]) / joint_error[0] if joint_error[0] else float("inf")
        )
        observations.append(
            f"With log(gap) in the design the height coefficient is {joint[0]:.6f} "
            f"+/- {joint_error[0]:.6f}, which is {joint[0] / predicted:.2f} times the "
            f"sigma^2/2 = {predicted:.6f} that the quadruple's own weight predicts. "
            f"Height alone gives {alone[0]:.6f} +/- {alone_error[0]:.6f}, a ratio of "
            f"{alone[0] / predicted:.2f}, over {len(star)} pairs."
        )
        observations.append(
            "THE TWO POINT ESTIMATES DO NOT ESTABLISH THE DIFFERENCE. They sit "
            f"{metrics['gap_separation_sigma']:.2f} sigma apart and the joint interval "
            "covers both. What establishes it is a shuffle control: permuting log(gap) "
            "against height keeps both marginals and destroys only their pairing, and "
            f"only {closeness:.3f} of {SHUFFLE_ROUNDS} permutations land as close to "
            "sigma^2/2 as the real gap does. That is the defensible claim. An earlier "
            "record asserted the correction from the point estimates alone, which they "
            "do not support."
        )
        observations.append(
            "THE FIT IS AT ONE SIZE AND DOES NOT TRANSFER. Both coefficients "
            f"above are measured at size {size}. The height law holds at every "
            "size tried, but its coefficient does not: fitting log(eta*) against "
            "size gives rates from -0.232 at height 17.6 to -0.123 at height "
            "48.9, so the height dependence and the size dependence do not "
            "separate. An earlier note read the size-20 coefficient as a "
            "property of the criterion and inferred a height ceiling from it; "
            "that inference is withdrawn in the module docstring."
        )
        observations.append(
            "A DESCRIPTION, NOT A MECHANISM, and the difference is recorded because "
            "the mechanism was tried and failed. eta* = sqrt(L0/c) from the curvature "
            "at eta = 0 predicts the bisected threshold to within 4-35% for the lowest "
            "pair, but measuring L0 and c across pairs gives a gap coefficient near "
            "zero where the direct fit gives about -0.8. The quadratic regime does not "
            "reach eta*, so the fitted form stands as a summary of measurements and "
            "nothing more."
        )

    return ExperimentResult(
        name="weil-sensitivity",
        parameters={"size": size, "sigma": sigma, "pairs": pairs},
        metrics=metrics,
        observations=observations,
    )
