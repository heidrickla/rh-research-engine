"""Weil's positivity criterion as a matrix eigenvalue problem.

RH is equivalent to `sum_rho h(gamma) >= 0` for every test function of the form
`h = |phi_hat|^2`. Taking `phi` to be Gaussians centred at `log m` for integers
`m = 1..M` gives

    h(r) = exp(-sigma^2 r^2) |sum_m c_m m^(i r)|^2   >=   0

so the functional is `c^T K c` for a matrix `K` built from PRIMES AND THE GAMMA
FACTOR ONLY. Under RH every `gamma` is real, `h >= 0` on the real line, and the
sum is non-negative for every `c` -- so `K` is positive semidefinite. A NEGATIVE
EIGENVALUE, at any `M` and any `sigma`, WOULD REFUTE RH.

WHY THIS ONE AND NOT THE OTHER TWO. `li_criterion` extends a sequence and
`baez_duarte` extends a sequence; both give a computer one direction to walk.
This gives it a SEARCH SPACE -- the basis and the width are ours to choose -- so
failing to find a violation is a statement about where we looked.

THE KERNEL IS EVEN IN `d = log(m_j/m_k)`, AND TWO CHOICES CONSPIRED TO BREAK
THAT. Zeros come in `+/- gamma` pairs, so the exact kernel is
`2 sum cos(gamma d)` and even by construction. The first build of this had:

  * the prime term as `2 G(log n)` instead of `G(log n) + G(-log n)`. The
    doubled form is only right for an EVEN `h`, and the elementary
    `h(r) = exp(-sigma^2 r^2 + i r d)` is even only at `d = 0` -- which is
    exactly where it was checked.
  * the assembly reading `K(|d|)` rather than `(K(d) + K(-d))/2`.

NEITHER IS A BUG ON ITS OWN, and that is the part worth keeping. Averaging the
doubled prime term over `+/- d` gives back `G(log n) + G(-log n)` exactly, so
the symmetrisation repairs it; and once the prime term is two-sided the kernel
really is even, so `K(|d|)` changes nothing -- measured, identical eigenvalues
at sizes 8, 12 and 16. Only together are they fatal: `|d|` removes the
averaging that would have cancelled the doubling.

Both accounts written before this one were wrong. The first called them two
independent bugs; the second, having measured that `K(|d|)` alone is harmless,
called it one. It is neither -- it is a pair of individually harmless choices
whose combination is wrong, which is why testing each in isolation cleared both.

The combination produced a negative eigenvalue -- what a refutation of RH looks
like -- and was caught by comparing the eigenvector against the zeros: the
matrix said `-5.8e-5` where 1.75M ordinates said `+2.5e-5`.

A NEGATIVE EIGENVALUE IS ONLY EVIDENCE WHILE THE ARITHMETIC CAN RESOLVE IT.
This form goes singular fast -- at `sigma = 0.2` its condition number reaches
`3e15` by size 12 -- and float64 reports `lambda_min = -3.3e-15` there against a
largest eigenvalue of `5.6e-3`. `weil_certified`, which computes the same matrix
in ball arithmetic from exact inputs, does not confirm it: the first version of
this experiment announced a refutation of the Riemann hypothesis at nothing more
exotic than `--size 12`.

THE FLOOR IS NOT THE EIGENSOLVER'S ALONE. It was first written as
`size * eps * lambda_max`, the backward error of `eigvalsh` and nothing about
the error already sitting in the matrix -- whose entries come from a
fifty-thousand-term prime sum and a quadrature. Measured against the certified
balls, an entry is wrong by up to `2e-15` absolute, and `lambda_min` moves by up
to `n` times that. The first floor was `1.5e-17` where the entry error alone
demands `2.4e-14`, so it let the spurious refutation through unchanged.

The two terms cross at `lambda_max = ENTRY_ERROR/eps`, near 23, so neither is
always the larger: at `sigma = 0.2` the entries dominate by a factor of a
thousand, and at `sigma = 0.03` and size 12, where `lambda_max` is 23.8, the
two are comparable.

That second half was measured at size 12 and is not known to survive size 24,
which has not been measured. Three mechanisms proposed for the size dependence
were each measured and refuted -- see `ENTRY_ERROR` -- so the sentence keeps
the range it was measured on rather than being generalised or replaced by a
fourth guess. What is known is that the entry error at sigma = 0.03 is small,
0.03x the constant at the worst ratio tried, so nothing suggests the order
reverses there.

An estimate, not a bound: `ENTRY_ERROR` is measured, and a measurement of an
error is not a proof of one. `weil_certified` is where the bound lives. This
path stays because it is a hundred times faster and answers the same question
everywhere the answer is not close.

Hence three verdicts and not two. `POSITIVE` and `REFUTED` are claims about
zeta; `UNRESOLVED` is a claim about the precision, and says nothing either way.
Collapsing it into either of the others is the same error as letting "not
tested" read as "refuted", which `patterns/ledger.py` exists to prevent.
"""

from __future__ import annotations

import numpy as np

from ..core.models import ExperimentResult

PRIME_LIMIT = 200_000
#: Nodes are doubled until two counts agree to this, relative.
#:
#: A PARAMETER, because the refinement dominates the cost -- up to seven
#: doublings from 600 nodes -- and the suite cannot afford production tolerance
#: on every entry. Loosening it for a test is honest as long as the test's own
#: assertions stay well above whatever it is loosened to.
QUADRATURE_TOLERANCE = 1e-12


def von_mangoldt(limit: int) -> tuple[np.ndarray, np.ndarray]:
    sieve = np.zeros(limit, dtype=np.float64)
    composite = np.zeros(limit, dtype=bool)
    for p in range(2, limit):
        if not composite[p]:
            composite[p * p :: p] = True
            power, log_p = p, float(np.log(p))
            while power < limit:
                sieve[power] = log_p
                power *= p
    n = np.nonzero(sieve)[0]
    return n.astype(np.float64), sieve[n]


def kernel(
    d: float,
    sigma: float,
    logs: np.ndarray,
    weight: np.ndarray,
    tolerance: float = QUADRATURE_TOLERANCE,
) -> float:
    """The explicit formula on `h(r) = exp(-sigma^2 r^2 + i r d)`.

    Every term below is even in `d` once written correctly, which is the check
    that the two sign errors in the module docstring both failed.
    """
    import mpmath as mp

    pole = 2.0 * np.exp(sigma**2 / 4.0) * np.cosh(d / 2.0)
    norm = 1.0 / (2.0 * sigma * np.sqrt(np.pi))
    g_zero = norm * np.exp(-(d**2) / (4.0 * sigma**2))

    span = 30.0 / sigma

    def integrate(count: int) -> float:
        nodes = list(np.linspace(-span, span, count + 1))
        return float(
            mp.quad(
                lambda r: mp.e ** (-(sigma**2) * r**2)
                * mp.cos(r * d)
                * mp.re(mp.digamma(mp.mpf(0.25) + 0.5j * r)),
                nodes,
            )
        ) / (2 * np.pi)

    # REFINED UNTIL IT AGREES WITH ITSELF. A fixed four-nodes-per-period rule
    # was exact at most `d` and 3.8e-5 wrong at `d = log(8/7)`, with nothing
    # separating the two -- a quadrature that happens to work rather than one
    # that is checked.
    period = 2 * np.pi / max(abs(d), 1e-9)
    count = int(min(600, max(12, 4 * span / period)))
    arch = integrate(count)
    for _ in range(7):
        finer = integrate(2 * count)
        if abs(finer - arch) <= tolerance * max(abs(finer), 1.0):
            arch = finer
            break
        arch, count = finer, 2 * count
    else:
        raise RuntimeError(f"archimedean quadrature did not converge at d={d:.6f}")

    # G(log n) + G(-log n), never 2 G(log n): see the module docstring.
    primes = float(
        np.sum(
            weight
            * norm
            * (
                np.exp(-((logs - d) ** 2) / (4.0 * sigma**2))
                + np.exp(-((logs + d) ** 2) / (4.0 * sigma**2))
            )
        )
    )
    return pole - g_zero * np.log(np.pi) + arch - primes


def form(
    size: int,
    sigma: float,
    prime_limit: int = PRIME_LIMIT,
    tolerance: float = QUADRATURE_TOLERANCE,
) -> np.ndarray:
    """The Weil quadratic form on the basis `m = 1..size`."""
    n, lam = von_mangoldt(prime_limit)
    logs, weight = np.log(n), lam / np.sqrt(n)
    ms = np.arange(1, size + 1, dtype=float)

    cache: dict[float, float] = {}
    matrix = np.zeros((size, size))
    for j, a in enumerate(ms):
        for k, b in enumerate(ms):
            d = float(np.log(a / b))
            for signed in (d, -d):
                key = round(signed, 12)
                if key not in cache:
                    cache[key] = kernel(signed, sigma, logs, weight, tolerance)
            matrix[j, k] = 0.5 * (cache[round(d, 12)] + cache[round(-d, 12)])
    return matrix


#: How wrong a single entry is, absolute. A SAMPLED MAXIMUM, not a bound.
#:
#: Measured against `weil_certified`'s balls, whose radii are near 1e-43 here,
#: so the comparison measures the float path and not the enclosure. The largest
#: value seen anywhere is 3.55e-15, over 19 ratios `d = log n` at sigma = 0.08
#: plus a sigma sweep at two ratios; 5e-15 covers all of it with margin. It was
#: 2e-15, taken over seven ratios at three widths, and that was exceeded.
#:
#: WHERE IT IS ACTUALLY USED IT IS SAFE BY AN ORDER. At sigma = 0.03, the width
#: of both recorded runs, the two worst ratios come in at 1.67e-16 and 6.66e-16
#: -- 0.03x and 0.13x of this constant. The exceedances live at sigma = 0.05 to
#: 0.08 with n >= 9, a region nothing on the record occupies.
#:
#: THREE SHAPES WERE PROPOSED AND ALL THREE WERE MEASURED AND REFUTED, which is
#: why this is a constant and not a function:
#:
#:   `log(size)`: sizes 4 and 6 give 8.88e-16 identically, and 9 and 12 give
#:   2.33e-15 identically, where log(size) wants 1.29x from 9 to 12. The worst
#:   ratio is never the largest available -- it is 4/1 and then 9/1.
#:
#:   prime-power-ness: the idea that `d = log(a/b)` landing on a `log n` with
#:   `Lambda(n) != 0` centres the Gaussian on a live term. `16 = 2^4` is a prime
#:   power and comes in at 0.15x, among the lowest; `12` is composite and
#:   reaches 0.83x; `23` is prime and sits at 0.39x. All three go the wrong way.
#:
#:   `1/sigma`: at ratio 9/1 the error runs 1.67e-16, 3.00e-15, 2.33e-15,
#:   8.33e-16, 2.02e-16, 5.68e-17 across sigma = 0.03 to 0.30 -- a peak in the
#:   middle, up 18x and down 53x, non-monotone in both directions.
#:
#: What survives is only that the error grows with n on average, with large
#: scatter and no monotonicity. A fitted surface over these points would be the
#: fourth guess in a row.
#:
#: THE UPSTREAM FIX DOES NOT WORK EITHER, and it was measured rather than
#: assumed. The float path forms `d = float(np.log(a/b))` while `kernel_ball`
#: takes the ratio as integers precisely because that is lossy, so forming `d`
#: in extended precision looks like the way to remove the phase error rather
#: than budget for it. Over every ratio with a, b <= 24 against 40-digit values,
#: the worst error is 2.69e-16 as computed and 2.17e-16 correctly rounded --
#: a factor of 1.2. `np.log(a/b)` is already nearly correctly rounded, so the
#: phase error is irreducible in float64 and the only way out of it is leaving
#: float64, which is what `weil_certified` is for.
#:
#: (With `delta d <= 2.7e-16` and `r` running to `30/sigma = 1000` the phase
#: error reaches 2.7e-13, yet entries are wrong by at most 3.5e-15: most of it
#: cancels in the integral. That is a reason not to expect a clean shape.)
#:
#: Measured by the rh-research-engine-da session.
ENTRY_ERROR = 5e-15


def noise_floor(eigenvalues: np.ndarray, size: int) -> float:
    """Below this, the sign of an eigenvalue is a fact about float64.

    Two contributions, and leaving out the second is what let a spurious
    refutation stand. A symmetric eigensolver returns the exact spectrum of a
    matrix within `O(n) * eps * norm` of the one it was given -- but the matrix
    it was given is already wrong by `ENTRY_ERROR` per entry, and Weyl's
    inequality moves `lambda_min` by up to `n` times that.
    """
    eigensolver = float(np.finfo(float).eps) * float(abs(eigenvalues[-1]))
    return size * (eigensolver + ENTRY_ERROR)


def classify(eigenvalues: np.ndarray, size: int) -> tuple[str, float]:
    """POSITIVE, REFUTED, or UNRESOLVED -- and the floor that separated them.

    Kept as a pure function of the spectrum so both branches can be forced from
    a test with a hand-built array, rather than by hunting for parameters that
    happen to be singular on the machine the suite runs on.
    """
    floor = noise_floor(eigenvalues, size)
    smallest = float(eigenvalues[0])
    if smallest < -floor:
        return "REFUTED", floor
    if smallest > floor:
        return "POSITIVE", floor
    return "UNRESOLVED", floor


def run(
    size: int = 12,
    sigma: float = 0.03,
    prime_limit: int = PRIME_LIMIT,
    tolerance: float = QUADRATURE_TOLERANCE,
) -> ExperimentResult:
    matrix = form(size, sigma, prime_limit, tolerance)
    eigenvalues = np.linalg.eigvalsh(matrix)
    verdict, floor = classify(eigenvalues, size)
    # Counted for the record, but NOT the verdict: at sigma = 0.2 and size 16
    # this is 5 while 60-digit arithmetic puts every eigenvalue above zero.
    negative = int(np.sum(eigenvalues < 0))

    return ExperimentResult(
        name="weil-positivity",
        parameters={"size": size, "sigma": sigma, "prime_limit": prime_limit},
        metrics={
            "size": size,
            "sigma": sigma,
            "smallest_eigenvalue": float(eigenvalues[0]),
            "largest_eigenvalue": float(eigenvalues[-1]),
            "negative_count": negative,
            "noise_floor": floor,
            "margin_over_floor": float(eigenvalues[0]) / floor if floor else float("inf"),
            "condition": float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] else float("inf"),
            "violated": float(1.0 if verdict == "REFUTED" else 0.0),
            "unresolved": float(1.0 if verdict == "UNRESOLVED" else 0.0),
        },
        observations=[
            f"Verdict {verdict}: smallest eigenvalue {eigenvalues[0]:.4g} against "
            f"a float64 noise floor of {floor:.3g}. UNRESOLVED is not POSITIVE -- "
            "it says the arithmetic ran out before zeta was asked anything. At "
            "sigma = 0.2 that happens by size 12, where float64 reports -3.3e-15 "
            "and weil-certified declines to confirm it.",
            "A falsification test that did not fire. RH implies this form is "
            "positive semidefinite for every basis size and every width; a "
            "negative eigenvalue would refute it. Positive here is consistent "
            "with RH and proves nothing.",
            "Built from primes and the Gamma factor only -- no zeros enter. The "
            "zeros are used solely to VALIDATE the kernel, which agrees with a "
            "direct sum over 1.75M ordinates to about 5e-15 per entry.",
            "Unlike Li's criterion and Baez-Duarte, this is a search rather "
            "than a sequence: the basis and the width are chosen. So a null "
            "result is a statement about where one looked, and enlarging the "
            "basis is the way to look further.",
            "The kernel is even in d = log(m_j/m_k) because zeros come in "
            "+/- gamma pairs. Writing the prime term as 2 G(log n) rather than "
            "G(log n) + G(-log n) breaks that and produced a negative "
            "eigenvalue, caught by testing the eigenvector against the zeros.",
        ],
    )
