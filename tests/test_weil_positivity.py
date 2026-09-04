"""The Weil form must agree with the zeros, not merely be positive.

A pair of individually harmless shortcuts produced a negative eigenvalue --
which is what a refutation of RH looks like -- and survived because the form
was checked only against itself. What exposed it was a direct sum over verified
ordinates, so that comparison is the test here.

POSITIVITY IS THE BLUNT CHECK, and that is the lesson worth keeping. The broken
form is still positive semidefinite at small sizes and only fails near size 16,
so eigenvalues alone would have let it through. Agreement with the zeros fails
immediately, at 1e-3 against a kernel otherwise good to 5e-15.

EVERY KERNEL VALUE IS CACHED ACROSS THE WHOLE FILE. The refining quadrature is
the entire cost -- unshared, these tests took eighteen minutes and would have
tripled the suite. The same `d` recurs across sizes and tests, so one dictionary
does most of the work.
"""

from __future__ import annotations

import numpy as np
import pytest

from rh_research_engine.experiments.weil_positivity import kernel, run, von_mangoldt

pytest.importorskip("mpmath")

SIGMA = 0.05
PRIMES = 50_000
#: Far looser than production's 1e-12, and still a thousand times tighter than
#: the 1e-9 these tests assert.
TOL = 1e-10

_N, _LAM = von_mangoldt(PRIMES)
LOGS, WEIGHT = np.log(_N), _LAM / np.sqrt(_N)
_CACHE: dict[float, float] = {}


def k(d: float) -> float:
    key = round(d, 12)
    if key not in _CACHE:
        _CACHE[key] = kernel(d, SIGMA, LOGS, WEIGHT, TOL)
    return _CACHE[key]


@pytest.fixture(scope="module")
def ordinates():
    """A few hundred ordinates is plenty: exp(-sigma^2 gamma^2) kills the rest."""
    from rh_research_engine.symbolic.riemann_siegel import first_zero_ordinates

    return first_zero_ordinates(400)


def _from_zeros(d: float, gammas: np.ndarray) -> float:
    """2 sum cos(gamma d) exp(-sigma^2 gamma^2), the kernel by definition."""
    return 2.0 * float(np.sum(np.exp(-(SIGMA**2) * gammas**2) * np.cos(gammas * d)))


def _matrix(size: int, entry) -> np.ndarray:
    ms = np.arange(1, size + 1, dtype=float)
    return np.array([[entry(float(np.log(a / b))) for b in ms] for a in ms])


@pytest.mark.parametrize("a,b", [(1, 1), (2, 1), (4, 3), (8, 7), (5, 4)])
def test_the_kernel_agrees_with_a_direct_sum_over_zeros(a, b, ordinates):
    """Primes and Gamma on one side, ordinates on the other.

    `(4,3)`, `(5,4)` and `(8,7)` are here on purpose: those are the `d` where a
    fixed-node quadrature was wrong by up to 3.8e-5 while every other entry was
    exact, which is why the routine refines until it agrees with itself.
    """
    d = float(np.log(a / b))
    assert abs(k(d) - _from_zeros(d, ordinates)) < 1e-9


def test_the_kernel_is_even_in_d():
    """Zeros come in +/- gamma pairs, so `2 sum cos(gamma d)` is even."""
    for ratio in (2.0, 8.0 / 7.0):
        d = float(np.log(ratio))
        assert abs(k(d) - k(-d)) < 1e-9


def test_the_quadratic_form_is_the_sum_over_zeros(ordinates):
    """c^T K c must equal sum_rho h(gamma), not merely be positive.

    A form can be positive semidefinite and still not be the functional it
    claims to be, which is exactly how the broken version survived.
    """
    size = 5
    matrix = _matrix(size, lambda d: 0.5 * (k(d) + k(-d)))
    ms = np.arange(1, size + 1, dtype=float)
    rng = np.random.default_rng(7)
    for _ in range(3):
        c = rng.normal(size=size)
        quadratic = float(c @ matrix @ c)
        amplitude = np.abs(np.exp(1j * np.outer(ordinates, np.log(ms))) @ c) ** 2
        direct = 2.0 * float(np.sum(np.exp(-(SIGMA**2) * ordinates**2) * amplitude))
        assert abs(quadratic - direct) < 1e-8, (quadratic, direct)
        assert direct >= 0.0, "h >= 0 and real gammas: this cannot be negative"


def test_the_form_is_positive_semidefinite():
    """The falsification test, not firing."""
    eigenvalues = np.linalg.eigvalsh(_matrix(6, lambda d: 0.5 * (k(d) + k(-d))))
    assert eigenvalues[0] > 0, eigenvalues


def test_run_reports_the_verdict():
    """The recorded experiment must carry the same answer the matrix gives."""
    result = run(size=4, sigma=SIGMA, prime_limit=PRIMES, tolerance=TOL)
    assert result.metrics["negative_count"] == 0
    assert result.metrics["violated"] == 0.0
    assert result.metrics["smallest_eigenvalue"] > 0


def test_neither_shortcut_is_wrong_alone_but_together_they_are(ordinates):
    """The failure was a conspiracy, so test the conspiracy.

    `2 G(log n)` for the prime term, and `K(|d|)` for the assembly. Averaging
    the first over `+/- d` reconstructs the correct two-sided term, so the
    symmetrisation repairs it; and the corrected kernel is even, so `|d|`
    changes nothing. Each alone is harmless. Together, `|d|` removes the
    averaging that would have cancelled the doubling.

    Two earlier accounts called this two bugs and then one. It is neither --
    which is precisely why checking them one at a time cleared both.
    """
    import mpmath as mp

    doubled_cache: dict[float, float] = {}

    def doubled(d: float) -> float:
        key = round(d, 12)
        if key in doubled_cache:
            return doubled_cache[key]
        norm = 1.0 / (2.0 * SIGMA * np.sqrt(np.pi))
        span = 30.0 / SIGMA
        arch = float(
            mp.quad(
                lambda r: mp.e ** (-(SIGMA**2) * r**2)
                * mp.cos(r * d)
                * mp.re(mp.digamma(mp.mpf(0.25) + 0.5j * r)),
                list(np.linspace(-span, span, 801)),
            )
        ) / (2 * np.pi)
        value = (
            2.0 * np.exp(SIGMA**2 / 4.0) * np.cosh(d / 2.0)
            - norm * np.exp(-(d**2) / (4.0 * SIGMA**2)) * np.log(np.pi)
            + arch
            - 2.0
            * float(np.sum(WEIGHT * norm * np.exp(-((LOGS - d) ** 2) / (4.0 * SIGMA**2))))
        )
        doubled_cache[key] = value
        return value

    # Wrong wherever d is not zero, and right at zero -- which is where it was
    # originally checked, and why it survived.
    assert abs(doubled(0.0) - _from_zeros(0.0, ordinates)) < 1e-8
    d = float(np.log(2.0))
    assert abs(doubled(d) - _from_zeros(d, ordinates)) > 1e-3

    size = 6
    correct = _matrix(size, lambda d: 0.5 * (k(d) + k(-d)))

    # Doubled prime term, SYMMETRISED: repaired.
    symmetrised = _matrix(size, lambda d: 0.5 * (doubled(d) + doubled(-d)))
    assert np.allclose(symmetrised, correct, atol=1e-6)
    # |d| with the correct kernel: harmless, because it is already even.
    assert np.allclose(_matrix(size, lambda d: k(abs(d))), correct, atol=1e-6)
    # Both together: no longer even, and positivity fails.
    assert np.linalg.eigvalsh(_matrix(size, lambda d: doubled(abs(d))))[0] < 0, (
        "the two shortcuts together must break positivity -- that is the "
        "failure this experiment was built out of"
    )
