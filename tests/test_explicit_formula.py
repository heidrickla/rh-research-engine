"""Rebuilding the prime staircase from the zeros.

The corpus records von Mangoldt's explicit formula and nothing had evaluated it.
What is under test is that the check is about the CORPUS's statement rather than
a copy of it, that it would notice a wrong record, and that a residual cannot be
filed as more than a residual.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import sympy as sp
from pydantic import ValidationError

from rh_research_engine.contracts.epistemic import RIGOROUS, Confidence
from rh_research_engine.symbolic import explicit_formula as module
from rh_research_engine.symbolic.explicit_formula import (
    ExplicitFormulaCheck,
    chebyshev_psi,
    check_explicit_formula,
    psi_from_zeros,
    recorded_equation,
    recorded_shape,
)


def test_the_recorded_formula_has_every_term_it_should():
    """Checked as MATHEMATICS, not as text.

    The first version searched the printed form for `log(2*pi)` and reported it
    missing, because SymPy canonicalises that to `log(pi) + log(2)`. A check on
    the notation is the exact failure this repository is built against, in a
    file written to catch it.
    """
    shape = recorded_shape()
    assert shape["constant_is_minus_log_two_pi"], (
        "a missing log(2 pi) leaves the residual a flat 1.8379 at every point, "
        "which reads as a bug in the summation rather than a wrong record"
    )
    assert shape["has_zero_sum"]
    assert shape["has_half_log_term"]
    assert shape["sums_from_one_to_infinity"]


def test_the_notation_check_would_have_missed_it():
    """Kept so the reason for checking mathematically is not lost."""
    printed = str(recorded_equation())
    assert "log(2*pi)" not in printed, (
        "if SymPy started printing it this way, the cautionary note above is "
        "stale and should be rewritten rather than left to mislead"
    )
    assert "log(pi)" in printed and "log(2)" in printed


def test_the_zeros_rebuild_the_staircase():
    """The measurement: psi(x) against the corpus's reconstruction."""
    result = check_explicit_formula([10.5, 20.5, 50.5], zeros=5000)
    assert result.confidence is Confidence.NUMERICAL
    assert abs(result.worst_residual) < 0.05, result.residuals
    for direct, rebuilt in zip(result.direct, result.reconstructed, strict=True):
        assert abs(direct - rebuilt) < 0.05


def test_the_residual_falls_as_more_zeros_are_used():
    """The sum is conditionally convergent; this is what convergence looks like."""
    few = check_explicit_formula([100.5], zeros=200)
    many = check_explicit_formula([100.5], zeros=10000)
    assert abs(many.worst_residual) < abs(few.worst_residual) / 5


def test_a_wrong_record_would_be_caught():
    """A missing constant parses, indexes and fingerprints exactly as well.

    The residual is what tells the two apart, so this confirms the residual
    actually moves when the formula does.
    """
    from rh_research_engine.symbolic.riemann_siegel import first_zero_ordinates

    ordinates = first_zero_ordinates(5000)
    right = psi_from_zeros(50.5, ordinates)
    without_constant = right + float(np.log(2 * np.pi))
    assert abs(chebyshev_psi(50.5) - right) < 0.05
    assert abs(chebyshev_psi(50.5) - without_constant) > 1.5


def test_a_corpus_without_the_formula_has_nothing_to_check(tmp_path, monkeypatch):
    empty = tmp_path / "index.json"
    empty.write_text(json.dumps([{"canonical": "Symbol('x')"}]), encoding="utf-8")
    monkeypatch.setattr(module, "INDEX_PATH", empty)
    with pytest.raises(LookupError, match="no longer records"):
        recorded_equation()


def test_a_missing_index_is_reported_as_such(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "INDEX_PATH", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="ingest"):
        recorded_equation()


def test_the_check_refuses_a_record_it_does_not_implement(tmp_path, monkeypatch):
    """Comparing against a formula this file does not implement tests this file.

    Forced by handing it a recorded equation with the constant dropped.
    """
    x, k = sp.Symbol("x"), sp.Symbol("k")
    from rh_research_engine.symbolic.functions import ChebyshevPsi, NthZetaZero

    crippled = sp.Eq(
        ChebyshevPsi(x),
        x
        - sp.log(1 - x**-2) / 2
        - 2 * sp.Sum(sp.re(x ** NthZetaZero(k) / NthZetaZero(k)), (k, 1, sp.oo)),
    )
    monkeypatch.setattr(module, "recorded_equation", lambda: crippled)
    with pytest.raises(LookupError, match="constant_is_minus_log_two_pi"):
        check_explicit_formula([10.5], zeros=200)


def test_psi_is_summed_over_prime_powers_not_primes():
    """`psi(10) = log 2 + log 3 + log 2 + log 5 + log 7 + log 2 + log 3`.

    The term at a prime power is `log p`, not `log p^k`. Getting that wrong is
    how a hand-rolled reference called a correct accumulation wrong elsewhere in
    this repository.
    """
    expected = float(
        np.log(2) * 3 + np.log(3) * 2 + np.log(5) + np.log(7)
    )
    assert abs(chebyshev_psi(10.5) - expected) < 1e-12


@pytest.mark.parametrize("confidence", sorted(RIGOROUS, key=str))
def test_a_residual_cannot_be_filed_as_a_proof(confidence):
    with pytest.raises(ValidationError) as caught:
        ExplicitFormulaCheck(
            points=[10.5],
            direct=[7.83],
            reconstructed=[7.83],
            residuals=[0.0009],
            zeros_used=20000,
            worst_residual=0.0009,
            worst_at=10.5,
            confidence=confidence,
        )
    assert "conditionally convergent" in str(caught.value)
