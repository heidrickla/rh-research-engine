"""The Baez-Duarte criterion has to be able to report a violation.

RH is equivalent to `c_k = O(k^(-3/4+eps))`, so this experiment is a
falsification test, and a falsification test nobody has watched fire is a
falsification test nobody has watched fire. The measurement that matters is the
ENVELOPE of `|c_k| k^(3/4)`: bounded is consistent with RH, growing like a power
is not.

The identity is checked too. Everything here rests on

    sum_j (-1)^j C(k,j)/zeta(2j+2) = sum_{n>=2} mu(n)/n^2 (1-1/n^2)^k

being the same sequence, because the experiment computes the right-hand side
and the criterion is stated for the left. If that is wrong, the whole thing
measures something nobody asked about.

AND THE VERDICT USED TO BE A TYPED CONSTANT. `violated` was the literal `0.0`,
so the tests below that assert it could not have failed for the right reason.
What makes them mean something now is that the envelope slope is measured over
every binning and a refutation requires all of them to agree -- because the bin
count and the upper-fraction cut, which nothing chose, move the slope's SIGN.

The tests appended below were very nearly written OVER this file rather than
after it. Five tests, including the identity check above, were deleted by a
rewrite that assumed the file was new; restoring them caught a ZeroDivisionError
in that same rewrite within a minute. A precondition belongs before the first
write, not after it.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from rh_research_engine.experiments import baez_duarte
from rh_research_engine.experiments.baez_duarte import (
    BIN_COUNTS,
    CRITERION_EXPONENT,
    MINIMUM_PEAKS,
    UPPER_FRACTIONS,
    envelope_peaks,
    mobius_sieve,
    run,
    sequence,
)

LIMIT = 200_000


def test_the_sieve_agrees_with_sympy():
    """Everything downstream is a weighted sum of these."""
    mu = mobius_sieve(2_000)
    assert [int(mu[n]) for n in range(2, 60)] == [int(sp.mobius(n)) for n in range(2, 60)]


@pytest.mark.parametrize("k", [10, 40, 100])
def test_the_mobius_form_is_the_binomial_definition(k):
    """The criterion is stated for the binomial sum; we compute the other one.

    The binomial form needs exact arithmetic -- in float64 it loses about two
    bits per k and returns 6.06e9 for c_100 -- so the comparison is done with
    sympy Rationals against a truncated Mobius sum, and agrees to the
    truncation.
    """
    exact = sum(
        (-1) ** j * sp.binomial(k, j) / sp.zeta(2 * j + 2) for j in range(k + 1)
    )
    approx = float(sequence(np.array([k]), LIMIT)[0])
    assert abs(float(exact) - approx) < 5e-6, (float(exact), approx)


def test_the_envelope_is_bounded_on_the_real_sequence():
    """The falsification test, not firing -- which is not proof of anything."""
    result = run(sieve_limit=LIMIT, k_low=2e3, k_high=2e4, points=60)
    assert result.metrics["violated"] == 0.0
    assert result.metrics["usable"] > 0.5 * result.metrics["samples"]
    # Bounded means the envelope does not climb. Generous, because this is a
    # short range run for the suite's sake; the recorded run reaches 2e6.
    assert result.metrics["envelope_slope"] < 0.35, result.metrics


def test_a_violating_sequence_would_be_reported(monkeypatch):
    """Feed it something that breaks the criterion and watch the envelope climb.

    `c_k = k^(-0.5)` decays, and decays too SLOWLY: RH needs -3/4, so
    `|c_k| k^(3/4)` grows like `k^(1/4)` and the exponent must come back clearly
    positive. Without this the test above passes on any decaying sequence at
    all, including ones that refute RH.
    """
    from rh_research_engine.experiments import baez_duarte

    def fake(ks, limit):  # noqa: ARG001
        return np.asarray(ks, dtype=float) ** -0.5

    monkeypatch.setattr(baez_duarte, "sequence", fake)
    result = baez_duarte.run(sieve_limit=LIMIT, k_low=2e3, k_high=2e4, points=60)

    expected = 1.0 - CRITERION_EXPONENT * 2  # -0.5 decay against a 0.75 scaling
    assert result.metrics["envelope_slope"] > 0.2, result.metrics["envelope_slope"]
    assert abs(result.metrics["envelope_slope"] - 0.25) < 0.05, (
        f"a k^-0.5 sequence must give an envelope of k^+0.25, got "
        f"{result.metrics['envelope_slope']:.3f}"
    )
    assert expected < 0  # the criterion's own arithmetic, stated once
    # And the VERDICT, which used to be a typed 0.0 and could not move.
    assert result.metrics["violated"] == 1.0
    assert result.metrics["sign_undetermined"] == 0.0
    assert result.metrics["envelope_slope_min"] > 0.0, (
        "every binning must agree before this counts as a refutation"
    )


def test_the_error_bar_is_the_two_sieve_difference_not_one_over_n():
    """The guard that discarded two thirds of a good run.

    `1/N` bounds the Mobius tail with no cancellation allowed for, and the tail
    has plenty: at k = 10^6 quadrupling N moves c_k by about 1.5%, so points
    that `1/N` called unusable were fine. The experiment compares two sieve
    limits instead, and that difference must be far below `1/N`.
    """
    # k = 2000 against a 50,000 coarse sieve. The pairing matters: the tail
    # beyond N is essentially independent of k, so a k whose c_k has fallen
    # below it is genuinely unmeasured -- at k = 100,000 here the difference is
    # six times the value, and the experiment's guard drops exactly those.
    ks = np.array([2_000])
    coarse = float(sequence(ks, LIMIT // 4)[0])
    fine = float(sequence(ks, LIMIT)[0])
    difference = abs(fine - coarse)

    assert difference < 0.05 * abs(fine), (coarse, fine)
    # The substantive claim: `1/N` overstates this by three orders of magnitude,
    # which is why it discarded points that were fine.
    assert difference < 1e-3 / (LIMIT // 4), (
        f"two-sieve difference {difference:.2e} against a 1/N bound of "
        f"{4.0 / LIMIT:.2e} -- if these are comparable, 1/N was not the problem"
    )


SMALL = dict(sieve_limit=LIMIT, k_low=2e3, k_high=2e4, points=60)


def test_a_sign_that_moves_with_the_binning_is_not_a_refutation(monkeypatch):
    """The bug that wiring the old comment's condition would have created.

    It prescribed `envelope_slope > 0` on the one recorded binning. Over
    eighteen equally defensible (bins, cut) choices the slope is positive on
    five, so that condition would announce a refutation of RH on five
    histograms. A refutation now requires every binning to agree.
    """
    real = baez_duarte.summarize

    def straddling(grids):
        summary = real(grids)
        return type(summary)(
            slope=0.01, intercept=summary.intercept, spread=0.019,
            lowest=-0.0574, highest=0.0282,
            sign_changes=summary.sign_changes, dynamic_range=summary.dynamic_range,
            dropped=summary.dropped, phases=summary.phases,
        )

    monkeypatch.setattr(baez_duarte, "summarize", straddling)
    result = run(**SMALL)
    assert result.metrics["envelope_slope_max"] > 0 > result.metrics["envelope_slope_min"]
    assert result.metrics["sign_undetermined"] == 1.0
    assert result.metrics["violated"] == 0.0, (
        "positive on some binnings is not positive on all, and only the second refutes"
    )


def test_the_binning_actually_changes_the_peaks():
    """Otherwise the spread is zero for a reason that has nothing to do with c_k.

    An `envelope_peaks` that ignored its arguments would report spread 0.0 and a
    determined sign -- the failure this rewrite exists to stop, arrived at
    through the guard instead of around it.
    """
    ks = np.unique(np.round(np.geomspace(2e3, 2e4, 60)).astype(np.int64))
    scaled = np.abs(np.sin(np.log(ks.astype(float)) * 7.0)) + 0.1
    usable = np.ones(len(ks), dtype=bool)

    shapes = {
        (bins, fraction): tuple(envelope_peaks(ks, scaled, usable, bins, fraction)[1])
        for bins in BIN_COUNTS
        for fraction in UPPER_FRACTIONS
    }
    assert len(set(shapes.values())) > 1, "the arguments must matter"
    assert len(envelope_peaks(ks, scaled, usable, 16, 0.6)[0]) > len(
        envelope_peaks(ks, scaled, usable, 10, 0.4)[0]
    )


def test_the_slope_error_is_not_the_scatter():
    """Two different quantities; the record used to carry only the second.

    The scatter is the residual spread about the fit. The standard error is the
    uncertainty on the number being tested against zero -- on the recorded run
    they are 0.0234 and 0.0154, and only the second says the slope is 0.84
    sigma from zero rather than distinguishable from it.
    """
    result = run(**SMALL)
    error = result.metrics["envelope_slope_error"]
    scatter = result.metrics["envelope_scatter"]
    assert error > 0 and scatter > 0 and error != scatter
    assert any("standard error" in o for o in result.observations)


def test_a_zero_standard_error_does_not_divide_by_zero():
    """The guard, forced through the seam rather than through the arithmetic.

    A synthetic power law fits exactly, so the residual and hence the standard
    error are zero, and the first rewrite divided by it to print "sigma from
    zero" -- ZeroDivisionError, caught by the pre-existing violation test that
    the rewrite had deleted and that was restored from git.

    THE FIRST VERSION OF THIS TEST ASSERTED `error == 0.0` ON THE REAL RUN. That
    held locally and failed in CI at 6.86e-16: whether a least-squares residual
    lands on exactly zero is a fact about the platform's BLAS, not about the
    code. A gate that reads the environment is a gate about the environment, so
    the branch is forced directly and the end-to-end case only has to be tiny.
    """
    assert "exactly determined" in baez_duarte._sigmas(0.25, 0.0)
    assert "sigma" in baez_duarte._sigmas(0.25, 0.1)
    assert baez_duarte._sigmas(0.5, 0.25).startswith("2.00")
    # And the value CI actually produced. `error == 0.0` would have printed
    # "362318840579710 sigma" here -- noise on an exact fit reading as
    # overwhelming significance, which is why the guard is relative.
    assert "exactly determined" in baez_duarte._sigmas(0.25, 6.855213374938573e-16)
    assert "sigma" in baez_duarte._sigmas(0.25, 1e-9), (
        "and a small-but-real error must still report a sigma"
    )


def test_an_exact_power_law_reports_a_negligible_error(monkeypatch):
    """End to end: it completes, and the error is at the noise floor.

    Not asserted to be exactly zero -- see above. What matters is that nothing
    raises and the reported uncertainty is far below anything that could change
    the verdict.
    """
    def exact(ks, limit):  # noqa: ARG001
        return np.asarray(ks, dtype=float) ** -0.5

    monkeypatch.setattr(baez_duarte, "sequence", exact)
    result = run(**SMALL)
    assert result.metrics["envelope_slope_error"] < 1e-12
    assert result.metrics["violated"] == 1.0


def test_dropped_binnings_are_counted_not_skipped():
    """A configuration that produced no fit is not one that agreed."""
    result = run(**SMALL)
    assert result.metrics["binnings"] + result.metrics["binnings_dropped"] == float(
        len(BIN_COUNTS) * len(UPPER_FRACTIONS)
    )
    assert result.metrics["binnings"] >= 2


def test_one_binning_cannot_show_the_binning_dependence(monkeypatch):
    """Refused rather than fitted, like every other single-configuration case."""
    monkeypatch.setattr(baez_duarte, "MINIMUM_PEAKS", 10_000)
    with pytest.raises(ValueError, match="cannot show"):
        run(**SMALL)
    assert MINIMUM_PEAKS == 4
