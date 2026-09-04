"""Counting the zeros in the strip, and what the count is allowed to mean.

Two things are under test and they are different. That the count is RIGHT --
checked against `mpmath.nzeros`, which reaches the same number by a completely
different route. And that the count cannot be filed as more than it is: the
phrase "verified to height T" is exactly the one that travels without its
qualifiers, so the records refuse a rigorous confidence at construction.
"""

from __future__ import annotations

import mpmath
import numpy as np
import pytest
from pydantic import ValidationError

from rh_research_engine.contracts.epistemic import NON_DEDUCTIVE, RIGOROUS, Confidence
from rh_research_engine.symbolic.argument_principle import (
    INITIAL_SAMPLES,
    MAX_INCREMENT,
    RIGHT_EDGE,
    LineVerification,
    StripCount,
    strip_zero_count,
    verify_zeros_on_the_line,
)


@pytest.mark.parametrize("height", [20.0, 50.0, 100.0, 500.0, 1000.0])
def test_the_strip_count_agrees_with_mpmath(height):
    """A different route to the same integer.

    `mpmath.nzeros` walks Gram and Rosser blocks and separates zeros by sign
    changes of Z. This tracks `arg zeta` along a path that never touches the
    critical line. Nothing is shared but the answer.
    """
    result = strip_zero_count(height)
    assert result.count == int(mpmath.nzeros(height))


@pytest.mark.parametrize("height", [50.0, 100.0, 1000.0, 5000.0])
def test_the_count_is_not_a_close_call(height):
    """`N(T)` is an integer, so landing near one is the sanity check.

    A total sitting halfway between two integers would mean the rounding, not
    the mathematics, chose the answer -- and the count would still look like a
    count.
    """
    result = strip_zero_count(height)
    assert result.distance_from_integer < 1e-6, result.distance_from_integer


def test_the_correction_term_stays_small():
    """`S(T)` is `O(log T)` and small in practice.

    A large value here is a tracking failure long before it is a discovery, and
    saying so is what keeps the failure from being read as the result.
    """
    for height in (50.0, 500.0, 5000.0, 25000.0):
        result = strip_zero_count(height)
        assert abs(result.correction) < 3.0, (height, result.correction)


def test_the_path_starts_where_zeta_cannot_wind():
    """The vertical leg contributes nothing, and that is why it is skipped.

    At `sigma = RIGHT_EDGE` the Euler product converges absolutely, so
    `arg zeta` stays in a neighbourhood of zero for every height. If that were
    false the horizontal leg alone would not be `S(T)`.
    """
    assert RIGHT_EDGE > 1.0
    for height in (10.0, 1000.0, 25000.0):
        angle = float(mpmath.arg(mpmath.zeta(complex(RIGHT_EDGE, height))))
        assert abs(angle) < np.pi / 2, (height, angle)


def test_the_horizontal_leg_needs_very_few_samples():
    """`arg zeta` barely moves along it, and the sampling should reflect that.

    `S(T)` is under one in absolute value at every height reached here, so the
    total change over the segment is a fraction of a radian. Four hundred
    samples -- the first version's default -- was four hundred `zeta`
    evaluations to reconstruct a curve with no features.
    """
    assert MAX_INCREMENT < np.pi / 2
    assert INITIAL_SAMPLES <= 64
    for height in (100.0, 1000.0, 25000.0):
        result = strip_zero_count(height)
        assert result.samples == INITIAL_SAMPLES, "no refinement should be needed"
        assert result.count == int(mpmath.nzeros(height))


def test_a_height_sitting_on_a_zero_is_refused():
    """`N(T)` is ambiguous there, so a number would be a wrong answer.

    `S(T)` is `arg zeta(1/2 + iT)`, which has no value when zeta vanishes, and
    the zero at that height is either counted or not. This is also the one case
    the refinement loop cannot fix: before there was a ceiling it doubled its
    way to eight hundred thousand `zeta` evaluations and then gave up.
    """
    first_zero = 14.134725141734693
    with pytest.raises(ValueError, match="zero"):
        strip_zero_count(first_zero)
    # Just off it the count is unambiguous again -- and this is where UNIFORM
    # sampling failed: arg zeta swings over the last stretch of the path while
    # the rest of it is featureless, so four thousand evenly spaced samples
    # still left an increment of 1.3 radians. The refinement is adaptive.
    below = strip_zero_count(first_zero - 1e-4)
    above = strip_zero_count(first_zero + 1e-4)
    assert below.count == 0, below
    assert above.count == 1, above
    assert max(below.samples, above.samples) > INITIAL_SAMPLES, (
        "the adaptive branch should have been exercised"
    )


def test_the_refusal_to_reconstruct_a_coarse_winding_is_reachable():
    """A guard that cannot fire is decoration.

    Forced by capping the refinement, since the loop repairs an ordinary coarse
    sample immediately.
    """
    with pytest.raises(RuntimeError, match="refusing to reconstruct"):
        strip_zero_count(14.134725141734693 + 1e-8, samples=3, max_samples=3)


def test_the_two_counts_agree_below_a_height():
    """The comparison the module exists for.

    Zeros in the strip against sign changes of Z on the line. Agreement means
    every zero below the height is on the critical line and simple -- as far as
    a finite floating-point computation can say so, which is what the record's
    confidence is for.
    """
    result = verify_zeros_on_the_line(1000.0)
    assert result.agrees
    assert result.strip == result.on_line == 649
    assert result.confidence is Confidence.NUMERICAL
    assert "as far as a finite floating-point computation" in result.evidence


def test_a_disagreement_is_reported_as_a_bug_report_first():
    """Because that is what it would almost certainly be.

    A genuine off-line zero would be the mathematical result of the century; a
    mistake in one of two numerical routines would not. The record has to say
    which is likelier, or the first arithmetic slip reads as a refutation of
    the Riemann hypothesis.
    """
    result = LineVerification(
        height=100.0,
        strip=29,
        on_line=28,
        agrees=False,
        evidence=(
            "the strip holds 29 zeros below 100 and only 28 sign changes of Z "
            "were found: either some zero is off the line or of even order, or "
            "one of the two computations is wrong -- and the second is far "
            "likelier"
        ),
        notes=["a disagreement is a bug report before it is a mathematical claim"],
    )
    assert not result.agrees
    assert "far likelier" in result.evidence


@pytest.mark.parametrize("confidence", sorted(RIGOROUS, key=str))
def test_neither_record_may_claim_a_rigorous_confidence(confidence):
    """"Verified to height T" must not become "proved".

    Refused at construction rather than checked at export, for the same reason
    a pattern finding is: a record that can be built wrong will eventually be
    built wrong, and this is the exact step where a finite computation would
    acquire an authority it has not got.
    """
    with pytest.raises(ValidationError) as caught:
        StripCount(
            height=100.0,
            count=29,
            smooth=29.0,
            correction=0.0,
            distance_from_integer=0.0,
            samples=400,
            confidence=confidence,
        )
    assert "every zero" in str(caught.value)

    with pytest.raises(ValidationError) as caught:
        LineVerification(
            height=100.0,
            strip=29,
            on_line=29,
            agrees=True,
            evidence="...",
            confidence=confidence,
        )
    assert "every zero" in str(caught.value)


@pytest.mark.parametrize("confidence", sorted(NON_DEDUCTIVE, key=str))
def test_a_non_deductive_confidence_is_allowed(confidence):
    record = StripCount(
        height=100.0,
        count=29,
        smooth=29.0,
        correction=0.0,
        distance_from_integer=0.0,
        samples=400,
        confidence=confidence,
    )
    assert record.confidence is confidence


def test_an_uncomfortable_rounding_is_flagged_rather_than_hidden():
    """A marginal count must not read as a clean one.

    `N(T)` is an integer, so the distance from one is the error bar. In practice
    it is around 1e-13, and this exercises the branch that fires when it is not
    -- otherwise the caveat would be a sentence nobody had ever seen printed.
    """
    from rh_research_engine.symbolic.argument_principle import (
        UNCOMFORTABLE_ROUNDING,
        _notes_for,
    )

    clean = verify_zeros_on_the_line(100.0)
    assert all("too far to be comfortable" not in note for note in clean.notes)

    marginal = StripCount(
        height=100.0,
        count=29,
        smooth=29.0,
        correction=0.0,
        distance_from_integer=UNCOMFORTABLE_ROUNDING + 0.3,
        samples=400,
    )
    notes = _notes_for(marginal, 29)
    assert any("too far to be comfortable" in note for note in notes)
    assert all("bug report" not in note for note in notes), (
        "the counts agreed; only the rounding was marginal"
    )

    disagreeing = _notes_for(marginal, 28)
    assert any("bug report" in note for note in disagreeing)
