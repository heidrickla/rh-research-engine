"""The certified verification, and the refusals that make it worth reading.

Two halves. The mathematics -- that the checks say what they claim and can
fail -- needs the backend and skips without it. The epistemics -- that a record
cannot be built at the wrong confidence, and that an absent backend refuses
rather than degrades -- must run everywhere, because the branch where the
backend is missing is exactly the branch nobody exercises.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rh_research_engine.contracts.epistemic import RIGOROUS, Confidence
from rh_research_engine.mathcert.arb_flint import (
    detect_arb_flint,
    envelope_confidence,
    interval_certificate,
)
from rh_research_engine.mathcert.verifiers import VerificationStatus
from rh_research_engine.symbolic.certified_line import (
    BackendUnavailable,
    CertifiedLineVerification,
    certified_count_envelope,
    certify_zeros_on_the_line,
)

BACKEND = detect_arb_flint()
needs_backend = pytest.mark.skipif(
    not BACKEND.available, reason="python-flint is not installed"
)


def _record(**overrides) -> dict:
    payload = {
        "height": 100.0,
        "counted": 29,
        "certified": 29,
        "checks": {"a check": True},
        "backend": "arb-flint",
        "backend_version": "0.9.0",
        "evidence": "twenty-nine zeros, certified",
    }
    payload.update(overrides)
    return payload


# --- what the record refuses ----------------------------------------------


@pytest.mark.parametrize("confidence", sorted(RIGOROUS, key=str))
def test_a_certified_record_still_cannot_claim_a_proof(confidence):
    """`rigorous_numerical` is not in RIGOROUS, and this is why that matters.

    A certified enclosure is rigorous about a finite computation. Every value
    in RIGOROUS says something about the mathematics itself, and the step from
    "certified below T" to "true" is the one this repository exists to stop --
    the more tempting here than anywhere, because the arithmetic really was
    certified.
    """
    with pytest.raises(ValidationError):
        CertifiedLineVerification(**_record(confidence=confidence))


def test_a_certified_record_cannot_be_filed_as_merely_numerical():
    """The other direction, which is the one nobody guards.

    Filed at `numerical` this record would be indistinguishable from the
    floating-point one beside it, with the certification thrown away by a
    default argument. The two verifications differ in kind; their records must
    not be able to agree in status.
    """
    with pytest.raises(ValidationError) as caught:
        CertifiedLineVerification(**_record(confidence=Confidence.NUMERICAL))
    assert "discard" in str(caught.value)


def test_a_record_that_did_not_place_every_zero_is_not_verified():
    """`verified` is the conjunction, not the checks alone.

    Every check can pass on a scan that isolated fewer zeros than the strip
    holds, and that scan has not shown the missing ones are on the line.
    """
    short = CertifiedLineVerification(**_record(counted=29, certified=28))
    assert not short.verified
    failed = CertifiedLineVerification(**_record(checks={"a check": False}))
    assert not failed.verified
    assert CertifiedLineVerification(**_record()).verified


# --- what happens without a backend ---------------------------------------


def test_an_absent_backend_refuses_rather_than_degrading(monkeypatch):
    """No enclosure, no record. Never a quiet fall back to floating point.

    The whole content of this verification is that the arithmetic was
    certified. A fallback would answer the question at the wrong confidence
    with nothing in the output saying so, and `confidence` would be a lie
    rather than a limit.
    """
    from rh_research_engine.mathcert import arb_flint
    from rh_research_engine.symbolic import certified_line

    monkeypatch.setattr(
        certified_line,
        "detect_arb_flint",
        lambda: arb_flint.ArbFlintCapability(
            available=False, reason="pretend it is absent"
        ),
    )
    with pytest.raises(BackendUnavailable) as caught:
        certify_zeros_on_the_line(100.0)
    assert "pretend it is absent" in str(caught.value)
    assert "no floating-point computation substitutes" in str(caught.value)


def test_the_placeholder_certificate_still_reports_unknown():
    """Computing an enclosure did not weaken the adapter that does not.

    `interval_certificate` takes endpoints somebody else worked out. It has no
    way to know they enclose anything, so it says UNKNOWN whether or not a
    backend is installed -- and installing one must not have quietly promoted
    it.
    """
    envelope = interval_certificate(expression="pi", lower="3.14", upper="3.15")
    assert envelope.status is VerificationStatus.UNKNOWN
    assert envelope_confidence(envelope) is not Confidence.RIGOROUS_NUMERICAL


# --- the mathematics ------------------------------------------------------


@needs_backend
def test_the_first_zeros_are_certified_on_the_critical_line():
    """The smallest real instance: 29 zeros below T = 100.

    Held to a literal count rather than to whatever comes back, so a backend
    that started answering a different question -- zeros in the upper half
    plane, say, or counted without multiplicity -- fails here rather than
    being believed.
    """
    result = certify_zeros_on_the_line(100.0)
    assert result.counted == 29
    assert result.certified == 29
    assert result.verified
    assert result.confidence is Confidence.RIGOROUS_NUMERICAL
    assert all(result.checks.values())
    assert result.backend == "arb-flint"


@needs_backend
def test_the_certified_count_agrees_with_the_floating_point_one():
    """Two implementations, one number, and neither told about the other.

    This is the validation of BOTH. `ZeroCount` reaches T = 10^6 by Turing's
    method in float64; Arb reaches it by the argument principle with rigorous
    error bounds. Every height this repository has published a figure for is
    checked here, so a regression in either shows up as a disagreement rather
    than as a number nobody compared.
    """
    import sympy as sp

    from rh_research_engine.mathcert.arb_flint import certified_zero_count
    from rh_research_engine.symbolic.functions import ZeroCount

    for height, recorded in ((50000, 63519), (100000, 138069), (1000000, 1747146)):
        certified = certified_zero_count(float(height))
        assert certified == recorded, f"Arb disagrees with the record at {height}"
        assert int(ZeroCount(sp.Integer(height))) == certified


@needs_backend
def test_an_undetermined_count_raises_rather_than_rounding(monkeypatch):
    """A ball is not an integer, however narrow.

    If the precision ever failed to pin the count down, Arb would return a
    ball with a radius. Rounding it to the nearest integer would produce a
    number indistinguishable from a determined one -- the same shape as
    reporting a relation refuted when it was only unexamined.

    Reached through `_raw_zero_count`, the seam in our own module, rather than
    by substituting the method on `flint.types.arb`. That substitution works
    against the abi3 wheel and raises `cannot set attribute of immutable type`
    against the cp313 one, so this guard ran on Python 3.12 and was silently
    skipped on 3.13 -- a gate exercised in only some of the environments it
    ships to, which is the failure this file is otherwise about.
    """
    from flint import arb, ctx

    from rh_research_engine.mathcert import arb_flint

    previous = ctx.prec
    try:
        ctx.prec = 128
        undetermined = arb(29, 0.4)
        assert undetermined.rad() != 0
        monkeypatch.setattr(
            arb_flint, "_raw_zero_count", lambda height, bits: undetermined
        )
        with pytest.raises(ValueError) as caught:
            arb_flint.certified_zero_count(100.0)
        assert "ball rather than an integer" in str(caught.value)
    finally:
        ctx.prec = previous


@needs_backend
def test_a_fractional_count_raises_rather_than_being_floored(monkeypatch):
    """And the other guard, through the same seam.

    `3/2` arrives from `man_exp` as `(3, -1)`; shifting right by one leaves
    `1`, a count that is wrong and looks exactly as definite as a right one.
    Nothing Arb returns should have a negative exponent, which is precisely
    why the branch needs reaching deliberately -- it guards the case where
    that assumption stopped holding, and would otherwise never run at all.
    """
    from flint import arb, ctx

    from rh_research_engine.mathcert import arb_flint

    previous = ctx.prec
    try:
        ctx.prec = 128
        fractional = arb(3) / 2
        assert fractional.rad() == 0, "the radius guard must not be what fires"
        assert int(fractional.mid().man_exp()[1]) < 0
        monkeypatch.setattr(
            arb_flint, "_raw_zero_count", lambda height, bits: fractional
        )
        with pytest.raises(ValueError) as caught:
            arb_flint.certified_zero_count(100.0)
        assert "not a whole number" in str(caught.value)
    finally:
        ctx.prec = previous


@needs_backend
def test_a_zero_above_the_height_is_caught():
    """The bounds check can fail, and it is the one most easily assumed.

    Asking for one enclosure more than the strip holds below T puts a zero
    above the height into the scan. Counting it would make "all N(T) zeros
    below T are on the line" a claim about a different set of zeros.
    """
    from rh_research_engine.mathcert.arb_flint import (
        certified_zero_count,
        certified_zeros_on_line,
    )

    height = 100.0
    counted = certified_zero_count(height)
    honest = certified_zeros_on_line(counted, height)
    assert all(honest.checks.values())

    greedy = certified_zeros_on_line(counted + 1, height)
    bounds, = [name for name in greedy.checks if name.startswith("every enclosure lies")]
    assert not greedy.checks[bounds]
    assert greedy.certified == 0, "one bad enclosure certifies nothing"


@needs_backend
def test_proximity_to_the_critical_line_is_not_being_on_it():
    """The check that would be easiest to write wrongly.

    A ball around 1/2 of radius 1e-30 contains points off the critical line,
    so an enclosure merely CONTAINING 1/2 establishes nothing. Arb marks a
    zero it has proved to be on the line with an exact real part; testing
    equality against 1/2 is testing that, and testing `abs(re - 1/2) < eps`
    would have been testing proximity.
    """
    from flint import arb, ctx

    previous = ctx.prec
    try:
        ctx.prec = 128
        half = arb(1) / 2
        nearly = arb(0.5, 1e-30)
        # `not (a == b)`, never `a != b`. Arb's `!=` means CERTAINLY unequal,
        # so two overlapping balls answer False to both -- and `!=` here would
        # be asserting the balls are provably different numbers, which is a
        # stronger claim than the code makes and than the data supports.
        assert not (nearly == half), "a ball around 1/2 must not compare equal"
        assert not (nearly != half), "nor is it certainly a different number"
        assert half == half and half.rad() == 0
    finally:
        ctx.prec = previous


@needs_backend
def test_arb_comparisons_are_true_only_when_certainly_true():
    """The soundness the three checks rest on.

    Every check compares balls, never floats read out of them. That is only
    sound because Arb's comparisons are conservative: two overlapping
    intervals compare False in BOTH directions rather than guessing. If that
    ever changed, the disjointness check would start accepting enclosures that
    might describe one zero twice, and nothing else would notice.
    """
    from flint import arb, ctx

    previous = ctx.prec
    try:
        ctx.prec = 53
        overlapping_a, overlapping_b = arb(1, 0.5), arb(1.2, 0.5)
        assert not (overlapping_a < overlapping_b)
        assert not (overlapping_a > overlapping_b)
        assert arb(5, 0.1) > overlapping_a
    finally:
        ctx.prec = previous


@needs_backend
def test_the_disjointness_check_separates_the_two_cases():
    """The third check can fail, on the same zeros.

    Arb will not hand back overlapping enclosures, so the way to see this
    check work is to widen every radius past the largest gap and confirm the
    same comparison then reports False. Nothing about the zeros changes; only
    what is known of them. If it answered True on overlapping intervals the
    scan would accept N enclosures describing fewer than N distinct zeros, and
    the multiplicity argument that gives simplicity would fail silently --
    still reporting CERTIFIED.
    """
    from flint import acb, arb, ctx

    previous = ctx.prec
    try:
        ctx.prec = 128
        ordered = sorted(acb.zeta_zeros(1, 12), key=lambda z: float(z.imag.mid()))
        assert all(a.imag < b.imag for a, b in zip(ordered, ordered[1:], strict=False))

        widest_gap = max(
            float(b.imag.mid()) - float(a.imag.mid())
            for a, b in zip(ordered, ordered[1:], strict=False)
        )
        fattened = [arb(float(z.imag.mid()), widest_gap) for z in ordered]
        assert not all(a < b for a, b in zip(fattened, fattened[1:], strict=False))
    finally:
        ctx.prec = previous


@needs_backend
def test_the_count_envelope_is_accepted_and_rigorous():
    """The adapter's first ACCEPTED envelope, and what makes it mean anything.

    `envelope_confidence` consults the registry rather than the envelope's own
    status, so ACCEPTED maps to `rigorous_numerical` only because the backend
    is independently detected as present. That is the property being checked:
    the status is a claim, and the registry is what makes it worth reading.
    """
    envelope = certified_count_envelope(1000.0)
    assert envelope.status is VerificationStatus.ACCEPTED
    assert envelope.verifier_family == "arb-flint"
    assert envelope_confidence(envelope) is Confidence.RIGOROUS_NUMERICAL
    assert envelope.certificate.value.lower.numerator == 649
    assert envelope.certificate.value.upper.numerator == 649


@needs_backend
def test_an_accepted_envelope_from_an_unregistered_family_is_not_rigorous():
    """The spoof this whole layer exists to stop, retested now it can fire.

    Before this build no envelope was ever ACCEPTED, so the check that an
    ACCEPTED envelope from an unregistered family maps to UNKNOWN was guarding
    a case that could not arise. It can now.
    """
    envelope = certified_count_envelope(100.0)
    spoof = envelope.model_copy(update={"verifier_family": "not-a-real-verifier"})
    assert spoof.status is VerificationStatus.ACCEPTED
    assert envelope_confidence(spoof) is Confidence.UNKNOWN


@needs_backend
def test_the_working_precision_is_put_back():
    """Arb's precision is global, and a computation must not leave it moved.

    A raised precision leaking out would silently change the cost and the
    results of everything that ran afterwards -- a computation depending on
    what happened to run before it, which is the shape of a bug nobody can
    reproduce.
    """
    from flint import ctx

    from rh_research_engine.mathcert.arb_flint import certified_zero_count

    ctx.prec = 53
    try:
        certified_zero_count(100.0)
        assert ctx.prec == 53
    finally:
        ctx.prec = 53
