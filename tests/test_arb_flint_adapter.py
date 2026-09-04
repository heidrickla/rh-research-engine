import pytest

from rh_research_engine.contracts.epistemic import RIGOROUS, Confidence
from rh_research_engine.mathcert import (
    VerificationStatus,
    VerifierEnvelope,
    detect_arb_flint,
    interval_certificate,
    validate_external_envelope,
    verifier_activation_status,
)
from rh_research_engine.mathcert.arb_flint import MAX_CONFIDENCE, envelope_confidence
from rh_research_engine.mathcert.verifiers import _test_only_registered_adapters


def test_arb_flint_capability_shape():
    capability = detect_arb_flint()
    assert capability.family == "arb-flint"
    assert capability.independence_group.startswith("math-verifier:arb-flint:")


def test_interval_certificate_fails_closed_without_accepted_verdict():
    envelope = interval_certificate(expression="zeta(2)", lower="1.64", upper="1.65")
    assert envelope.verifier_family == "arb-flint"
    assert envelope.status is VerificationStatus.UNKNOWN
    assert envelope.independence_group.startswith("math-verifier:arb-flint:")
    assert validate_external_envelope(envelope, allowed_families={"arb-flint"}) == []


def test_mpmath_cannot_be_relabelled_as_arb_flint():
    envelope = interval_certificate(expression="zeta(2)", lower="1.64", upper="1.65")
    envelope.certificate.verifier.method = "mpmath"
    errors = validate_external_envelope(envelope, allowed_families={"arb-flint"})
    assert "certificate verifier method does not match envelope verifier family" in errors


def _accepted_envelope() -> VerifierEnvelope:
    """A caller-constructed envelope asserting a verification that never ran."""
    envelope = interval_certificate(expression="zeta(2)", lower="1.64", upper="1.65")
    envelope.status = VerificationStatus.ACCEPTED
    envelope.checks = ["enclosure verified"]
    return envelope


def _attributed_envelope() -> VerifierEnvelope:
    """The same assertion, carrying the attribution a real one carries.

    The difference between this and `_accepted_envelope` is the whole of what
    separates a verification from a claim to have done one, once a backend is
    actually installed.
    """
    envelope = _accepted_envelope()
    envelope.certificate.verifier.worker_version = "0.0.0-test"
    envelope.certificate.verifier.worker_hash = "0" * 64
    envelope.certificate.verifier.source_hash = "1" * 64
    return envelope


def test_no_public_function_accepts_an_adapter_registry_override():
    """A caller that can register its own family has registered itself.

    The same shape as a caller-supplied trusted-engine set, and closed the same
    way: the registry is module-private and resolved from capability detection.
    """
    import inspect

    from rh_research_engine.mathcert import verifiers

    for name, obj in vars(verifiers).items():
        if name.startswith("_") or not inspect.isfunction(obj):
            continue
        params = inspect.signature(obj).parameters
        assert "registered_adapters" not in params, f"verifiers.{name}"


def test_a_self_declared_acceptance_is_not_a_verification():
    """`status=ACCEPTED` is caller-supplied; no backend ran to produce it.

    The absence is FORCED rather than read off the machine. This test passed
    for months because nothing had python-flint installed, so it was a
    statement about the developer's environment wearing the shape of a
    statement about the code -- and it started failing the day a backend
    arrived, which is the day it was supposed to start mattering.
    """
    with _test_only_registered_adapters(set()):
        errors = validate_external_envelope(
            _accepted_envelope(), allowed_families={"arb-flint"}
        )
    assert any("no registered execution adapter" in error for error in errors)


def test_a_self_declared_acceptance_is_not_a_verification_with_a_backend_either():
    """The case that only became reachable once a backend was installed.

    With the family registered, "no adapter" is no longer the objection --
    and an envelope naming a real, present backend is the more convincing
    spoof of the two. What refuses it is attribution: an ACCEPTED envelope
    must say which build did the work, or its metadata can be edited to agree
    with whatever it claims.
    """
    with _test_only_registered_adapters({"arb-flint"}):
        errors = validate_external_envelope(
            _accepted_envelope(), allowed_families={"arb-flint"}
        )
    assert not any("no registered execution adapter" in error for error in errors)
    assert any("worker_hash" in error for error in errors)
    assert any("source_hash" in error for error in errors)


def test_confidence_follows_the_adapter_not_the_status_field():
    """Mapping `status` straight through was the bypass.

    `VerifierEnvelope` is public and freely constructible, so reading its own
    `status` meant a caller could hand itself RIGOROUS_NUMERICAL without any
    enclosure having been computed.
    """
    with _test_only_registered_adapters(set()):
        assert envelope_confidence(_accepted_envelope()) is Confidence.UNKNOWN


def test_an_unattributed_acceptance_is_unknown_even_with_the_adapter_present():
    """The registry check was a complete answer only while nothing was installed.

    With python-flint present the family IS registered, so the guard above
    passes and a hand-built envelope naming `arb-flint` would have earned
    RIGOROUS_NUMERICAL with no computation behind it. The hole was closed by
    absence, and absence stopped being the case the moment a backend was
    added -- so the same attribution `validate_external_envelope` demands is
    demanded here.
    """
    with _test_only_registered_adapters({"arb-flint"}):
        assert envelope_confidence(_accepted_envelope()) is Confidence.UNKNOWN


def test_a_connected_backend_earns_rigorous_numerical_and_no_more():
    with _test_only_registered_adapters({"arb-flint"}):
        confidence = envelope_confidence(_attributed_envelope())
    assert confidence is MAX_CONFIDENCE
    assert confidence is Confidence.RIGOROUS_NUMERICAL
    # The ceiling. A certified enclosure is rigorous about a finite
    # computation, which is not the same as rigorous about a theorem.
    assert confidence not in RIGOROUS


@pytest.mark.parametrize("family", ["arb", "flint", "pari", "mpfi", "lean"])
def test_naming_an_external_family_without_an_adapter_is_refused(family):
    envelope = _accepted_envelope()
    envelope.verifier_family = family
    envelope.certificate.verifier.method = family
    errors = validate_external_envelope(envelope, allowed_families={family})
    assert any("not connected in this build" in error for error in errors)


def test_verifier_activation_reports_inert_without_exposing_the_registry():
    """Inert when nothing is connected -- and the emptiness is forced.

    Read off the machine, this asserted "the developer has not installed
    python-flint", which stopped being true and was never the claim.
    """
    with _test_only_registered_adapters(set()):
        status = verifier_activation_status()
    assert status["rigorous_verification_active"] is False
    assert status["registered_adapter_count"] == 0
    assert status["max_confidence"] == "rigorous_numerical"


def test_verifier_activation_reports_active_when_a_backend_is_connected():
    """And says so, still without naming what is registered.

    The count and the boolean move; the family names never appear. Reading
    this report grants nothing, which is why it can be public.
    """
    with _test_only_registered_adapters({"arb-flint"}):
        status = verifier_activation_status()
    assert status["rigorous_verification_active"] is True
    assert status["registered_adapter_count"] == 1
    assert status["max_confidence"] == "rigorous_numerical"
    assert "arb-flint" not in repr(status)
