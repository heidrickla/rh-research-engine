"""Activation tests for DRE receipt work-product certification.

These are the guard on `contracts/receipts.py::_authenticate` -- the seam where
a real signature, sealed-store, or deterministic-replay backend will connect.
The seam is unusual in that its correct behaviour today is *to refuse
everything*, which is indistinguishable from the seam not being called at all.
So these tests check the refusal, the reason given, and the one thing a no-op
would not do: bind authentication to the receipt's content rather than to what
the receipt says about itself.

Deleting the body of `_authenticate` must fail tests in this file. If it ever
stops doing so, the seam has become decorative.
"""

from __future__ import annotations

import pytest

import helpers_discharge
from rh_research_engine.contracts.discharge import resolve_discharges
from rh_research_engine.contracts.receipts import (
    AUTHENTICATED_MECHANISMS,
    DreReceipt,
    ReceiptAuthentication,
    _test_only_authenticated_receipts,
    _test_only_trusted_engines,
)


def _resolve(registry):
    obligations, evidence, decisions, receipts = registry
    return resolve_discharges(
        ["OBL-1"],
        obligations=obligations,
        evidence=evidence,
        decisions=decisions,
        receipts=receipts,
    )


@pytest.mark.parametrize(
    "mechanism,expected",
    [
        # Signature has a real backend now, so it refuses further along: the
        # fixture receipt declares the mechanism and carries no signature.
        (ReceiptAuthentication.SIGNATURE, "carries no signature"),
        (ReceiptAuthentication.SEALED_STORE, "no sealed-store verifier"),
        (ReceiptAuthentication.DETERMINISTIC_REPLAY, "no deterministic-replay verifier"),
    ],
)
def test_declared_authentication_does_not_certify_the_work_product(mechanism, expected):
    """A trusted fingerprint plus a mechanism string is still only a claim.

    Every mechanism is covered, not just the one the fixtures happen to use. A
    member of `AUTHENTICATED_MECHANISMS` that no backend implements but that
    verifies by declaration would be worse than an absent one: it reads as a
    checked property.

    The two unimplemented mechanisms refuse for want of a backend; signature
    refuses because a declaration is not a signature. Both are the same point.
    """
    registry = helpers_discharge.accepted_registry()
    registry[3]["DEC-1"] = registry[3]["DEC-1"].model_copy(
        update={"receipt_authentication": mechanism}
    )

    with _test_only_trusted_engines(helpers_discharge.TRUSTED):
        resolution = _resolve(registry)

    assert resolution.qualifying == []
    assert expected in resolution.explain()


def test_no_declared_mechanism_falls_through_the_dispatch():
    """Every member either verifies for a stated reason or refuses for one.

    A member reaching `_authenticate` with no branch would return `None`, which
    the caller reads as success -- the one outcome this seam must never produce
    by omission.
    """
    registry = helpers_discharge.accepted_registry()
    for mechanism in AUTHENTICATED_MECHANISMS:
        registry[3]["DEC-1"] = registry[3]["DEC-1"].model_copy(
            update={"receipt_authentication": mechanism}
        )
        with _test_only_trusted_engines(helpers_discharge.TRUSTED):
            resolution = _resolve(registry)
        assert resolution.qualifying == [], mechanism
        assert resolution.explain().strip(), mechanism


def test_the_empty_test_hook_authenticates_nothing():
    """Regression: the empty set used to mean "authenticate everything".

    That inverted the convention its sibling `_test_only_trusted_engines`
    follows, where empty means trust nothing. The consequence was that the
    common call -- `trusting_test_engine()` with no arguments -- bypassed the
    seam, so thirteen call sites read as though they exercised an authenticator
    that none of them reached.
    """
    registry = helpers_discharge.accepted_registry()
    with _test_only_trusted_engines(helpers_discharge.TRUSTED):
        with _test_only_authenticated_receipts(frozenset()):
            resolution = _resolve(registry)
    assert resolution.qualifying == []
    # Falls through to the real signature backend, which refuses: the fixture
    # receipt declares the mechanism and carries no signature.
    assert "carries no signature" in resolution.explain()


def test_authentication_is_bound_to_the_receipt_not_to_its_declaration():
    """Naming a *different* receipt does not authenticate this one.

    This is what a no-op authenticator would not reproduce: it distinguishes
    "the seam ran and matched" from "the seam did not run".
    """
    registry = helpers_discharge.accepted_registry()
    unrelated = DreReceipt(
        decision_ref="DEC-OTHER",
        dre_decision_hash="e" * 64,
        dre_input_hash="a" * 64,
        dre_model_pack_hash="b" * 64,
        dre_engine_fingerprint=helpers_discharge.TEST_ENGINE,
        dre_proof_hash="c" * 64,
        receipt_authentication=ReceiptAuthentication.SIGNATURE,
    )
    with _test_only_trusted_engines(helpers_discharge.TRUSTED):
        with _test_only_authenticated_receipts({unrelated.receipt_hash()}):
            resolution = _resolve(registry)
    assert resolution.qualifying == []


def test_an_authenticated_receipt_stops_authenticating_once_it_is_edited():
    """The binding is the content hash, so editing the receipt breaks it.

    A receipt authenticated by identity rather than by content would let an
    accepted work product be swapped underneath its own certification.
    """
    registry = helpers_discharge.accepted_registry()
    original = registry[3]["DEC-1"]

    with helpers_discharge.trusting_registry(registry):
        assert _resolve(registry).qualifying == ["OBL-1"]

    # Same authenticated hash set; the receipt itself now differs.
    with _test_only_trusted_engines(helpers_discharge.TRUSTED):
        with _test_only_authenticated_receipts({original.receipt_hash()}):
            registry[3]["DEC-1"] = original.model_copy(
                update={"dre_input_hash": "f" * 64}
            )
            resolution = _resolve(registry)
    assert resolution.qualifying == []


def test_the_authenticator_is_not_reachable_through_any_public_name():
    """A caller that can call the authenticator can decide its own answer."""
    from rh_research_engine.contracts import receipts

    assert set(receipts.__all__) == {
        "DreReceipt",
        "ReceiptAuthentication",
        "ReceiptError",
        "activation_status",
    }
    for name in (
        "_authenticate",
        "_authenticate_signature",
        "_authenticate_sealed_store",
        "_authenticate_deterministic_replay",
        "_test_only_authenticated_receipts",
    ):
        assert name.startswith("_"), name
        assert name not in receipts.__all__


def test_production_authenticates_nothing_at_all():
    """The Phase 1 posture, with no test hook open anywhere."""
    from rh_research_engine.contracts import receipts

    assert receipts._TEST_ONLY_AUTHENTICATED_RECEIPTS == frozenset()
    assert _resolve(helpers_discharge.accepted_registry()).qualifying == []
