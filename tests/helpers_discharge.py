"""Registries for testing discharge resolution.

A discharge counts only when the reference resolves to a DISCHARGED
ProofObligation *and* a DRE-accepted ObligationDischargeDecision binds the
obligation hash, the evidence hashes, the direction, and every requirement the
obligation states -- with an authenticated receipt from a trusted engine behind
it. Each helper below returns ``(obligations, evidence, decisions, receipts)``
exercising one branch. Nothing is pre-verified: `resolve_discharges` verifies at
the point of use.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager

from rh_research_engine.contracts.artifacts import (
    Artifact,
    DreDecisionStatus,
    ObligationDischargeDecision,
    ObligationStatus,
    ProofObligation,
    PropertyAssertion,
)
from rh_research_engine.contracts.epistemic import Confidence
from rh_research_engine.contracts.receipts import DreReceipt, ReceiptAuthentication

#: Tests opt in to a fake trusted engine. Production `_TRUSTED_DRE_ENGINES` is
#: empty, so nothing verifies there -- which is the point: connecting a real
#: engine is a deliberate act, not something a caller can assert.
TEST_ENGINE = "dre-test-engine@0.22.0"
#: The model pack the fixture receipts are issued under. Trust is the *pair*:
#: which engine ruled, and under which rules.
TEST_MODEL_PACK = "b" * 64
TRUSTED = {TEST_ENGINE: frozenset({TEST_MODEL_PACK})}


@contextmanager
def trusting_test_engine(receipts: Iterable[DreReceipt]):
    """Scoped widening of the private test trust roots. Tests only.

    Production has no equivalent: no public function accepts a trust set or a
    receipt-authenticator override, and this reaches for private context
    managers rather than exported hooks. Receipt certification is bound to the
    exact fixture receipt hashes, not to the mechanism string they declare.

    `receipts` is required, with no default. A default of `()` meant the common
    call authenticated everything, so every positive test read as though it
    exercised the authenticator while bypassing it. Naming the receipts makes
    each test say which work products it is standing in for a real backend on.
    """
    from rh_research_engine.contracts.receipts import (
        _test_only_authenticated_receipts,
        _test_only_trusted_engines,
    )

    receipt_hashes = {receipt.receipt_hash() for receipt in receipts}
    with _test_only_trusted_engines(TRUSTED), _test_only_authenticated_receipts(
        receipt_hashes
    ):
        yield


@contextmanager
def trusting_registry(registry):
    """`trusting_test_engine` for a whole registry tuple.

    Saves call sites from indexing into the tuple to reach the receipts, which
    is the shape that invites passing the wrong element.
    """
    with trusting_test_engine(registry[3].values()):
        yield


REQUIREMENT = "prove the forward implication without assuming RH"
DIRECTION = "A => RH"
CONVERSE = "RH => A"


def _evidence(
    artifact_id: str = "EV-1",
    *,
    confidence: Confidence = Confidence.PROVED,
    rh_equivalent: bool = False,
    conditions: list[str] | None = None,
    value: str = "forward direction",
) -> Artifact:
    return PropertyAssertion(
        artifact_id=artifact_id,
        created_by="external-verifier",
        method_family="lean",
        method_version="4.0",
        object_id="obj",
        property_kind="implication",
        value=value,
        epistemic_status=confidence,
        rh_equivalent=rh_equivalent,
        conditions=list(conditions or []),
    )


def _obligation(
    artifact_id: str = "OBL-1",
    *,
    status: ObligationStatus = ObligationStatus.DISCHARGED,
    evidence_refs: list[str] | None = None,
    requires: list[str] | None = None,
    required_direction: str | None = None,
) -> ProofObligation:
    return ProofObligation(
        artifact_id=artifact_id,
        created_by="rh-math-worker",
        method_family="python",
        method_version="0.11.0",
        statement="prove the forward direction without assuming RH",
        status=status,
        discharge_requires=list(requires if requires is not None else [REQUIREMENT]),
        discharge_evidence=list(evidence_refs if evidence_refs is not None else ["EV-1"]),
        required_direction=required_direction,
    )


def _decision(
    obligation: ProofObligation,
    evidence: list[Artifact],
    *,
    status: DreDecisionStatus = DreDecisionStatus.ACCEPTED,
    requirements: list[str] | None = None,
    obligation_hash: str | None = None,
    evidence_hashes: list[str] | None = None,
    evidence_refs: list[str] | None = None,
    direction: str = DIRECTION,
) -> ObligationDischargeDecision:
    return ObligationDischargeDecision(
        artifact_id="DEC-1",
        created_by="dre",
        method_family="dre",
        method_version="0.22.0",
        obligation_ref=obligation.artifact_id,
        obligation_hash=obligation_hash or obligation.artifact_hash(),
        evidence_refs=(
            evidence_refs
            if evidence_refs is not None
            else [e.artifact_id for e in evidence]
        ),
        evidence_hashes=(
            evidence_hashes
            if evidence_hashes is not None
            else [e.artifact_hash() for e in evidence]
        ),
        discharged_direction=direction,
        requirements_satisfied=(
            requirements if requirements is not None else [REQUIREMENT]
        ),
        dre_decision_ref="dre-run-0001",
        dre_decision_status=status,
        dre_pack_version="0.1.0",
    )


def _receipt(
    decision: ObligationDischargeDecision,
    *,
    engine: str = TEST_ENGINE,
    decision_hash: str | None = None,
    model_pack: str = TEST_MODEL_PACK,
    authentication: ReceiptAuthentication = ReceiptAuthentication.SIGNATURE,
    drop_field: str | None = None,
) -> DreReceipt:
    fields = dict(
        decision_ref=decision.artifact_id,
        dre_decision_hash=decision_hash or decision.artifact_hash(),
        dre_input_hash="a" * 64,
        dre_model_pack_hash=model_pack,
        dre_engine_fingerprint=engine,
        dre_proof_hash="c" * 64,
        receipt_authentication=authentication,
        notes=[],
    )
    if drop_field is None:
        return DreReceipt(**fields)
    # `model_construct` skips validation on purpose: the field validator already
    # refuses a blank digest at construction, so the only way to reach the
    # runtime check is to simulate a record that arrived without passing through
    # it -- from a store, a cache, or an older schema.
    fields[drop_field] = ""
    return DreReceipt.model_construct(**fields)


def _registry(obligation, decision, **receipt_kwargs):
    """Raw decision plus receipt, the shape `resolve_discharges` now takes.

    Nothing here is pre-verified: verification happens at the point of use,
    against a trust set no caller can widen. Tests open that set with the
    private `_test_only_trusted_engines` context manager.
    """
    return (
        {obligation.artifact_id: decision},
        {decision.artifact_id: _receipt(decision, **receipt_kwargs)},
    )


def _standard(
    *,
    decision_kwargs: dict | None = None,
    receipt_kwargs: dict | None = None,
    obligation: ProofObligation | None = None,
    evidence: Artifact | None = None,
):
    """The common shape: one obligation, one evidence artifact, one decision."""
    obligation = obligation if obligation is not None else _obligation()
    evidence = evidence if evidence is not None else _evidence()
    decision = _decision(obligation, [evidence], **(decision_kwargs or {}))
    return (
        {obligation.artifact_id: obligation},
        {evidence.artifact_id: evidence},
        *_registry(obligation, decision, **(receipt_kwargs or {})),
    )


def accepted_registry():
    """The one combination that actually discharges."""
    return _standard()


def unreceipted_decision_registry():
    """An accepted-looking decision with no receipt behind it.

    This is what every test used to construct, and what a worker could mint.
    """
    obligation, evidence = _obligation(), _evidence()
    return {obligation.artifact_id: obligation}, {evidence.artifact_id: evidence}, {}, {}


def untrusted_engine_registry():
    """A receipt from an engine this build does not trust."""
    return _standard(receipt_kwargs={"engine": "some-other-engine@9.9.9"})


def invented_fingerprint_registry():
    """A fingerprint that reads like the real one but was never registered.

    Plausibility is not membership: the string travels inside the receipt, so
    anything able to write a receipt can write a convincing engine name.
    """
    return _standard(receipt_kwargs={"engine": TEST_ENGINE.replace("0.22.0", "0.22.1")})


def unpinned_model_pack_registry():
    """A trusted engine ruling under a model pack the build never pinned.

    The engine is genuine; the rules it applied are not the ones this build
    accepts rulings under. Naming a pack is not the same as having been judged
    by it.
    """
    return _standard(receipt_kwargs={"model_pack": "9" * 64})


def unauthenticated_receipt_registry():
    """A receipt from a trusted engine that nothing actually authenticated.

    The Phase 1 stand-in for a failed replay: no signature, no sealed-store
    retrieval, no reproduction of `dre_proof_hash`.
    """
    return _standard(receipt_kwargs={"authentication": ReceiptAuthentication.NONE})


def stale_receipt_registry():
    """A receipt bound to a different version of the decision."""
    return _standard(receipt_kwargs={"decision_hash": "0" * 64})


def missing_input_hash_registry():
    return _standard(receipt_kwargs={"drop_field": "dre_input_hash"})


def missing_model_pack_hash_registry():
    return _standard(receipt_kwargs={"drop_field": "dre_model_pack_hash"})


def missing_proof_hash_registry():
    return _standard(receipt_kwargs={"drop_field": "dre_proof_hash"})


def no_decision_registry():
    """Obligation closed, evidence rigorous -- but nobody ruled on it."""
    obligation, evidence = _obligation(), _evidence()
    return {obligation.artifact_id: obligation}, {evidence.artifact_id: evidence}, {}, {}


def pending_decision_registry():
    """A decision DRE has not accepted."""
    return _standard(decision_kwargs={"status": DreDecisionStatus.PENDING})


def rejected_decision_registry():
    """A decision DRE explicitly refused."""
    return _standard(decision_kwargs={"status": DreDecisionStatus.REJECTED})


def stale_obligation_binding_registry():
    """The decision was issued against an earlier version of the obligation."""
    return _standard(decision_kwargs={"obligation_hash": "0" * 64})


def stale_evidence_hash_registry():
    """The decision binds a digest that is not the evidence's current one."""
    return _standard(decision_kwargs={"evidence_hashes": ["d" * 64]})


def missing_evidence_registry():
    """The decision names evidence that is not registered at all."""
    obligation, evidence = _obligation(), _evidence()
    decision = _decision(obligation, [evidence], evidence_refs=["EV-MISSING"])
    return (
        {obligation.artifact_id: obligation},
        {evidence.artifact_id: evidence},
        *_registry(obligation, decision),
    )


def wrong_direction_registry():
    """The converse was discharged; the direction that was asked for is open."""
    return _standard(
        obligation=_obligation(required_direction=DIRECTION),
        decision_kwargs={"direction": CONVERSE},
    )


def unsatisfied_requirement_registry():
    """Rigour without relevance: the decision does not cover what was asked."""
    obligation = _obligation(requires=[REQUIREMENT, "bound the error term uniformly"])
    return _standard(
        obligation=obligation, decision_kwargs={"requirements": [REQUIREMENT]}
    )


def swapped_evidence_registry():
    """Accepted decision, then the evidence was replaced underneath it."""
    obligation, original = _obligation(), _evidence()
    decision = _decision(obligation, [original])
    swapped = _evidence(value="something else entirely")
    return (
        {obligation.artifact_id: obligation},
        {swapped.artifact_id: swapped},
        *_registry(obligation, decision),
    )


def worker_minted_evidence_registry():
    """A worker labelling its own output CERTIFIED, with no DRE decision.

    `CERTIFIED` is not worker-forbidden, so the artifact constructs; the point
    is that constructing it confers no authority.
    """
    obligation = _obligation()
    evidence = PropertyAssertion(
        artifact_id="EV-1",
        created_by="rh-math-worker",
        method_family="python-numpy",
        method_version="0.11.0",
        object_id="obj",
        property_kind="implication",
        value="forward direction",
        epistemic_status=Confidence.CERTIFIED,
    )
    return {obligation.artifact_id: obligation}, {evidence.artifact_id: evidence}, {}, {}


def numerical_evidence_registry():
    return _standard(evidence=_evidence(confidence=Confidence.NUMERICAL))


def rigorous_numerical_evidence_registry():
    """A certified enclosure over a finite computation is not a theorem."""
    return _standard(evidence=_evidence(confidence=Confidence.RIGOROUS_NUMERICAL))


def synthetic_evidence_registry():
    """True of a constructed model, which is not the same as true of zeta."""
    return _standard(evidence=_evidence(confidence=Confidence.SYNTHETIC))


def conditional_evidence_registry():
    """Rigorous, but still resting on an unproved hypothesis."""
    return _standard(evidence=_evidence(conditions=["assumes a Lindelof-type bound"]))


def circular_evidence_registry():
    return _standard(evidence=_evidence(rh_equivalent=True))


def open_obligation_registry():
    obligation = _obligation(status=ObligationStatus.OPEN, evidence_refs=[])
    evidence = _evidence()
    return {obligation.artifact_id: obligation}, {evidence.artifact_id: evidence}, {}, {}


#: Every registry that must never yield a discharge under the production trust
#: set, plus `accepted_registry` -- included deliberately, because the one
#: combination that works under the scoped test trust set has to fail without
#: it, or the test set is not what is doing the work.
ALL_REGISTRIES = [
    "accepted_registry",
    "unreceipted_decision_registry",
    "untrusted_engine_registry",
    "invented_fingerprint_registry",
    "unpinned_model_pack_registry",
    "unauthenticated_receipt_registry",
    "stale_receipt_registry",
    "missing_input_hash_registry",
    "missing_model_pack_hash_registry",
    "missing_proof_hash_registry",
    "no_decision_registry",
    "pending_decision_registry",
    "rejected_decision_registry",
    "stale_obligation_binding_registry",
    "stale_evidence_hash_registry",
    "missing_evidence_registry",
    "wrong_direction_registry",
    "unsatisfied_requirement_registry",
    "swapped_evidence_registry",
    "worker_minted_evidence_registry",
    "numerical_evidence_registry",
    "rigorous_numerical_evidence_registry",
    "synthetic_evidence_registry",
    "conditional_evidence_registry",
    "circular_evidence_registry",
    "open_obligation_registry",
]

#: The subset that must be refused even with a trusted engine connected, paired
#: with the phrase the refusal has to contain. A gate that fails without saying
#: which check failed sends a researcher looking in the wrong place.
REFUSAL_REASONS = {
    "no_decision_registry": "no DRE decision covers this obligation",
    "unreceipted_decision_registry": "no DRE decision covers this obligation",
    "untrusted_engine_registry": "not trusted by this build",
    "invented_fingerprint_registry": "not trusted by this build",
    "unpinned_model_pack_registry": "not for model pack",
    "unauthenticated_receipt_registry": "receipt_authentication",
    "stale_receipt_registry": "edited after the ruling",
    "missing_input_hash_registry": "has no dre_input_hash",
    "missing_model_pack_hash_registry": "has no dre_model_pack_hash",
    "missing_proof_hash_registry": "has no dre_proof_hash",
    "pending_decision_registry": "not accepted",
    "rejected_decision_registry": "not accepted",
    "stale_obligation_binding_registry": "different version",
    "stale_evidence_hash_registry": "changed since the decision",
    "missing_evidence_registry": "is not registered",
    "wrong_direction_registry": "closing the converse",
    "unsatisfied_requirement_registry": "rigour is not relevance",
    "swapped_evidence_registry": "changed since the decision",
    "worker_minted_evidence_registry": "no DRE decision covers this obligation",
    "numerical_evidence_registry": "no deductive force",
    "rigorous_numerical_evidence_registry": "no deductive force",
    "synthetic_evidence_registry": "no deductive force",
    "conditional_evidence_registry": "still conditional on",
    "circular_evidence_registry": "closes a loop",
    "open_obligation_registry": "not discharged",
}
