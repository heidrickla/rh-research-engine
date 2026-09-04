"""Resolving discharge claims against DRE-accepted decisions.

A discharge is an *authority* question, and the earlier version answered it by
inspecting labels: an artifact counted if its own ``epistemic_status`` was in
``RIGOROUS``, it was unconditional, and it was not RH-equivalent. Two holes
followed from that.

**Labels are not credentials.** Only ``proved``, ``known`` and
``formally_verified`` are worker-forbidden, so a worker can construct
``CERTIFIED`` or ``RIGOROUS_DERIVED`` evidence for itself. ``created_by`` is a
free string, so ``"external-verifier"`` is a claim, not a credential. Either
route let a worker mint its own discharge.

**Rigour is not relevance.** A perfectly rigorous artifact about something
unrelated satisfied every check, because ``discharge_requires`` was never
consulted. Nothing tied the evidence to the obligation it was supposed to close,
or said which direction of an equivalence it closed.

So the authority is now an :class:`ObligationDischargeDecision` -- DRE's ruling,
binding the obligation hash, the evidence hashes, the direction, and the
requirements it satisfies. Content-bound, so swapping the evidence under an
accepted decision invalidates it instead of inheriting its authority.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import (
    Artifact,
    ObligationDischargeDecision,
    ObligationStatus,
    ProofObligation,
)
from .epistemic import RIGOROUS, Confidence
from .receipts import DreReceipt, _verify_all


class DischargeProvenance(BaseModel):
    """The complete replay identity of one accepted discharge.

    Written into the derived record, not held in a transient registry. A
    conclusion that is rigorous *because* a circular premise was discharged has
    to carry the reason with it: a reader looking at the stored graph alone
    otherwise sees a rigorous derived bound with nothing explaining why the
    equivalence was allowed to drive it, and no way to re-check the ruling.

    Every field here is part of the artifact identity of something -- the
    obligation, the evidence, the decision, the receipt, or the DRE run. None of
    them is a label.
    """

    model_config = ConfigDict(extra="forbid")

    obligation_ref: str
    obligation_hash: str
    evidence_refs: list[str] = Field(default_factory=list)
    evidence_hashes: list[str] = Field(default_factory=list)
    discharged_direction: str
    requirements_satisfied: list[str] = Field(default_factory=list)
    decision_ref: str
    decision_hash: str
    receipt_ref: str
    receipt_hash: str
    dre_engine_fingerprint: str
    dre_model_pack_hash: str
    dre_input_hash: str
    dre_proof_hash: str
    receipt_authentication: str


class DischargeVerdict(BaseModel):
    """Whether one discharge reference actually discharges anything."""

    reference: str
    resolved: bool
    qualifies: bool
    reason: str
    #: Set when the discharge counted. Recorded on derived properties so the
    #: serialized result explains why it is rigorous, not merely that it is.
    receipt_hash: str | None = None
    engine_fingerprint: str | None = None
    #: The full replay identity, for persisting into derived provenance.
    provenance: DischargeProvenance | None = None

    @property
    def counts(self) -> bool:
        return self.resolved and self.qualifies


class DischargeResolution(BaseModel):
    """The result of resolving every discharge reference on a record."""

    verdicts: list[DischargeVerdict] = Field(default_factory=list)

    @property
    def qualifying(self) -> list[str]:
        return [v.reference for v in self.verdicts if v.counts]

    @property
    def unresolved(self) -> list[str]:
        return [v.reference for v in self.verdicts if not v.counts]

    @property
    def receipts(self) -> dict[str, str]:
        """Obligation reference -> receipt hash, for the qualifying discharges."""
        return {
            v.reference: v.receipt_hash
            for v in self.verdicts
            if v.counts and v.receipt_hash is not None
        }

    @property
    def provenance(self) -> dict[str, DischargeProvenance]:
        """Obligation reference -> full replay identity, for the ones that count."""
        return {
            v.reference: v.provenance
            for v in self.verdicts
            if v.counts and v.provenance is not None
        }

    def provenance_payload(self) -> dict[str, dict]:
        """The same, JSON-ready, for writing into a stored record."""
        return {
            reference: record.model_dump(mode="json")
            for reference, record in sorted(self.provenance.items())
        }

    def provenance_note(self) -> list[str]:
        """Human-readable attribution for each discharge that counted."""
        return [
            f"discharge {v.reference} authorized by DRE engine "
            f"{v.engine_fingerprint} (receipt {v.receipt_hash[:16]}...)"
            for v in self.verdicts
            if v.counts and v.receipt_hash is not None
        ]

    def explain(self) -> str:
        if not self.verdicts:
            return "no discharge references supplied"
        return "; ".join(f"{v.reference}: {v.reason}" for v in self.verdicts)


def _evidence_still_disqualifies(evidence: Artifact) -> str | None:
    """Defence in depth, after the decision has already been checked.

    DRE is the authority, but these two would be errors in the ruling itself,
    and it costs nothing to refuse them here as well.
    """
    if evidence.rh_equivalent:
        return (
            f"evidence {evidence.artifact_id!r} is itself RH-equivalent, so using it "
            "to discharge an RH-equivalence closes a loop rather than a gap"
        )
    if evidence.epistemic_status not in RIGOROUS:
        return (
            f"evidence {evidence.artifact_id!r} is "
            f"{evidence.epistemic_status.value!r}, which carries no deductive force"
        )
    if evidence.open_qualifiers:
        return (
            f"evidence {evidence.artifact_id!r} is still conditional on "
            f"{evidence.open_qualifiers}"
        )
    return None


def _check_decision(
    reference: str,
    obligation: ProofObligation,
    verified,
    evidence_index: dict[str, Artifact],
    rejection: str | None = None,
) -> tuple[bool, str]:
    if verified is None:
        return False, rejection or (
            "no DRE decision covers this obligation. A raw "
            "ObligationDischargeDecision is an ordinary artifact whose "
            "dre_decision_status and created_by are caller-supplied, so a record "
            "saying 'accepted' is a claim; only a receipt authenticated against a "
            "trusted engine makes it authority"
        )
    decision = verified.decision

    # Content binding. Names are cheap; hashes pin what was actually ruled on.
    actual_obligation_hash = obligation.artifact_hash()
    if decision.obligation_hash != actual_obligation_hash:
        return False, (
            f"decision {decision.artifact_id!r} was issued against a different "
            f"version of {reference!r} (bound {decision.obligation_hash[:16]}..., "
            f"actual {actual_obligation_hash[:16]}...); the obligation changed after "
            "it was ruled on"
        )

    # Direction. An equivalence has two, and closing the other one closes
    # nothing -- but a decision naming *some* direction reads as directional, so
    # without the obligation stating which it needs, any direction passed.
    required = obligation.required_direction
    if required is not None and decision.discharged_direction != required:
        return False, (
            f"decision {decision.artifact_id!r} discharges "
            f"{decision.discharged_direction!r}, but {reference!r} needs "
            f"{required!r}; closing the converse of an equivalence leaves the "
            "direction that was actually asked for still open"
        )

    # Every requirement the obligation states must be covered.
    outstanding = [
        requirement
        for requirement in obligation.discharge_requires
        if requirement not in decision.requirements_satisfied
    ]
    if outstanding:
        return False, (
            f"decision {decision.artifact_id!r} leaves {outstanding} unsatisfied; "
            "rigour is not relevance, and an obligation is closed only when what it "
            "asked for is what was supplied"
        )

    if not decision.evidence_refs:
        return False, f"decision {decision.artifact_id!r} binds no evidence"

    for reference_id, bound_hash in zip(
        decision.evidence_refs, decision.evidence_hashes, strict=True
    ):
        artifact = evidence_index.get(reference_id)
        if artifact is None:
            return False, (
                f"evidence {reference_id!r} named by the decision is not registered"
            )
        actual = artifact.artifact_hash()
        if actual != bound_hash:
            return False, (
                f"evidence {reference_id!r} changed since the decision "
                f"(bound {bound_hash[:16]}..., actual {actual[:16]}...); swapping "
                "evidence under an accepted decision does not inherit its authority"
            )
        disqualified = _evidence_still_disqualifies(artifact)
        if disqualified is not None:
            return False, disqualified

    return True, (
        f"decision {decision.artifact_id!r} verified against DRE engine "
        f"{verified.engine_fingerprint} (receipt {verified.receipt_hash[:16]}...), "
        f"discharging {decision.discharged_direction!r}"
    )


def _provenance(obligation: ProofObligation, verified) -> DischargeProvenance:
    """Everything needed to re-check the ruling, from the verified decision."""
    decision = verified.decision
    receipt = verified.receipt
    return DischargeProvenance(
        obligation_ref=obligation.artifact_id,
        obligation_hash=obligation.artifact_hash(),
        evidence_refs=list(decision.evidence_refs),
        evidence_hashes=list(decision.evidence_hashes),
        discharged_direction=decision.discharged_direction,
        requirements_satisfied=list(decision.requirements_satisfied),
        decision_ref=decision.artifact_id,
        decision_hash=decision.artifact_hash(),
        receipt_ref=receipt.decision_ref,
        receipt_hash=verified.receipt_hash,
        dre_engine_fingerprint=receipt.dre_engine_fingerprint,
        dre_model_pack_hash=receipt.dre_model_pack_hash,
        dre_input_hash=receipt.dre_input_hash,
        dre_proof_hash=receipt.dre_proof_hash,
        receipt_authentication=receipt.receipt_authentication.value,
    )


def resolve_discharges(
    references: list[str],
    *,
    obligations: dict[str, ProofObligation] | None = None,
    evidence: dict[str, Artifact] | None = None,
    decisions: dict[str, ObligationDischargeDecision] | None = None,
    receipts: dict[str, DreReceipt] | None = None,
) -> DischargeResolution:
    """Resolve discharge references against obligations, evidence and receipts.

    ``decisions`` and ``receipts`` are *raw* records, verified here rather than
    on arrival. An earlier version took a pre-verified value, which was the
    bypass: the type was a promise nothing enforced, so a caller could construct
    one and skip the trust check entirely.

    With no registry nothing resolves. The safe reading of "I cannot check this
    claim" is that the claim does not hold, so a caller without the registries
    gets no frontier credit rather than the benefit of the doubt.

    Keyed by the obligation reference, so a lookup is by obligation rather than
    by decision ID.
    """
    obligation_index = obligations or {}
    evidence_index = evidence or {}
    # Verified here, at the point of use. An earlier version accepted a
    # pre-verified value, which was a bypass: the type was a promise nothing
    # enforced, so a caller could construct one and skip the trust check.
    verified_index, rejections = _verify_all(decisions or {}, receipts or {})
    verdicts: list[DischargeVerdict] = []

    for reference in references:
        obligation = obligation_index.get(reference)
        if obligation is None:
            verdicts.append(
                DischargeVerdict(
                    reference=reference,
                    resolved=False,
                    qualifies=False,
                    reason=(
                        "does not resolve to a ProofObligation artifact; a prose "
                        "description of a discharge is an assertion, not a discharge"
                    ),
                )
            )
            continue
        if obligation.status is not ObligationStatus.DISCHARGED:
            verdicts.append(
                DischargeVerdict(
                    reference=reference,
                    resolved=True,
                    qualifies=False,
                    reason=f"obligation is {obligation.status.value!r}, not discharged",
                )
            )
            continue
        if not obligation.discharge_evidence:
            verdicts.append(
                DischargeVerdict(
                    reference=reference,
                    resolved=True,
                    qualifies=False,
                    reason="obligation names no discharge evidence",
                )
            )
            continue

        verified = verified_index.get(reference)
        qualifies, reason = _check_decision(
            reference, obligation, verified, evidence_index, rejections.get(reference)
        )
        counts = qualifies and verified is not None
        verdicts.append(
            DischargeVerdict(
                reference=reference,
                resolved=True,
                qualifies=qualifies,
                reason=reason,
                receipt_hash=verified.receipt_hash if counts else None,
                engine_fingerprint=verified.engine_fingerprint if counts else None,
                provenance=_provenance(obligation, verified) if counts else None,
            )
        )

    return DischargeResolution(verdicts=verdicts)


def qualifying_discharges(
    references: list[str],
    *,
    obligations: dict[str, ProofObligation] | None = None,
    evidence: dict[str, Artifact] | None = None,
    decisions: dict[str, ObligationDischargeDecision] | None = None,
    receipts: dict[str, DreReceipt] | None = None,
) -> list[str]:
    """Just the references that actually discharge something."""
    return resolve_discharges(
        references,
        obligations=obligations,
        evidence=evidence,
        decisions=decisions,
        receipts=receipts,
    ).qualifying


#: Confidences that can never close a proof obligation, for callers that want to
#: check before building an artifact. Note this is necessary, not sufficient:
#: a rigorous status still needs an accepted DRE decision behind it.
NON_DISCHARGING = frozenset(c for c in Confidence if c not in RIGOROUS)
