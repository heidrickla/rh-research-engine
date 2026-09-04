from __future__ import annotations

from ..contracts.epistemic import Confidence
from ..properties.extract import stable_property_id
from ..properties.inventory import stable_object_id
from ..properties.models import (
    EpistemicStatus,
    MathematicalRole,
    PropertyKind,
    PropertyRecord,
    Provenance,
)
from .models import Hypothesis


def extract_from_hypothesis(hypothesis: Hypothesis) -> list[PropertyRecord]:
    """Expose supervisor hypotheses to the property graph without promotion."""

    object_id = stable_object_id(hypothesis.id)
    prov = Provenance(
        source_type="hypothesis",
        source_id=hypothesis.id,
        method="supervisor-hypothesis-extractor",
        assumptions=list(hypothesis.assumptions),
        evidence_hash=hypothesis.stable_hash(),
    )
    # The epistemic axis, and only the epistemic axis.
    #
    # An earlier version also mapped `lifecycle is BLOCKED` to BLOCKED, which
    # re-created the exact conflation the split removed: "blocked" is a
    # workflow position meaning work cannot proceed *yet*, usually because a
    # dependency is unresolved. It says nothing about whether the statement is
    # true, and turning it into a refutation would let a scheduling fact
    # permanently mark a live hypothesis as a dead route.
    status = (
        EpistemicStatus.BLOCKED
        if hypothesis.epistemic_status is Confidence.REFUTED
        else EpistemicStatus.SYMBOLIC_DERIVED
    )
    kind = PropertyKind.EQUIVALENCE if hypothesis.rh_equivalent else PropertyKind.INVARIANT
    return [
        PropertyRecord(
            id=stable_property_id(object_id, kind, hypothesis.statement, hypothesis.id),
            object_id=object_id,
            kind=kind,
            value=hypothesis.statement,
            status=status,
            provenance=[prov],
            assumptions=list(hypothesis.assumptions),
            conditions=[gap.description for gap in hypothesis.open_proof_gaps],
            role=MathematicalRole.EQUIVALENCE if hypothesis.rh_equivalent else MathematicalRole.CLAIM,
            rh_equivalent=hypothesis.rh_equivalent,
            discharged_obligations=list(hypothesis.discharged_obligations),
            # Inputs only. `frontier_relevant`, `advances_frontier` and
            # `actionable` used to be copied in here, which put a stored
            # snapshot of a derived verdict next to the facts it derives from --
            # two representations of one thing, free to drift the moment either
            # side is edited. The record already carries every input, so the
            # reader derives the verdict rather than trusting a stale copy.
            metadata={
                "lifecycle": hypothesis.lifecycle.value,
                "epistemic_status": hypothesis.epistemic_status.value,
            },
        )
    ]
