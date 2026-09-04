from __future__ import annotations

from fractions import Fraction

from ..contracts.frontier import assess
from ..contracts.roles import META_ROLES
from ..core.bounds import THETA_FLOOR
from .inventory import stable_object_id
from .models import (
    ClosureMode,
    EpistemicStatus,
    ImplicationRule,
    PropertyEdge,
    PropertyGraph,
    PropertyKind,
    PropertyRecord,
    Provenance,
)

DEFAULT_RULES = [
    ImplicationRule(
        id="RH-PROP-001",
        premise_kind=PropertyKind.GROWTH_BOUND,
        conclusion_kind=PropertyKind.THETA_BOUND,
        description="A rigorous R_q(X)=O(X^theta) style remainder bound implies a Theta upper-bound relation.",
    )
]


def _resolve(prop, obligations, evidence, decisions, receipts):
    from ..contracts.discharge import resolve_discharges

    return resolve_discharges(
        prop.discharged_obligations,
        obligations=obligations,
        evidence=evidence,
        decisions=decisions,
        receipts=receipts,
    )


def implication_closure(
    graph: PropertyGraph,
    *,
    mode: ClosureMode = ClosureMode.RIGOROUS,
    rules: list[ImplicationRule] | None = None,
    obligations: dict | None = None,
    evidence: dict | None = None,
    decisions: dict | None = None,
    receipts: dict | None = None,
) -> PropertyGraph:
    """Close the graph under the implication rules.

    The discharge registries are threaded through to the frontier assessment.
    Without them this called ``prop.frontier`` with nothing, so no discharge
    could ever resolve and a legitimately discharged equivalence was as
    permanently blocked as an undischarged one -- the gate was not strict, it
    was stuck.
    """
    rules = rules or DEFAULT_RULES
    properties = {item.id: item for item in graph.properties}
    edges = list(graph.edges)
    for prop in sorted(graph.properties, key=lambda item: item.id):
        for rule in rules:
            if prop.kind != rule.premise_kind:
                continue
            # Meta records carry no mathematical content and imply nothing.
            if prop.canonical_role in META_ROLES:
                continue

            # `produces_frontier_claim` is the actual gate, not a label.
            #
            # A frontier rule asserts the RH frontier moved, so its premise must
            # itself *advance the frontier* -- rigorous, unconditional, and not a
            # restatement of RH. Requiring only rigour was too weak: a classical
            # `A <=> RH` is rigorous, and letting it drive a frontier rule is
            # exactly the circular step. A rewriting rule is held to the weaker
            # `is_usable_as_rule`, so an established equivalence remains usable
            # as a transformation while earning nothing.
            resolution = _resolve(prop, obligations, evidence, decisions, receipts)
            assessment = assess(
                role=prop.canonical_role,
                confidence=prop.confidence,
                rh_equivalent=prop.rh_equivalent,
                qualifying_discharges=resolution.qualifying,
                open_qualifiers=prop.open_qualifiers,
            )
            if rule.produces_frontier_claim:
                if mode is ClosureMode.RIGOROUS and not assessment.advances_frontier:
                    continue
                if mode is ClosureMode.EXPLORATORY and not assessment.frontier_relevant:
                    continue
            elif mode is ClosureMode.RIGOROUS and not prop.is_usable_as_rule:
                continue
            if rule.id == "RH-PROP-001":
                theta = _extract_exponent(prop.value)
                if theta is None:
                    if mode is ClosureMode.EXPLORATORY:
                        theta = "theta"
                    else:
                        continue
                if not theta_is_possible(theta):
                    # Implies Theta < 1/2, which is impossible. Emitting it
                    # would publish a provably false bound as a derived result.
                    continue
                status = (
                    EpistemicStatus.RIGOROUS_DERIVED
                    if mode is ClosureMode.RIGOROUS
                    else EpistemicStatus.HEURISTIC
                )
                # The receipt travels with the conclusion. Without it the
                # registry changed the outcome but left no trace in the
                # serialized record, so a reader could see a rigorous derived
                # bound and find nothing explaining *why* the circular premise
                # was allowed to drive it.
                provenance = [
                    Provenance(
                        source_type="symbolic",
                        source_id=prop.id,
                        method=f"implication-closure:{rule.id}:{mode.value}",
                        assumptions=prop.open_qualifiers + resolution.provenance_note(),
                    )
                ]
                object_id = stable_object_id("Theta")
                value = f"1/2 + ({theta})/2"
                conclusion = PropertyRecord(
                    id=f"{prop.id}:closure:{rule.id}:{mode.value}",
                    object_id=object_id,
                    kind=PropertyKind.THETA_BOUND,
                    value=value,
                    status=status,
                    provenance=provenance,
                    # Both lists travel with the conclusion. A derived bound is
                    # no less conditional than the premise it came from.
                    assumptions=list(prop.assumptions),
                    conditions=list(prop.conditions),
                    role=prop.role,
                    rh_equivalent=prop.rh_equivalent,
                    discharged_obligations=list(prop.discharged_obligations),
                    metadata={
                        "premise": prop.id,
                        # Which *version* of the premise. The ID alone does not
                        # distinguish the record that was actually read from one
                        # edited afterwards.
                        "premise_hash": prop.record_hash(),
                        "rule": rule.id,
                        "mode": mode.value,
                        # Obligation reference -> DRE receipt hash, so the
                        # authority behind the discharge is auditable from the
                        # stored graph alone.
                        **(
                            {
                                "discharge_receipts": resolution.receipts,
                                # And the complete replay identity: obligation,
                                # evidence, decision, receipt, engine, model
                                # pack, and proof hashes. Held only in the
                                # caller's registries, none of it would survive
                                # the run that produced this record -- so a
                                # reader could see a rigorous bound derived from
                                # a circular premise with no way to re-check why
                                # that was allowed.
                                "discharge_provenance": resolution.provenance_payload(),
                            }
                            if resolution.receipts
                            else {}
                        ),
                    },
                )
                properties.setdefault(conclusion.id, conclusion)
                edges.append(
                    PropertyEdge(
                        source_id=prop.id,
                        target_id=conclusion.id,
                        relation="implies",
                        rule_id=rule.id,
                        status=status,
                        provenance=provenance,
                    )
                )
    return PropertyGraph(
        objects=graph.objects,
        properties=sorted(properties.values(), key=lambda item: item.id),
        edges=sorted(edges, key=lambda item: (item.source_id, item.target_id, item.relation)),
    )


def _extract_exponent(value: str) -> str | None:
    compact = value.replace(" ", "")
    if compact.startswith("X^"):
        return compact[2:]
    if compact.startswith("X**"):
        return compact[3:]
    return None


def _implied_theta(exponent: str) -> Fraction | None:
    """Numeric value of 1/2 + exponent/2, or None when the exponent is symbolic."""
    try:
        return Fraction(1, 2) + Fraction(exponent) / 2
    except (ValueError, ZeroDivisionError):
        return None


def theta_is_possible(exponent: str) -> bool:
    """Reject an exponent implying Theta < 1/2.

    Theta >= 1/2 holds unconditionally because zeta has zeros on the critical
    line, so a smaller value is not a stronger result -- it means the premise is
    invalid. The conclusion here is built as a *string* (`"1/2 + (theta)/2"`),
    so none of the numeric guards in core.bounds or symbolic.exponents sit on
    this path; the check has to happen here.

    A symbolic exponent (`"theta"`) is not rejected -- it is simply not yet a
    number to check.
    """
    implied = _implied_theta(exponent)
    return implied is None or implied >= Fraction(THETA_FLOOR).limit_denominator()
