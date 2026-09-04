from __future__ import annotations

import hashlib
import re

from ..core.knowledge import KnowledgeItem, KnowledgeStatus
from ..symbolic import fingerprint, simplify_with_trace
from ..symbolic.formula_index import FormulaRecord
from .inventory import stable_object_id
from .models import (
    EpistemicStatus,
    MathematicalRole,
    PropertyKind,
    PropertyRecord,
    Provenance,
)

_BIG_O = re.compile(r"(?P<object>[A-Za-z][A-Za-z0-9_]*(?:_[A-Za-z0-9]+)?)\s*=\s*O\((?P<value>[^)]+)\)")
_THETA = re.compile(r"Theta\s*(?:<=|\\le)\s*(?P<value>[0-9./ +*\-theta]+)")


def stable_property_id(object_id: str, kind: PropertyKind, value: str, source_id: str) -> str:
    payload = f"{object_id}|{kind.value}|{value}|{source_id}"
    return "prop:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def extract_from_formula(record: FormulaRecord) -> list[PropertyRecord]:
    prov = Provenance(
        source_type="formula_index",
        source_id=record.id,
        source_ref=record.source,
        method="formula-property-extractor",
    )
    properties: list[PropertyRecord] = []
    expression = record.expression
    try:
        fp = fingerprint(record.lhs or expression)
        conditions = list(fp.metadata.get("domain_conditions", []))
    except Exception as exc:
        conditions = [f"domain extraction failed: {type(exc).__name__}"]

    object_name = (record.lhs or expression).split("(", 1)[0].split("=", 1)[0].strip()
    if object_name:
        object_id = stable_object_id(object_name)
        value = " and ".join(conditions) if conditions else "entire-expression-domain"
        properties.append(
            PropertyRecord(
                id=stable_property_id(object_id, PropertyKind.DOMAIN, value, record.id),
                object_id=object_id,
                kind=PropertyKind.DOMAIN,
                value=value,
                status=EpistemicStatus.SYMBOLIC_DERIVED,
                provenance=[prov],
                conditions=conditions,
            )
        )

    for match in _BIG_O.finditer(expression):
        object_id = stable_object_id(match.group("object"))
        value = match.group("value").strip()
        properties.append(
            PropertyRecord(
                id=stable_property_id(object_id, PropertyKind.GROWTH_BOUND, value, record.id),
                object_id=object_id,
                kind=PropertyKind.GROWTH_BOUND,
                value=value,
                status=EpistemicStatus.SYMBOLIC_DERIVED,
                provenance=[prov],
                conditions=conditions,
            )
        )

    if record.kind == "equation" and record.lhs and record.rhs:
        try:
            result = simplify_with_trace(f"({record.lhs})-({record.rhs})")
            if result.simplified in {"0", "0.0"}:
                properties.append(
                    PropertyRecord(
                        id=stable_property_id(
                            stable_object_id(record.lhs),
                            PropertyKind.EQUIVALENCE,
                            record.rhs,
                            record.id,
                        ),
                        object_id=stable_object_id(record.lhs),
                        kind=PropertyKind.EQUIVALENCE,
                        value=record.rhs,
                        status=EpistemicStatus.SYMBOLIC_DERIVED,
                        provenance=[prov],
                        assumptions=result.assumptions,
                    )
                )
        except Exception:
            pass
    return properties


def extract_from_knowledge(item: KnowledgeItem) -> list[PropertyRecord]:
    if not is_property_extractable(item):
        # A meta-rule is not a statement about zeta; minting a property from it
        # would let later code mistake policy for mathematics.
        return []
    status = _status_from_knowledge(item.status)
    role = _role_from_knowledge(item.status)
    assumptions = _assumptions_for(item)
    rh_equivalent = item.status in _RH_EQUIVALENT
    prov = Provenance(
        source_type="knowledge",
        source_id=item.id,
        source_ref=item.domain,
        method="knowledge-property-extractor",
        assumptions=assumptions,
    )
    properties: list[PropertyRecord] = []
    text = " ".join([item.statement, *item.formulas])
    for match in _BIG_O.finditer(text):
        object_id = stable_object_id(match.group("object"))
        value = match.group("value").strip()
        properties.append(
            PropertyRecord(
                id=stable_property_id(object_id, PropertyKind.GROWTH_BOUND, value, item.id),
                object_id=object_id,
                kind=PropertyKind.GROWTH_BOUND,
                value=value,
                status=status,
                provenance=[prov],
                assumptions=list(assumptions),
                role=role,
                rh_equivalent=rh_equivalent,
            )
        )
    theta_match = _THETA.search(text)
    if theta_match:
        object_id = stable_object_id("Theta")
        value = theta_match.group("value").strip()
        properties.append(
            PropertyRecord(
                id=stable_property_id(object_id, PropertyKind.THETA_BOUND, value, item.id),
                object_id=object_id,
                kind=PropertyKind.THETA_BOUND,
                value=value,
                status=status,
                provenance=[prov],
                assumptions=list(assumptions),
                role=role,
                rh_equivalent=rh_equivalent,
            )
        )
    return properties


#: Explicit KnowledgeStatus -> EpistemicStatus mapping.
#:
#: This was substring matching (`"derived" in status.value`,
#: `status.value.startswith("exact")`), which promoted 14 of 21 statuses to
#: rigorous -- including `derived_symbolic`, which
#: docs/RH_MATHEMATICAL_MEMORY.md defines as "derived in this research but not
#: yet independently formalized or literature-checked end to end", and
#: `derived_symbolic_needs_external_check`, whose name says the check has not
#: happened. Substring rules also mean any status added later is classified by
#: accident of spelling.
#:
#: Only two families are rigorous here: identities checked algebraically from
#: stated definitions, and established external mathematics. Everything derived
#: inside this research is `symbolic_derived` until something outside it says
#: otherwise.
_STATUS_MAP: dict[KnowledgeStatus, EpistemicStatus] = {
    # "identity checked algebraically from the stated definitions"
    KnowledgeStatus.EXACT: EpistemicStatus.CERTIFIED,
    KnowledgeStatus.EXACT_ALGEBRA: EpistemicStatus.CERTIFIED,
    KnowledgeStatus.EXACT_CALCULUS: EpistemicStatus.CERTIFIED,
    KnowledgeStatus.EXACT_CONSTRUCTION: EpistemicStatus.CERTIFIED,
    KnowledgeStatus.EXACT_DISTRIBUTIONAL: EpistemicStatus.CERTIFIED,
    # Exact *within a model*. The algebra is exact; the model is conjectural,
    # so the result is not a rigorous fact about zeta.
    KnowledgeStatus.EXACT_ALGEBRA_IN_MODEL: EpistemicStatus.SYMBOLIC_DERIVED,
    # "established external mathematics"
    KnowledgeStatus.KNOWN: EpistemicStatus.KNOWN,
    KnowledgeStatus.KNOWN_FRAMEWORK: EpistemicStatus.KNOWN,
    KnowledgeStatus.KNOWN_OR_STANDARD_CONSEQUENCE: EpistemicStatus.KNOWN,
    # A known *equivalence* framework is a restatement of RH; see _RH_EQUIVALENT.
    KnowledgeStatus.KNOWN_EQUIVALENCE_FRAMEWORK: EpistemicStatus.KNOWN,
    # An input to a conjectural model is not established mathematics.
    KnowledgeStatus.KNOWN_MODEL_INPUT: EpistemicStatus.HEURISTIC,
    # Derived in this research, not independently checked.
    KnowledgeStatus.DERIVED_SYMBOLIC: EpistemicStatus.SYMBOLIC_DERIVED,
    KnowledgeStatus.DERIVED_SYMBOLIC_NEEDS_EXTERNAL_CHECK: EpistemicStatus.SYMBOLIC_DERIVED,
    KnowledgeStatus.DERIVED_FROM_ABSCISSA: EpistemicStatus.SYMBOLIC_DERIVED,
    KnowledgeStatus.DERIVED_FROM_STANDARD_EXPONENT_RELATION: EpistemicStatus.SYMBOLIC_DERIVED,
    KnowledgeStatus.ASYMPTOTIC_DERIVED: EpistemicStatus.SYMBOLIC_DERIVED,
    KnowledgeStatus.CLASSICAL_FAMILY_REPACKAGED: EpistemicStatus.SYMBOLIC_DERIVED,
    # Held under RH; carries an explicit assumption, see _assumptions_for.
    KnowledgeStatus.CONDITIONAL_ON_RH_STANDARD: EpistemicStatus.SYMBOLIC_DERIVED,
    # "a sufficient target, not an established theorem"
    KnowledgeStatus.RESEARCH_TARGET: EpistemicStatus.SYMBOLIC_DERIVED,
    # "explicitly falsified or shown insufficient; must not be silently revived"
    KnowledgeStatus.FALSE_ROUTE: EpistemicStatus.BLOCKED,
    # Not a mathematical claim at all, so no mathematical status applies. It was
    # previously BLOCKED, which reads as "this route was examined and cannot
    # proceed" -- wrong for a rule like "numerical evidence cannot promote a
    # theorem to proved", which is a meta-rule about how the engine reasons.
    KnowledgeStatus.GOVERNANCE: EpistemicStatus.AUTHORITATIVE_POLICY,
}

#: Axis 2. What kind of thing each status describes, independent of how well
#: established it is. Meta roles are never property-extractable.
_ROLE_MAP: dict[KnowledgeStatus, MathematicalRole] = {
    KnowledgeStatus.EXACT: MathematicalRole.IDENTITY,
    KnowledgeStatus.EXACT_ALGEBRA: MathematicalRole.IDENTITY,
    KnowledgeStatus.EXACT_CALCULUS: MathematicalRole.IDENTITY,
    KnowledgeStatus.EXACT_DISTRIBUTIONAL: MathematicalRole.IDENTITY,
    KnowledgeStatus.EXACT_ALGEBRA_IN_MODEL: MathematicalRole.IDENTITY,
    KnowledgeStatus.EXACT_CONSTRUCTION: MathematicalRole.CONSTRUCTION,
    KnowledgeStatus.KNOWN: MathematicalRole.CLAIM,
    KnowledgeStatus.KNOWN_FRAMEWORK: MathematicalRole.CLAIM,
    KnowledgeStatus.KNOWN_OR_STANDARD_CONSEQUENCE: MathematicalRole.CLAIM,
    KnowledgeStatus.KNOWN_MODEL_INPUT: MathematicalRole.CLAIM,
    KnowledgeStatus.KNOWN_EQUIVALENCE_FRAMEWORK: MathematicalRole.EQUIVALENCE,
    KnowledgeStatus.CONDITIONAL_ON_RH_STANDARD: MathematicalRole.EQUIVALENCE,
    KnowledgeStatus.DERIVED_SYMBOLIC: MathematicalRole.CLAIM,
    KnowledgeStatus.DERIVED_SYMBOLIC_NEEDS_EXTERNAL_CHECK: MathematicalRole.CLAIM,
    KnowledgeStatus.DERIVED_FROM_ABSCISSA: MathematicalRole.BOUND,
    KnowledgeStatus.DERIVED_FROM_STANDARD_EXPONENT_RELATION: MathematicalRole.BOUND,
    KnowledgeStatus.ASYMPTOTIC_DERIVED: MathematicalRole.BOUND,
    KnowledgeStatus.CLASSICAL_FAMILY_REPACKAGED: MathematicalRole.EQUIVALENCE,
    KnowledgeStatus.RESEARCH_TARGET: MathematicalRole.BOUND,
    KnowledgeStatus.FALSE_ROUTE: MathematicalRole.NO_GO,
    KnowledgeStatus.GOVERNANCE: MathematicalRole.GOVERNANCE,
}

#: Statuses whose content restates RH rather than advancing toward it.
_RH_EQUIVALENT = frozenset(
    {
        KnowledgeStatus.KNOWN_EQUIVALENCE_FRAMEWORK,
        KnowledgeStatus.CONDITIONAL_ON_RH_STANDARD,
    }
)

#: Assumptions implied by a status, independent of the statement text.
_STATUS_ASSUMPTIONS: dict[KnowledgeStatus, list[str]] = {
    KnowledgeStatus.CONDITIONAL_ON_RH_STANDARD: ["conditional on RH"],
    KnowledgeStatus.EXACT_ALGEBRA_IN_MODEL: ["holds within the stated model, not unconditionally"],
    KnowledgeStatus.KNOWN_MODEL_INPUT: ["input to a conjectural model"],
}


def _status_from_knowledge(status: KnowledgeStatus) -> EpistemicStatus:
    if status not in _STATUS_MAP:
        # Fail closed. An unmapped status is an unreviewed status, and the safe
        # reading of "unreviewed" is "not rigorous".
        return EpistemicStatus.SYMBOLIC_DERIVED
    return _STATUS_MAP[status]


def _role_from_knowledge(status: KnowledgeStatus) -> MathematicalRole:
    return _ROLE_MAP.get(status, MathematicalRole.CLAIM)


def is_property_extractable(item: KnowledgeItem) -> bool:
    """Whether an item carries mathematical content at all.

    Governance and procedural records describe how the engine reasons, not what
    is true about zeta, so no property should be minted from them.

    Decided on the *canonical* role. The legacy `MathematicalRole` is the
    storage vocabulary; branching on it here would have been a second copy of
    the meta/mathematical split, free to drift from the one in
    `contracts.roles` that everything else consults.
    """
    # Function-scope: `contracts.mappings` reads this package's enums, so a
    # module-level import would close the loop.
    from ..contracts.mappings import role_from_knowledge_status
    from ..contracts.roles import META_ROLES as CANONICAL_META_ROLES

    return role_from_knowledge_status(item.status) not in CANONICAL_META_ROLES


def _assumptions_for(item: KnowledgeItem) -> list[str]:
    """Assumptions every property extracted from this item must carry.

    Previously only the `_BIG_O` branch attached these, so a
    `conditional_on_RH_standard` item stating "Under RH, Theta <= 1/2" produced
    an unconditional THETA_BOUND -- an RH-conditional bound arriving as an
    unconditional fact, which is the circular step this whole layer exists to
    prevent.
    """
    return list(_STATUS_ASSUMPTIONS.get(item.status, []))
