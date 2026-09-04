"""Total mappings from every legacy vocabulary onto the canonical axes.

This is the only module in ``contracts/`` permitted to import the rest of the
package. The leaf axis modules stay dependency-free so every subsystem can
import them; the back-references live here, where they are explicitly a
transitional concern.

**Every mapping is total and explicit.** No substring rules, no name
similarity, no default branch that guesses. A legacy value with no entry raises
:class:`UnmappedValueError` naming the value and the table it is missing from,
and a test asserts each table covers its enum exactly -- so a value added later
fails the suite until someone classifies it deliberately.

That is not stylistic. Classification by spelling
(``status.value.startswith("exact")``) promoted 14 of 21 knowledge statuses to
rigorous, including one whose name says an external check has not happened.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypeVar

from ..core.knowledge import KnowledgeStatus
from ..core.models import ClaimStatus, EvidenceClass
from ..mathcert.verifiers import VerificationStatus
from ..properties.models import EpistemicStatus, MathematicalRole
from ..supervisor.compat import HypothesisState
from .epistemic import Confidence
from .lifecycle import HypothesisLifecycle
from .roles import Role

E = TypeVar("E", bound=StrEnum)


class UnmappedValueError(ValueError):
    """A legacy value has no canonical mapping.

    Raised rather than defaulted. A default here would silently classify an
    unreviewed value, which is exactly the failure the contract layer exists to
    prevent -- and it would classify it as *something*, when the honest answer
    is that nobody has decided yet.
    """

    def __init__(self, value: object, table: str, known: list[str]) -> None:
        super().__init__(
            f"{value!r} has no entry in {table}. Every legacy value must be mapped "
            f"deliberately; add it to the table rather than relying on a default. "
            f"Known values: {known}"
        )
        self.value = value
        self.table = table


def _lookup(table: dict, value, table_name: str):
    try:
        return table[value]
    except KeyError:
        raise UnmappedValueError(
            value, table_name, sorted(str(k) for k in table)
        ) from None


# --------------------------------------------------------------------------
# ClaimStatus -> Confidence
# --------------------------------------------------------------------------
CLAIM_STATUS_TO_CONFIDENCE: dict[ClaimStatus, Confidence] = {
    ClaimStatus.HYPOTHESIS: Confidence.CONJECTURAL,
    ClaimStatus.NUMERICAL: Confidence.NUMERICAL,
    ClaimStatus.SYMBOLIC: Confidence.SYMBOLIC_DERIVED,
    ClaimStatus.PROVED: Confidence.PROVED,
    ClaimStatus.KNOWN: Confidence.KNOWN,
    # An inference made here carries no deductive force until someone checks
    # it, so it maps to HEURISTIC and not to RIGOROUS_DERIVED. Promoting it is
    # a decision, and a decision needs somewhere to be recorded.
    ClaimStatus.INFERRED: Confidence.HEURISTIC,
    # An RH-equivalence is not a confidence level: it is a claim about
    # usefulness wearing a confidence field. Confidence goes to KNOWN, and the
    # circularity is carried on the frontier axis by CLAIM_STATUS_RH_EQUIVALENT.
    ClaimStatus.EQUIVALENT_RH: Confidence.KNOWN,
    ClaimStatus.FALSE: Confidence.REFUTED,
}

CLAIM_STATUS_TO_ROLE: dict[ClaimStatus, Role] = {
    ClaimStatus.HYPOTHESIS: Role.CLAIM,
    ClaimStatus.NUMERICAL: Role.CLAIM,
    ClaimStatus.SYMBOLIC: Role.CLAIM,
    ClaimStatus.PROVED: Role.CLAIM,
    ClaimStatus.KNOWN: Role.CLAIM,
    ClaimStatus.INFERRED: Role.CLAIM,
    ClaimStatus.EQUIVALENT_RH: Role.EQUIVALENCE,
    ClaimStatus.FALSE: Role.NO_GO,
}

#: Legacy statuses whose meaning includes "this restates RH".
CLAIM_STATUS_RH_EQUIVALENT: frozenset[ClaimStatus] = frozenset(
    {ClaimStatus.EQUIVALENT_RH}
)


# --------------------------------------------------------------------------
# EvidenceClass -> Confidence
# --------------------------------------------------------------------------
EVIDENCE_CLASS_TO_CONFIDENCE: dict[EvidenceClass, Confidence] = {
    EvidenceClass.NUMERICAL: Confidence.NUMERICAL,
    EvidenceClass.RIGOROUS_NUMERICAL: Confidence.RIGOROUS_NUMERICAL,
    EvidenceClass.SYMBOLIC: Confidence.SYMBOLIC_DERIVED,
    EvidenceClass.PROVED: Confidence.PROVED,
    EvidenceClass.KNOWN: Confidence.KNOWN,
    EvidenceClass.HEURISTIC: Confidence.HEURISTIC,
    EvidenceClass.COUNTEREXAMPLE: Confidence.REFUTED,
}


# --------------------------------------------------------------------------
# KnowledgeStatus -> Confidence and Role
# --------------------------------------------------------------------------
KNOWLEDGE_STATUS_TO_CONFIDENCE: dict[KnowledgeStatus, Confidence] = {
    # "identity checked algebraically from the stated definitions"
    KnowledgeStatus.EXACT: Confidence.CERTIFIED,
    KnowledgeStatus.EXACT_ALGEBRA: Confidence.CERTIFIED,
    KnowledgeStatus.EXACT_CALCULUS: Confidence.CERTIFIED,
    KnowledgeStatus.EXACT_CONSTRUCTION: Confidence.CERTIFIED,
    KnowledgeStatus.EXACT_DISTRIBUTIONAL: Confidence.CERTIFIED,
    # Exact *within a model*: the algebra is exact, the model is conjectural.
    KnowledgeStatus.EXACT_ALGEBRA_IN_MODEL: Confidence.SYMBOLIC_DERIVED,
    # "established external mathematics"
    KnowledgeStatus.KNOWN: Confidence.KNOWN,
    KnowledgeStatus.KNOWN_FRAMEWORK: Confidence.KNOWN,
    KnowledgeStatus.KNOWN_OR_STANDARD_CONSEQUENCE: Confidence.KNOWN,
    KnowledgeStatus.KNOWN_EQUIVALENCE_FRAMEWORK: Confidence.KNOWN,
    # An input to a conjectural model is not established mathematics.
    KnowledgeStatus.KNOWN_MODEL_INPUT: Confidence.HEURISTIC,
    # "derived in this research but not yet independently formalized or
    # literature-checked end to end"
    KnowledgeStatus.DERIVED_SYMBOLIC: Confidence.SYMBOLIC_DERIVED,
    KnowledgeStatus.DERIVED_SYMBOLIC_NEEDS_EXTERNAL_CHECK: Confidence.SYMBOLIC_DERIVED,
    KnowledgeStatus.DERIVED_FROM_ABSCISSA: Confidence.SYMBOLIC_DERIVED,
    KnowledgeStatus.DERIVED_FROM_STANDARD_EXPONENT_RELATION: Confidence.SYMBOLIC_DERIVED,
    KnowledgeStatus.ASYMPTOTIC_DERIVED: Confidence.SYMBOLIC_DERIVED,
    KnowledgeStatus.CLASSICAL_FAMILY_REPACKAGED: Confidence.SYMBOLIC_DERIVED,
    KnowledgeStatus.CONDITIONAL_ON_RH_STANDARD: Confidence.SYMBOLIC_DERIVED,
    # "a sufficient target, not an established theorem"
    KnowledgeStatus.RESEARCH_TARGET: Confidence.CONJECTURAL,
    # "explicitly falsified or shown insufficient"
    KnowledgeStatus.FALSE_ROUTE: Confidence.REFUTED,
    # Methodology, not mathematics.
    KnowledgeStatus.GOVERNANCE: Confidence.AUTHORITATIVE_POLICY,
}

KNOWLEDGE_STATUS_TO_ROLE: dict[KnowledgeStatus, Role] = {
    KnowledgeStatus.EXACT: Role.IDENTITY,
    KnowledgeStatus.EXACT_ALGEBRA: Role.IDENTITY,
    KnowledgeStatus.EXACT_CALCULUS: Role.IDENTITY,
    KnowledgeStatus.EXACT_DISTRIBUTIONAL: Role.IDENTITY,
    KnowledgeStatus.EXACT_ALGEBRA_IN_MODEL: Role.IDENTITY,
    KnowledgeStatus.EXACT_CONSTRUCTION: Role.CONSTRUCTION,
    KnowledgeStatus.KNOWN: Role.CLAIM,
    KnowledgeStatus.KNOWN_FRAMEWORK: Role.CLAIM,
    KnowledgeStatus.KNOWN_OR_STANDARD_CONSEQUENCE: Role.CLAIM,
    KnowledgeStatus.KNOWN_MODEL_INPUT: Role.CLAIM,
    KnowledgeStatus.KNOWN_EQUIVALENCE_FRAMEWORK: Role.EQUIVALENCE,
    KnowledgeStatus.CONDITIONAL_ON_RH_STANDARD: Role.EQUIVALENCE,
    KnowledgeStatus.CLASSICAL_FAMILY_REPACKAGED: Role.EQUIVALENCE,
    KnowledgeStatus.DERIVED_SYMBOLIC: Role.CLAIM,
    KnowledgeStatus.DERIVED_SYMBOLIC_NEEDS_EXTERNAL_CHECK: Role.CLAIM,
    KnowledgeStatus.DERIVED_FROM_ABSCISSA: Role.BOUND,
    KnowledgeStatus.DERIVED_FROM_STANDARD_EXPONENT_RELATION: Role.BOUND,
    KnowledgeStatus.ASYMPTOTIC_DERIVED: Role.BOUND,
    KnowledgeStatus.RESEARCH_TARGET: Role.BOUND,
    KnowledgeStatus.FALSE_ROUTE: Role.NO_GO,
    KnowledgeStatus.GOVERNANCE: Role.GOVERNANCE,
}

#: Knowledge statuses whose content restates RH.
KNOWLEDGE_STATUS_RH_EQUIVALENT: frozenset[KnowledgeStatus] = frozenset(
    {
        KnowledgeStatus.KNOWN_EQUIVALENCE_FRAMEWORK,
        KnowledgeStatus.CONDITIONAL_ON_RH_STANDARD,
    }
)

#: Assumptions a status implies regardless of what its statement text says.
KNOWLEDGE_STATUS_ASSUMPTIONS: dict[KnowledgeStatus, tuple[str, ...]] = {
    KnowledgeStatus.CONDITIONAL_ON_RH_STANDARD: ("conditional on RH",),
    KnowledgeStatus.EXACT_ALGEBRA_IN_MODEL: (
        "holds within the stated model, not unconditionally",
    ),
    KnowledgeStatus.KNOWN_MODEL_INPUT: ("input to a conjectural model",),
}


# --------------------------------------------------------------------------
# properties.EpistemicStatus -> Confidence, MathematicalRole -> Role
# --------------------------------------------------------------------------
PROPERTY_STATUS_TO_CONFIDENCE: dict[EpistemicStatus, Confidence] = {
    EpistemicStatus.KNOWN: Confidence.KNOWN,
    EpistemicStatus.PROVED: Confidence.PROVED,
    EpistemicStatus.CERTIFIED: Confidence.CERTIFIED,
    EpistemicStatus.RIGOROUS_DERIVED: Confidence.RIGOROUS_DERIVED,
    EpistemicStatus.SYMBOLIC_DERIVED: Confidence.SYMBOLIC_DERIVED,
    EpistemicStatus.HEURISTIC: Confidence.HEURISTIC,
    EpistemicStatus.SYNTHETIC: Confidence.SYNTHETIC,
    EpistemicStatus.BLOCKED: Confidence.REFUTED,
    EpistemicStatus.AUTHORITATIVE_POLICY: Confidence.AUTHORITATIVE_POLICY,
}

PROPERTY_ROLE_TO_ROLE: dict[MathematicalRole, Role] = {
    MathematicalRole.CLAIM: Role.CLAIM,
    MathematicalRole.IDENTITY: Role.IDENTITY,
    MathematicalRole.BOUND: Role.BOUND,
    MathematicalRole.EQUIVALENCE: Role.EQUIVALENCE,
    MathematicalRole.CONSTRUCTION: Role.CONSTRUCTION,
    MathematicalRole.NO_GO: Role.NO_GO,
    MathematicalRole.GOVERNANCE: Role.GOVERNANCE,
    MathematicalRole.PROCEDURAL: Role.PROCEDURAL,
}


# --------------------------------------------------------------------------
# VerificationStatus -> Confidence
# --------------------------------------------------------------------------
VERIFICATION_STATUS_TO_CONFIDENCE: dict[VerificationStatus, Confidence] = {
    # ACCEPTED means a registered adapter checked an enclosure. That is
    # rigorous about a finite computation and nothing more.
    VerificationStatus.ACCEPTED: Confidence.RIGOROUS_NUMERICAL,
    VerificationStatus.REJECTED: Confidence.REFUTED,
    VerificationStatus.UNKNOWN: Confidence.UNKNOWN,
}


# --------------------------------------------------------------------------
# HypothesisState -> HypothesisLifecycle  (deprecated; see supervisor/compat.py)
# --------------------------------------------------------------------------
HYPOTHESIS_STATE_TO_LIFECYCLE: dict[HypothesisState, HypothesisLifecycle] = {
    HypothesisState.PROPOSED: HypothesisLifecycle.PROPOSED,
    # "actionable" conflated a lifecycle position with a derived predicate.
    # The position it meant is "triaged and ready to work".
    HypothesisState.ACTIONABLE: HypothesisLifecycle.TRIAGED,
    HypothesisState.TESTING: HypothesisLifecycle.ACTIVE,
    HypothesisState.BLOCKED: HypothesisLifecycle.BLOCKED,
    # Both terminal states become RESOLVED; how they resolved is the epistemic
    # axis, which is the whole point of splitting them apart.
    HypothesisState.FALSIFIED: HypothesisLifecycle.RESOLVED,
    HypothesisState.ADVANCED: HypothesisLifecycle.RESOLVED,
}

#: The confidence each legacy terminal state implied. Lifecycle alone cannot
#: carry this, which is precisely why the split was needed.
HYPOTHESIS_STATE_TO_CONFIDENCE: dict[HypothesisState, Confidence] = {
    HypothesisState.PROPOSED: Confidence.CONJECTURAL,
    HypothesisState.ACTIONABLE: Confidence.CONJECTURAL,
    HypothesisState.TESTING: Confidence.CONJECTURAL,
    HypothesisState.BLOCKED: Confidence.CONJECTURAL,
    HypothesisState.FALSIFIED: Confidence.REFUTED,
    # ADVANCED asserted progress without recording how strongly. RIGOROUS_DERIVED
    # is the weakest reading that still justifies the claim it made.
    HypothesisState.ADVANCED: Confidence.RIGOROUS_DERIVED,
}


# --------------------------------------------------------------------------
# Public accessors
# --------------------------------------------------------------------------
def confidence_from_claim_status(status: ClaimStatus) -> Confidence:
    return _lookup(CLAIM_STATUS_TO_CONFIDENCE, status, "CLAIM_STATUS_TO_CONFIDENCE")


def role_from_claim_status(status: ClaimStatus) -> Role:
    return _lookup(CLAIM_STATUS_TO_ROLE, status, "CLAIM_STATUS_TO_ROLE")


def confidence_from_evidence_class(evidence: EvidenceClass) -> Confidence:
    return _lookup(EVIDENCE_CLASS_TO_CONFIDENCE, evidence, "EVIDENCE_CLASS_TO_CONFIDENCE")


def confidence_from_knowledge_status(status: KnowledgeStatus) -> Confidence:
    return _lookup(KNOWLEDGE_STATUS_TO_CONFIDENCE, status, "KNOWLEDGE_STATUS_TO_CONFIDENCE")


def role_from_knowledge_status(status: KnowledgeStatus) -> Role:
    return _lookup(KNOWLEDGE_STATUS_TO_ROLE, status, "KNOWLEDGE_STATUS_TO_ROLE")


def assumptions_from_knowledge_status(status: KnowledgeStatus) -> list[str]:
    return list(KNOWLEDGE_STATUS_ASSUMPTIONS.get(status, ()))


def confidence_from_property_status(status: EpistemicStatus) -> Confidence:
    return _lookup(PROPERTY_STATUS_TO_CONFIDENCE, status, "PROPERTY_STATUS_TO_CONFIDENCE")


def role_from_property_role(role: MathematicalRole) -> Role:
    return _lookup(PROPERTY_ROLE_TO_ROLE, role, "PROPERTY_ROLE_TO_ROLE")


def confidence_from_verification_status(status: VerificationStatus) -> Confidence:
    return _lookup(
        VERIFICATION_STATUS_TO_CONFIDENCE, status, "VERIFICATION_STATUS_TO_CONFIDENCE"
    )


def lifecycle_from_hypothesis_state(state: HypothesisState) -> HypothesisLifecycle:
    return _lookup(HYPOTHESIS_STATE_TO_LIFECYCLE, state, "HYPOTHESIS_STATE_TO_LIFECYCLE")


def confidence_from_hypothesis_state(state: HypothesisState) -> Confidence:
    return _lookup(HYPOTHESIS_STATE_TO_CONFIDENCE, state, "HYPOTHESIS_STATE_TO_CONFIDENCE")


#: Every mapping table, for the exhaustiveness tests. Keyed by table name so a
#: failure names the table that is missing an entry.
LEGACY_TABLES: dict[str, tuple[type[StrEnum], dict]] = {
    "CLAIM_STATUS_TO_CONFIDENCE": (ClaimStatus, CLAIM_STATUS_TO_CONFIDENCE),
    "CLAIM_STATUS_TO_ROLE": (ClaimStatus, CLAIM_STATUS_TO_ROLE),
    "EVIDENCE_CLASS_TO_CONFIDENCE": (EvidenceClass, EVIDENCE_CLASS_TO_CONFIDENCE),
    "KNOWLEDGE_STATUS_TO_CONFIDENCE": (KnowledgeStatus, KNOWLEDGE_STATUS_TO_CONFIDENCE),
    "KNOWLEDGE_STATUS_TO_ROLE": (KnowledgeStatus, KNOWLEDGE_STATUS_TO_ROLE),
    "PROPERTY_STATUS_TO_CONFIDENCE": (EpistemicStatus, PROPERTY_STATUS_TO_CONFIDENCE),
    "PROPERTY_ROLE_TO_ROLE": (MathematicalRole, PROPERTY_ROLE_TO_ROLE),
    "VERIFICATION_STATUS_TO_CONFIDENCE": (
        VerificationStatus,
        VERIFICATION_STATUS_TO_CONFIDENCE,
    ),
    "HYPOTHESIS_STATE_TO_LIFECYCLE": (HypothesisState, HYPOTHESIS_STATE_TO_LIFECYCLE),
    "HYPOTHESIS_STATE_TO_CONFIDENCE": (HypothesisState, HYPOTHESIS_STATE_TO_CONFIDENCE),
}
