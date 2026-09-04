from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover
    from ..contracts.epistemic import Confidence
    from ..contracts.roles import Role


class EpistemicStatus(StrEnum):
    """Axis 1: how strongly established a statement is.

    This axis answers only "how confident are we that this is true?". It says
    nothing about whether the statement is *useful* -- a classical equivalence
    to RH is fully established mathematics and simultaneously worth no progress
    toward a proof. Those are separate questions and get separate fields
    (`MathematicalRole`, `rh_equivalent`), because collapsing them means the
    engine downgrades mathematical confidence as a proxy for research value.
    """

    KNOWN = "known"
    PROVED = "proved"
    CERTIFIED = "certified"
    SYMBOLIC_DERIVED = "symbolic_derived"
    RIGOROUS_DERIVED = "rigorous_derived"
    HEURISTIC = "heuristic"
    SYNTHETIC = "synthetic"
    #: A mathematical route that was examined and cannot proceed.
    BLOCKED = "blocked"
    #: A meta-rule governing the reasoning engine, not a mathematical claim.
    #: Its "truth" is a matter of policy, so no mathematical status applies.
    AUTHORITATIVE_POLICY = "authoritative_policy"


class MathematicalRole(StrEnum):
    """Axis 2: what kind of thing a record is.

    Orthogonal to `EpistemicStatus`. A governance rule such as "numerical
    evidence cannot promote a theorem to proved" is not a blocked mathematical
    route -- it is not a mathematical route at all, and filing it as BLOCKED
    invites later code to read it as a failed proof attempt.
    """

    # Mathematical roles: statements about the mathematics.
    CLAIM = "mathematical_claim"
    IDENTITY = "mathematical_identity"
    BOUND = "mathematical_bound"
    EQUIVALENCE = "mathematical_equivalence"
    CONSTRUCTION = "mathematical_construction"
    NO_GO = "mathematical_no_go"
    # Meta roles: statements about how the research is conducted.
    GOVERNANCE = "governance"
    PROCEDURAL = "procedural"


MATHEMATICAL_ROLES = frozenset(
    {
        MathematicalRole.CLAIM,
        MathematicalRole.IDENTITY,
        MathematicalRole.BOUND,
        MathematicalRole.EQUIVALENCE,
        MathematicalRole.CONSTRUCTION,
        MathematicalRole.NO_GO,
    }
)

#: Roles that carry no mathematical content and are never property-extractable.
META_ROLES = frozenset({MathematicalRole.GOVERNANCE, MathematicalRole.PROCEDURAL})


RIGOROUS_STATUSES = frozenset(
    {
        EpistemicStatus.KNOWN,
        EpistemicStatus.PROVED,
        EpistemicStatus.CERTIFIED,
        EpistemicStatus.RIGOROUS_DERIVED,
    }
)

#: Statuses nothing in this package may produce.
#:
#: No extractor, miner, or closure rule can establish a proof. `PROVED` stays in
#: the vocabulary because an external formal checker could one day supply it,
#: but it has to arrive through a path that does not exist yet -- so a record
#: carrying it today came from a hand edit or a corrupted graph file.
PACKAGE_FORBIDDEN_STATUSES = frozenset({EpistemicStatus.PROVED})


class PropertyKind(StrEnum):
    DOMAIN = "domain"
    CODOMAIN = "codomain"
    SYMMETRY = "symmetry"
    INVARIANT = "invariant"
    SINGULARITY = "singularity"
    ZERO = "zero"
    POLE = "pole"
    RESIDUE = "residue"
    GROWTH_BOUND = "growth_bound"
    THETA_BOUND = "theta_bound"
    POSITIVITY = "positivity"
    ANALYTICITY = "analyticity"
    EQUIVALENCE = "equivalence"
    DISCRIMINATOR = "discriminator"


class ObjectKind(StrEnum):
    FUNCTION = "function"
    SEQUENCE = "sequence"
    OPERATOR = "operator"
    KERNEL = "kernel"
    TRANSFORM = "transform"
    PARAMETER = "parameter"
    CLAIM = "claim"
    EXPRESSION = "expression"
    UNKNOWN = "unknown"


class Provenance(BaseModel):
    source_type: Literal[
        "formula_index",
        "knowledge",
        "math_certificate",
        "symbolic",
        "dre",
        "hypothesis",
        "synthetic",
        "manual",
    ]
    source_id: str
    source_ref: str | None = None
    method: str
    assumptions: list[str] = Field(default_factory=list)
    evidence_hash: str | None = None

    @model_validator(mode="after")
    def stable_hash(self):
        """Content-address this provenance record.

        This is an *identifier*, not tamper evidence: it recomputes over
        whatever the record currently holds, so editing a field simply produces
        a different valid hash. Integrity for the graph as a whole comes from
        the sidecar seal in `PropertyGraphStore`.
        """
        if self.evidence_hash is None:
            payload = self.model_dump(mode="json", exclude={"evidence_hash"})
            text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            self.evidence_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self


class MathObject(BaseModel):
    id: str
    name: str
    kind: ObjectKind = ObjectKind.UNKNOWN
    expression: str | None = None
    aliases: list[str] = Field(default_factory=list)
    provenance: list[Provenance] = Field(default_factory=list)


class ForbiddenStatusError(ValueError):
    """Raised when a record claims a status nothing in this package can establish."""


class PropertyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object_id: str
    kind: PropertyKind
    value: str
    status: EpistemicStatus
    provenance: list[Provenance]
    conditions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    implies: list[str] = Field(default_factory=list)
    #: Axis 2. Meta roles are excluded from property extraction entirely.
    role: MathematicalRole = MathematicalRole.CLAIM
    #: Axis 3. True when the statement restates RH rather than advancing toward
    #: it. Deliberately independent of `status`: `A <=> RH` for classical `A` is
    #: rigorous, established mathematics *and* worth no frontier progress.
    rh_equivalent: bool = False
    #: The escape hatch on `rh_equivalent`. An equivalence stops being circular
    #: exactly when one direction has been genuinely discharged -- at that point
    #: it is a proof step, not a restatement. Naming the discharged obligations
    #: is what distinguishes the two.
    discharged_obligations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _reject_derived_inputs(cls, data):
        from ..contracts.frontier import reject_derived_inputs

        return reject_derived_inputs(data, "PropertyRecord")

    @model_validator(mode="after")
    def _reject_unearnable_status(self):
        if self.status in PACKAGE_FORBIDDEN_STATUSES:
            raise ForbiddenStatusError(
                f"status {self.status.value!r} cannot be produced by this package; no extractor, "
                "miner, or closure rule establishes a proof"
            )
        return self

    # -- canonical axes ----------------------------------------------------
    #
    # `status` and `role` remain the *stored* representation for compatibility
    # with existing property graphs. All live reasoning happens on the canonical
    # Confidence/Role, so there is one set of rules rather than a second copy
    # that can drift. The mappings are imported lazily: contracts.mappings reads
    # this module's enums, so a top-level import would close the loop.

    @property
    def confidence(self) -> Confidence:
        from ..contracts.mappings import confidence_from_property_status

        return confidence_from_property_status(self.status)

    @property
    def canonical_role(self) -> Role:
        from ..contracts.mappings import role_from_property_role

        return role_from_property_role(self.role)

    def frontier_with(
        self,
        *,
        obligations: dict | None = None,
        evidence: dict | None = None,
        decisions: dict | None = None,
        receipts: dict | None = None,
    ):
        """Frontier axes via the single canonical implementation.

        With no obligation registry nothing discharges, so a prose entry in
        `discharged_obligations` cannot lift an RH-equivalence into progress.
        """
        from ..contracts.discharge import resolve_discharges
        from ..contracts.frontier import assess

        resolution = resolve_discharges(
            self.discharged_obligations,
            obligations=obligations,
            evidence=evidence,
            decisions=decisions,
            receipts=receipts,
        )
        return assess(
            role=self.canonical_role,
            confidence=self.confidence,
            rh_equivalent=self.rh_equivalent,
            qualifying_discharges=resolution.qualifying,
            open_qualifiers=self.open_qualifiers,
        )

    @property
    def frontier(self):
        return self.frontier_with()

    @property
    def is_rigorous(self) -> bool:
        """Axis 1 only: established, with nothing left hanging.

        Deliberately does not consult `rh_equivalent`. An RH-equivalence can be
        perfectly rigorous; whether it is *useful* is `advances_frontier`.
        """
        from ..contracts.epistemic import is_rigorous as _is_rigorous

        return _is_rigorous(self.confidence) and not self.open_qualifiers

    @property
    def frontier_relevant(self) -> bool:
        return self.frontier.frontier_relevant

    @property
    def advances_frontier(self) -> bool:
        return self.frontier.advances_frontier

    @property
    def is_usable_as_rule(self) -> bool:
        """May this be applied as an established rewriting step?

        Weaker than `advances_frontier` on purpose: a rigorous RH-equivalence is
        a perfectly good transformation rule even though invoking it earns
        nothing.
        """
        from ..contracts.frontier import usable_as_rule

        return usable_as_rule(
            role=self.canonical_role,
            confidence=self.confidence,
            open_qualifiers=self.open_qualifiers,
        )

    @property
    def open_qualifiers(self) -> list[str]:
        """Everything that must hold for this property to apply, in one list."""
        return sorted(set(self.assumptions) | set(self.conditions))

    def record_hash(self) -> str:
        """Canonical identity for this record, excluding `metadata`.

        Same rule as `Artifact.canonical_json`: metadata is annotation, and
        nothing that decides promotion or closure may read a field the identity
        does not cover. A derived record cites this hash for its premise, so
        "which version of the premise produced this?" has an answer that
        editing the premise cannot quietly change.
        """
        from ..contracts.hashing import canonical_hash

        return canonical_hash(self.model_dump(mode="json", exclude={"metadata"}))


class ImplicationRule(BaseModel):
    id: str
    premise_kind: PropertyKind
    conclusion_kind: PropertyKind
    description: str
    rigorous_only: bool = True
    #: True when firing this rule asserts movement of the RH frontier, as
    #: opposed to rewriting a statement into an equivalent form. Frontier rules
    #: require a premise that `advances_frontier`; pure transformation rules only
    #: require one that `is_usable_as_rule`, so an established RH-equivalence can
    #: still be applied as a rewriting step.
    produces_frontier_claim: bool = True


class PropertyEdge(BaseModel):
    source_id: str
    target_id: str
    relation: str
    rule_id: str | None = None
    status: EpistemicStatus
    provenance: list[Provenance] = Field(default_factory=list)


class PropertyGraph(BaseModel):
    schema_version: str = "1"
    objects: list[MathObject] = Field(default_factory=list)
    properties: list[PropertyRecord] = Field(default_factory=list)
    edges: list[PropertyEdge] = Field(default_factory=list)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def graph_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class ClosureMode(StrEnum):
    RIGOROUS = "rigorous"
    EXPLORATORY = "exploratory"


class DiscriminatorResult(BaseModel):
    object_id: str
    property_id: str
    critical_line_value: str
    off_line_value: str
    status: EpistemicStatus
    promoted_to_proof: bool = False
    reason: str
    provenance: list[Provenance] = Field(default_factory=list)
