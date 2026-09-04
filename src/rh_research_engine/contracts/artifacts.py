"""Versioned artifact contracts.

Schemas only. Phase 1 establishes what these records *are*; the engines that
produce them are later workstreams. A placeholder algorithm pretending to
implement C4 would be worse than an honest empty slot, because it would produce
artifacts that look real.

Every artifact carries the same envelope, so provenance, hashing, and epistemic
status work identically no matter which subsystem emitted it.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from .epistemic import WORKER_FORBIDDEN, Confidence
from .frontier import FrontierAssessment, assess, reject_derived_inputs
from .hashing import canonical_json, validate_sha256, validate_sha256_list
from .roles import Role

SCHEMA_VERSION = "1"


class ArtifactType(StrEnum):
    HYPOTHESIS = "hypothesis"
    PROOF_STEP = "proof_step"
    PROOF_OBLIGATION = "proof_obligation"
    PROPERTY_ASSERTION = "property_assertion"
    MATHEMATICAL_OBJECT = "mathematical_object"
    MODEL_FAMILY_SPEC = "model_family_spec"
    SYNTHETIC_MODEL = "synthetic_model"
    COUNTEREXAMPLE = "counterexample"
    TRANSFORM_DERIVATION = "transform_derivation"
    KERNEL_ANALYSIS = "kernel_analysis"
    OPERATOR_EXPERIMENT = "operator_experiment"
    INEQUALITY_CERTIFICATE = "inequality_certificate"
    MATH_CERTIFICATE = "math_certificate"
    LITERATURE_MATCH = "literature_match"
    FORMALIZATION_REPORT = "formalization_report"
    OBLIGATION_DISCHARGE_DECISION = "obligation_discharge_decision"
    SUPERVISOR_DECISION = "supervisor_decision"
    RESEARCH_RUN_MANIFEST = "research_run_manifest"


class ArtifactError(ValueError):
    """An artifact violates a contract rule."""


class Artifact(BaseModel):
    """Common envelope for every research artifact.

    ``artifact_hash`` content-addresses the record. Like every other hash in
    this repository it is an *identifier*, not tamper evidence: it recomputes
    over whatever the record currently holds. Integrity comes from the sealed
    stores, not from a field the same edit can update.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    artifact_id: str
    artifact_type: ArtifactType
    created_by: str
    method_family: str
    method_version: str

    #: Hash of the producing worker's source, where the producer supplies one.
    source_hash: str | None = None
    #: Hashes of the artifacts this one was computed from.
    input_hashes: list[str] = Field(default_factory=list)
    #: Artifact IDs this record depends on.
    dependencies: list[str] = Field(default_factory=list)
    #: Certificates supporting this record.
    certificate_refs: list[str] = Field(default_factory=list)

    assumptions: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)

    epistemic_status: Confidence = Confidence.UNKNOWN
    mathematical_role: Role = Role.CLAIM
    rh_equivalent: bool = False
    discharged_obligations: list[str] = Field(default_factory=list)

    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _reject_derived_inputs(cls, data):
        return reject_derived_inputs(data, cls.__name__)

    @model_validator(mode="after")
    def _reject_worker_asserted_proof(self):
        if self.epistemic_status in WORKER_FORBIDDEN and self.created_by != "external-verifier":
            raise ArtifactError(
                f"{self.created_by!r} may not assert epistemic_status="
                f"{self.epistemic_status.value!r}. 'proved', 'known', and "
                "'formally_verified' require a formal checker or cited literature "
                "outside this package; artifacts carrying them must be created_by "
                "'external-verifier'."
            )
        return self

    @property
    def open_qualifiers(self) -> list[str]:
        """Everything still hanging on this record, assumptions and conditions."""
        return sorted(set(self.assumptions) | set(self.conditions))

    def frontier_with(
        self,
        *,
        obligations: dict | None = None,
        evidence: dict | None = None,
        decisions: dict | None = None,
        receipts: dict | None = None,
    ) -> FrontierAssessment:
        """Frontier axes, resolving discharges against DRE-accepted decisions."""
        from .discharge import resolve_discharges

        resolution = resolve_discharges(
            self.discharged_obligations,
            obligations=obligations,
            evidence=evidence,
            decisions=decisions,
            receipts=receipts,
        )
        return assess(
            role=self.mathematical_role,
            confidence=self.epistemic_status,
            rh_equivalent=self.rh_equivalent,
            qualifying_discharges=resolution.qualifying,
            open_qualifiers=self.open_qualifiers,
        )

    @property
    def frontier(self) -> FrontierAssessment:
        """Derived, never stored. With no registry, nothing discharges."""
        return self.frontier_with()

    @property
    def frontier_relevant(self) -> bool:
        return self.frontier.frontier_relevant

    @property
    def advances_frontier(self) -> bool:
        return self.frontier.advances_frontier

    @property
    def property_extractable(self) -> bool:
        return self.frontier.property_extractable

    def canonical_json(self) -> str:
        """Identity payload. ``metadata`` is excluded, and so is not authoritative.

        Nothing that decides promotion, closure, or replay may read a field the
        identity does not cover: a value outside the hash can be edited without
        changing what the record claims to be, so depending on it would make two
        different artifacts indistinguishable.
        """
        return canonical_json(self.model_dump(mode="json", exclude={"metadata"}))

    def artifact_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Workstream A: proof analysis
# --------------------------------------------------------------------------
class ObligationStatus(StrEnum):
    OPEN = "open"
    DISCHARGED = "discharged"
    #: Shown to be unnecessary; the surrounding argument does not need it.
    ELIMINATED = "eliminated"


class ProofObligation(Artifact):
    """A named gap that must be closed for an argument to stand."""

    artifact_type: Literal[ArtifactType.PROOF_OBLIGATION] = ArtifactType.PROOF_OBLIGATION
    statement: str
    status: ObligationStatus = ObligationStatus.OPEN
    #: What would close it: a theorem, a certificate, a formalization.
    discharge_requires: list[str] = Field(default_factory=list)
    #: Evidence offered as a discharge. Present does not mean accepted.
    discharge_evidence: list[str] = Field(default_factory=list)
    #: Which direction of an equivalence this obligation needs closed, when it
    #: is directional. An equivalence has two, and a decision that closes the
    #: other one satisfies nothing -- but a decision naming *some* direction
    #: reads as directional and passed unchecked while nothing stated which
    #: direction was wanted.
    required_direction: str | None = None
    blocking: bool = True

    @property
    def open(self) -> bool:
        return self.status is ObligationStatus.OPEN

    @model_validator(mode="after")
    def _discharged_must_name_evidence(self):
        if self.status is ObligationStatus.DISCHARGED and not self.discharge_evidence:
            raise ArtifactError(
                f"obligation {self.artifact_id!r} is marked discharged but names no "
                "evidence. An obligation closed without saying what closed it is "
                "indistinguishable from one that was simply deleted."
            )
        return self


class DreDecisionStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PENDING = "pending"


class ObligationDischargeDecision(Artifact):
    """DRE's ruling that a specific obligation was closed by specific evidence.

    This exists because an artifact's own labels are not authority. A worker can
    construct ``CERTIFIED`` or ``RIGOROUS_DERIVED`` evidence -- only ``proved``,
    ``known`` and ``formally_verified`` are forbidden -- and ``created_by`` is a
    free string, so ``"external-verifier"`` is a claim rather than a credential.
    Trusting either would let a worker mint its own discharge and lift an
    RH-equivalence into frontier credit.

    The decision binds *content*, not names. ``obligation_hash`` and
    ``evidence_hashes`` pin the exact artifacts that were ruled on, so swapping
    the evidence under an accepted decision invalidates it rather than
    inheriting its authority.
    """

    artifact_type: Literal[ArtifactType.OBLIGATION_DISCHARGE_DECISION] = (
        ArtifactType.OBLIGATION_DISCHARGE_DECISION
    )
    obligation_ref: str
    #: `artifact_hash()` of the obligation as ruled on.
    obligation_hash: str
    evidence_refs: list[str] = Field(default_factory=list)
    #: `artifact_hash()` of each evidence artifact as ruled on, same order.
    evidence_hashes: list[str] = Field(default_factory=list)
    #: Which direction of the equivalence this closes. An equivalence has two,
    #: and closing the wrong one discharges nothing.
    discharged_direction: str
    #: Which entries of the obligation's `discharge_requires` this satisfies.
    requirements_satisfied: list[str] = Field(default_factory=list)
    dre_decision_ref: str
    dre_decision_status: DreDecisionStatus = DreDecisionStatus.PENDING
    dre_pack_version: str
    mathematical_role: Role = Role.PROCEDURAL
    epistemic_status: Confidence = Confidence.AUTHORITATIVE_POLICY

    @field_validator("obligation_hash")
    @classmethod
    def _obligation_hash_is_a_digest(cls, value: str) -> str:
        return validate_sha256(value, "ObligationDischargeDecision.obligation_hash")

    @field_validator("evidence_hashes")
    @classmethod
    def _evidence_hashes_are_digests(cls, values: list[str]) -> list[str]:
        return validate_sha256_list(
            values, "ObligationDischargeDecision.evidence_hashes"
        )

    @model_validator(mode="after")
    def _accepted_decisions_must_be_substantiated(self):
        if self.dre_decision_status is DreDecisionStatus.ACCEPTED:
            if not self.evidence_refs:
                raise ArtifactError(
                    f"decision {self.artifact_id!r} is accepted but names no evidence"
                )
            if len(self.evidence_refs) != len(self.evidence_hashes):
                raise ArtifactError(
                    f"decision {self.artifact_id!r} names {len(self.evidence_refs)} "
                    f"evidence refs but {len(self.evidence_hashes)} hashes; the binding "
                    "must be one-to-one or it pins nothing"
                )
            if not self.discharged_direction.strip():
                raise ArtifactError(
                    f"decision {self.artifact_id!r} does not say which direction it "
                    "discharges; an equivalence has two and closing the wrong one "
                    "closes nothing"
                )
            if not self.dre_decision_ref.strip():
                raise ArtifactError(
                    f"decision {self.artifact_id!r} is accepted but references no DRE "
                    "decision. The engine is the authority; this artifact only records "
                    "what it ruled."
                )
        return self


class PropertyAssertion(Artifact):
    """A property attributed to a mathematical object."""

    artifact_type: Literal[ArtifactType.PROPERTY_ASSERTION] = ArtifactType.PROPERTY_ASSERTION
    object_id: str
    property_kind: str
    value: str
    #: Obligations that must close before this is unconditional.
    open_obligation_refs: list[str] = Field(default_factory=list)


class MathematicalObject(Artifact):
    """A named object: a function, kernel, operator, sequence, parameter."""

    artifact_type: Literal[ArtifactType.MATHEMATICAL_OBJECT] = ArtifactType.MATHEMATICAL_OBJECT
    name: str
    object_kind: str = "unknown"
    expression: str | None = None
    aliases: list[str] = Field(default_factory=list)
    mathematical_role: Role = Role.DEFINITION


# --------------------------------------------------------------------------
# Workstream D: adversarial models
# --------------------------------------------------------------------------
class ModelFamilySpec(Artifact):
    """A family of synthetic zeta-like systems.

    ``satisfied_properties`` and ``violated_properties`` are required and must
    not overlap: a model that does not declare which genuine zeta properties it
    fails is a model whose results cannot be interpreted.
    """

    artifact_type: Literal[ArtifactType.MODEL_FAMILY_SPEC] = ArtifactType.MODEL_FAMILY_SPEC
    family_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    satisfied_properties: list[str] = Field(default_factory=list)
    violated_properties: list[str] = Field(default_factory=list)
    epistemic_status: Confidence = Confidence.SYNTHETIC
    mathematical_role: Role = Role.CONSTRUCTION

    @model_validator(mode="after")
    def _declare_what_it_is_not(self):
        if not self.satisfied_properties and not self.violated_properties:
            raise ArtifactError(
                f"model family {self.family_name!r} declares no relationship to real "
                "zeta properties. Results from an undeclared model cannot be "
                "interpreted, so the declaration is mandatory."
            )
        overlap = sorted(set(self.satisfied_properties) & set(self.violated_properties))
        if overlap:
            raise ArtifactError(
                f"model family {self.family_name!r} lists {overlap} as both satisfied "
                "and violated"
            )
        return self


class SyntheticModel(Artifact):
    """A concrete instance drawn from a model family."""

    artifact_type: Literal[ArtifactType.SYNTHETIC_MODEL] = ArtifactType.SYNTHETIC_MODEL
    family_ref: str
    instance_parameters: dict[str, Any] = Field(default_factory=dict)
    epistemic_status: Confidence = Confidence.SYNTHETIC
    mathematical_role: Role = Role.CONSTRUCTION


class CounterexampleVerdict(StrEnum):
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    SEPARATES_SYNTHETIC_CLASSES = "separates_synthetic_classes"
    INAPPLICABLE = "inapplicable"
    INCONCLUSIVE = "inconclusive"


class CounterexampleArtifact(Artifact):
    """The result of running a criterion against a synthetic model."""

    artifact_type: Literal[ArtifactType.COUNTEREXAMPLE] = ArtifactType.COUNTEREXAMPLE
    criterion_ref: str
    model_ref: str
    verdict: CounterexampleVerdict
    detail: str = ""
    epistemic_status: Confidence = Confidence.SYNTHETIC

    @model_validator(mode="after")
    def _separation_is_not_a_theorem(self):
        if (
            self.verdict is CounterexampleVerdict.SEPARATES_SYNTHETIC_CLASSES
            and self.epistemic_status is not Confidence.SYNTHETIC
        ):
            raise ArtifactError(
                "a separator found on synthetic classes is SYNTHETIC evidence. It "
                "cannot become a theorem without an analytic proof, and relabelling "
                "the artifact is not that proof."
            )
        return self


# --------------------------------------------------------------------------
# Workstream B: analytic compiler
# --------------------------------------------------------------------------
class TransformDerivation(Artifact):
    """A transform applied to a kernel, with its convergence conditions."""

    artifact_type: Literal[ArtifactType.TRANSFORM_DERIVATION] = ArtifactType.TRANSFORM_DERIVATION
    transform: str
    input_expression: str
    output_expression: str
    #: Contour shifts, interchanges, and growth estimates the result rests on.
    justification_gaps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _gaps_are_conditions(self):
        missing = [gap for gap in self.justification_gaps if gap not in self.conditions]
        if missing:
            raise ArtifactError(
                f"transform {self.transform!r} lists justification gaps {missing} that "
                "are not among its conditions. An unjustified interchange is a "
                "condition on the result, not a footnote."
            )
        return self


class KernelAnalysis(Artifact):
    """Computed properties of a kernel."""

    artifact_type: Literal[ArtifactType.KERNEL_ANALYSIS] = ArtifactType.KERNEL_ANALYSIS
    kernel_name: str
    kernel_expression: str
    properties: dict[str, str] = Field(default_factory=dict)
    #: Properties established exactly, as opposed to numerically sampled.
    exact_properties: list[str] = Field(default_factory=list)


class OperatorExperiment(Artifact):
    """A discretized operator spectrum experiment.

    Always numerical. A finite discretization of a candidate Hilbert-Polya
    operator is evidence about the discretization.
    """

    artifact_type: Literal[ArtifactType.OPERATOR_EXPERIMENT] = ArtifactType.OPERATOR_EXPERIMENT
    operator_name: str
    discretization: str
    dimension: int
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)
    epistemic_status: Confidence = Confidence.NUMERICAL

    @model_validator(mode="after")
    def _finite_discretization_stays_numerical(self):
        allowed = {Confidence.NUMERICAL, Confidence.RIGOROUS_NUMERICAL, Confidence.SYNTHETIC}
        if self.epistemic_status not in allowed:
            raise ArtifactError(
                f"operator experiment {self.operator_name!r} claims "
                f"{self.epistemic_status.value!r}. A finite discretization is evidence "
                "about the discretization; matching Riemann zero counts does not make "
                "it a spectral determinant."
            )
        return self


# --------------------------------------------------------------------------
# Workstream C: rigorous numerics and inequalities
# --------------------------------------------------------------------------
class CertificateVerification(StrEnum):
    """Whether an independent checker accepted the certificate.

    A closed vocabulary rather than a free string. As a `str` field this was
    compared against the literal ``"accepted"``, so a typo or a new spelling
    read as "not accepted" -- fail-closed by luck rather than by design, and
    the mirror-image mistake would have read as accepted.

    Mirrors ``mathcert.verifiers.VerificationStatus``. Declared here rather than
    imported so ``contracts`` stays a leaf that every subsystem can depend on;
    the two are kept in step by ``VERIFICATION_STATUS_TO_CONFIDENCE``.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class InequalityCertificate(Artifact):
    """A solver-produced certificate for an inequality.

    The solver run is non-authoritative. Only a replayable, independently
    checked certificate may enter DRE reasoning -- so the fields that matter are
    the certificate and its verification, not the search that found it. Replay
    verifies the certificate; it does not reproduce solver ordering.
    """

    artifact_type: Literal[ArtifactType.INEQUALITY_CERTIFICATE] = ArtifactType.INEQUALITY_CERTIFICATE
    statement: str
    input_hash: str
    solver_family: str
    solver_version: str
    solver_options_hash: str
    random_seed: int | None = None
    certificate_format: str
    certificate_hash: str
    verifier_family: str | None = None
    verifier_version: str | None = None
    verification_status: CertificateVerification = CertificateVerification.UNKNOWN
    epistemic_status: Confidence = Confidence.UNKNOWN

    @model_validator(mode="after")
    def _unverified_certificate_is_not_evidence(self):
        verified = self.verification_status is CertificateVerification.ACCEPTED
        if not verified and self.epistemic_status not in {
            Confidence.UNKNOWN,
            Confidence.HEURISTIC,
            Confidence.NUMERICAL,
        }:
            raise ArtifactError(
                f"certificate {self.artifact_id!r} claims "
                f"{self.epistemic_status.value!r} with verification_status="
                f"{self.verification_status.value!r}. A floating SDP or SMT result "
                "counts for nothing until rationalized or enclosed and independently "
                "checked."
            )
        if verified and not (self.verifier_family and self.verifier_version):
            raise ArtifactError(
                f"certificate {self.artifact_id!r} is accepted but names no verifier. "
                "An acceptance nobody is accountable for is not an acceptance."
            )
        return self


# --------------------------------------------------------------------------
# Workstream E: literature
# --------------------------------------------------------------------------
class LiteratureVerdict(StrEnum):
    KNOWN_IDENTICAL = "known_identical"
    KNOWN_EQUIVALENT = "known_equivalent"
    KNOWN_STRONGER = "known_stronger"
    KNOWN_WEAKER = "known_weaker"
    LIKELY_RELATED = "likely_related"
    POTENTIALLY_NEW = "potentially_new"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class LiteratureMatch(Artifact):
    """A frozen literature comparison.

    Retrieval and extraction happen live and outside deterministic replay; what
    the supervisor consumes is this artifact, by hash. Replay reuses it and does
    not re-query the network or the model, because a supervisor decision that
    changes when a search index changes is not replayable.
    """

    artifact_type: Literal[ArtifactType.LITERATURE_MATCH] = ArtifactType.LITERATURE_MATCH
    query: str
    retrieval_provider: str
    #: Logical tick, not wall-clock: decisions must not observe time.
    retrieved_at_tick: int
    source_identifiers: list[str] = Field(default_factory=list)
    source_hashes: list[str] = Field(default_factory=list)
    normalized_theorem_statements: list[str] = Field(default_factory=list)
    theorem_assumptions: list[str] = Field(default_factory=list)
    extraction_model: str | None = None
    extraction_version: str | None = None
    verdict: LiteratureVerdict = LiteratureVerdict.INSUFFICIENT_EVIDENCE
    comparison_rationale: str = ""

    @model_validator(mode="after")
    def _a_match_must_be_citable(self):
        decisive = {
            LiteratureVerdict.KNOWN_IDENTICAL,
            LiteratureVerdict.KNOWN_EQUIVALENT,
            LiteratureVerdict.KNOWN_STRONGER,
            LiteratureVerdict.KNOWN_WEAKER,
        }
        if self.verdict in decisive:
            if not self.source_identifiers:
                raise ArtifactError(
                    f"verdict {self.verdict.value!r} names no source. A claimed "
                    "equivalence to the literature that cannot be cited is a "
                    "hallucination with a status field."
                )
            if not self.theorem_assumptions:
                raise ArtifactError(
                    f"verdict {self.verdict.value!r} records no theorem assumptions. "
                    "Two statements are not equivalent until their hypotheses are "
                    "compared."
                )
        return self


# --------------------------------------------------------------------------
# Workstream F: formalization
# --------------------------------------------------------------------------
class FormalizationReport(Artifact):
    """What a formalization attempt actually established.

    ``axioms_introduced`` is the load-bearing field. A Lean file that compiles
    because an axiom stands in for the missing theorem has proved nothing, and
    the ledger is what stops "it compiles" from being read as "it is proved".
    """

    artifact_type: Literal[ArtifactType.FORMALIZATION_REPORT] = ArtifactType.FORMALIZATION_REPORT
    target: str
    steps_formalized: list[str] = Field(default_factory=list)
    steps_assumed: list[str] = Field(default_factory=list)
    axioms_introduced: list[str] = Field(default_factory=list)
    missing_library_results: list[str] = Field(default_factory=list)
    remaining_obligations: list[str] = Field(default_factory=list)

    @property
    def fully_formalized(self) -> bool:
        return not (self.steps_assumed or self.axioms_introduced or self.remaining_obligations)

    @model_validator(mode="after")
    def _axioms_forbid_formal_verification(self):
        if self.epistemic_status is Confidence.FORMALLY_VERIFIED and not self.fully_formalized:
            raise ArtifactError(
                f"{self.target!r} claims formally verified while resting on "
                f"{len(self.axioms_introduced)} axiom(s), {len(self.steps_assumed)} "
                f"assumed step(s), and {len(self.remaining_obligations)} open "
                "obligation(s). A file that compiles with an axiom is not a proof."
            )
        return self


# --------------------------------------------------------------------------
# Supervisor
# --------------------------------------------------------------------------
class SupervisorDecision(Artifact):
    """The supervisor's classification of a hypothesis after a run."""

    artifact_type: Literal[ArtifactType.SUPERVISOR_DECISION] = ArtifactType.SUPERVISOR_DECISION
    hypothesis_ref: str
    classification: str
    survived: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    minimal_blocking_obligations: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class ResearchRunManifest(Artifact):
    """An immutable record of one research run.

    ``inputs_hash`` plus ``engine_fingerprint`` is the replay identity: the same
    inputs under the same engine must produce the same decisions, and if they do
    not, one of these two changed.
    """

    artifact_type: Literal[ArtifactType.RESEARCH_RUN_MANIFEST] = ArtifactType.RESEARCH_RUN_MANIFEST
    run_id: str
    #: Logical tick, not wall-clock.
    started_at_tick: int
    inputs_hash: str
    engine_fingerprint: str
    produced_artifacts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    mathematical_role: Role = Role.PROCEDURAL
    epistemic_status: Confidence = Confidence.AUTHORITATIVE_POLICY

    @field_serializer("produced_artifacts", "decisions")
    def _sorted(self, values: list[str]) -> list[str]:
        return sorted(values)


#: Every artifact contract, for exhaustiveness tests and registry lookup.
ARTIFACT_MODELS: dict[ArtifactType, type[Artifact]] = {
    ArtifactType.PROOF_OBLIGATION: ProofObligation,
    ArtifactType.PROPERTY_ASSERTION: PropertyAssertion,
    ArtifactType.MATHEMATICAL_OBJECT: MathematicalObject,
    ArtifactType.MODEL_FAMILY_SPEC: ModelFamilySpec,
    ArtifactType.SYNTHETIC_MODEL: SyntheticModel,
    ArtifactType.COUNTEREXAMPLE: CounterexampleArtifact,
    ArtifactType.TRANSFORM_DERIVATION: TransformDerivation,
    ArtifactType.KERNEL_ANALYSIS: KernelAnalysis,
    ArtifactType.OPERATOR_EXPERIMENT: OperatorExperiment,
    ArtifactType.INEQUALITY_CERTIFICATE: InequalityCertificate,
    ArtifactType.LITERATURE_MATCH: LiteratureMatch,
    ArtifactType.FORMALIZATION_REPORT: FormalizationReport,
    ArtifactType.OBLIGATION_DISCHARGE_DECISION: ObligationDischargeDecision,
    ArtifactType.SUPERVISOR_DECISION: SupervisorDecision,
    ArtifactType.RESEARCH_RUN_MANIFEST: ResearchRunManifest,
}
