from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from ..contracts.epistemic import Confidence
from ..contracts.frontier import FrontierAssessment, assess, reject_derived_inputs
from ..contracts.lifecycle import CLOSED_LIFECYCLES, OPEN_LIFECYCLES, HypothesisLifecycle
from ..contracts.roles import Role

# `HypothesisState` moved to `.compat` and is deprecated; `contracts.mappings`
# owns the translation. It is deliberately not imported here -- this module
# stores the canonical axes, and reading a legacy record goes through
# `_migrate_legacy_state` below rather than through the old enum.


class ProofGap(BaseModel):
    id: str
    description: str
    blocking: bool = True
    discharged_by: list[str] = Field(default_factory=list)

    @property
    def open(self) -> bool:
        return self.blocking and not self.discharged_by


class FalsificationTest(BaseModel):
    id: str
    description: str
    cost: int = Field(ge=0)
    command: str | None = None
    expected_failure_mode: str | None = None


class Hypothesis(BaseModel):
    """A research hypothesis, carrying each axis separately.

    ``lifecycle`` says where work stands. ``epistemic_status`` says how well
    established the statement is. ``mathematical_role`` says what kind of thing
    it is. The frontier axes are *derived* from those plus circularity, never
    stored -- a stored copy of a derived fact is a second thing that can
    disagree with the first.
    """

    # extra="forbid" so an unknown key is an error rather than a silent drop.
    model_config = ConfigDict(extra="forbid")

    id: str
    statement: str
    assumptions: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    tags: set[str] = Field(default_factory=set)

    lifecycle: HypothesisLifecycle = HypothesisLifecycle.PROPOSED
    epistemic_status: Confidence = Confidence.CONJECTURAL
    mathematical_role: Role = Role.CLAIM

    proof_gaps: list[ProofGap] = Field(default_factory=list)
    falsification_tests: list[FalsificationTest] = Field(default_factory=list)

    rh_equivalent: bool = False
    discharged_obligations: list[str] = Field(default_factory=list)
    open_obligation_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    blocker_refs: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_serializer("tags")
    def _sort_tags(self, tags: set[str]) -> list[str]:
        return sorted(tags)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_state(cls, data: Any) -> Any:
        """Translate a stored ``state`` into the canonical axes.

        Migration on read, so an existing queue keeps loading. The legacy field
        carried two facts at once, and both are recovered: the workflow position
        goes to ``lifecycle`` and the verdict to ``epistemic_status``. A record
        that already has ``lifecycle`` is left alone.
        """
        data = reject_derived_inputs(data, "Hypothesis")
        if not isinstance(data, dict) or "state" not in data:
            return data
        from .compat import migrate_legacy_payload

        # The legacy `advanced` state implied advancement. It maps to an
        # epistemic status, never to the verdict itself, so a migrated record
        # earns advancement the same way any other record does or not at all.
        return migrate_legacy_payload(data)

    # There is deliberately no validator forbidding "RH-equivalent and
    # advancing". The previous model had one, because `state=ADVANCED` was an
    # explicit assertion of progress that could contradict the facts. Progress
    # is now *derived*, so the contradiction is unrepresentable rather than
    # merely rejected -- a stronger guarantee, and the reason
    # test_rh_equivalence_can_never_advance_without_discharge checks it across
    # the whole input product instead of on one hand-picked case.

    # -- derived -----------------------------------------------------------
    @property
    def open_proof_gaps(self) -> list[ProofGap]:
        return [gap for gap in self.proof_gaps if gap.open]

    @property
    def open_qualifiers(self) -> list[str]:
        """Everything still hanging on the statement.

        Assumptions and unclosed proof gaps are the same kind of thing from the
        frontier's point of view: a reason the result is not yet unconditional.
        """
        return sorted(
            set(self.assumptions)
            | {gap.description for gap in self.open_proof_gaps}
            | set(self.open_obligation_refs)
        )

    def frontier_with(
        self,
        *,
        obligations: dict | None = None,
        evidence: dict | None = None,
        decisions: dict | None = None,
        receipts: dict | None = None,
    ) -> FrontierAssessment:
        """Frontier axes, resolving discharge references against a registry.

        With no registry nothing resolves, so an RH-equivalence stays circular.
        A prose discharge cannot unlock progress: the reference has to name a
        DISCHARGED ProofObligation whose evidence is rigorous, unconditional and
        not itself RH-equivalent.
        """
        from ..contracts.discharge import resolve_discharges

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
        """Frontier axes with no obligation registry -- nothing discharges."""
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

    @property
    def cheapest_falsification_test(self) -> FalsificationTest | None:
        if not self.falsification_tests:
            return None
        return sorted(self.falsification_tests, key=lambda test: (test.cost, test.id))[0]

    @property
    def actionable(self) -> bool:
        """Can work proceed on this now?

        Distinct from frontier value: a hypothesis can be worth pursuing and
        still be unworkable because it is blocked or has no cheap way to fail.
        """
        if self.lifecycle in CLOSED_LIFECYCLES:
            return False
        if self.lifecycle is HypothesisLifecycle.BLOCKED or self.blocker_refs:
            return False
        return self.frontier_relevant and self.cheapest_falsification_test is not None

    def explain_actionability(self) -> str:
        if self.lifecycle in CLOSED_LIFECYCLES:
            return f"lifecycle is {self.lifecycle.value}; work has concluded"
        if self.lifecycle is HypothesisLifecycle.BLOCKED or self.blocker_refs:
            blockers = self.blocker_refs or ["unspecified"]
            return f"blocked on {blockers}"
        if not self.frontier_relevant:
            return self.frontier.explain()
        if self.cheapest_falsification_test is None:
            return "no falsification test is registered, so there is no cheap way to be wrong"
        return f"actionable via {self.cheapest_falsification_test.id}"

    def stable_hash(self) -> str:
        payload = self.model_dump(mode="json")
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NextStep(BaseModel):
    hypothesis_id: str
    falsification_test_id: str
    description: str
    command: str | None = None
    reason: str


#: Bumped when `state` split into lifecycle + epistemic_status.
QUEUE_SCHEMA_VERSION = "2"

#: Identifies the translation applied, so a record says which rules produced it.
QUEUE_MIGRATION_ID = "hypothesis-contract-v1"
QUEUE_MAPPING_VERSION = 1

#: Every on-disk version this build can read. Anything else is refused rather
#: than normalised, in either direction.
KNOWN_QUEUE_SCHEMA_VERSIONS = frozenset({"1", "2"})


class QueueSchemaError(ValueError):
    """The stored queue declares a schema this build cannot safely read."""


def _payload_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class QueueMigrationRecord(BaseModel):
    """What was translated, by which rules, and three distinct hashes.

    Three, not two, because they answer different questions and a rollback
    needs all of them:

    ``source_file_sha256``    the bytes on disk. What a rollback restores, and
                              the only one that changes when the file is merely
                              reformatted.
    ``source_semantic_hash``  the content as parsed, before translation. Two
                              files that differ only in whitespace share this.
    ``normalized_semantic_hash``  the content after translation. Equal to the
                              source semantic hash exactly when the migration
                              was a no-op.

    Collapsing them loses the ability to tell a reformat from a translation, or
    "this migration ran" from "this file was already in the target shape".
    """

    migration_id: str = QUEUE_MIGRATION_ID
    mapping_version: int = QUEUE_MAPPING_VERSION
    source_schema_version: str
    target_schema_version: str = QUEUE_SCHEMA_VERSION
    #: Set by the store, which is the only layer that sees the raw bytes.
    source_file_sha256: str | None = None
    source_semantic_hash: str
    normalized_semantic_hash: str | None = None


class HypothesisQueue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = QUEUE_SCHEMA_VERSION
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    #: Present only when this queue was read from an older on-disk schema.
    migration: QueueMigrationRecord | None = None

    @model_validator(mode="before")
    @classmethod
    def _record_and_upgrade_schema_version(cls, data: Any) -> Any:
        """Loading normalises every record, so the version must say so.

        Records are migrated on read, so a file loaded as v1 holds v2 records
        the moment it is in memory. Leaving the declared version at "1" would
        mean re-saving a file that claims a schema it no longer has -- and the
        next migration would read that claim and skip work it needed to do.

        This only changes what is held in memory. Writing is the store's job,
        so loading a v1 file never touches it on disk.
        """
        if not isinstance(data, dict):
            return data
        # A *stored* queue with no declared version is rejected by the store,
        # which is the layer that knows it is reading a file. In-memory
        # construction legitimately omits it and takes the current default.
        stored = data.get("schema_version")
        if stored is None or stored == QUEUE_SCHEMA_VERSION:
            return data
        if stored not in KNOWN_QUEUE_SCHEMA_VERSIONS:
            # A version this build does not know is not something to normalise.
            # Treating "3" as a legacy version would *downgrade* a file written
            # by a newer build and record the downgrade as a migration -- data
            # loss dressed as provenance.
            raise QueueSchemaError(
                f"queue schema_version {stored!r} is not one this build understands "
                f"(known: {sorted(KNOWN_QUEUE_SCHEMA_VERSIONS)}). Refusing to read it: "
                "a newer file must be handled by a newer build, not silently "
                "rewritten to an older shape."
            )
        payload = dict(data)
        payload["schema_version"] = QUEUE_SCHEMA_VERSION
        payload.setdefault(
            "migration",
            {
                "migration_id": QUEUE_MIGRATION_ID,
                "mapping_version": QUEUE_MAPPING_VERSION,
                "source_schema_version": stored,
                "target_schema_version": QUEUE_SCHEMA_VERSION,
                "source_semantic_hash": _payload_hash(
                    {k: v for k, v in data.items() if k != "schema_version"}
                ),
            },
        )
        return payload

    @model_validator(mode="after")
    def _stamp_normalized_hash(self):
        if self.migration is not None and self.migration.normalized_semantic_hash is None:
            self.migration.normalized_semantic_hash = _payload_hash(
                self.model_dump(mode="json", exclude={"migration", "schema_version"})
            )
        return self

    def sorted_hypotheses(self) -> list[Hypothesis]:
        return sorted(
            self.hypotheses,
            key=lambda h: (
                not h.advances_frontier,
                not h.actionable,
                len(h.open_proof_gaps),
                h.cheapest_falsification_test.cost if h.cheapest_falsification_test else 10**9,
                h.id,
            ),
        )

    def next_step(self) -> NextStep | None:
        for hypothesis in self.sorted_hypotheses():
            test = hypothesis.cheapest_falsification_test
            if not hypothesis.actionable or test is None:
                continue
            return NextStep(
                hypothesis_id=hypothesis.id,
                falsification_test_id=test.id,
                description=test.description,
                command=test.command,
                reason="frontier-relevant hypothesis with the cheapest available falsification test",
            )
        return None

    def open_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.lifecycle in OPEN_LIFECYCLES]

    def upsert(self, hypothesis: Hypothesis) -> None:
        remaining = [item for item in self.hypotheses if item.id != hypothesis.id]
        remaining.append(hypothesis)
        self.hypotheses = sorted(remaining, key=lambda item: item.id)
