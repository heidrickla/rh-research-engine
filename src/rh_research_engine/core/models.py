from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_serializer, model_validator

from .. import __version__


class ClaimStatus(StrEnum):
    """How a claim came to be believed.

    `INFERRED` was missing, and its absence had a cost. Reasoning that is
    neither measured nor read from a source had nowhere to go, so it went to
    `KNOWN` -- which maps to `Confidence.KNOWN`, the class whose own definition
    is "established external mathematics, needs a literature citation", and
    which `ArtifactRecord` refuses to let any worker assert for exactly that
    reason. C006 sat there: its evidence line is a known fact about the singular
    series, and its statement is a conclusion drawn from that fact about THIS
    project's localized Gamma shell, which no literature could be citing.

    The distinction the other statuses do not draw is between how a conclusion
    was reached and what was fed into it. `KNOWN` and `INFERRED` can rest on
    the same evidence line and differ entirely in what they assert.
    """

    HYPOTHESIS = "hypothesis"
    NUMERICAL = "numerical"
    SYMBOLIC = "symbolic"
    PROVED = "proved"
    KNOWN = "known"
    #: Concluded here by reasoning, from evidence that may itself be cited.
    #: Carries no deductive force: an inference nobody has checked is a
    #: hypothesis with an argument attached.
    INFERRED = "inferred"
    EQUIVALENT_RH = "equivalent_rh"
    FALSE = "false"


class EvidenceClass(StrEnum):
    """How strong a piece of evidence is.

    Defined here rather than in ``dre`` because an ``ExperimentResult`` must
    declare its own class at the point of production. Letting the caller pick
    the class at export time is exactly how a numerical run becomes a proof.
    """

    NUMERICAL = "numerical"
    RIGOROUS_NUMERICAL = "rigorous-numerical"
    SYMBOLIC = "symbolic"
    PROVED = "proved"
    KNOWN = "known"
    HEURISTIC = "heuristic"
    COUNTEREXAMPLE = "counterexample"


#: Classes a deterministic math worker is never allowed to assert about its own
#: output. A worker computes; it does not decide that what it computed is a
#: proof or an established theorem. Both require a human or a formal checker
#: outside this package.
WORKER_FORBIDDEN_CLASSES = frozenset({EvidenceClass.PROVED, EvidenceClass.KNOWN})

#: Classes that carry no deductive force on their own.
NON_DEDUCTIVE_CLASSES = frozenset(
    {EvidenceClass.NUMERICAL, EvidenceClass.RIGOROUS_NUMERICAL, EvidenceClass.HEURISTIC}
)


class Claim(BaseModel):
    id: str
    statement: str
    status: ClaimStatus = ClaimStatus.HYPOTHESIS
    assumptions: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    tags: set[str] = Field(default_factory=set)
    evidence: list[str] = Field(default_factory=list)
    #: Where a KNOWN claim is established, outside this package. There was no
    #: such field, so `KNOWN` promised a citation the model could not hold, and
    #: the one claim carrying that status had none. `evidence` is not a
    #: substitute: it says what was fed in, not where the conclusion is proved.
    citation: str = ""
    implied_theta_upper: float | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _known_needs_a_citation(self) -> Claim:
        """The guard `ArtifactRecord` already has, on the path claims take.

        `WORKER_FORBIDDEN_CLASSES` stops an experiment asserting KNOWN about its
        own output, and nothing did the same for a claim -- so the unguarded
        path is the one that got used. A guard that is not on the path is not a
        guard.
        """
        if self.status is ClaimStatus.KNOWN and not self.citation.strip():
            raise ValueError(
                f"claim {self.id!r} is status 'known' with no citation. "
                "'known' means established outside this package; a conclusion "
                "drawn here from cited inputs is 'inferred', however sound."
            )
        return self

    @field_serializer("tags")
    def _sort_tags(self, tags: set[str]) -> list[str]:
        """Serialize tags in a stable order.

        A ``set`` iterates in hash order, which is seeded per process, so
        without this the same claim registry serializes to different bytes on
        different runs and any content hash over ``claims.json`` is worthless.
        """
        return sorted(tags)


class ExperimentResult(BaseModel):
    name: str
    parameters: dict[str, Any]
    metrics: dict[str, float | int | str | bool]
    observations: list[str] = Field(default_factory=list)

    # Provenance, declared by the worker that produced the result. These are
    # not caller-supplied labels: they travel with the record so the DRE
    # exporter cannot invent a stronger class or a fresh independence group.
    evidence_class: EvidenceClass = EvidenceClass.NUMERICAL
    method_family: str = "python-numpy"
    worker_version: str = __version__
    assumptions: list[str] = Field(default_factory=list)


class NoGoRule(BaseModel):
    id: str
    trigger_tags: set[str] = Field(default_factory=set)
    #: Case-insensitive substrings matched against a claim's statement, so a
    #: refuted route cannot be resurrected simply by renaming its tag.
    trigger_phrases: list[str] = Field(default_factory=list)
    message: str
    fatal: bool = True

    @field_serializer("trigger_tags")
    def _sort_trigger_tags(self, tags: set[str]) -> list[str]:
        return sorted(tags)
