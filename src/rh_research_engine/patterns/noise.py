"""Filters for findings that have been VERIFIED as noise.

WHY THIS RATHER THAN A CLEVERER DETECTOR. There is always noise and there are
occasionally surprises, and no scoring rule separates them in advance -- the
first blind run over the corpus's own functions ranked `n - (n+1) = -1` above
`psi(x) = sum Lambda(n)`, because it has no way to know the first column was
defined as `n+1` a moment earlier. A detector tuned hard enough to drop that
would also drop the thing worth finding.

So noise is retired by triage instead of by prediction: a finding is looked at
once, judged an artifact, and recorded with the reason. It stops appearing.
Everything not yet judged keeps appearing, which is the point -- the surprise
is in what has not been explained yet.

WHAT KEEPS THIS HONEST.

  * A rule must carry a REASON. "Noise" without one is indistinguishable from
    a result somebody found inconvenient.
  * A rule names its COLUMNS. Suppressing every integrality finding would also
    suppress an integrality that mattered; suppressing integrality OF ONE
    COLUMN, because that column counts things, does not.
  * Suppression is COUNTED and reported, never silent. A filter that hides how
    much it hid is a filter that will eventually hide the discovery.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .models import PatternFinding, RegularityKind


class NoiseGround(StrEnum):
    """How a finding was established to be noise.

    Recorded because the grounds differ in how much they should be trusted
    later. An artifact of how a column was built is settled forever; "a human
    looked at it once" is a judgement that a later reader may want to revisit,
    and cannot revisit if the record does not say that is what it was.
    """

    #: The relation follows from how a column was constructed. `n_plus_1` was
    #: defined as `n + 1`, so `n - n_plus_1 = -1` is arithmetic, not structure.
    CONSTRUCTION = "construction"
    #: A known identity, already in the literature and already indexed. Real
    #: mathematics, and not news. Distinct from ELEMENTARY: this one CITES.
    KNOWN_IDENTITY = "known_identity"
    #: Settled by a short argument, written out in the reason. Real
    #: mathematics, and not something the scan discovered.
    #:
    #: Added because the two findings it first retired -- `phi(n) >= Omega(n)`
    #: tight on {2, 4, 6}, and `n - phi(n) >= Omega(n)` tight on the primes
    #: together with 4 -- fit none of the others. They are not artifacts of a
    #: column, they are not in a paper anybody can point at, and calling them
    #: TRIAGED would file a checkable argument under "the weakest ground, and
    #: the one most worth re-examining". Recording something false about how a
    #: decision was reached is worse than a coarse vocabulary.
    #:
    #: **This is not an epistemic status.** It says a person wrote down an
    #: argument, not that this package established anything: a `PatternFinding`
    #: still refuses every rigorous confidence, and retiring one does not touch
    #: its confidence at all. The value is deliberately not spelled `proved` --
    #: `Confidence.PROVED` exists and a ground that collided with it would be
    #: read as one, which `test_no_noise_ground_is_spelled_like_a_confidence`
    #: exists to prevent.
    ELEMENTARY = "elementary"
    #: True of the generator rather than the mathematics -- integrality of a
    #: column that counts things.
    GENERATOR = "generator"
    #: Judged uninteresting by a person. The weakest ground, and the one most
    #: worth re-examining.
    TRIAGED = "triaged"


class NoiseRule(BaseModel):
    """One retired finding.

    Matches on the columns and optionally the kind. Deliberately narrow: a
    rule that matched on kind alone would retire a whole class of relation
    across every column, which is how a filter starts eating results.
    """

    model_config = ConfigDict(extra="forbid")

    columns: list[str]
    reason: str
    ground: NoiseGround
    #: None matches any kind over those columns.
    kind: RegularityKind | None = None

    def matches(self, finding: PatternFinding) -> bool:
        if self.kind is not None and finding.kind is not self.kind:
            return False
        return set(self.columns) == set(finding.columns)


class Retirement(BaseModel):
    """One finding, and the rule that retired it.

    The pairing matters downstream. A count of what was hidden is enough to
    audit the filter, and it is not enough to tell the open ledger that a
    relation it holds an entry for was EXPLAINED rather than lost -- for which
    the ledger needs the finding, to recognise it, and the reason, to record.
    """

    model_config = ConfigDict(extra="forbid")

    finding: PatternFinding
    reason: str
    ground: NoiseGround


class Suppression(BaseModel):
    """What a filtering pass kept, and what it removed."""

    model_config = ConfigDict(extra="forbid")

    kept: list[PatternFinding] = Field(default_factory=list)
    #: Reason -> how many findings it retired this pass. Reported rather than
    #: dropped: a filter that cannot say how much it hid is not auditable.
    removed: dict[str, int] = Field(default_factory=dict)
    #: What was retired, in full, each with the rule that retired it.
    #:
    #: The counts above cannot answer "is this open finding gone, or
    #: explained?", and answering it wrong is the failure this package is
    #: built around: a triage decision that reads as a refutation puts "was a
    #: regularity of the narrower range" into the record about a relation
    #: nothing has touched. See `ledger.RevisitVerdict.RETIRED`.
    retired: list[Retirement] = Field(default_factory=list)

    @property
    def removed_total(self) -> int:
        return sum(self.removed.values())


class NoiseRegistry(BaseModel):
    """The accumulated triage."""

    model_config = ConfigDict(extra="forbid")

    rules: list[NoiseRule] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> NoiseRegistry:
        if not path.exists():
            return cls()
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="",
        )

    def add(self, rule: NoiseRule) -> bool:
        """Record a rule. False when an identical one is already present."""
        for existing in self.rules:
            if (
                set(existing.columns) == set(rule.columns)
                and existing.kind == rule.kind
            ):
                return False
        self.rules.append(rule)
        return True

    def apply(self, findings: list[PatternFinding]) -> Suppression:
        """Filter, counting what went and why."""
        result = Suppression()
        for finding in findings:
            rule = next((r for r in self.rules if r.matches(finding)), None)
            if rule is None:
                result.kept.append(finding)
            else:
                result.removed[rule.reason] = result.removed.get(rule.reason, 0) + 1
                result.retired.append(
                    Retirement(
                        finding=finding, reason=rule.reason, ground=rule.ground
                    )
                )
        return result
