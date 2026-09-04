"""Records for the pattern-detection research function.

WHY THIS EXISTS. The productive move in an autonomous mathematics run is often
not the one the task asked for. In Anthropic's Riemann-zeta run the sub-agent
briefed to develop a "negative index" idea argued, fifty minutes in and before
any computation, that the brief's own quantity "is IDENTICALLY ZERO in every
computable range ... There is nothing to fit"; it then measured a quantity the
brief had asserted but not asked to have measured, found an exact regularity in
the counts of zeros -- equality in thirteen cases out of thirteen -- and
followed that instead. The result was the theorem; the assigned route was
empty.

Three separable capabilities, and this package makes each a function rather
than a piece of luck:

  1. AUDIT THE PREMISE before fitting it. A quantity that is constant over the
     computable range has nothing to fit, and finding that out is a result.
  2. MEASURE WHAT WAS NOT ASKED FOR. Scan every column and every pair, not the
     one named in the task.
  3. ESCALATE AN EXACT REGULARITY. `>=` holding with equality in n of n cases
     is not a statistical trend, it is a structure asking to be proved.

WHAT THIS IS NOT, structurally. A regularity in sampled data is a CONJECTURE.
`PatternFinding` refuses any confidence in the rigorous set outright -- see
`_reject_rigorous_confidence` -- so no amount of agreement in the data can turn
into a claim that something is established. Thirteen of thirteen is a reason to
go looking for a proof, and it is not one.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..contracts.epistemic import RIGOROUS, Confidence
from ..properties.models import MathematicalRole


class PremiseVerdict(StrEnum):
    """Whether an assigned quantity has anything in it to find."""

    #: It varies. Fitting it is a real question.
    LIVE = "live"
    #: Constant over every sampled point -- there is nothing to fit.
    EMPTY = "empty"
    #: It varies, but only in a way the target already forces. Fitting it
    #: would recover the assumption rather than test it.
    DEGENERATE = "degenerate"
    #: Too few points, or none evaluated. Says nothing either way, and says so
    #: rather than defaulting to LIVE.
    INCONCLUSIVE = "inconclusive"


class RegularityKind(StrEnum):
    """The shape of an exact relation found in sampled data."""

    #: Every value is a whole number where nothing required it to be.
    INTEGRALITY = "integrality"
    #: Two columns agree at every sampled point.
    EXACT_EQUALITY = "exact_equality"
    #: `a >= b` throughout, AND equal in every case. A bound that is never
    #: slack is not a bound, it is an identity nobody has written down yet --
    #: this is the shape of the Anthropic finding.
    SATURATED_BOUND = "saturated_bound"
    #: `a / b` takes the same value throughout.
    CONSTANT_RATIO = "constant_ratio"
    #: `a - b` takes the same value throughout.
    CONSTANT_DIFFERENCE = "constant_difference"
    #: Never changes sign, where nothing required that.
    SIGN_CONSTANCY = "sign_constancy"


class Observation(BaseModel):
    """One named column of measured values.

    Values are carried as strings so an exact input stays exact. A `Fraction`
    or a SymPy `Rational` that is squeezed through a float on the way in cannot
    be recovered, and "equality in 13 of 13" is a claim about exactness -- it
    is not the same finding at 1e-16.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    values: list[str]
    #: What each row IS -- the parameter value it was measured at.
    #:
    #: Without these a tight subset is reported as "12 of 39", which is a
    #: number. With them it is `{2,3,5,7,11,...}`, which a reader recognises
    #: on sight. The whole value of a partial saturation is in which cases.
    labels: list[str] | None = None
    #: What produced it, and whether the task asked for it. An unrequested
    #: column is the interesting kind; recording which is which is what makes
    #: "it was not asked for" reportable afterwards.
    requested: bool = True
    source: str | None = None


class PremiseAudit(BaseModel):
    """The verdict on whether an assigned quantity is worth fitting."""

    model_config = ConfigDict(extra="forbid")

    quantity: str
    verdict: PremiseVerdict
    sampled: int
    #: Plain sentence naming what was found, for a report a human reads.
    evidence: str
    #: Never rigorous, and never promotable. An audit is an observation about a
    #: finite sample.
    confidence: Confidence = Confidence.NUMERICAL


class PatternFinding(BaseModel):
    """An exact regularity observed in sampled data.

    A candidate conjecture with its evidence attached, and nothing more. The
    validator below is the whole safety property of this package.
    """

    model_config = ConfigDict(extra="forbid")

    kind: RegularityKind
    #: The relation, written out, so a human can decide whether to chase it.
    statement: str
    columns: list[str]
    #: Other names that carried the SAME values on every examined row, per
    #: column. The relation holds under those names too, and holds no harder
    #: for it.
    #:
    #: Recorded rather than dropped, because a merge that says nothing is
    #: indistinguishable from a comparison that never happened -- and because
    #: the agreement is range-dependent. Two implementations equal on the rows
    #: examined may part company higher up, and then they really are two
    #: quantities; the record has to say which claim it is making.
    aliases: dict[str, list[str]] = Field(default_factory=dict)
    #: Cases where the relation held, out of cases examined. Reported as a
    #: pair because "13 of 13" and "13 of 400" are different findings and a
    #: single ratio hides which one this is.
    support: int
    sampled: int
    #: True when the relation holds EXACTLY rather than within a tolerance.
    #: A saturated bound that is only saturated to 1e-12 is a different claim.
    exact: bool
    #: Whether any column involved was measured without being asked for.
    from_unrequested: bool = False
    #: The rows where the relation was TIGHT, when that is a proper subset.
    witnesses: list[str] = Field(default_factory=list)
    #: One line describing the set the relation is tight on -- the primes, a
    #: residue class, or UNRECOGNISED. The last is the interesting one: a
    #: relation holding exactly on a set with no name is structure nobody has
    #: written down, which is the reason to look at all.
    character: str = ""
    #: The predicate that named the tight set, when one did. Empty means
    #: nothing named it.
    #:
    #: A FIELD rather than a substring of `character`, because selecting the
    #: unexplained findings is what the open ledger is for, and selecting them
    #: by searching a display string for "UNRECOGNISED" is a filter that
    #: silently empties the day somebody rewords the headline.
    characterised: str = ""
    #: How much this constrains. Zero for a property of a single column:
    #: integrality of a column of counts is a fact about the generator, not
    #: about the mathematics, and reporting it at the same weight as a
    #: relation buries the relation.
    surprise: int = 0
    confidence: Confidence = Confidence.CONJECTURAL
    role: MathematicalRole = MathematicalRole.CLAIM
    evidence: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _reject_rigorous_confidence(cls, value: Confidence) -> Confidence:
        """A regularity in data cannot be filed as established mathematics.

        This is the one invariant that matters here. Pattern detection exists
        to point at things worth proving, and the failure mode it invites is
        exactly the one this repository is built against: agreement in every
        sampled case reads like certainty, and thirteen of thirteen is a
        reason to look for a proof rather than a substitute for one.

        Refused at construction rather than checked at export, because a
        record that can be built wrong will eventually be built wrong.
        """
        if value in RIGOROUS:
            raise ValueError(
                f"a pattern finding may not claim {value.value!r}: a regularity "
                "in sampled data is a conjecture, however many cases agree"
            )
        return value

    @property
    def unexplained(self) -> bool:
        """A tight set was described, and nothing named it.

        The ledger's selector. Note it is FALSE when no tight set was found at
        all: an unanimous identity has nothing to characterise, and calling
        that "unexplained" would file every identity as an open question.
        """
        return bool(self.character) and not self.characterised

    @property
    def unanimous(self) -> bool:
        """Held in every case examined -- the shape worth escalating."""
        return self.sampled > 0 and self.support == self.sampled
