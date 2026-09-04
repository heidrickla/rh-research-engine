"""Axis: research value -- does this move RH forward?

Two questions, deliberately kept separate:

    frontier_relevant   would proving this matter?
    advances_frontier   has a qualifying result actually moved the frontier?

A research target is relevant and not advancing. A classical ``A <=> RH`` is
neither: proving yet another reformulation of ``A`` moves nothing. **Only
``advances_frontier`` may gate rigorous closure or progress accounting.**

This module is the single implementation. ``Hypothesis`` and ``PropertyRecord``
previously each derived ``frontier_relevant`` themselves, with the same rule
written twice -- which is how two copies of a rule start disagreeing.

Every verdict comes back with its reasons. A bare ``False`` tells a researcher
nothing about what would change it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .epistemic import Confidence, is_rigorous
from .roles import NON_ADVANCING_ROLES, Role, is_mathematical

#: Names that are always *derived*, never stored or supplied.
#:
#: Pydantic defaults to ``extra="ignore"``, so passing one of these used to
#: construct successfully and drop the value on the floor -- the caller gets no
#: error, reasonably concludes it worked, and reads the opposite later with no
#: way to explain it. Every model carrying a frontier axis forbids them by name.
DERIVED_FIELD_NAMES = frozenset(
    {
        "frontier_relevant",
        "advances_frontier",
        "property_extractable",
        "is_rigorous",
        "usable_as_rule",
        "actionable",
        "open_qualifiers",
        "frontier",
    }
)


class DerivedFieldError(ValueError):
    """A derived verdict was supplied as input."""


def reject_derived_inputs(data: object, model_name: str) -> object:
    """Refuse any attempt to supply a derived verdict.

    Applies to constructors, ``model_validate`` from stored JSON, and legacy
    migrations alike -- all three funnel through the same before-validator, so
    a record cannot acquire frontier advancement by being written down.
    """
    if not isinstance(data, dict):
        return data
    supplied = sorted(DERIVED_FIELD_NAMES & set(data))
    if supplied:
        raise DerivedFieldError(
            f"{model_name} cannot accept {supplied}: these are derived from "
            "mathematical_role, epistemic_status, rh_equivalent, "
            "discharged_obligations and open qualifiers. Set those instead. "
            "Supplying a verdict directly is how a restatement of RH acquires "
            "frontier credit by assertion."
        )
    return data


class FrontierAssessment(BaseModel):
    """The frontier axes, with the reasoning that produced them."""

    frontier_relevant: bool
    advances_frontier: bool
    property_extractable: bool
    #: Why each negative verdict was reached, in the order the checks ran.
    reasons: list[str] = Field(default_factory=list)

    def explain(self) -> str:
        if not self.reasons:
            return "advances the frontier: rigorous, mathematical, and not a restatement"
        return "; ".join(self.reasons)


def assess(
    *,
    role: Role,
    confidence: Confidence,
    rh_equivalent: bool = False,
    qualifying_discharges: list[str] | None = None,
    open_qualifiers: list[str] | None = None,
) -> FrontierAssessment:
    """Derive the frontier axes from the other two axes plus circularity.

    ``qualifying_discharges`` must already be *resolved*: references that were
    checked against real ProofObligation artifacts carrying qualifying evidence.
    Raw prose does not belong here. Resolution lives in
    :mod:`rh_research_engine.contracts.discharge` and is performed by the
    caller, which keeps this module a leaf that every subsystem can import.

    The default is the empty list, so a caller that supplies no registry gets no
    frontier credit. The safe reading of "this claim cannot be checked" is that
    it does not hold.

    ``open_qualifiers`` is every assumption and domain condition still
    attached. A statement with something hanging on it is conditional, and a
    conditional result has not moved an unconditional frontier.
    """
    discharged = list(qualifying_discharges or [])
    qualifiers = list(open_qualifiers or [])
    reasons: list[str] = []

    mathematical = is_mathematical(role)
    if not mathematical:
        reasons.append(
            f"role {role.value!r} is meta, not mathematics: it describes how the "
            "research is conducted, so it neither yields properties nor moves a frontier"
        )

    relevant = mathematical
    if mathematical and role in NON_ADVANCING_ROLES:
        relevant = False
        reasons.append(
            f"role {role.value!r} cannot move a frontier: a definition is true by "
            "construction and a no-go route is a dead end"
        )
    if relevant and rh_equivalent and not discharged:
        relevant = False
        reasons.append(
            "restates RH rather than advancing toward it; this changes only when a "
            "discharge reference resolves to a DISCHARGED ProofObligation whose "
            "evidence is rigorous, unconditional and not itself RH-equivalent"
        )

    advancing = relevant
    if advancing and not is_rigorous(confidence):
        advancing = False
        reasons.append(
            f"confidence {confidence.value!r} is not rigorous, so nothing has been "
            "established yet -- worth proving, not proved"
        )
    if advancing and qualifiers:
        advancing = False
        reasons.append(
            f"still conditional on {len(qualifiers)} open qualifier(s): "
            f"{sorted(qualifiers)}"
        )

    return FrontierAssessment(
        frontier_relevant=relevant,
        advances_frontier=advancing,
        property_extractable=mathematical,
        reasons=reasons,
    )


def usable_as_rule(*, role: Role, confidence: Confidence, open_qualifiers: list[str] | None = None) -> bool:
    """Whether this may be applied as an established rewriting step.

    Weaker than ``advances_frontier`` on purpose: a rigorous RH-equivalence is a
    perfectly good transformation rule even though invoking it earns nothing.
    """
    return (
        is_mathematical(role)
        and is_rigorous(confidence)
        and not list(open_qualifiers or [])
    )
