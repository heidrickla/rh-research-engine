"""Axis: what kind of thing a record is.

Orthogonal to confidence. A governance rule such as "numerical evidence cannot
promote a theorem to proved" is fully authoritative and carries no mathematical
confidence at all; a conjecture is a mathematical claim with very little. The
two facts are independent and must not share a field.

Meta roles are never property-extractable: minting a mathematical property from
a policy statement lets later code mistake the rules of the game for a fact
about zeta.
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Canonical mathematical role."""

    # --- mathematical: statements about the mathematics ---------------------
    #: Introduces notation or fixes a construction. Carries no truth claim.
    DEFINITION = "mathematical_definition"
    #: An equality that holds identically on its stated domain.
    IDENTITY = "mathematical_identity"
    #: An upper or lower estimate.
    BOUND = "mathematical_bound"
    #: A two-way implication. When one side is RH, see ``rh_equivalent``.
    EQUIVALENCE = "mathematical_equivalence"
    #: A one-way implication.
    IMPLICATION = "mathematical_implication"
    #: A test that a candidate either passes or fails.
    CRITERION = "mathematical_criterion"
    #: An explicit construction: a kernel, an operator, a model.
    CONSTRUCTION = "mathematical_construction"
    #: A general assertion that does not fit a narrower role.
    CLAIM = "mathematical_claim"
    #: A route falsified or shown insufficient. Retained deliberately: losing
    #: one lets a refuted line of attack be rediscovered as novel.
    NO_GO = "mathematical_no_go"

    # --- meta: statements about how the research is conducted ---------------
    #: A rule constraining the engine's own reasoning.
    GOVERNANCE = "governance"
    #: A workflow or process record.
    PROCEDURAL = "procedural"


MATHEMATICAL_ROLES: frozenset[Role] = frozenset(
    {
        Role.DEFINITION,
        Role.IDENTITY,
        Role.BOUND,
        Role.EQUIVALENCE,
        Role.IMPLICATION,
        Role.CRITERION,
        Role.CONSTRUCTION,
        Role.CLAIM,
        Role.NO_GO,
    }
)

#: Roles carrying no mathematical content. Never property-extractable.
META_ROLES: frozenset[Role] = frozenset({Role.GOVERNANCE, Role.PROCEDURAL})

#: Roles that cannot be a step toward proving anything, even when rigorous.
#: A definition is true by construction and a no-go is a dead end; neither
#: moves a frontier.
NON_ADVANCING_ROLES: frozenset[Role] = frozenset({Role.DEFINITION, Role.NO_GO})


def is_mathematical(role: Role) -> bool:
    return role in MATHEMATICAL_ROLES


def is_property_extractable(role: Role) -> bool:
    """Whether a record in this role may yield mathematical properties."""
    return role in MATHEMATICAL_ROLES
