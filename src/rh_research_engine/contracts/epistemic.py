"""Axis: how strongly established a statement is.

One question only -- "how confident are we that this is true?" -- and never a
proxy for anything else. In particular it is never lowered because a statement
is useless (a classical RH-equivalence is fully established *and* worth no
progress) nor raised because a statement is important.

This is the canonical vocabulary. The legacy vocabularies -- ``ClaimStatus``,
``EvidenceClass``, ``KnowledgeStatus``, ``EpistemicStatus``,
``VerificationStatus`` -- remain as on-disk formats and are mapped onto this one
by :mod:`rh_research_engine.contracts.mappings`, exhaustively and explicitly.
"""

from __future__ import annotations

from enum import StrEnum


class Confidence(StrEnum):
    """Canonical epistemic status, strongest first.

    Ordering is by :data:`CONFIDENCE_RANK`, not by declaration order; do not
    rely on enum position.
    """

    #: Machine-checked by an external formal system, with no axiom stubs.
    FORMALLY_VERIFIED = "formally_verified"
    #: Proved, checked by a human or a formal checker outside this package.
    PROVED = "proved"
    #: Established external mathematics. Needs a literature citation.
    KNOWN = "known"
    #: Mechanically exact from stated definitions: exact algebra or calculus.
    CERTIFIED = "certified"
    #: Derived here by a checked, rigorous argument.
    RIGOROUS_DERIVED = "rigorous_derived"
    #: Enclosed by certified interval arithmetic over a finite computation.
    #: Rigorous about what it covers, and what it covers is always finite.
    RIGOROUS_NUMERICAL = "rigorous_numerical"
    #: Derived here symbolically, not independently formalized or
    #: literature-checked end to end.
    SYMBOLIC_DERIVED = "symbolic_derived"
    #: Floating-point evidence. Never a proof, at any precision or sample count.
    NUMERICAL = "numerical"
    #: Plausible, not deductively established.
    HEURISTIC = "heuristic"
    #: True of a constructed model, which is not the same as true of zeta.
    SYNTHETIC = "synthetic"
    #: Proposed, unassessed.
    CONJECTURAL = "conjectural"
    #: Shown false, or shown insufficient for its purpose.
    REFUTED = "refuted"
    #: A meta-rule about how the engine reasons. Not a mathematical claim, so
    #: no mathematical confidence applies.
    AUTHORITATIVE_POLICY = "authoritative_policy"
    #: Could not be established either way. The fail-closed default: an
    #: unmapped or unrecognised input lands here, never somewhere stronger.
    UNKNOWN = "unknown"


#: Comparable strength. Higher is stronger. Non-mathematical and negative
#: verdicts sit at or below zero so no arithmetic on this accidentally treats
#: "refuted" or "policy" as weak-but-positive evidence.
CONFIDENCE_RANK: dict[Confidence, int] = {
    Confidence.FORMALLY_VERIFIED: 100,
    Confidence.PROVED: 90,
    Confidence.KNOWN: 80,
    Confidence.CERTIFIED: 70,
    Confidence.RIGOROUS_DERIVED: 60,
    Confidence.RIGOROUS_NUMERICAL: 45,
    Confidence.SYMBOLIC_DERIVED: 40,
    Confidence.NUMERICAL: 20,
    Confidence.HEURISTIC: 10,
    Confidence.SYNTHETIC: 5,
    Confidence.CONJECTURAL: 1,
    Confidence.UNKNOWN: 0,
    Confidence.AUTHORITATIVE_POLICY: 0,
    Confidence.REFUTED: -100,
}

#: Confidences that establish a mathematical statement outright.
#:
#: ``RIGOROUS_NUMERICAL`` is deliberately absent. A certified enclosure is
#: rigorous about a *finite* computation; treating it as rigorous in general is
#: the step from "checked to height T" to "true", which is the whole failure
#: this vocabulary exists to prevent.
RIGOROUS: frozenset[Confidence] = frozenset(
    {
        Confidence.FORMALLY_VERIFIED,
        Confidence.PROVED,
        Confidence.KNOWN,
        Confidence.CERTIFIED,
        Confidence.RIGOROUS_DERIVED,
    }
)

#: Confidences carrying no deductive force on their own.
NON_DEDUCTIVE: frozenset[Confidence] = frozenset(
    {
        Confidence.RIGOROUS_NUMERICAL,
        Confidence.NUMERICAL,
        Confidence.HEURISTIC,
        Confidence.SYNTHETIC,
        Confidence.CONJECTURAL,
        Confidence.UNKNOWN,
    }
)

#: Confidences a deterministic math worker may never assert about its own
#: output. A worker computes; deciding that what it computed is a proof or an
#: established theorem requires a formal checker or cited literature.
WORKER_FORBIDDEN: frozenset[Confidence] = frozenset(
    {Confidence.FORMALLY_VERIFIED, Confidence.PROVED, Confidence.KNOWN}
)

#: Confidences that are not mathematical verdicts at all.
NON_MATHEMATICAL: frozenset[Confidence] = frozenset({Confidence.AUTHORITATIVE_POLICY})


def is_rigorous(confidence: Confidence) -> bool:
    return confidence in RIGOROUS


def rank(confidence: Confidence) -> int:
    return CONFIDENCE_RANK[confidence]


def at_least(confidence: Confidence, floor: Confidence) -> bool:
    """Whether ``confidence`` is at least as strong as ``floor``.

    Only meaningful between mathematical verdicts. Comparing against
    ``AUTHORITATIVE_POLICY`` or ``REFUTED`` is a category error, so it raises
    rather than returning a plausible-looking boolean.
    """
    for value in (confidence, floor):
        if value in NON_MATHEMATICAL or value is Confidence.REFUTED:
            raise ValueError(
                f"{value.value!r} is not a point on the confidence scale; "
                "comparing it to another status is a category error"
            )
    return CONFIDENCE_RANK[confidence] >= CONFIDENCE_RANK[floor]
