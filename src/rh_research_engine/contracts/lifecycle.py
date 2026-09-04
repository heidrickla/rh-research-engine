"""Axis: where a hypothesis sits in the research workflow.

Strictly operational. It records *where work stands*, never what is known, what
kind of statement it is, or whether it counts as progress -- those are the
epistemic, role, and frontier axes respectively (ADR-001).

The laboratory plan proposed ten values, but they mix four questions:
``FALSIFIED`` and ``RIGOROUSLY_ESTABLISHED`` are epistemic verdicts,
``EQUIVALENT_REFORMULATION`` is a classification, and
``PROOF_OBLIGATION_IDENTIFIED`` restates a fact already carried by an open
``ProofObligation`` reference. Encoding them here would store the same fact
twice and let the two copies disagree.
"""

from __future__ import annotations

from enum import StrEnum


class HypothesisLifecycle(StrEnum):
    """Operational position only."""

    #: Recorded, not yet assessed.
    PROPOSED = "proposed"
    #: Assessed and prioritised; work has not started.
    TRIAGED = "triaged"
    #: Being worked on. Evidence and obligations accumulate against it.
    ACTIVE = "active"
    #: Cannot proceed until something else resolves. See ``blocker_refs``.
    BLOCKED = "blocked"
    #: Work has concluded. *How* it concluded is the epistemic axis, not this
    #: one: refuted, established, and shown-to-be-a-reformulation are all
    #: RESOLVED here and differ in ``epistemic_status`` and ``role``.
    RESOLVED = "resolved"
    #: Retained for the record; no further work intended.
    ARCHIVED = "archived"


#: Lifecycles in which work can still be done.
OPEN_LIFECYCLES = frozenset(
    {
        HypothesisLifecycle.PROPOSED,
        HypothesisLifecycle.TRIAGED,
        HypothesisLifecycle.ACTIVE,
    }
)

#: Lifecycles in which no further work is expected.
CLOSED_LIFECYCLES = frozenset(
    {HypothesisLifecycle.RESOLVED, HypothesisLifecycle.ARCHIVED}
)

#: Lifecycles where work is wanted but cannot start.
STALLED_LIFECYCLES = frozenset({HypothesisLifecycle.BLOCKED})


def is_open(lifecycle: HypothesisLifecycle) -> bool:
    return lifecycle in OPEN_LIFECYCLES


def is_closed(lifecycle: HypothesisLifecycle) -> bool:
    return lifecycle in CLOSED_LIFECYCLES
