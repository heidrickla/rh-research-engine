"""Deprecated hypothesis vocabulary, kept for one release.

``HypothesisState`` conflated four questions in one field. ``ACTIONABLE`` was a
lifecycle position *and* a derived predicate; ``FALSIFIED`` and ``ADVANCED``
were epistemic verdicts wearing a workflow label. It is replaced by
:class:`~rh_research_engine.contracts.lifecycle.HypothesisLifecycle` plus the
orthogonal epistemic, role, and frontier axes.

Nothing outside this module and ``contracts.mappings`` may import it; a test
enforces that, so the deprecation cannot quietly spread while it waits to be
removed. Stored records carrying ``state`` are translated on read by
``Hypothesis``, so no data needs rewriting before the removal.
"""

from __future__ import annotations

import warnings
from enum import StrEnum

#: Remove after v0.13.0. Until then, reading a legacy record still works.
REMOVE_AFTER = "0.13.0"


class HypothesisState(StrEnum):
    """Deprecated. Use ``HypothesisLifecycle`` and the epistemic axis."""

    PROPOSED = "proposed"
    ACTIONABLE = "actionable"
    TESTING = "testing"
    BLOCKED = "blocked"
    FALSIFIED = "falsified"
    ADVANCED = "advanced"


def migrate_legacy_payload(data: dict) -> dict:
    """Translate a stored ``state`` into the canonical axes.

    Lives here rather than in ``models.py`` so every reference to the
    deprecated enum sits in one module -- the containment test enforces that,
    and it is what makes the eventual removal a single-file deletion.

    The legacy field carried two facts at once and both are recovered: the
    workflow position becomes ``lifecycle``, the verdict becomes
    ``epistemic_status``. Explicitly supplied axes win, so a record already
    migrated is left alone.
    """
    from ..contracts.mappings import (
        confidence_from_hypothesis_state,
        lifecycle_from_hypothesis_state,
    )

    payload = dict(data)
    state = HypothesisState(payload.pop("state"))
    payload.setdefault("lifecycle", lifecycle_from_hypothesis_state(state).value)
    payload.setdefault("epistemic_status", confidence_from_hypothesis_state(state).value)
    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("migrated_from_state", state.value)
    payload["metadata"] = metadata
    return payload


def warn_deprecated(context: str) -> None:
    """Emit the deprecation once per call site, naming what to use instead."""
    warnings.warn(
        f"{context} uses the deprecated HypothesisState. It conflates lifecycle "
        "with epistemic verdict; use Hypothesis.lifecycle plus "
        "Hypothesis.epistemic_status. Removed after v" + REMOVE_AFTER,
        DeprecationWarning,
        stacklevel=3,
    )
