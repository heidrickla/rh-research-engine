"""Canonical research contracts.

One authoritative definition per axis. Subsystems import from here; they do not
declare local variants. See ``docs/ADR_001_CANONICAL_STATUS_CONTRACT.md``.

    lifecycle   where work stands
    epistemic   how strongly established
    roles       what kind of thing it is
    frontier    whether it moves RH forward
    mappings    total, explicit translations from the legacy vocabularies

The axis modules import nothing else from the package, so any subsystem can
depend on them. Only ``mappings`` reaches back into the legacy vocabularies,
because translating them is explicitly transitional.
"""

from .artifacts import ARTIFACT_MODELS, Artifact, ArtifactError, ArtifactType
from .epistemic import (
    CONFIDENCE_RANK,
    NON_DEDUCTIVE,
    NON_MATHEMATICAL,
    RIGOROUS,
    WORKER_FORBIDDEN,
    Confidence,
    at_least,
    is_rigorous,
    rank,
)
from .frontier import FrontierAssessment, assess, usable_as_rule
from .lifecycle import (
    CLOSED_LIFECYCLES,
    OPEN_LIFECYCLES,
    STALLED_LIFECYCLES,
    HypothesisLifecycle,
    is_closed,
    is_open,
)
from .receipts import (
    DreReceipt,
    ReceiptAuthentication,
    ReceiptError,
    activation_status,
)
from .roles import (
    MATHEMATICAL_ROLES,
    META_ROLES,
    NON_ADVANCING_ROLES,
    Role,
    is_mathematical,
    is_property_extractable,
)

__all__ = [
    "ARTIFACT_MODELS",
    "CLOSED_LIFECYCLES",
    "CONFIDENCE_RANK",
    "MATHEMATICAL_ROLES",
    "META_ROLES",
    "NON_ADVANCING_ROLES",
    "NON_DEDUCTIVE",
    "NON_MATHEMATICAL",
    "OPEN_LIFECYCLES",
    "RIGOROUS",
    "STALLED_LIFECYCLES",
    "WORKER_FORBIDDEN",
    "Artifact",
    "ArtifactError",
    "ArtifactType",
    "Confidence",
    "DreReceipt",
    "ReceiptAuthentication",
    "ReceiptError",
    "FrontierAssessment",
    "HypothesisLifecycle",
    "Role",
    "assess",
    "at_least",
    "is_closed",
    "is_mathematical",
    "is_open",
    "is_property_extractable",
    "is_rigorous",
    "rank",
    "activation_status",
    "usable_as_rule",
]
