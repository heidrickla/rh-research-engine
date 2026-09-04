"""Pattern detection: audit a premise, measure what was not asked, escalate
an exact regularity.

See `models` for why this is a first-class research function rather than
something an operator is expected to remember to do.
"""

from .character import Characterisation, WitnessCharacter, characterise
from .detect import (
    MINIMUM_SAMPLE,
    RELATION_KINDS,
    audit_premise,
    escalate,
    scan_columns,
    scan_for_regularities,
    scan_universe,
)
from .ledger import (
    CLOSING,
    OpenFinding,
    OpenLedger,
    Revisit,
    RevisitVerdict,
    judge,
)
from .models import (
    Observation,
    PatternFinding,
    PremiseAudit,
    PremiseVerdict,
    RegularityKind,
)
from .noise import NoiseGround, NoiseRegistry, NoiseRule, Suppression

__all__ = [
    "CLOSING",
    "MINIMUM_SAMPLE",
    "Characterisation",
    "WitnessCharacter",
    "NoiseGround",
    "NoiseRegistry",
    "NoiseRule",
    "OpenFinding",
    "OpenLedger",
    "RELATION_KINDS",
    "Revisit",
    "RevisitVerdict",
    "Suppression",
    "Observation",
    "PatternFinding",
    "PremiseAudit",
    "PremiseVerdict",
    "RegularityKind",
    "audit_premise",
    "characterise",
    "escalate",
    "judge",
    "scan_columns",
    "scan_for_regularities",
    "scan_universe",
]
