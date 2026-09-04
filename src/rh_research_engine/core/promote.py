"""The single gate every export must pass through.

The guards in this package were previously a library, not a gate: six of them
had zero callers outside their own unit tests, and the only end-to-end path --
run an experiment, export it to DRE -- touched none. Demonstrating a check in a
test proves it works; it does not put it between input and output.

Everything an artifact must survive before it leaves the worker lives here, and
`cli.py` calls nothing else.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from .models import (
    NON_DEDUCTIVE_CLASSES,
    WORKER_FORBIDDEN_CLASSES,
    Claim,
    ClaimStatus,
)
from .nogo import explain as explain_nogo


class GateSeverity(StrEnum):
    BLOCK = "block"
    WARN = "warn"


class GateFinding(BaseModel):
    gate: str
    severity: GateSeverity
    message: str


class PromotionDecision(BaseModel):
    allowed: bool
    findings: list[GateFinding] = Field(default_factory=list)

    @property
    def blocks(self) -> list[GateFinding]:
        return [f for f in self.findings if f.severity is GateSeverity.BLOCK]

    @property
    def warnings(self) -> list[GateFinding]:
        return [f for f in self.findings if f.severity is GateSeverity.WARN]

    def render(self) -> list[str]:
        return [f"[{f.severity.value}] {f.gate}: {f.message}" for f in self.findings]


class PromotionBlocked(RuntimeError):
    def __init__(self, decision: PromotionDecision) -> None:
        super().__init__("; ".join(f.message for f in decision.blocks))
        self.decision = decision


def evaluate_export(
    envelope,
    *,
    claim: Claim | None = None,
    knowledge_path: Path | None = None,
) -> PromotionDecision:
    """Decide whether a DRE evidence envelope may be written."""
    # Function-scope: `dre` imports `core`, so a module-level import here would
    # close the loop core.promote -> dre -> dre.export -> core.promote.
    from ..dre.contracts import ClaimEffect

    findings: list[GateFinding] = []

    def block(gate: str, message: str) -> None:
        findings.append(GateFinding(gate=gate, severity=GateSeverity.BLOCK, message=message))

    def warn(gate: str, message: str) -> None:
        findings.append(GateFinding(gate=gate, severity=GateSeverity.WARN, message=message))

    # 1. A worker never asserts proof about its own output.
    if envelope.evidence_class in WORKER_FORBIDDEN_CLASSES:
        block(
            "evidence-class",
            f"evidence_class={envelope.evidence_class.value!r} cannot be asserted by a math worker",
        )

    # 2. Non-deductive evidence carries no Theta bound.
    if envelope.evidence_class in NON_DEDUCTIVE_CLASSES and envelope.theta_upper is not None:
        block(
            "theta-bound",
            f"{envelope.evidence_class.value} evidence cannot assert theta_upper="
            f"{envelope.theta_upper}",
        )
    if envelope.theta_upper is not None and envelope.theta_upper < 0.5:
        block("theta-bound", f"theta_upper={envelope.theta_upper} is below 1/2 and impossible")

    # 3. Assumptions must be visible, and must never be dropped in silence.
    if envelope.assumptions:
        warn(
            "assumptions",
            f"{len(envelope.assumptions)} assumption(s) attached; every conclusion is conditional",
        )
        if envelope.claim_effect is ClaimEffect.SUPPORTS and envelope.rh_equivalent:
            block(
                "circularity",
                "assumption-bearing evidence cannot support an RH-equivalent claim: "
                "the premise would be doing the work",
            )

    # 4. RH-equivalent evidence is never progress.
    if envelope.rh_equivalent and envelope.claim_effect is ClaimEffect.SUPPORTS:
        warn(
            "rh-equivalence",
            "supports an RH-equivalent statement; this is a reformulation, not progress toward "
            "a proof",
        )

    # 5. Independent verification must be earned.
    if envelope.independently_verified and envelope.evidence_class in NON_DEDUCTIVE_CLASSES:
        block(
            "independence",
            "independently_verified cannot accompany non-deductive evidence without a "
            "corroborating envelope from a distinct method family",
        )

    # 6. The claim this attaches to must not be a known dead end.
    if claim is not None:
        for rule, why in explain_nogo(claim):
            block("no-go", f"{rule.id} ({why}): {rule.message}")
        if (
            claim.status is ClaimStatus.FALSE
            and envelope.claim_effect is ClaimEffect.SUPPORTS
        ):
            block("no-go", f"{claim.id} is recorded as refuted; it cannot be supported")

    # 7. Durable memory must be loadable, or nothing downstream can be trusted.
    if knowledge_path is not None:
        from .knowledge import KnowledgeBase, KnowledgeIntegrityError

        try:
            KnowledgeBase(knowledge_path).load()
        except KnowledgeIntegrityError as exc:
            block("durable-memory", f"durable memory failed integrity check: {exc}")
        except FileNotFoundError:
            # Blocking, not a warning. Every no-go check downstream consults
            # durable memory; with none present they all pass vacuously, so an
            # export written in that state carries no route checking at all.
            block(
                "durable-memory",
                f"no durable memory at {knowledge_path}. No-go and route checks read "
                "from it, so exporting without it means nothing was checked.",
            )

    return PromotionDecision(allowed=not any(f.severity is GateSeverity.BLOCK for f in findings), findings=findings)


def require_export_allowed(envelope, **kwargs) -> PromotionDecision:
    decision = evaluate_export(envelope, **kwargs)
    if not decision.allowed:
        raise PromotionBlocked(decision)
    return decision
