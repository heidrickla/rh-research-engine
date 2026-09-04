from __future__ import annotations

from .models import ProofGap, ProofStep

_RIGOROUS = {"proved", "known", "exact", "formal", "formally_verified", "rigorous"}


def extract_proof_gaps(steps: list[ProofStep]) -> list[ProofGap]:
    by_id = {step.id: step for step in steps}
    gaps: list[ProofGap] = []
    for step in steps:
        status = step.status.lower()
        blocking = [dep for dep in step.depends_on if dep not in by_id or by_id[dep].status.lower() not in _RIGOROUS]
        if status not in _RIGOROUS:
            gaps.append(ProofGap(step_id=step.id, statement=step.statement, reason=f"step status is {step.status}, not rigorous", blocking_dependencies=blocking))
        elif blocking:
            gaps.append(ProofGap(step_id=step.id, statement=step.statement, reason="rigorous-labeled step depends on unresolved/non-rigorous input", blocking_dependencies=blocking))
    return gaps
