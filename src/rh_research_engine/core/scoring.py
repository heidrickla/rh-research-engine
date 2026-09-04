from __future__ import annotations

from dataclasses import dataclass

from .models import Claim, ClaimStatus


@dataclass(frozen=True)
class Score:
    total: float
    progress: float
    rigor: float
    assumption_penalty: float
    explanation: str


_RIGOR = {
    ClaimStatus.FALSE: -10.0,
    ClaimStatus.HYPOTHESIS: 0.0,
    ClaimStatus.NUMERICAL: 0.5,
    ClaimStatus.SYMBOLIC: 1.5,
    ClaimStatus.KNOWN: 2.0,
    ClaimStatus.PROVED: 3.0,
    ClaimStatus.EQUIVALENT_RH: -2.0,
}

#: Only a claim that has actually been established may contribute quantitative
#: progress. Previously any claim could set ``implied_theta_upper`` freely, and
#: because the progress term spans 0..10 while rigor spans -10..3, a bare
#: hypothesis asserting Theta <= 1/2 outscored a proved result.
_PROGRESS_ELIGIBLE = frozenset({ClaimStatus.PROVED, ClaimStatus.KNOWN})

#: Statuses that mean "this is RH again, wearing a hat". They can never be net
#: positive: restating the target is not progress toward it.
_CIRCULAR_STATUSES = frozenset({ClaimStatus.EQUIVALENT_RH})

#: Historic tag spellings that all mean the same thing. Keyed off status where
#: possible, but tags remain honoured so existing registries keep working --
#: `rh_equivalence` in particular was in the shipped seed data and matched none
#: of the previously checked tags.
_CIRCULAR_TAGS = frozenset(
    {
        "assumes_rh",
        "rh_equivalent_assumption",
        "rh_equivalence",
        "rh_equivalent",
        "equivalent_to_rh",
    }
)

CIRCULARITY_PENALTY = 10.0


def is_circular(claim: Claim) -> bool:
    return claim.status in _CIRCULAR_STATUSES or bool(_CIRCULAR_TAGS & claim.tags)


def score_claim(claim: Claim) -> Score:
    circular = is_circular(claim)

    # Progress is measured against the trivial global bound Theta <= 1.
    if claim.implied_theta_upper is None or claim.status not in _PROGRESS_ELIGIBLE or circular:
        progress = 0.0
    else:
        progress = max(0.0, 1.0 - claim.implied_theta_upper) * 10.0

    rigor = _RIGOR[claim.status]
    assumption_penalty = 0.75 * len(claim.assumptions)
    if circular:
        assumption_penalty += CIRCULARITY_PENALTY
    total = progress + rigor - assumption_penalty

    # A refuted or circular claim must never present as net progress, whatever
    # else it carries.
    if circular or claim.status is ClaimStatus.FALSE:
        total = min(total, 0.0)

    notes = []
    if circular:
        notes.append("circular: restates RH rather than advancing it")
    if claim.implied_theta_upper is not None and claim.status not in _PROGRESS_ELIGIBLE:
        notes.append(
            f"implied_theta_upper ignored: status {claim.status.value!r} is not established"
        )
    suffix = (" [" + "; ".join(notes) + "]") if notes else ""

    return Score(
        total=total,
        progress=progress,
        rigor=rigor,
        assumption_penalty=assumption_penalty,
        explanation=(
            f"progress={progress:.2f}, rigor={rigor:.2f}, "
            f"assumption_penalty={assumption_penalty:.2f}{suffix}"
        ),
    )
