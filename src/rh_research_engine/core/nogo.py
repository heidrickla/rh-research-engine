from __future__ import annotations

import re

from .models import Claim, NoGoRule

# Each rule carries both tags and statement phrases. Tag-only matching meant a
# refuted route could be resurrected verbatim just by renaming its tag: the
# identical statement with a new tag produced no violation at all.
DEFAULT_RULES = [
    NoGoRule(
        id="boundary-unitarity",
        trigger_tags={"boundary_unitarity_only"},
        trigger_phrases=[
            "boundary unitarity",
            "all-pass",
            "blaschke",
            "unit boundary modulus",
            "phase rigidity",
            "scattering ratio",
            "scattering quotient",
        ],
        message="Boundary unitarity alone is compatible with off-line zeros via Blaschke/all-pass factors.",
    ),
    NoGoRule(
        id="generic-theta-positivity",
        trigger_tags={"generic_theta_positivity"},
        trigger_phrases=["theta kernel positivity", "generic positivity", "de bruijn-newman"],
        message="Generic positivity/log-concavity of the theta kernel is insufficient; de Bruijn-Newman deformations provide stress tests.",
    ),
    NoGoRule(
        id="finite-euler-normal-limit",
        trigger_tags={"finite_euler_normal_convergence"},
        trigger_phrases=["finite euler product", "normal convergence", "hurwitz"],
        message="Finite Euler products are zero-free; normal convergence across the strip cannot create isolated zeta zeros (Hurwitz obstruction).",
    ),
    NoGoRule(
        id="iterated-log-concavity",
        trigger_tags={"all_iterated_log_concavity"},
        trigger_phrases=["iterated log-concavity", "iterated log concavity", "curvature hierarchy"],
        message="The iterated theta curvature hierarchy fails at higher level for the actual kernel; it cannot be the RH proof route.",
    ),
    NoGoRule(
        id="generic-selfadjoint-resonances",
        trigger_tags={"selfadjoint_implies_real_resonances"},
        trigger_phrases=["self-adjoint", "selfadjoint", "hilbert-polya operator alone"],
        message="Self-adjoint scattering Hamiltonians can have complex resonances; self-adjointness alone does not imply RH.",
    ),
    NoGoRule(
        id="toroidality-alone",
        trigger_tags={"toroidality_alone"},
        trigger_phrases=["toroidality", "toroidal"],
        message="Toroidality selects zeta zeros through a scalar L-factor and does not constrain their horizontal position.",
    ),
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _phrase_hits(rule: NoGoRule, claim: Claim) -> list[str]:
    haystack = _normalize(" ".join([claim.statement, claim.notes, *claim.evidence]))
    return [phrase for phrase in rule.trigger_phrases if _normalize(phrase) in haystack]


def violations(claim: Claim) -> list[NoGoRule]:
    """Rules this claim trips, by tag *or* by statement wording."""
    hits: list[NoGoRule] = []
    for rule in DEFAULT_RULES:
        by_tag = bool(rule.trigger_tags) and rule.trigger_tags <= claim.tags
        if by_tag or _phrase_hits(rule, claim):
            hits.append(rule)
    return hits


def explain(claim: Claim) -> list[tuple[NoGoRule, str]]:
    """Violations paired with why each fired, for audit output."""
    out: list[tuple[NoGoRule, str]] = []
    for rule in DEFAULT_RULES:
        if bool(rule.trigger_tags) and rule.trigger_tags <= claim.tags:
            out.append((rule, f"tag match: {sorted(rule.trigger_tags)}"))
            continue
        matched = _phrase_hits(rule, claim)
        if matched:
            out.append((rule, f"statement match: {matched}"))
    return out
