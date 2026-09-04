from __future__ import annotations

import re

from .models import ConjectureMinimization

_BIG_O_RE = re.compile(r"O\(X\^\(?([+-]?[0-9]*\.?[0-9]+)\)?\)")


def minimize_conjecture(statement: str, goal: str = "RH") -> ConjectureMinimization:
    """Return a weaker sufficient target when a registered implication is known."""
    compact = " ".join(statement.split())
    reasons: list[str] = []
    weaker = compact
    rule_id: str | None = None

    if "R_q" in compact or "mathscr R" in compact or "screening remainder" in compact.lower():
        if "O(1)" in compact or "bounded" in compact.lower():
            weaker = "R_q(X) = X^{o(1)} for one fixed q > 0"
            reasons.append("subpower screening is already sufficient to force the zero-edge exponent to 1/2")
            rule_id = "MIN-RH-SCREENING-SUBPOWER"
        elif match := _BIG_O_RE.search(compact):
            theta = float(match.group(1))
            implied = 0.5 + theta / 2
            if theta < 0:
                # Theta >= 1/2 unconditionally, so this target implies something
                # provably false. Narrating it as "useful partial progress" is
                # exactly the overstatement this minimizer exists to avoid.
                weaker = compact
                reasons.append(
                    f"target exponent {theta:g} is negative, implying Theta <= {implied:g} < 1/2, "
                    "which is impossible; no weakening offered because the premise is invalid"
                )
                rule_id = "MIN-REJECT-IMPOSSIBLE-EXPONENT"
            else:
                weaker = f"R_q(X) = O(X^{theta})"
                reasons.append(
                    f"this yields the graded consequence Theta <= {implied:g}; it is useful "
                    "partial progress even when not RH"
                )
                rule_id = "MIN-SCREENING-GRADED"

    if ("d_k" in compact or "d~" in compact or "binomial" in compact.lower()) and "O(k^-3/4)" in compact.replace(" ", ""):
        weaker = "For graded progress, prove corrected_d_k = O(k^{-alpha}) for any alpha > 1/2; alpha = 3/4 is the RH endpoint"
        reasons.append("the exact spectral exponent relation is Theta <= 2 - 2 alpha")
        rule_id = "MIN-BINOMIAL-GRADED"

    if "S_q" in compact and "O(X^1/2)" in compact.replace(" ", ""):
        weaker = "For graded progress, prove S_q(X) = O(X^theta) with any fixed theta < 1"
        reasons.append("any fixed theta < 1 gives a global zero-free strip Re(rho) <= theta; theta=1/2 is the RH endpoint")
        rule_id = "MIN-GAMMA-FILTER-GRADED"

    changed = weaker != compact
    if not reasons:
        reasons.append("no registered weakening rule matched; statement returned unchanged")
    return ConjectureMinimization(original=compact, goal=goal, weaker_target=weaker, changed=changed, rule_id=rule_id, rationale=reasons)
