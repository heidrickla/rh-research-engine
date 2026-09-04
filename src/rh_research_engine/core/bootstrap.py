from __future__ import annotations

from .models import Claim, ClaimStatus


def seed_claims() -> list[Claim]:
    return [
        Claim(
            id="C001",
            statement="For fixed q>0, a rigorous bound S_q(X)=O(X^theta) implies Re(rho)<=theta for all nontrivial zeta zeros.",
            status=ClaimStatus.SYMBOLIC,
            tags={"gamma_filter", "spectral_exponent"},
            implied_theta_upper=None,
            evidence=["Mellin transform with zero-free Gamma multiplier."],
        ),
        Claim(
            # Status is EQUIVALENT_RH, not SYMBOLIC: the statement says "RH is
            # equivalent to ...", so it restates the target rather than
            # advancing it. Filed as SYMBOLIC with only the tag `rh_equivalence`
            # -- which matched none of the penalty tags -- it scored as ordinary
            # progress.
            id="C002",
            statement="RH is equivalent to corrected safe-binomial coefficients satisfying corrected_d_k=O(k^-3/4).",
            status=ClaimStatus.EQUIVALENT_RH,
            tags={"safe_values", "norlund_rice", "rh_equivalence", "rh_equivalent_assumption"},
            evidence=["Simple-pole residues of zeta'/zeta and explicit trivial-zero correction."],
        ),
        Claim(
            id="C005",
            statement="For one fixed q>0, if the localized Hardy--Littlewood screening model is O_q(1) and the actual screening remainder is O(X^theta), then Theta<=1/2+theta/2.",
            status=ClaimStatus.SYMBOLIC,
            assumptions=["rigorous uniform screening-remainder bound", "fixed q", "bounded deterministic model contribution"],
            tags={"correlation_lab", "screening_remainder", "spectral_exponent"},
            evidence=["Exact zero response of the localized Gamma shell has energy exponent 2(Re(rho)-1/2)."],
        ),
        Claim(
            id="C006",
            statement="In the Hardy--Littlewood singular-series model, the universal secondary shift bias cancels the leading A_q log X diagonal shot noise for the localized Gamma shell.",
            # INFERRED, not KNOWN. The evidence line below is established
            # mathematics; the statement is a conclusion drawn from it about
            # this project's own localized Gamma shell, which nothing in the
            # literature is about. The status said "someone else proved this".
            status=ClaimStatus.INFERRED,
            tags={"correlation_lab", "hardy_littlewood_model", "screening"},
            evidence=["Weighted average of the singular series has universal -1/2 log H secondary term."],
        ),
        Claim(
            id="C003",
            statement="Boundary unitarity of the zeta scattering ratio alone proves RH.",
            status=ClaimStatus.FALSE,
            tags={"boundary_unitarity_only"},
            evidence=["Off-line zeros appear as all-pass/Blaschke factors with unit boundary modulus."],
        ),
        Claim(
            id="C004",
            statement="All iterated theta log-concavity levels remain positive.",
            status=ClaimStatus.FALSE,
            tags={"all_iterated_log_concavity"},
            evidence=["Higher-level numerical/formal recursion fails for the actual theta kernel."],
        ),
    ]
