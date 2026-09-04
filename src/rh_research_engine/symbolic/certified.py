from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

from ..mathcert.models import MathCertificate, RealInterval, SymbolicExpression
from ..mathcert.predicates import contains_zero, definitely_negative, definitely_positive
from .equivalence import domain_conditions, fingerprint


class CertifiedSymbolicCheck(BaseModel):
    expression_match: bool
    canonical_match: bool | None = None
    certificate_hash: str
    predicates: dict[str, str] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    domain_gap: list[str] = Field(default_factory=list)
    usable: bool = True
    warnings: list[str] = Field(default_factory=list)


def check_certificate_against_expression(
    certificate: MathCertificate, expression: str
) -> CertifiedSymbolicCheck:
    """Check that a certificate actually covers the expression being asked about.

    The canonical fallback used to compare algebra only, so a certificate whose
    enclosure was computed for `x+1` was accepted for `(x**2-1)/(x-1)` -- a
    function undefined at x = 1 -- with an empty warning list. Fingerprints are
    now domain-aware, and any remaining difference in domain is reported as a
    gap that makes the certificate unusable for the request.
    """
    raw_hash = hashlib.sha256(expression.encode("utf-8")).hexdigest()
    expression_match = raw_hash == certificate.expression_hash
    warnings: list[str] = []
    domain_gap: list[str] = []
    canonical_match: bool | None = None
    usable = True

    if not expression_match:
        try:
            canonical_match = fingerprint(expression).sha256 == fingerprint(certificate.expression).sha256
        except Exception:
            canonical_match = None
        try:
            requested_domain = set(domain_conditions(expression))
            certified_domain = set(domain_conditions(certificate.expression))
            domain_gap = sorted(requested_domain ^ certified_domain)
        except Exception:
            domain_gap = []
        if domain_gap:
            usable = False
            warnings.append(
                "certificate domain does not match the requested expression: "
                + ", ".join(domain_gap)
            )
        if canonical_match is not True:
            usable = False
            warnings.append("certificate expression does not match requested expression")

    if certificate.assumptions:
        warnings.append(
            f"certificate carries {len(certificate.assumptions)} assumption(s); any conclusion "
            "drawn from it is conditional"
        )

    predicates: dict[str, str] = {}
    if isinstance(certificate.value, RealInterval):
        predicates = {
            "definitely_positive": definitely_positive(certificate.value).value,
            "definitely_negative": definitely_negative(certificate.value).value,
            "contains_zero": contains_zero(certificate.value).value,
        }
    elif isinstance(certificate.value, SymbolicExpression):
        if certificate.value.fingerprint is not None:
            try:
                matches = fingerprint(certificate.value.expression).sha256 == certificate.value.fingerprint
                predicates["symbolic_fingerprint_matches"] = str(matches).lower()
            except Exception:
                warnings.append("could not recompute symbolic fingerprint")

    return CertifiedSymbolicCheck(
        expression_match=expression_match,
        canonical_match=canonical_match,
        certificate_hash=certificate.certificate_hash(),
        predicates=predicates,
        assumptions=list(certificate.assumptions),
        domain_gap=domain_gap,
        usable=usable,
        warnings=warnings,
    )
