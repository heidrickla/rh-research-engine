from __future__ import annotations

from .models import MathCertificate, RealInterval
from .predicates import contains_zero, definitely_negative, definitely_positive


def certificate_predicates(certificate: MathCertificate) -> dict[str, str | int | bool]:
    """Return DRE-safe facts while keeping high-precision endpoints in the artifact.

    Assumptions cross this boundary. They used to be dropped, which turned a
    conditional enclosure into an unconditional three-valued fact: a
    certificate annotated "assumes RH" arrived at DRE indistinguishable from an
    unconditional one, and an RH-assuming certificate then counted as evidence
    for RH. The endpoints stay out (that is deliberate -- DRE bans floats and
    the full certificate remains the evidence artifact), but the *conditions*
    must travel with the predicates.
    """
    facts: dict[str, str | int | bool] = {
        "certificate_hash": certificate.certificate_hash(),
        "expression_hash": certificate.expression_hash or "",
        "verifier_method": certificate.verifier.method,
        "assumption_count": len(certificate.assumptions),
        "assumptions_present": bool(certificate.assumptions),
        "unconditional": not certificate.assumptions,
    }
    for index, assumption in enumerate(sorted(certificate.assumptions), start=1):
        facts[f"assumption_{index}"] = assumption
    if certificate.verifier.precision_bits is not None:
        facts["precision_bits"] = certificate.verifier.precision_bits
    if certificate.verifier.worker_hash:
        facts["worker_hash"] = certificate.verifier.worker_hash
    if isinstance(certificate.value, RealInterval):
        facts["definitely_positive"] = definitely_positive(certificate.value).value
        facts["definitely_negative"] = definitely_negative(certificate.value).value
        facts["contains_zero"] = contains_zero(certificate.value).value
    return facts
