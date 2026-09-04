"""External verifier envelopes: what a backend checked, versus what it claims.

A ``VerifierEnvelope`` is public and freely constructible, because a caller has
to be able to *supply* one. Its ``status`` field is therefore caller-supplied
too, and ``ACCEPTED`` written into a JSON file is a claim about a computation
nobody in this build performed.

WHY THE REGISTRY IS PRIVATE. This module used to take the registry as a
keyword argument -- ``registered_adapters=`` -- with a public default. That is
the same shape as a caller-supplied trusted-engine set: a caller passing its own
family name has registered itself, and the check then confirms only that the
caller agrees with the caller. The registry is now module-private and resolved
from *capability detection*: which backends are actually importable in this
build. Nothing a caller passes can widen it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum

from pydantic import BaseModel, Field

from .models import MathCertificate


class VerificationStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


#: Families registered unconditionally, independent of what is installed.
#: Empty, and private. A family reaches the effective registry only by being
#: detected as actually present -- see :func:`_registered_adapters`.
_REGISTERED_ADAPTERS: frozenset[str] = frozenset()

#: Overridden only by the private test-only context manager below.
_TEST_ADAPTERS: frozenset[str] | None = None


def _registered_adapters() -> frozenset[str]:
    """Verifier families with a real execution adapter in this build.

    Detection, not declaration. ``detect_arb_flint`` answers "is python-flint
    importable here?", which is a fact about the environment; a caller-supplied
    set would only have been a fact about the caller.
    """
    if _TEST_ADAPTERS is not None:
        return _TEST_ADAPTERS
    # Function-scope: `arb_flint` imports this module, so a top-level import
    # would close the loop verifiers -> arb_flint -> verifiers.
    from .arb_flint import registered_arb_flint_families

    return _REGISTERED_ADAPTERS | registered_arb_flint_families()


@contextmanager
def _test_only_registered_adapters(families: set[str] | frozenset[str]) -> Iterator[None]:
    """Temporarily pretend a backend is connected. **Tests only.**

    Private and context-managed so the widening is always scoped and always
    reverted. Production has no equivalent: no public function takes an adapter
    set, and this one is not exported.
    """
    global _TEST_ADAPTERS
    previous = _TEST_ADAPTERS
    _TEST_ADAPTERS = frozenset(families)
    try:
        yield
    finally:
        _TEST_ADAPTERS = previous


def verifier_activation_status() -> dict[str, object]:
    """Report whether any rigorous verifier backend is connected.

    Returns a count and a boolean rather than the family names, so reading it
    grants nothing -- the same shape as the DRE activation report.
    """
    registered = _registered_adapters()
    return {
        "registered_adapter_count": len(registered),
        "rigorous_verification_active": bool(registered),
        "max_confidence": "rigorous_numerical",
        "activation_requires": [
            "an importable backend, detected rather than declared",
            "a pinned dependency version and environment",
            "expression hash, interval endpoints, and precision bound into the "
            "certificate",
        ],
    }


#: Families whose names are reserved for real external workers. Naming one
#: without an adapter is a spoof attempt, not a typo.
KNOWN_EXTERNAL_FAMILIES = frozenset(
    {"arb", "flint", "arb-flint", "pari", "mpfi", "lean", "sympy-exact"}
)


class VerifierEnvelope(BaseModel):
    verifier_family: str
    verifier_version: str
    certificate: MathCertificate
    status: VerificationStatus
    checks: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def independence_group(self) -> str:
        return f"math-verifier:{self.verifier_family}:{self.verifier_version}"

    def envelope_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_external_envelope(
    envelope: VerifierEnvelope,
    *,
    allowed_families: set[str],
) -> list[str]:
    """Structural and provenance checks on a verifier envelope.

    ``allowed_families`` is required rather than defaulting to None. Defaulting
    to "no restriction" meant the common call accepted any family name at all.
    It is a *policy* filter -- which families this deployment permits -- and is
    deliberately not the registry: a caller may narrow what it will accept, and
    can never widen what this build can actually verify.

    This does **not** re-verify the mathematics: it cannot, because no
    enclosure-checking backend is bundled. What it can do is refuse to let an
    unbacked assertion present itself as a completed verification.
    """
    errors: list[str] = []
    verifier = envelope.certificate.verifier
    registered_adapters = _registered_adapters()

    if envelope.verifier_family not in allowed_families:
        errors.append(f"unapproved verifier family: {envelope.verifier_family}")
    if verifier.method != envelope.verifier_family:
        errors.append("certificate verifier method does not match envelope verifier family")
    if verifier.precision_bits is not None and verifier.precision_bits <= 0:
        errors.append("precision_bits must be positive")

    if envelope.status == VerificationStatus.ACCEPTED:
        if not envelope.checks:
            errors.append("accepted verification must name at least one completed check")
        if envelope.verifier_family not in registered_adapters:
            errors.append(
                f"verifier family {envelope.verifier_family!r} has no registered execution "
                "adapter in this build, so ACCEPTED cannot be substantiated; the certificate "
                "is an assertion, not a verification"
            )
        # An accepted result has to be attributable to a specific worker build,
        # otherwise metadata can be edited freely and the recomputed hash simply
        # agrees with whatever it was edited to.
        if not verifier.worker_hash:
            errors.append("accepted verification must carry verifier.worker_hash")
        if not verifier.source_hash:
            errors.append("accepted verification must carry verifier.source_hash")
        if not verifier.worker_version:
            errors.append("accepted verification must carry verifier.worker_version")

    if (
        envelope.verifier_family in KNOWN_EXTERNAL_FAMILIES
        and envelope.verifier_family not in registered_adapters
        and envelope.status != VerificationStatus.UNKNOWN
    ):
        errors.append(
            f"{envelope.verifier_family!r} names an external verifier that is not connected in "
            "this build; report status 'unknown' rather than claiming a verdict"
        )
    return errors
