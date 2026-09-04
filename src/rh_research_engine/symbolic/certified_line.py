"""A rigorous verification that the zeros below a height are on the line.

WHAT THIS IS FOR. `argument_principle.verify_zeros_on_the_line` answers the
same question in floating point, and refuses a rigorous confidence at
construction because a float64 `zeta` carries no error bound at all. That
refusal is honest and it is also a standing invitation: the README has said
since the count first worked that reaching `rigorous_numerical` "would need
the certified enclosures the `mathcert` interval machinery exists for, which
are not wired to it". This wires them.

WHAT IS ACTUALLY PROVED, and it is worth being precise because the phrase
"verified to height T" travels without its qualifiers. The claim is:

    every zero of zeta with 0 < Im(s) <= T has Re(s) = 1/2 and is simple

and it decomposes into two halves that cost wildly different amounts:

  * HOW MANY zeros the strip holds below T. Arb's `zeta_nzeros` settles this
    by the argument principle with Turing's method and rigorous error bounds,
    and returns an exact integer -- a ball of radius zero. It is free: N(T)
    for T = 10^8, which is 248008025 zeros, comes back in a hundredth of a
    second.

  * WHERE they are. Isolating each zero to a certified enclosure with real
    part exactly 1/2 costs milliseconds per zero and grows with height. This
    is the half that bounds the reach.

So the reach of the two halves differs by four orders of magnitude, and saying
so is the point rather than an apology for it. The count half now rigorously
confirms every figure this repository has recorded, including the 1747146
zeros below T = 10^6 that took the floating-point argument principle 24
minutes; the position half reaches T = 10^4 in under a minute.

WHY THE THREE CHECKS. `zeta_zeros(1, N)` hands back N enclosures and it would
be easy to conclude too much from them:

  1. Each has real part exactly 1/2, with radius zero. An enclosure merely
     *containing* 1/2 says nothing -- a ball around the critical line contains
     off-line points, so "Re = 1/2 to within 1e-30" is not the claim.
  2. Each lies inside (0, T]. Ask for one zero too many and the extra sits
     above the height, and a count that includes it is not a count below T.
  3. They are pairwise disjoint. Without this, N enclosures could describe
     fewer than N distinct zeros, and the multiplicity argument below fails.

Given all three: there are N distinct zeros on the line with 0 < Im <= T, and
the strip holds N counted WITH multiplicity, so every one is simple and there
is no other -- on the line or off it.

WHAT IT IS NOT. Rigorous about a finite computation, which is what
`RIGOROUS_NUMERICAL` means and why that value is deliberately not in
`contracts.epistemic.RIGOROUS`. It is not a proof of the Riemann hypothesis
and not evidence of one: RH is a statement about every zero, and this is a
statement about the first ten thousand. It rests on Arb being correct, which
is a different assumption from float64 being lucky but is still an assumption
-- recorded in the certificate as the backend family and version rather than
left implicit.

Nor is it new mathematics. Zeros have been verified rigorously to heights
enormously beyond anything here. What is new *to this repository* is that its
own verification can now be filed at a rigorous confidence instead of being
refused one.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..contracts.epistemic import Confidence
from ..mathcert.arb_flint import (
    CertifiedZeroScan,
    certified_zero_count,
    certified_zeros_on_line,
    detect_arb_flint,
)
from ..mathcert.verifiers import VerifierEnvelope


class CertifiedLineVerification(BaseModel):
    """Zeros below a height, certified on the critical line and simple.

    The counterpart of `argument_principle.LineVerification`, and the
    validators run in opposite directions. That one REFUSES a rigorous
    confidence because there is no enclosure behind it; this one refuses
    anything else, because a record that could be built at `numerical` from a
    certified computation would let a caller quietly discard the certification
    -- and a record that could be built at `rigorous_numerical` without one
    would be the spoof the whole `mathcert` layer exists to prevent.
    """

    model_config = ConfigDict(extra="forbid")

    height: float
    #: Zeros in the strip with 0 < Im <= height, counted with multiplicity.
    #: Exact: an Arb ball of radius zero, not a rounded float.
    counted: int
    #: Enclosures with real part exactly 1/2, inside (0, height], pairwise
    #: disjoint. Equal to `counted` when the verification succeeded.
    certified: int
    #: Every check, with what it decided. Named individually because "it
    #: passed" is not a result a later reader can audit.
    checks: dict[str, bool] = Field(default_factory=dict)
    #: The backend that did the work, and its version. Part of the claim: a
    #: certified enclosure is only as good as the library that produced it,
    #: and which one that was must not be implicit.
    backend: str
    backend_version: str
    evidence: str
    confidence: Confidence = Confidence.RIGOROUS_NUMERICAL

    @field_validator("confidence")
    @classmethod
    def _require_the_certified_confidence(cls, value: Confidence) -> Confidence:
        """Exactly `rigorous_numerical`, neither stronger nor weaker.

        Stronger is the familiar failure: `proved` or `known` would turn a
        finite computation into a theorem, and `RIGOROUS_NUMERICAL` is kept out
        of `contracts.epistemic.RIGOROUS` precisely so that cannot happen by
        accident.

        Weaker is the one worth spelling out. Nothing stops a caller building
        this record at `numerical`, and the record would then be indistinct
        from the floating-point one beside it -- the certification thrown away
        by a default argument. The two verifications differ in kind, so their
        records must not be able to agree in status.
        """
        if value is not Confidence.RIGOROUS_NUMERICAL:
            raise ValueError(
                f"a certified line verification is {Confidence.RIGOROUS_NUMERICAL.value!r} "
                f"and cannot be filed as {value.value!r}: stronger would make a "
                "finite computation into a theorem, and weaker would discard "
                "the enclosure that distinguishes it from the floating-point "
                "check"
            )
        return value

    @property
    def verified(self) -> bool:
        """Every check passed and every zero below the height was placed."""
        return bool(self.checks) and all(self.checks.values()) and (
            self.certified == self.counted
        )


class BackendUnavailable(RuntimeError):
    """No certified-enclosure backend is importable here.

    Raised rather than returning a degraded record. A verification that
    silently falls back to floating point is the one mistake this module
    cannot be allowed to make: its whole content is that the arithmetic was
    certified, and a caller reading `confidence` would have no way to tell.
    """


def certify_zeros_on_the_line(height: float) -> CertifiedLineVerification:
    """Certify that every zero below `height` is on the line and simple.

    Fails closed. Without a backend this raises `BackendUnavailable` instead
    of producing a record, because there is no honest record to produce.
    """
    capability = detect_arb_flint()
    if not capability.available:
        raise BackendUnavailable(
            f"{capability.reason}; a certified verification needs an interval "
            "backend, and no floating-point computation substitutes for one"
        )

    counted = certified_zero_count(height)
    scan = certified_zeros_on_line(counted, height)
    evidence = _evidence_for(height, counted, scan)
    return CertifiedLineVerification(
        height=height,
        counted=counted,
        certified=scan.certified,
        checks=dict(scan.checks),
        backend=capability.family,
        backend_version=capability.version or "unknown",
        evidence=evidence,
    )


def _evidence_for(height: float, counted: int, scan: CertifiedZeroScan) -> str:
    """A sentence a reader can repeat without losing the qualifiers."""
    if scan.certified == counted and all(scan.checks.values()):
        return (
            f"every one of the {counted} zeros of zeta with 0 < Im <= "
            f"{height:g} lies on the critical line and is simple, by certified "
            "interval arithmetic over that finite range -- which is a "
            "statement about those zeros and not about the rest"
        )
    failed = sorted(name for name, passed in scan.checks.items() if not passed)
    return (
        f"the strip holds {counted} zeros below {height:g} and "
        f"{scan.certified} were certified on the line"
        + (f"; failing checks: {', '.join(failed)}" if failed else "")
        + " -- either a zero is off the line or of even order, or the "
        "enclosures are too wide to separate at this precision"
    )


def certified_count_envelope(height: float) -> VerifierEnvelope:
    """The count alone, as a verifier envelope, for any height.

    Split out because the two halves have wildly different reach and a caller
    wanting N(T) at T = 10^8 should not have to isolate 248 million zeros to
    get it. This is the half that confirms the floating-point argument
    principle at every height this repository has recorded.
    """
    from ..mathcert.arb_flint import zero_count_certificate

    return zero_count_certificate(height)
