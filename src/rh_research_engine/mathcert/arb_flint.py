from __future__ import annotations

import importlib.metadata
import importlib.util
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..contracts.epistemic import Confidence
from .models import MathCertificate, RealInterval, VerifierMetadata
from .verifiers import VerificationStatus, VerifierEnvelope


@dataclass(frozen=True)
class ArbFlintCapability:
    available: bool
    family: str = "arb-flint"
    module_name: str | None = None
    version: str | None = None
    reason: str = ""

    @property
    def independence_group(self) -> str:
        return f"math-verifier:{self.family}:{self.version or 'unavailable'}"


def detect_arb_flint() -> ArbFlintCapability:
    spec = importlib.util.find_spec("flint")
    if spec is None:
        return ArbFlintCapability(
            available=False,
            reason="python-flint module 'flint' is not importable",
        )
    try:
        version = importlib.metadata.version("python-flint")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return ArbFlintCapability(
        available=True,
        module_name="flint",
        version=version,
        reason="python-flint module is importable",
    )


def registered_arb_flint_families() -> frozenset[str]:
    capability = detect_arb_flint()
    return frozenset({capability.family}) if capability.available else frozenset()


def interval_certificate(
    *,
    expression: str,
    lower: str,
    upper: str,
    precision_bits: int = 256,
) -> VerifierEnvelope:
    """Return an Arb/FLINT verifier envelope, failing closed when unavailable.

    This adapter intentionally supports only externally supplied interval
    endpoints. It does not parse arbitrary mathematics. When Arb/FLINT is not
    importable, or when no rigorous enclosure computation was performed, the
    envelope status remains UNKNOWN.
    """

    capability = detect_arb_flint()
    interval = RealInterval.from_decimals(lower, upper)
    cert = MathCertificate(
        expression=expression,
        value=interval,
        verifier=VerifierMetadata(
            method=capability.family,
            precision_bits=precision_bits,
            worker_version=capability.version,
        ),
        assumptions=[] if capability.available else ["Arb/FLINT backend unavailable"],
    )
    if not capability.available:
        return VerifierEnvelope(
            verifier_family=capability.family,
            verifier_version="unavailable",
            certificate=cert,
            status=VerificationStatus.UNKNOWN,
            checks=[],
            notes=[
                capability.reason,
                "No rigorous interval verification was performed.",
            ],
        )
    return VerifierEnvelope(
        verifier_family=capability.family,
        verifier_version=capability.version or "unknown",
        certificate=cert,
        status=VerificationStatus.UNKNOWN,
        checks=[],
        notes=[
            "Arb/FLINT capability detected, but this adapter has not been given a backend-specific enclosure proof for the expression.",
            "Status remains unknown rather than relabelling numeric output as rigorous.",
        ],
    )


def envelope_confidence(envelope: VerifierEnvelope) -> Confidence:
    """The canonical confidence this envelope justifies.

    The registry is consulted here, at the point of use. Mapping ``status``
    straight through was a bypass: ``VerifierEnvelope`` is public and its
    ``status`` is caller-supplied, so constructing one with
    ``status=ACCEPTED`` yielded ``RIGOROUS_NUMERICAL`` without any backend
    having run. A family with no registered adapter now maps to ``UNKNOWN``
    whatever the envelope says about itself.

    The ceiling is ``RIGOROUS_NUMERICAL`` and that is deliberately *not* in
    ``contracts.epistemic.RIGOROUS``: a certified enclosure is rigorous about a
    finite computation, and treating it as rigorous in general is the step from
    "checked to height T" to "true".
    """
    # Imported here, not at module scope: `contracts.mappings` reaches back
    # into `mathcert.verifiers`, so a top-level import closes the loop
    # mathcert -> contracts -> mathcert.
    from ..contracts.mappings import confidence_from_verification_status
    from .verifiers import _registered_adapters

    if envelope.status is VerificationStatus.ACCEPTED:
        if envelope.verifier_family not in _registered_adapters():
            return Confidence.UNKNOWN
        # AND attributable to a build. The registry check alone was a complete
        # answer only while no backend existed anywhere: with one installed,
        # the family IS registered, so a hand-built envelope naming it earned
        # RIGOROUS_NUMERICAL with no computation behind it. The hole was closed
        # by absence, and absence stopped being the case.
        #
        # These are the same three fields `validate_external_envelope` demands
        # of an ACCEPTED envelope, for the reason it gives there: metadata that
        # can be edited freely attributes a result to whatever it was edited
        # to. Required here as well because this function is the one a caller
        # reaches for when it wants a confidence rather than a list of errors.
        verifier = envelope.certificate.verifier
        if not (verifier.worker_hash and verifier.source_hash and verifier.worker_version):
            return Confidence.UNKNOWN
    return confidence_from_verification_status(envelope.status)


#: The strongest confidence any Arb/FLINT envelope can justify, whatever its
#: metadata claims.
MAX_CONFIDENCE = Confidence.RIGOROUS_NUMERICAL


# --- computed enclosures ---------------------------------------------------
#
# Everything above this line is capability detection and an honest placeholder:
# `interval_certificate` takes endpoints somebody else computed and reports
# UNKNOWN, because a certificate for an enclosure nothing computed is not a
# certificate. What follows actually computes one.
#
# Every `import flint` in the package is here. A backend reachable from several
# places is a backend that will eventually be reached from one that forgot to
# check whether it is installed.


DEFAULT_PRECISION_BITS = 128


@contextmanager
def _at_precision(bits: int) -> Iterator[None]:
    """Arb's working precision, set and put back.

    `flint.ctx.prec` is global mutable state. Leaving it raised would silently
    change the cost and the results of every later call in the process, which
    is the kind of action-at-a-distance that makes a computation depend on what
    happened to run before it.
    """
    import flint

    previous = flint.ctx.prec
    flint.ctx.prec = bits
    try:
        yield
    finally:
        flint.ctx.prec = previous


def _require_backend() -> ArbFlintCapability:
    capability = detect_arb_flint()
    if not capability.available:
        raise RuntimeError(capability.reason)
    return capability


def _digest(paths: list[str]) -> str:
    """A digest over file bytes, path-independent and order-independent.

    The paths differ between machines and the bytes do not, so only the bytes
    and their lengths go in. Without the lengths a digest over concatenated
    files would be blind to a byte moving from the end of one to the start of
    the next.
    """
    import hashlib

    running = hashlib.sha256()
    for blob in sorted(Path(path).read_bytes() for path in paths):
        running.update(len(blob).to_bytes(8, "big"))
        running.update(blob)
    return running.hexdigest()


def _worker_attribution() -> tuple[str, str]:
    """`(worker_hash, source_hash)` for a certificate this adapter issues.

    `validate_external_envelope` refuses an ACCEPTED envelope that carries
    neither, and the reason it gives is the whole point: without them
    "metadata can be edited freely and the recomputed hash simply agrees with
    whatever it was edited to". A version string alone is metadata; these are
    the bytes that did the work.

    The worker is the compiled extension actually called -- `flint.types.arb`
    and `flint.types.acb`, not the `flint` package's `__init__.py`, which is a
    few lines of re-exports and would be identical across builds linked
    against different FLINT libraries. The source is this file, which decides
    what the enclosure is asked for and what is checked of it: a certificate
    is only as good as both, and naming one without the other would attribute
    the result to half of what produced it.
    """
    import flint.types.acb
    import flint.types.arb

    worker = _digest(
        [flint.types.arb.__file__, flint.types.acb.__file__]
    )
    return worker, _digest([__file__])


def _raw_zero_count(height: float, precision_bits: int):
    """Arb's count, as the ball it returns, before anything is concluded.

    A named seam rather than a call inline, so the two guards below can be
    reached in a test. The obvious alternative -- substituting the method on
    `flint.types.arb` -- works against the abi3 wheel and raises `cannot set
    attribute of immutable type` against the cp313 one, so the guard would be
    exercised on one Python and silently skipped on the next. A gate that runs
    in only some of the environments it ships to is the failure this file's
    tests were themselves caught committing.
    """
    from flint import arb

    with _at_precision(precision_bits):
        return arb(height).zeta_nzeros()


def certified_zero_count(
    height: float, *, precision_bits: int = DEFAULT_PRECISION_BITS
) -> int:
    """`N(T)`: zeros with `0 < Im(s) <= height`, counted with multiplicity.

    Rigorous, by Arb's argument-principle count with Turing's method and
    explicit error bounds, and returned as an INTEGER because Arb returns a
    ball of radius zero -- an exact count, not a rounded one. A ball with a
    radius reaching here would mean the count was not determined, and raises
    rather than being rounded into a number that looks just as definite.

    Free at any height worth asking about: `N(10^8) = 248008025` comes back in
    about a hundredth of a second, where the floating-point argument principle
    in `symbolic/argument_principle.py` needed 24 minutes for `N(10^6)`.
    """
    _require_backend()

    counted = _raw_zero_count(height, precision_bits)
    if counted.rad() != 0:
        raise ValueError(
            f"the zero count below {height:g} came back as {counted}, which "
            f"is a ball rather than an integer: {precision_bits} bits did not "
            "determine it, and rounding an undetermined count would produce a "
            "number indistinguishable from a determined one"
        )
    mantissa, exponent = counted.mid().man_exp()

    # Read as mantissa x 2^exponent rather than through a float, because a
    # count above 2^53 would lose its last digits on the way through one and
    # come back looking exactly as definite. A NEGATIVE exponent means the
    # value is not an integer at all, and shifting right would quietly floor
    # it -- 3/2 arrives as (3, -1) and leaves as 1. A count that is not a whole
    # number is not a count, so it raises.
    if int(exponent) < 0:
        raise ValueError(
            f"the zero count below {height:g} is {counted}, which is not a "
            "whole number: a count with a fractional part is not a count, and "
            "rounding it would hide that"
        )
    return int(mantissa) << int(exponent)


@dataclass(frozen=True)
class CertifiedZeroScan:
    """Enclosures for the zeros below a height, and what was checked of them."""

    #: Enclosures that passed every check. The caller compares this against the
    #: strip count; equality is the whole verification.
    certified: int
    #: Each check by name, with what it decided. Named individually because
    #: "it passed" is not something a later reader can audit, and because the
    #: three say different things when they fail.
    checks: dict[str, bool]
    #: The largest ordinate placed. A float, and only for a report: nothing is
    #: decided from it.
    top_ordinate: float


def certified_zeros_on_line(
    count: int,
    height: float,
    *,
    precision_bits: int = DEFAULT_PRECISION_BITS,
) -> CertifiedZeroScan:
    """Isolate `count` zeros and certify each is on the line, below `height`.

    THE CHECKS ARE THE ARGUMENT. Arb hands back `count` enclosures and it would
    be easy to conclude too much from them:

      * `Re = 1/2 exactly`. An enclosure CONTAINING 1/2 says nothing: a ball
        around the critical line contains off-line points, so "Re = 1/2 to
        within 1e-30" is a different and much weaker claim. Arb represents a
        zero it has proved to be on the line with an exact real part, radius
        zero, and that -- not proximity -- is what is tested.
      * `inside (0, height]`. Ask for one zero too many and the extra sits
        above the height; a count including it is not a count below it.
      * `pairwise disjoint`. Without it, `count` enclosures could describe
        fewer than `count` distinct zeros, and the multiplicity argument that
        gives simplicity fails.

    Every comparison is Arb's, on balls, never on floats read out of them.
    Arb's `<`, `>` and `==` are true only when true of every point in the
    balls: two overlapping intervals compare False in BOTH directions, and a
    ball of radius 1e-30 around 1/2 is not equal to 1/2. Converting to float
    first would have thrown that away and turned three sound tests into three
    approximate ones.
    """
    _require_backend()
    from flint import acb, arb

    with _at_precision(precision_bits):
        zeros = list(acb.zeta_zeros(1, count)) if count else []
        half = arb(1) / 2
        limit = arb(height)
        origin = arb(0)

        ordered = sorted(zeros, key=lambda z: float(z.imag.mid()))
        checks = {
            "every enclosure has real part exactly 1/2": all(
                z.real == half for z in zeros
            ),
            f"every enclosure lies in (0, {height:g}]": all(
                z.imag > origin and z.imag <= limit for z in zeros
            ),
            "the enclosures are pairwise disjoint": all(
                a.imag < b.imag for a, b in zip(ordered, ordered[1:], strict=False)
            ),
        }
        certified = count if all(checks.values()) else 0
        top = float(ordered[-1].imag.mid()) if ordered else 0.0
    return CertifiedZeroScan(certified=certified, checks=checks, top_ordinate=top)


def zero_count_certificate(
    height: float, *, precision_bits: int = DEFAULT_PRECISION_BITS
) -> VerifierEnvelope:
    """`N(T)` as an ACCEPTED envelope -- the first one this adapter can issue.

    `interval_certificate` above returns UNKNOWN whatever it is handed, and
    says why: it was given endpoints rather than a computation. This one runs
    the enclosure, so it may say ACCEPTED, and `envelope_confidence` then maps
    it to `rigorous_numerical` -- but only because `_registered_adapters`
    independently detects the backend. Constructing an ACCEPTED envelope
    without one still yields UNKNOWN, which is the property that makes the
    status worth reading at all.

    The count is exact, so the enclosure is the degenerate interval `[N, N]`.
    That is not a weakness of the certificate; it is what a rigorous integer
    count looks like.
    """
    capability = _require_backend()
    counted = certified_zero_count(height, precision_bits=precision_bits)
    worker_hash, source_hash = _worker_attribution()
    return VerifierEnvelope(
        verifier_family=capability.family,
        verifier_version=capability.version or "unknown",
        certificate=MathCertificate(
            expression=f"N({height:g})",
            value=RealInterval.from_decimals(str(counted), str(counted)),
            verifier=VerifierMetadata(
                method=capability.family,
                precision_bits=precision_bits,
                worker_version=capability.version,
                worker_hash=worker_hash,
                source_hash=source_hash,
            ),
            assumptions=[
                "Arb's zero counting is correct",
                "the count is of zeros with 0 < Im(s) <= T, with multiplicity",
            ],
        ),
        status=VerificationStatus.ACCEPTED,
        checks=[
            "counted by the argument principle with Turing's method",
            "returned as a ball of radius zero, so the integer is determined",
        ],
        notes=[
            "Rigorous about a finite computation, which is what "
            "rigorous_numerical means. It says nothing about zeros above T.",
        ],
    )
