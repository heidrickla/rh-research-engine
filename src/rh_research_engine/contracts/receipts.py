"""DRE receipts: the difference between "says accepted" and "was authorized".

``ObligationDischargeDecision`` is an ordinary artifact. Its
``dre_decision_status``, ``dre_decision_ref``, ``created_by`` and
``method_family`` are all caller-supplied, so anything able to construct one can
write ``accepted`` on it. The obligation and evidence hashes prove the *records
agree with each other*; they prove nothing about who authorized the ruling.

A receipt binds the decision to a specific DRE run:

    dre_input_hash          what the engine was given
    dre_model_pack_hash     which rules it applied
    dre_engine_fingerprint  which build applied them
    dre_decision_hash       the decision content it ruled on
    dre_proof_hash          the engine's own derivation record

WHY VERIFICATION IS NOT A TYPE. An earlier version exposed a public
``VerifiedDecision`` model and a ``trusted_engines=`` override. Both were
bypasses: a caller could construct a ``VerifiedDecision`` directly -- the name
was a promise nothing enforced -- or simply pass its own fingerprint as trusted.
Verification now happens at the point of use. ``resolve_discharges`` takes raw
decisions and raw receipts and revalidates them here, against a module-private
trust set that no public function can widen.

ACTIVATION REQUIREMENT. Registering a fingerprint and its pinned model-pack
hashes in ``_TRUSTED_DRE_ENGINES`` is **necessary and not sufficient**. A fingerprint is
just a string in this file; on its own it would restore the same
self-authentication this module exists to remove. ``receipt_authentication``
records which of the three real mechanisms established the receipt:

  * :attr:`ReceiptAuthentication.SIGNATURE` -- a cryptographic signature over
    the receipt, verified against a key held outside this repository;
  * :attr:`ReceiptAuthentication.SEALED_STORE` -- retrieval from a sealed DRE
    decision store, addressed by ``dre_decision_hash``, whose seal is checked
    the way durable memory's is;
  * :attr:`ReceiptAuthentication.DETERMINISTIC_REPLAY` -- re-running the engine
    at ``dre_engine_fingerprint`` on ``dre_input_hash`` with
    ``dre_model_pack_hash`` and reproducing ``dre_proof_hash``.

The default is :attr:`ReceiptAuthentication.NONE`, and ``NONE`` never verifies.
Declaring one of the other three is a *claim* about how the receipt was
established, not the check itself -- which is why the claim alone gets a
receipt no further than a trusted fingerprint does: both are inputs to a
verifier that Phase 1 deliberately leaves unbuilt. Until it exists,
``_TRUSTED_DRE_ENGINES`` stays empty and discharge authority is inert.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .artifacts import DreDecisionStatus, ObligationDischargeDecision
from .hashing import canonical_hash, canonical_json, validate_sha256

__all__ = [
    "DreReceipt",
    "ReceiptAuthentication",
    "ReceiptError",
    "activation_status",
]

#: Engine fingerprint -> the model-pack hashes that engine is trusted for.
#: Empty, and private.
#:
#: Private because a public trust set is a public bypass: a caller that can pass
#: its own fingerprint has authorized itself. Nothing in the public API widens
#: this, and adding to it is not sufficient on its own -- see the module
#: docstring's activation requirement.
#:
#: A mapping rather than a set of fingerprints, because "which engine ruled" and
#: "under which rules" are two different questions and only the pair is a trust
#: root. `dre_model_pack_hash` was previously checked for non-emptiness alone,
#: so a trusted engine could have ruled under any pack it named -- including one
#: written for the occasion. The pack has to be pinned by the build, not chosen
#: by the receipt.
_TRUSTED_DRE_ENGINES: dict[str, frozenset[str]] = {}

#: Receipt hashes that tests have explicitly marked as authenticated work
#: products. Empty in production, private, and scoped by context manager.
#:
#: Empty means *nothing authenticates*, the same way an empty
#: `_TRUSTED_DRE_ENGINES` means nothing is trusted. Two adjacent private hooks
#: whose empty argument meant opposite things would be a trap: a caller reading
#: one and reasoning about the other would get blanket authentication from a
#: value that looks like it grants none.
_TEST_ONLY_AUTHENTICATED_RECEIPTS: frozenset[str] = frozenset()

#: Guards `_VerifiedDecision` construction. A "verified" value that any caller
#: can build is a label, not a verification.
_VERIFIER_TOKEN = object()


class ReceiptError(ValueError):
    """A receipt does not establish that DRE authorized the decision."""


class ReceiptAuthentication(StrEnum):
    """How a receipt was established as genuinely issued by the named engine.

    The point of the enum is that there is no fourth option. "The record says
    accepted", "the hashes agree with each other" and "the fingerprint is in
    the set" are all self-consistency, and self-consistency is what a forged
    receipt has too.
    """

    #: Nothing authenticates this receipt. The fail-closed default.
    NONE = "none"
    #: Signed over by a key held outside this repository.
    SIGNATURE = "signature"
    #: Retrieved from a sealed DRE decision store, addressed by decision hash.
    SEALED_STORE = "sealed_store"
    #: Reproduced by re-running the pinned engine on the pinned inputs.
    DETERMINISTIC_REPLAY = "deterministic_replay"


#: The mechanisms that could establish a receipt. Membership here is not itself
#: authentication -- see the module docstring.
AUTHENTICATED_MECHANISMS: frozenset[ReceiptAuthentication] = frozenset(
    {
        ReceiptAuthentication.SIGNATURE,
        ReceiptAuthentication.SEALED_STORE,
        ReceiptAuthentication.DETERMINISTIC_REPLAY,
    }
)


class DreReceipt(BaseModel):
    """Evidence that a specific DRE run produced a specific decision.

    Public, because a caller has to be able to *supply* one. Supplying it is not
    the same as it being believed: everything that decides whether it counts is
    private to this module.
    """

    model_config = ConfigDict(extra="forbid")

    decision_ref: str
    #: `artifact_hash()` of the decision this receipt covers.
    dre_decision_hash: str
    dre_input_hash: str
    dre_model_pack_hash: str
    dre_engine_fingerprint: str
    #: The engine's own derivation record for the ruling.
    dre_proof_hash: str
    #: Which mechanism established this receipt. Defaults to NONE, which never
    #: verifies, so a receipt that says nothing about its own provenance is
    #: refused rather than accepted by omission.
    receipt_authentication: ReceiptAuthentication = ReceiptAuthentication.NONE
    #: Detached Ed25519 signature over everything else in this record, hex.
    signature: str | None = None
    #: Which key produced `signature`. Signed over, so it cannot be swapped for
    #: the id of a key that happens to verify some other payload.
    signing_key_id: str | None = None
    notes: list[str] = Field(default_factory=list)

    def signing_payload(self) -> str:
        """The bytes a signature covers: this record without the signature.

        `signing_key_id` is deliberately inside the payload. A signature that
        did not cover the key id would let an attacker keep a valid signature
        and relabel which key made it.
        """
        return canonical_json(self.model_dump(mode="json", exclude={"signature"}))

    @field_validator(
        "dre_decision_hash", "dre_input_hash", "dre_model_pack_hash", "dre_proof_hash"
    )
    @classmethod
    def _digests_are_digests(cls, value: str, info) -> str:
        return validate_sha256(value, f"DreReceipt.{info.field_name}")

    def receipt_hash(self) -> str:
        """Canonical-JSON identity for this receipt.

        Not delimiter concatenation: joining these fields with ``"|"`` made the
        digest ambiguous for any field that could contain the delimiter, and
        positional, so adding a field would silently redefine it.
        """
        return canonical_hash(self.model_dump(mode="json"))


class _VerifiedDecision:
    """A decision a trusted DRE engine is known to have authorized.

    Deliberately not a Pydantic model and not exported. Construction requires a
    module-private token, so this cannot be forged from outside even by a caller
    that imports the name.
    """

    __slots__ = ("decision", "receipt", "receipt_hash", "engine_fingerprint")

    def __init__(
        self,
        token: object,
        *,
        decision: ObligationDischargeDecision,
        receipt: DreReceipt,
        receipt_hash: str,
        engine_fingerprint: str,
    ) -> None:
        if token is not _VERIFIER_TOKEN:
            raise ReceiptError(
                "_VerifiedDecision cannot be constructed directly. Verification "
                "happens inside contracts.receipts, against a trust set no caller "
                "can widen; a value you can build yourself is a label, not a "
                "verification."
            )
        self.decision = decision
        self.receipt = receipt
        self.receipt_hash = receipt_hash
        self.engine_fingerprint = engine_fingerprint

    @property
    def obligation_ref(self) -> str:
        return self.decision.obligation_ref


def _authenticate(receipt: DreReceipt) -> None:
    """Authenticate the receipt's claimed work-product certification.

    The mechanism declaration has already been checked before this is called.
    This function is the private seam where real signature, sealed-store, or
    deterministic-replay verification is connected. Until one is connected, a
    declaration remains only a claim and fails closed.

    Tests reach the positive path by naming *exact receipt hashes* through
    `_test_only_authenticated_receipts`. Binding to the hash rather than to the
    mechanism string is the point: a test that opted in by declaring `signature`
    would be asserting the very thing under test.
    """
    if receipt.receipt_hash() in _TEST_ONLY_AUTHENTICATED_RECEIPTS:
        return

    if receipt.receipt_authentication is ReceiptAuthentication.SIGNATURE:
        _authenticate_signature(receipt)
    elif receipt.receipt_authentication is ReceiptAuthentication.SEALED_STORE:
        _authenticate_sealed_store(receipt)
    elif receipt.receipt_authentication is ReceiptAuthentication.DETERMINISTIC_REPLAY:
        _authenticate_deterministic_replay(receipt)
    else:  # Defensive: callers should have tripped the mechanism check first.
        raise ReceiptError(
            f"receipt declares receipt_authentication={receipt.receipt_authentication.value!r}, "
            "which cannot authenticate a DRE work product"
        )


def _authentication_backend_missing(receipt: DreReceipt, mechanism: str) -> ReceiptError:
    return ReceiptError(
        f"receipt for {receipt.decision_ref!r} declares receipt_authentication="
        f"{receipt.receipt_authentication.value!r}, but no {mechanism} verifier "
        "is connected. The field is a claim about DRE work-product certification, "
        "not the certification itself."
    )


#: Environment variable naming a JSON file of trusted DRE signing keys,
#: ``{"key-id": "<64 hex chars of an Ed25519 public key>"}``.
#:
#: An environment variable rather than a repository path, because a key
#: committed alongside the thing it authenticates proves only that both were
#: written by the same author. Unset means no keys, which means no signature
#: verifies -- the fail-closed default.
SIGNING_KEY_ENV = "RHRE_DRE_PUBLIC_KEYS"

#: Key ids withdrawn from service. Checked *before* the trusted set, so
#: revoking beats registering and a key cannot be un-revoked by re-adding it.
_REVOKED_SIGNING_KEYS: frozenset[str] = frozenset()

#: Set only by the private test hook below.
_TEST_ONLY_SIGNING_KEYS: dict[str, bytes] | None = None


def _signature_backend():
    """The Ed25519 verifier, or ``None`` when the optional dependency is absent.

    Capability detection, the same shape as ``mathcert.arb_flint``: whether a
    backend exists is a fact about the environment, not something a caller
    asserts. Absent means signature authentication cannot happen here, which is
    reported rather than skipped.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:  # pragma: no cover - exercised by the absent-backend test
        return None
    return Ed25519PublicKey, InvalidSignature


def _load_trusted_signing_keys() -> dict[str, bytes]:
    """Registered public keys, read fresh from the operator-designated file.

    Re-read per verification rather than cached at import, so withdrawing a key
    from the file takes effect immediately. A malformed file yields no keys:
    the safe reading of "the trust root cannot be parsed" is that nothing is
    trusted.
    """
    if _TEST_ONLY_SIGNING_KEYS is not None:
        return _TEST_ONLY_SIGNING_KEYS
    path = os.environ.get(SIGNING_KEY_ENV)
    if not path:
        return {}
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        return {
            str(key_id): bytes.fromhex(str(public_hex))
            for key_id, public_hex in document.items()
        }
    except (OSError, ValueError, AttributeError):
        return {}


def _authenticate_signature(receipt: DreReceipt) -> None:
    backend = _signature_backend()
    if backend is None:
        raise _authentication_backend_missing(receipt, "signature")
    public_key_type, invalid_signature = backend

    if not receipt.signature:
        raise ReceiptError(
            f"receipt for {receipt.decision_ref!r} declares signature "
            "authentication but carries no signature"
        )
    key_id = receipt.signing_key_id
    if not key_id:
        raise ReceiptError(
            f"receipt for {receipt.decision_ref!r} carries a signature but names no "
            "signing key. An unattributed signature cannot be checked against "
            "anything, and cannot be revoked."
        )
    if key_id in _REVOKED_SIGNING_KEYS:
        raise ReceiptError(
            f"signing key {key_id!r} has been revoked. Receipts it signed are no "
            "longer authoritative, including ones signed before the revocation: "
            "revocation exists precisely for the case where the key was in the "
            "wrong hands earlier than anyone noticed."
        )
    trusted = _load_trusted_signing_keys()
    public_bytes = trusted.get(key_id)
    if public_bytes is None:
        raise ReceiptError(
            f"signing key {key_id!r} is not registered in this build "
            f"{sorted(trusted) or '(none)'}. Register it in the file named by "
            f"{SIGNING_KEY_ENV}, which is held outside this repository."
        )
    try:
        public_key_type.from_public_bytes(public_bytes).verify(
            bytes.fromhex(receipt.signature),
            receipt.signing_payload().encode("utf-8"),
        )
    except invalid_signature as exc:
        raise ReceiptError(
            f"signature on the receipt for {receipt.decision_ref!r} does not verify "
            f"under key {key_id!r}. Either the receipt was edited after signing, or "
            "it was signed by a different key."
        ) from exc
    except ValueError as exc:
        raise ReceiptError(
            f"signature on the receipt for {receipt.decision_ref!r} is malformed: "
            f"{exc}"
        ) from exc


# The two remaining backends, unimplemented. Each raises rather than returning,
# so connecting one is a deliberate edit to a function that currently refuses --
# not the removal of a guard. See docs/PHASE2_ACTIVATION.md item A1.2.


def _authenticate_sealed_store(receipt: DreReceipt) -> None:
    raise _authentication_backend_missing(receipt, "sealed-store")


def _authenticate_deterministic_replay(receipt: DreReceipt) -> None:
    raise _authentication_backend_missing(receipt, "deterministic-replay")


def _verify_decision(
    decision: ObligationDischargeDecision, receipt: DreReceipt | None
) -> _VerifiedDecision:
    """Verify against the module-private trust set, or raise.

    No ``trusted_engines`` parameter, on purpose. Every failure names what was
    missing, because "verification failed" does not tell a researcher whether
    the engine is disconnected, the receipt is stale, or the decision was edited
    after the ruling.
    """
    if receipt is None:
        raise ReceiptError(
            f"decision {decision.artifact_id!r} has no DRE receipt. Its "
            "dre_decision_status is caller-supplied, so the record saying "
            "'accepted' establishes nothing about who authorized it."
        )
    if decision.dre_decision_status is not DreDecisionStatus.ACCEPTED:
        raise ReceiptError(
            f"decision {decision.artifact_id!r} is "
            f"{decision.dre_decision_status.value!r}, not accepted"
        )
    if receipt.decision_ref != decision.artifact_id:
        raise ReceiptError(
            f"receipt covers {receipt.decision_ref!r}, not {decision.artifact_id!r}"
        )

    actual = decision.artifact_hash()
    if receipt.dre_decision_hash != actual:
        raise ReceiptError(
            f"receipt binds a different version of {decision.artifact_id!r} "
            f"(receipt {receipt.dre_decision_hash[:16]}..., actual {actual[:16]}...); "
            "the decision was edited after the ruling"
        )

    if not receipt.dre_engine_fingerprint:
        raise ReceiptError(f"receipt for {decision.artifact_id!r} names no engine")
    pinned_packs = _TRUSTED_DRE_ENGINES.get(receipt.dre_engine_fingerprint)
    if pinned_packs is None:
        raise ReceiptError(
            f"engine {receipt.dre_engine_fingerprint!r} is not trusted by this build "
            f"{sorted(_TRUSTED_DRE_ENGINES) or '(empty)'}. No DRE engine is connected, "
            "so no decision can currently be authoritative -- the honest state, not a "
            "bug. Activation needs a fingerprint AND receipt authentication by "
            "signature, sealed-store retrieval, or deterministic replay."
        )
    for field in ("dre_input_hash", "dre_model_pack_hash", "dre_proof_hash"):
        if not getattr(receipt, field):
            raise ReceiptError(
                f"receipt for {decision.artifact_id!r} has no {field}; without it the "
                "ruling cannot be replayed or attributed to a specific run"
            )
    if receipt.dre_model_pack_hash not in pinned_packs:
        raise ReceiptError(
            f"engine {receipt.dre_engine_fingerprint!r} is trusted, but not for model "
            f"pack {receipt.dre_model_pack_hash[:16]}... (pinned: "
            f"{sorted(h[:16] + '...' for h in pinned_packs) or '(none)'}). The pack is "
            "the set of rules the ruling was made under, and a receipt naming its own "
            "pack is describing the rules it would like to have been judged by."
        )

    # A trusted fingerprint says which engine *claims* to have ruled. It does
    # not say that this engine actually did: the fingerprint travels inside the
    # receipt, so anything able to write a receipt can write the fingerprint
    # too. Authentication is the separate question, and NONE is the honest
    # default rather than an accepted omission.
    if receipt.receipt_authentication not in AUTHENTICATED_MECHANISMS:
        raise ReceiptError(
            f"receipt for {decision.artifact_id!r} declares receipt_authentication="
            f"{receipt.receipt_authentication.value!r}. A trusted fingerprint is "
            "necessary and not sufficient -- it names the claimed issuer, and the "
            "claim travels inside the receipt. One of "
            f"{sorted(m.value for m in AUTHENTICATED_MECHANISMS)} must have "
            "established it."
        )
    _authenticate(receipt)

    return _VerifiedDecision(
        _VERIFIER_TOKEN,
        decision=decision,
        receipt=receipt,
        receipt_hash=receipt.receipt_hash(),
        engine_fingerprint=receipt.dre_engine_fingerprint,
    )


def _verify_all(
    decisions: dict[str, ObligationDischargeDecision],
    receipts: dict[str, DreReceipt],
) -> tuple[dict[str, _VerifiedDecision], dict[str, str]]:
    """Verify a batch, returning the verified ones and why the rest failed."""
    verified: dict[str, _VerifiedDecision] = {}
    rejected: dict[str, str] = {}
    for obligation_ref, decision in decisions.items():
        try:
            verified[obligation_ref] = _verify_decision(
                decision, receipts.get(decision.artifact_id)
            )
        except ReceiptError as exc:
            rejected[obligation_ref] = str(exc)
    return verified, rejected


@contextmanager
def _test_only_trusted_engines(engines: dict[str, frozenset[str]]) -> Iterator[None]:
    """Temporarily trust fingerprint/model-pack pairs. **Tests only.**

    Private and context-managed so the widening is always scoped and always
    reverted. Production code has no equivalent: no public function takes a
    trust set, and this one is not exported.
    """
    global _TRUSTED_DRE_ENGINES
    previous = _TRUSTED_DRE_ENGINES
    _TRUSTED_DRE_ENGINES = {
        fingerprint: frozenset(packs) for fingerprint, packs in engines.items()
    }
    try:
        yield
    finally:
        _TRUSTED_DRE_ENGINES = previous


@contextmanager
def _test_only_authenticated_receipts(receipt_hashes: set[str] | frozenset[str]) -> Iterator[None]:
    """Temporarily mark specific receipt hashes as authenticated. **Tests only.**

    Preserves positive-path tests without teaching production to trust a
    mechanism declaration. Authentication is bound to the exact hashes supplied:
    an empty set authenticates nothing, matching `_test_only_trusted_engines`.

    An earlier version treated the empty set as "authenticate everything", for
    compatibility with positive tests written before this seam existed. That
    made the common call -- the one with no arguments -- bypass the seam
    entirely, so the authenticator was exercised by exactly one test in the
    suite while thirteen call sites read as though they covered it.
    """
    global _TEST_ONLY_AUTHENTICATED_RECEIPTS
    previous = _TEST_ONLY_AUTHENTICATED_RECEIPTS
    _TEST_ONLY_AUTHENTICATED_RECEIPTS = frozenset(receipt_hashes)
    try:
        yield
    finally:
        _TEST_ONLY_AUTHENTICATED_RECEIPTS = previous


@contextmanager
def _test_only_signing_keys(keys: dict[str, bytes]) -> Iterator[None]:
    """Temporarily register Ed25519 public keys. **Tests only.**

    Private and context-managed, like the other two hooks. An empty mapping
    registers nothing, so the same convention holds throughout: empty grants
    nothing.
    """
    global _TEST_ONLY_SIGNING_KEYS
    previous = _TEST_ONLY_SIGNING_KEYS
    _TEST_ONLY_SIGNING_KEYS = dict(keys)
    try:
        yield
    finally:
        _TEST_ONLY_SIGNING_KEYS = previous


@contextmanager
def _test_only_revoked_keys(key_ids: set[str] | frozenset[str]) -> Iterator[None]:
    """Temporarily withdraw key ids. **Tests only.**"""
    global _REVOKED_SIGNING_KEYS
    previous = _REVOKED_SIGNING_KEYS
    _REVOKED_SIGNING_KEYS = frozenset(key_ids)
    try:
        yield
    finally:
        _REVOKED_SIGNING_KEYS = previous


def activation_status() -> dict[str, object]:
    """Report whether discharge authority is active, without exposing the set.

    Public so tooling can say "inert" honestly; returns a count rather than the
    fingerprints, so reading it grants nothing.
    """
    return {
        "trusted_engine_count": len(_TRUSTED_DRE_ENGINES),
        "pinned_model_pack_count": sum(len(p) for p in _TRUSTED_DRE_ENGINES.values()),
        "discharge_authority_active": bool(_TRUSTED_DRE_ENGINES),
        "accepted_authentication_mechanisms": sorted(
            m.value for m in AUTHENTICATED_MECHANISMS
        ),
        "signature_backend_available": _signature_backend() is not None,
        "registered_signing_key_count": len(_load_trusted_signing_keys()),
        "revoked_signing_key_count": len(_REVOKED_SIGNING_KEYS),
        "activation_requires": [
            "a registered engine fingerprint",
            "a pinned model-pack hash for that fingerprint",
            "receipt authentication by signature, sealed-store retrieval, "
            "or deterministic replay",
        ],
    }
