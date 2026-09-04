"""Ed25519 signature authentication for DRE receipts (activation item A1.2).

The first mechanism that can actually turn `receipt_authentication: signature`
from a claim into a fact. Everything else in the discharge path was already
content-bound; this is the only step that establishes *issuer* identity, which
is why HMAC was ruled out for it -- a shared secret that can verify a receipt
can also mint one.

The positive path here is the first in the repository where a receipt genuinely
authenticates. It runs entirely on an ephemeral keypair generated in-process and
registered through a private test hook, so nothing about it is reachable from
production: `SIGNING_KEY_ENV` is unset, no key is registered, and the engine
trust registry is empty regardless.
"""

from __future__ import annotations

import json

import pytest

import helpers_discharge
from rh_research_engine.contracts.discharge import resolve_discharges
from rh_research_engine.contracts.receipts import (
    SIGNING_KEY_ENV,
    DreReceipt,
    ReceiptAuthentication,
    _load_trusted_signing_keys,
    _signature_backend,
    _test_only_revoked_keys,
    _test_only_signing_keys,
    _test_only_trusted_engines,
    activation_status,
)

cryptography = pytest.importorskip(
    "cryptography", reason="signature authentication needs the optional backend"
)
from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: E402
    Ed25519PrivateKey,
)

KEY_ID = "dre-signing-key-2026-08"


def _keypair():
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes_raw()
    return private, public


def _sign(receipt: DreReceipt, private, *, key_id: str = KEY_ID) -> DreReceipt:
    """Attach a detached signature, the way a real DRE issuer would.

    `signing_key_id` is set before the payload is computed, because the payload
    covers it.
    """
    unsigned = receipt.model_copy(update={"signing_key_id": key_id, "signature": None})
    signature = private.sign(unsigned.signing_payload().encode("utf-8"))
    return unsigned.model_copy(update={"signature": signature.hex()})


def _signed_registry(private, *, key_id: str = KEY_ID):
    registry = helpers_discharge.accepted_registry()
    registry[3]["DEC-1"] = _sign(registry[3]["DEC-1"], private, key_id=key_id)
    return registry


def _resolve(registry):
    obligations, evidence, decisions, receipts = registry
    return resolve_discharges(
        ["OBL-1"],
        obligations=obligations,
        evidence=evidence,
        decisions=decisions,
        receipts=receipts,
    )


# ---------------------------------------------------------------------------
# The positive path
# ---------------------------------------------------------------------------


def test_a_genuinely_signed_receipt_authenticates():
    """The first receipt in this repository that actually verifies."""
    private, public = _keypair()
    registry = _signed_registry(private)

    with _test_only_trusted_engines(helpers_discharge.TRUSTED):
        with _test_only_signing_keys({KEY_ID: public}):
            resolution = _resolve(registry)

    assert resolution.qualifying == ["OBL-1"]
    provenance = resolution.provenance_payload()["OBL-1"]
    assert provenance["receipt_authentication"] == "signature"


def test_the_positive_path_still_needs_a_trusted_engine():
    """Defence in depth: a valid signature does not substitute for trust.

    The engine registry is checked before authentication, so a perfectly signed
    receipt from an engine this build does not trust is refused before its
    signature is ever examined. Both halves have to hold.
    """
    private, public = _keypair()
    registry = _signed_registry(private)

    with _test_only_signing_keys({KEY_ID: public}):
        resolution = _resolve(registry)

    assert resolution.qualifying == []
    assert "not trusted by this build" in resolution.explain()


# ---------------------------------------------------------------------------
# A1.5 negative cases
# ---------------------------------------------------------------------------


def _refusal(registry, *, keys=None, revoked=frozenset()):
    with _test_only_trusted_engines(helpers_discharge.TRUSTED):
        with _test_only_signing_keys(keys or {}):
            with _test_only_revoked_keys(revoked):
                return _resolve(registry)


def test_an_edited_receipt_no_longer_verifies():
    """Stale signature: the record changed after it was signed."""
    private, public = _keypair()
    registry = _signed_registry(private)
    registry[3]["DEC-1"] = registry[3]["DEC-1"].model_copy(
        update={"dre_input_hash": "f" * 64}
    )

    resolution = _refusal(registry, keys={KEY_ID: public})
    assert resolution.qualifying == []
    assert "does not verify" in resolution.explain()


def test_a_signature_from_a_different_key_does_not_verify():
    private, _ = _keypair()
    _, other_public = _keypair()
    registry = _signed_registry(private)

    resolution = _refusal(registry, keys={KEY_ID: other_public})
    assert resolution.qualifying == []
    assert "does not verify" in resolution.explain()


def test_relabelling_which_key_signed_invalidates_the_signature():
    """`signing_key_id` is inside the signed payload, so it cannot be swapped.

    Without that, an attacker holding a valid signature could point it at
    whichever registered key happened to verify some other payload.
    """
    private, public = _keypair()
    registry = _signed_registry(private)
    registry[3]["DEC-1"] = registry[3]["DEC-1"].model_copy(
        update={"signing_key_id": "some-other-key"}
    )

    resolution = _refusal(registry, keys={KEY_ID: public, "some-other-key": public})
    assert resolution.qualifying == []
    assert "does not verify" in resolution.explain()


def test_an_unregistered_key_is_refused():
    private, _ = _keypair()
    resolution = _refusal(_signed_registry(private), keys={})
    assert resolution.qualifying == []
    assert "is not registered in this build" in resolution.explain()


def test_a_revoked_key_is_refused_even_though_it_is_registered():
    """Revocation beats registration, and applies retroactively.

    A key is revoked precisely when it may have been in the wrong hands earlier
    than anyone noticed, so honouring signatures it made before the revocation
    would defeat the point.
    """
    private, public = _keypair()
    resolution = _refusal(
        _signed_registry(private), keys={KEY_ID: public}, revoked={KEY_ID}
    )
    assert resolution.qualifying == []
    assert "has been revoked" in resolution.explain()


def test_declaring_signature_authentication_without_a_signature_is_refused():
    registry = helpers_discharge.accepted_registry()
    resolution = _refusal(registry, keys={KEY_ID: b"\x00" * 32})
    assert resolution.qualifying == []
    assert "carries no signature" in resolution.explain()


def test_a_signature_with_no_key_id_is_refused():
    """An unattributed signature cannot be checked, and cannot be revoked."""
    private, public = _keypair()
    registry = _signed_registry(private)
    registry[3]["DEC-1"] = registry[3]["DEC-1"].model_copy(
        update={"signing_key_id": None}
    )
    resolution = _refusal(registry, keys={KEY_ID: public})
    assert resolution.qualifying == []
    assert "names no signing key" in resolution.explain()


def test_a_malformed_signature_is_refused_as_malformed():
    private, public = _keypair()
    registry = _signed_registry(private)
    registry[3]["DEC-1"] = registry[3]["DEC-1"].model_copy(
        update={"signature": "not-hex"}
    )
    resolution = _refusal(registry, keys={KEY_ID: public})
    assert resolution.qualifying == []
    assert "malformed" in resolution.explain()


def test_an_absent_backend_fails_closed(monkeypatch):
    """No `cryptography` installed means signatures cannot be checked here."""
    from rh_research_engine.contracts import receipts

    private, public = _keypair()
    monkeypatch.setattr(receipts, "_signature_backend", lambda: None)
    resolution = _refusal(_signed_registry(private), keys={KEY_ID: public})
    assert resolution.qualifying == []
    assert "no signature verifier" in resolution.explain()


# ---------------------------------------------------------------------------
# Key custody
# ---------------------------------------------------------------------------


def test_keys_load_from_the_operator_designated_file(tmp_path, monkeypatch):
    """Outside the repository, named by an environment variable."""
    _, public = _keypair()
    key_file = tmp_path / "dre-public-keys.json"
    key_file.write_text(json.dumps({KEY_ID: public.hex()}), encoding="utf-8", newline="")
    monkeypatch.setenv(SIGNING_KEY_ENV, str(key_file))

    assert _load_trusted_signing_keys() == {KEY_ID: public}


@pytest.mark.parametrize(
    "content", ["not json at all", '{"k": "zz"}', '["a", "list"]'], ids=range(3)
)
def test_an_unparseable_key_file_registers_nothing(tmp_path, monkeypatch, content):
    """The safe reading of "the trust root cannot be parsed" is that it is empty."""
    key_file = tmp_path / "broken.json"
    key_file.write_text(content, encoding="utf-8", newline="")
    monkeypatch.setenv(SIGNING_KEY_ENV, str(key_file))

    assert _load_trusted_signing_keys() == {}


def test_an_unset_environment_variable_registers_nothing(monkeypatch):
    monkeypatch.delenv(SIGNING_KEY_ENV, raising=False)
    assert _load_trusted_signing_keys() == {}


def test_no_signing_key_is_committed_to_this_repository():
    """A key stored beside what it authenticates proves only common authorship.

    Asks git what is *tracked*, because that is what "committed" means. An
    earlier version walked the working tree, which flagged
    `.venv/.../certifi/cacert.pem` on any machine with an in-tree virtualenv --
    a CA bundle, not key material, and not committed. A security check that
    cries wolf is one people learn to skip, so the false positive mattered more
    than the missing coverage would have.
    """
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo,
        capture_output=True,
        text=True, encoding="utf-8",
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout; 'committed' is not defined here")

    offenders = [
        name
        for name in result.stdout.splitlines()
        if name and Path(name).suffix in {".pem", ".key", ".pub"}
    ]
    assert offenders == [], f"key material committed at {offenders}"


# ---------------------------------------------------------------------------
# Production posture
# ---------------------------------------------------------------------------


def test_production_registers_no_signing_keys(monkeypatch):
    monkeypatch.delenv(SIGNING_KEY_ENV, raising=False)
    status = activation_status()
    assert status["registered_signing_key_count"] == 0
    assert status["discharge_authority_active"] is False
    # The backend being present is a fact about the environment, and reporting
    # it honestly is the point -- it grants nothing on its own.
    assert status["signature_backend_available"] is (_signature_backend() is not None)


def test_a_signed_receipt_still_discharges_nothing_in_production(monkeypatch):
    """The whole mechanism, exercised against the real registries."""
    monkeypatch.delenv(SIGNING_KEY_ENV, raising=False)
    private, _ = _keypair()
    assert _resolve(_signed_registry(private)).qualifying == []


def test_the_signature_hook_is_private_and_unexported():
    from rh_research_engine.contracts import receipts

    for name in ("_test_only_signing_keys", "_test_only_revoked_keys"):
        assert name not in receipts.__all__
    assert "signature" in {m.value for m in ReceiptAuthentication}
