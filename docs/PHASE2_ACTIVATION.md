# Phase 2 activation queue

Work that Phase 1 deliberately left undone, recorded here so it is queued
rather than remembered.

Phase 1 closed with both trust registries empty and both authority paths inert.
That is the correct posture for a phase whose job was to establish what the
records *are*, and it is why nothing here is urgent: the system currently
refuses everything it cannot substantiate, which is the safe direction to be
wrong in. What it means is that two capabilities are implemented at the contract
boundary and cannot yet be switched on honestly.

Nothing in this document changes Phase 1. The invariant Phase 1 established
stays in force throughout:

> With the production DRE and verifier trust registries empty, no public
> constructor, public function, CLI command, migration path, or closure path
> can produce a qualifying discharge, accepted rigorous authority,
> RH-equivalent frontier advancement, or a worker-declared proof.

Each item below states what it would take to relax that safely, and what must
still hold afterwards. Source: `docs/reviews/PHASE1_FINAL_CLOSURE.md`
(*Activation requirements*, *Open limitations*) and §2.5 / §11 of the Phase 1
Final One-Shot Closure Gate.

---

## A1 — Receipt authentication verification

**Status:** Done, except for two backends that cannot be built until the
things they verify against exist. Signature authentication works end to end.

### Where it stands

`DreReceipt.receipt_authentication` records *which* mechanism established a
receipt — `signature`, `sealed_store`, `deterministic_replay`, or the
fail-closed default `none`. `contracts/receipts.py::_verify_decision` refuses
`none` outright, and now routes every other declaration through a private
`_authenticate` seam whose three backends all refuse.

A declaration is therefore no longer taken at face value, and one mechanism can
now turn one into a fact: `signature` verifies a detached Ed25519 signature over
the receipt's canonical JSON. The other two still refuse for want of something
to verify against.

**Discharge authority remains inert in production**, because the engine trust
registry is empty and no signing key is registered. Those are two independent
gates and a receipt has to pass both, which is why the signature work could land
without changing the posture at all.

### The work

**A1.1 — An authenticator seam. Done** (`7aa6a7f`, hardened in `69a5875`).
Private `_authenticate(receipt)` in `contracts/receipts.py`, called from
`_verify_decision` after the mechanism-declaration check, taking no
caller-supplied parameter.

> Worth recording, because it is the failure mode this kind of seam invites.
> The seam landed correct and almost entirely untested: its test hook treated an
> *empty* set of authenticated receipts as "authenticate everything", and all
> thirteen call sites used the argument-free form that produced exactly that.
> Neutering `_authenticate` to `return` failed one test. The hook now fails
> closed on an empty set like its sibling `_test_only_trusted_engines`, every
> call site names the receipts it stands in for, and the same mutation now
> fails six.
>
> The lesson carries to A1.2: a seam whose correct behaviour today is *to refuse
> everything* is indistinguishable from a seam that is never called. Each
> backend needs at least one test that a no-op version of it would fail.

**A1.3 — Pinning. Done.** `_TRUSTED_DRE_ENGINES` is now a mapping from engine
fingerprint to the model-pack hashes that engine is trusted *for*, and a receipt
naming an unpinned pack is refused. `dre_model_pack_hash` was previously checked
for non-emptiness alone, so a trusted engine could have ruled under any pack it
named — including one written for the occasion. Both halves stay empty in
production.

**A1.2 — At least one real mechanism. Done, for signature.**

Ed25519 over `DreReceipt.signing_payload()` — the whole record except the
signature itself. `signing_key_id` is *inside* that payload deliberately: a
signature that did not cover the key id would let a holder keep a valid
signature and relabel which key made it, pointing it at whichever registered key
verifies some other payload.

The backend is an optional dependency, detected rather than assumed, the same
shape as the Arb/FLINT adapter. Absent `cryptography`, receipts declaring
signature authentication are refused for want of a verifier. It is in the `dev`
extra as well as its own, so CI exercises the real path rather than the
absent-backend branch.

The remaining two, and what each is waiting for:

- *Sealed store* — retrieve the decision from an immutable DRE decision store
  addressed by `dre_decision_hash`, checking the store's seal the way
  `core/knowledge.py` checks durable memory's. Reuses machinery that already
  exists and is already tested.
- *Deterministic replay* — re-run the engine at `dre_engine_fingerprint` on
  `dre_input_hash` with `dre_model_pack_hash` and reproduce `dre_proof_hash`.
  Strongest, and the only one that needs no trusted third party — but it needs a
  DRE engine to be connected, which is its own Phase 2 workstream.

**A1.4 — Rotation and revocation. Done.** Procedures in
[`EPISTEMIC_BOUNDARIES.md`](EPISTEMIC_BOUNDARIES.md), backed by code: revocation
is checked before the registered set, so revoking beats registering, and it
applies retroactively to signatures the key already made. Both were ordinary
until the day they became urgent, and a trust root with no revocation path is
one that cannot be withdrawn after a compromise.

**A1.5 — Negative tests. Done for signature.** Edited receipt, wrong key,
unregistered key, revoked key, relabelled `signing_key_id`, missing signature,
missing key id, malformed signature, and absent backend — twenty cases in
`tests/test_receipt_signatures.py`. Replay reproducing a *different*
`dre_proof_hash` waits on the replay backend.
Add them to `tests/helpers_discharge.py::ALL_REGISTRIES` and
`REFUSAL_REASONS`, so they are covered by the existing invariant tests rather
than by a parallel set that can drift. `unpinned_model_pack_registry` is the
worked example.

### How the signature decision was settled

| Mechanism | Needs | Status |
|---|---|---|
| Signature | An asymmetric-signature library, and a public key held outside the repo | **Built.** `cryptography` (Ed25519), key file named by `RHRE_DRE_PUBLIC_KEYS` |
| Sealed store | An immutable DRE decision store to retrieve from | Does not exist yet |
| Deterministic replay | A connected DRE engine to re-run | Does not exist yet |

HMAC was considered and rejected. A shared secret that can verify a receipt can
also mint one, so it establishes content integrity where the requirement is
*issuer* identity — the whole point of the mechanism. That ruled out the
standard library, which has `hmac` and no asymmetric signatures, and made a
dependency unavoidable.

### Acceptance

- No public function accepts an authenticator, a key, or a trust set.
  `tests/test_public_api_attack_matrix.py::test_no_public_function_takes_a_trust_set`
  covers this and its `TRUST_OVERRIDE_PARAMETERS` set should grow to match.
- Every mechanism in `AUTHENTICATED_MECHANISMS` is either implemented or removed
  from the enum. An unimplemented member that verifies by declaration is worse
  than an absent one, because it reads as a checked property.
  `test_every_declared_mechanism_has_a_backend_that_refuses_it` holds that line
  until one is built.
- `activation_status()` keeps reporting a count rather than the fingerprints.

### What must not change — and one thing that quietly does

The invariant test `test_no_public_composition_produces_a_qualifying_discharge`
runs every registry against the production trust set. It keeps passing after
activation, because `helpers_discharge.TEST_ENGINE` is a fake fingerprint that
no real registration would ever include.

That is worth stating plainly, because it means the test's *meaning* changes the
moment a real engine is registered. Today it establishes "nothing discharges".
Afterwards it establishes only "this fake engine does not discharge" — a much
weaker claim wearing the same name. Activation must therefore add a companion
test: with the real engine registered and A1 built, a receipt from that engine
carrying `receipt_authentication` it cannot substantiate is still refused.
Without that companion, the suite would look unchanged while covering strictly
less.

**Done** — `tests/test_receipt_authentication_activation.py` is that companion,
parametrized over all three mechanisms, plus a regression for the empty-hook
inversion that made it nearly the only test exercising the seam.

---

## A2 — Branch-protection enforcement

**Decided: not doing it server-side.** Closed rather than queued.

### The constraint

`GET /repos/.../branches/main/protection` returns **403 — "Upgrade to GitHub
Pro or make this repository public to enable this feature."** The modern
rulesets API (`/rulesets`) returns the same 403, so both routes are gated
identically. This is not a token-scope problem: the token carries `repo`, and
the account holds `admin: true` on the repository. It is plan plus visibility.

That leaves two unlocks, and both were declined:

- **Make the repository public.** Free, and immediate. Rejected: it holds
  unpublished RH research, and publishing it to obtain a merge button trades
  priority on the work for a CI feature. That is the wrong trade at any price.
- **GitHub Pro.** Keeps it private, unlocks both APIs. Declined.

So server-side enforcement is unavailable by decision, not by oversight, and
this section records that so it is not re-proposed as though it were an
oversight.

### What compensates

`tools/githooks/pre-push` refuses a push to `main` when the gate fails.
Install once per clone:

```bash
git config core.hooksPath tools/githooks
```

It runs `phase1-final-gate.py --fast --no-report`: fifteen gates in roughly
fifteen seconds, writing nothing, so the working tree is clean after a
successful push. Speed is the design constraint rather than a nicety — a guard
slow enough to be annoying gets bypassed out of habit, and a habitually
bypassed guard covers nothing.

**It is strictly weaker than branch protection, in four specific ways.** It
runs on the author's machine, so it protects one clone rather than the
repository. `git push --no-verify` skips it. It cannot see what CI sees on the
other three platform/version combinations. And it never fires for a pull
request merged through the GitHub UI, because that merge happens server-side
with no local hook involved.

That last one bounds what this covers: pushing straight to `main`, which is the
path with no review surface at all. A pull request at least puts the check
results on the page next to the merge button.

What it does cover is the realistic failure mode for a single maintainer: not a
determined bypass, but an absent-minded push. CI still runs everything on every
push and pull request, so nothing is unverified — only unblocked.

### If the constraint ever changes

The exact call, with the current check names. A context that never reports is a
required check that blocks forever; a misspelled one never blocks at all.

```bash
gh api -X PUT repos/heidrickla/rh-research-engine/branches/main/protection --input - <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "checks": [
      { "context": "Lint and test (ubuntu-latest, py3.12)" },
      { "context": "Lint and test (ubuntu-latest, py3.13)" },
      { "context": "Lint and test (windows-latest, py3.12)" },
      { "context": "Lint and test (windows-latest, py3.13)" },
      { "context": "Determinism (ubuntu-latest)" },
      { "context": "Determinism (windows-latest)" },
      { "context": "Phase 1 final gate" }
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
```

Two settings are judgement calls, not defaults to accept unread:

- **`enforce_admins: true`** applies the rules to the repository owner as well.
  Honest, and on a single-maintainer repository it means no bypass when a gate
  is wrong at an inconvenient moment. Setting it `false` keeps an escape hatch
  and makes the protection advisory for the one account that matters most.
- **`required_pull_request_reviews: null`** is deliberate. With one maintainer,
  requiring an approving review blocks every merge, because the author cannot
  approve their own pull request.

### A2.1 — Tag protection

Same permission gate, and worth doing in the same pass. The frozen baseline tag
is load-bearing: `tools/phase1-final-gate.py` reads
`rh-lab-v1-baseline-df7016d` to answer "has durable memory moved since the
program started?". If that tag is deleted or repointed, the gate either breaks
loudly or — worse — compares against a different tree and reports no drift.

```bash
gh api -X POST repos/heidrickla/rh-research-engine/tags/protection -f pattern='rh-lab-v1-*'
```

### Acceptance, if it is ever revisited

- A pull request with a failing gate cannot be merged through the UI.
- `GET /repos/.../branches/main/protection` returns the seven contexts above.
- The *Open limitations* entry in `tools/phase1-final-gate.py` is updated in the
  same change, so the closure report stops describing a compensating control
  that is no longer the only one. A stale limitation is as misleading as a
  missing one.

---

## Ordering

A2 is independent of A1 and is now closed. A1 is complete to the limit of what
can be built without the systems it verifies against.

Neither is a prerequisite for connecting a mathematical engine — but A1 **is** a
prerequisite for trusting one. The order that stays safe is: build the
authenticator, then register a fingerprint. Reversing it produces a system that
accepts rulings on the strength of a string in a source file, which is the state
Phase 1 was built to make impossible.

### Next

A1 is done to the limit of what can be built without the systems it verifies
against. What remains genuinely cannot be written yet:

- **Sealed-store authentication** — needs an immutable DRE decision store to
  retrieve from. The seal-checking machinery already exists in
  `core/knowledge.py` and would be reused rather than rewritten.
- **Deterministic-replay authentication** — needs a connected DRE engine. This
  is the strongest of the three, and the only one needing no trusted third
  party, so it is worth building when the engine arrives even though signature
  authentication already works.
- **A2** — closed. Local enforcement is in place; server-side enforcement is
  declined for as long as the repository stays private, which is indefinitely.

Neither missing backend blocks anything: a real issuer can sign today.
