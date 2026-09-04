# Epistemic Boundaries

This document is the single reference for what the system is allowed to
conclude, and where each boundary is enforced. It exists because an adversarial
review found that the boundaries were documented and unit-tested but not
actually enforced anywhere in the running path: six guard functions had zero
callers outside their own tests, and the only end-to-end path — run an
experiment, export it to DRE — touched none of them.

The rule that follows from that: **a guard that is not on the path is not a
guard.** Everything below names the code that enforces it.

## The one-line invariant

> A deterministic math worker computes. It never decides that what it computed
> is a proof.

## Status vocabularies

There are four, and they are deliberately not interchangeable.

| Vocabulary | Type | Where | Purpose |
|---|---|---|---|
| `ClaimStatus` | `StrEnum` | `core/models.py` | What the research registry believes about a claim |
| `EvidenceClass` | `StrEnum` | `core/models.py` | How strong one experimental result is |
| `KnowledgeStatus` | `StrEnum` | `core/knowledge.py` | What durable memory records about a route |
| `proof_status` | pack symbol | `dre/model-packs/.../ontology.yaml` | What DRE concluded |

Mapping between them:

| `EvidenceClass` | Worker may emit? | DRE rule | Resulting `proof_status` | Promotion |
|---|---|---|---|---|
| `numerical` | yes (default) | RH001 | `not-proved` | ❌ |
| `rigorous-numerical` | yes | RH002 | `not-proved` | ❌ |
| `heuristic` | yes | RH005 | `not-proved` | ❌ |
| `counterexample` | yes | RH006 / RH010 | `not-proved`, candidate `refuted` | ❌ |
| `symbolic` | yes | RH003 | `needs-formal-proof` | ❌ |
| `known` | **no** | — | requires a citation from outside this package | ❌ |
| `proved` | **no** | RH004 | `proved` | ✅ |

`WORKER_FORBIDDEN_CLASSES` in `core/models.py` is the enforcement point:
constructing a `DreEvidenceEnvelope` with `proved` or `known` raises. There is
no CLI flag that sets the class at all — it travels with the
`ExperimentResult` that produced the data.

`KnowledgeStatus` deliberately contains no `proved`, `verified`, or `theorem`
value. Durable memory records what is known *about routes*; it is not a place
where a claim acquires proof status by being written down.

## Promotion boundary matrix

| Source evidence | Maximum status | Enforced by |
|---|---|---|
| numerical interval | `numerical` | `DreEvidenceEnvelope._enforce_epistemic_boundaries` |
| symbolic identity with assumptions | conditional `symbolic` | pack rule RH007; `certificate_predicates` emits `assumption_count` |
| known theorem citation | `known`, only from outside the worker | `WORKER_FORBIDDEN_CLASSES` |
| RH-equivalent premise | never progress | `scoring.is_circular`; pack rule RH020 |
| DRE predicate from numerical evidence | `numerical` | RH001 |
| recovered durable state | recomputed or quarantined | `KnowledgeBase.load` + seal |

## Quantitative bounds

Θ = sup Re(ρ) over nontrivial zeros. Zeta provably has zeros on the critical
line, so **Θ ≥ 1/2 holds unconditionally**. Any computed bound below 1/2 is not
a strong result — it is proof that an input was invalid.

Every exponent map asserts this as a postcondition:

- `core/bounds.py` — `correlation_remainder_to_theta` rejects negative
  exponents rather than clamping them. Clamping silently rewrote noise-driven
  fits, which routinely go negative on finite ranges, into exactly the RH
  endpoint.
- `symbolic/exponents.py` — raises `ImpossibleBoundError`.
- `mathcert/exponents.py` — raises `ImpossibleIntervalError`.
- `symbolic/conjecture.py` — refuses to narrate a target whose implied Θ is
  below 1/2.

Reaching the RH endpoint additionally requires `rigorous=True`: a fitted
log-log slope is a diagnostic, not a proved estimate.

## Assumptions

Assumptions cross every boundary or the boundary is broken.

- `MathCertificate.assumptions` → `certificate_predicates` emits
  `assumption_count`, `assumptions_present`, `unconditional`, and each
  assumption verbatim.
- `simplify_with_trace` attaches the domain conditions a rewrite discarded, so
  cancelling `(x²−1)/(x−1)` to `x+1` records `x - 1 != 0`.
- `equivalent()` reports a domain gap in `assumptions` and marks the method
  `..._conditional`.
- `fingerprint()` covers the domain, so two expressions agreeing only off a
  pole do not collide.
- Pack rule RH007 blocks promotion whenever `assumptions_present` is true.

## Provenance and independence

`independence_group` is `method_family:worker_version`, both taken from the
`ExperimentResult`. There is no flag to relabel them: one numpy run used to
become three "independent" corroborating witnesses simply by passing
`--method-family` three times, and because the old `result_hash` covered the
labels, hash-based dedup could not catch it.

Hashing is now split:

- `payload_hash` — what was computed. Floats are canonicalised to 12
  significant digits, absorbing cross-machine ULP drift. **Dedup on this.**
- `provenance_hash` — who computed it. **Never dedup on this.**

## Certificates

There is no Arb, FLINT, MPFI, or interval backend in this package;
`mathcert` imports only stdlib and pydantic. Every certificate is therefore an
assertion until an adapter exists.

`REGISTERED_ADAPTERS` in `mathcert/verifiers.py` is empty on purpose. An
envelope cannot reach `ACCEPTED` for a family with no registered adapter; the
honest verdict for a disconnected backend is `unknown`. `allowed_families` is a
required argument, because defaulting it to "no restriction" meant the common
call accepted any family name at all.

## Discharge authority

An `ObligationDischargeDecision` is an ordinary artifact: its
`dre_decision_status`, `dre_decision_ref`, `created_by` and `method_family` are
all caller-supplied. On its own it establishes nothing about who authorized a
ruling. Authority comes from a `DreReceipt` binding the decision to a specific
run — `dre_input_hash`, `dre_model_pack_hash`, `dre_engine_fingerprint`,
`dre_decision_hash`, `dre_proof_hash` — verified inside
`contracts/receipts.py` against a **private** trust registry.

That registry maps an engine fingerprint to the model-pack hashes that engine
is trusted *for*. "Which engine ruled" and "under which rules" are two
questions, and only the pair is a trust root: `dre_model_pack_hash` used to be
checked for non-emptiness alone, so a trusted engine could have ruled under any
pack it named — including one written for the occasion.

Verification happens at the point of use. `resolve_discharges` takes raw
decisions and raw receipts and revalidates them; there is no pre-verified value
to construct and no public parameter that widens the trust set. A caller that
could pass its own fingerprint would have authorized itself.

**Discharge authority is currently inert, deliberately.** `activation_status()`
reports `discharge_authority_active: false`, and no combination of public
constructors or functions can produce a qualifying discharge — a test asserts
this across every registry, including the one that succeeds under the scoped
test trust set.

### Activation requirement

Registering a fingerprint is **necessary and not sufficient**. A fingerprint is
just a string in a source file; trusting one on its own would restore exactly
the self-authentication this design removes. Before any engine is trusted, a
receipt must be authenticated by at least one of:

1. **Signature** — a cryptographic signature over the receipt, verified against
   a key held outside this repository.
2. **Sealed-store retrieval** — fetched from a sealed DRE decision store,
   addressed by `dre_decision_hash`, whose seal is checked the way durable
   memory's is.
3. **Deterministic replay** — re-running the engine at
   `dre_engine_fingerprint` on `dre_input_hash` with `dre_model_pack_hash` and
   reproducing `dre_proof_hash`.

`DreReceipt.receipt_authentication` records which of the three applies, and
defaults to `none`, which never verifies. Declaring one of the other three is a
*claim* about how the receipt was established, not the check itself — which is
why it gets a receipt no further than a trusted fingerprint does. Both are
inputs to a verifier Phase 1 deliberately leaves unbuilt. Until it exists, the
empty trust set is what makes discharge authority inert.

One of the three is now built. `signature` verifies a detached Ed25519
signature over the receipt's canonical JSON, against a public key registered by
key id. The other two still refuse for want of a backend, and the trust
registry is empty either way, so nothing discharges in production.

### Signing keys: custody, rotation, revocation

Keys live **outside this repository**, in a JSON file named by the
`RHRE_DRE_PUBLIC_KEYS` environment variable:

```json
{ "dre-signing-key-2026-08": "<64 hex characters of an Ed25519 public key>" }
```

A key committed alongside the artifacts it authenticates proves only that both
were written by the same author, so a test refuses any `.pem`, `.key`, or
`.pub` file in the tree. Unset variable, unreadable file, or unparseable
contents all register nothing: the safe reading of "the trust root cannot be
read" is that nothing is trusted.

The file is re-read on every verification rather than cached at import, so
withdrawing a key takes effect immediately rather than at the next restart.

**Rotation.** Add the new key id alongside the old one, cut the issuer over,
then remove the old id once no unprocessed receipts reference it. Both are
registered during the overlap, and `signing_key_id` travels inside the signed
payload, so a receipt is always checked against the key that actually signed it
rather than against whichever key happens to be current.

**Revocation.** Add the key id to `_REVOKED_SIGNING_KEYS`. Revocation is
checked *before* the registered set, so revoking beats registering and a key
cannot be quietly restored by re-adding it to the file. It also applies
retroactively: signatures that key made before the revocation stop verifying.
That is deliberate — a key is revoked precisely when it may have been in the
wrong hands earlier than anyone noticed, and honouring its earlier signatures
would defeat the purpose.

The remaining two mechanisms are queued as item **A1** in
[`docs/PHASE2_ACTIVATION.md`](PHASE2_ACTIVATION.md), along with the one
non-obvious consequence of activating any of this: the empty-trust invariant
test keeps passing afterwards while establishing something strictly weaker, so
it needs a companion test in the same change.

### Rigorous verifier authority

The same shape, one layer over. A `VerifierEnvelope` is public and freely
constructible, so `status: accepted` in a JSON file is a claim about a
computation nobody in this build performed.

`mathcert/verifiers.py` used to take the adapter registry as a keyword
argument. That was a caller-supplied trust set by another name: a caller
passing its own family name had registered itself, and the check then confirmed
only that the caller agreed with the caller. The registry is now module-private
and resolved from **capability detection** — which backends are actually
importable here — and `envelope_confidence` consults it at the point of use, so
an unbacked `accepted` maps to `UNKNOWN` rather than to `RIGOROUS_NUMERICAL`.

`verifier_activation_status()` reports `rigorous_verification_active: false`.

### What a discharge persists

A conclusion reached *because* an obligation was discharged carries the whole
replay identity into its stored provenance: obligation reference and hash,
evidence references and hashes, the discharged direction, the requirements it
satisfied, decision reference and hash, receipt reference and hash, engine
fingerprint, model-pack hash, input hash, proof hash, and the authentication
mechanism. Held only in the caller's registries, none of that survives the
process — and a reader of the stored graph would see a rigorous bound derived
from a premise that restates RH, with nothing recording why that was allowed.

## The Phase 1 closure gate

```bash
python tools/phase1-final-gate.py
```

Seventeen gates in one command — line endings, lint, the full suite, four drift
gates, durable-memory validation, three AST containment scans, the public API
attack matrix, the empty-trust invariant, determinism, the worker boundary, and
the missing-memory end-to-end run. It writes
`docs/reviews/PHASE1_FINAL_CLOSURE.md` and `docs/reviews/phase1-final-gate.json`,
and CI runs it as the `phase1-final-gate` job.

The claim it establishes is bounded and re-runnable:

> With the production DRE and verifier trust registries empty, no public
> constructor, public function, CLI command, migration path, or closure path
> can produce a qualifying discharge, accepted rigorous authority,
> RH-equivalent frontier advancement, or a worker-declared proof.

A later finding is classified against the invariant it violates, gets a
regression test, and is fixed as a normal defect.

Branch protection would normally require every one of those gates. It is not
configured and will not be: GitHub gates both the protection and rulesets APIs
behind a paid plan for private repositories, and this repository stays private
because it holds unpublished research. Publishing it to obtain a merge button
would trade priority on the work for a CI feature.

`tools/githooks/pre-push` compensates, refusing a push to `main` when the gate
fails — fifteen gates in about fifteen seconds. Install with `git config
core.hooksPath tools/githooks`. It is strictly weaker: it runs on one machine,
`--no-verify` skips it, and it cannot see the other platforms. It covers the
absent-minded push rather than the determined bypass, which is the failure mode
a single maintainer actually has. Recorded as item **A2** in
[`docs/PHASE2_ACTIVATION.md`](PHASE2_ACTIVATION.md).

## Durable memory

`research_state/authoritative/knowledge/math_knowledge.json` is the authoritative
research record. It moved there from `research_state/math_knowledge.json`; the
relocation manifest is `docs/contracts/knowledge-path-migration.json`.

- Trailing bytes are a hard error. The previous recovery path used
  `raw_decode`, which parses the longest valid *prefix* — so a file truncated
  at an item boundary and then closed loaded as complete, silently dropping
  every entry after the cut, including all four `false_route` records.
- A sidecar `math_knowledge.json.sha256` catches truncation that still happens
  to be syntactically valid. Regenerate it deliberately with
  `rhre knowledge seal` after an intentional edit.
- Unknown statuses are quarantined and reported, never loaded.

## Line endings and encoding

Certificate hashes, payload hashes, and formula-index digests are computed over
exact bytes, and DRE computes `ModelHash` over the raw bytes of a pack — so
CRLF and LF copies of an artifact are **different models**.

- `.gitattributes` pins `eol=lf`.
- `tools/check-line-endings.py` verifies the pin held in the *working tree*,
  which `.gitattributes` alone cannot do; CI runs it before anything else.
- `tools/githooks/pre-commit` runs the same check (install with
  `git config core.hooksPath tools/githooks`).
- Every writer passes `newline=""`; every reader passes `encoding="utf-8"`.
  Without the latter, a hand-edited UTF-8 `claims.json` round-trips correctly
  on Linux and is silently mojibaked on Windows.

Measurement note: `grep -c $'\r'` lies under MSYS/Git Bash. `git ls-files --eol`
is the authoritative diagnostic.

## The gate

`core/promote.py` is the single chokepoint. `rhre dre export-latest` calls
`evaluate_export` and writes nothing when the decision is blocked. The gate
checks, in order: evidence class, Θ bounds, assumptions, RH-equivalence,
independent verification, no-go rules against the target claim, and durable
memory integrity.

Adding a new export path means routing it through this function. That is the
whole point.
