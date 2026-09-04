# Build 0.12 Notes

Everything since the 0.11.0 bump. Three research-loop accelerators, then the
RH Research Laboratory program — Phase 0 and Phase 1 — which reworked the
epistemic boundaries rather than merely preserving them.

The headline for this build is not a feature. It is that the engine now refuses,
mechanically and at the point of use, to manufacture mathematical authority it
does not have.

## Research Supervisor

The supervisor queue stores structured hypotheses in
`research_state/hypotheses.json`.

Each hypothesis records assumptions, proof gaps, cheapest falsification tests,
`frontier_relevant`, `advances_frontier`, and actionable next-step state.

An RH-equivalent restatement is not frontier progress. A hypothesis with
`rh_equivalent=true` can become frontier-relevant only when
`discharged_obligations` explicitly names the proof obligations that were
settled. Attempting to mark an RH-equivalent hypothesis as `advanced` without
those obligations is rejected at model validation time.

```bash
rhre supervisor add --id H001 --statement "R_q = O(X^0.49)" --proof-gap "uniformity in q" --falsification finite-window:1:"run synthetic adversary"
rhre supervisor list
rhre supervisor next --json
```

Supervisor hypotheses are also exposed to the property graph as
`symbolic_derived` or `blocked` records. They are queryable research state, not
proof sources.

## Synthetic Adversarial Counterexamples

`rhre experiment synthetic-adversary` builds zeta-like synthetic zero systems
with conjugation and functional-equation symmetry. The system supports critical
line zeros and configurable off-line zeros, then evaluates candidate criteria
against the known synthetic ground truth.

```bash
rhre experiment synthetic-adversary --eta 0.02 --gamma 14.1347251417
rhre experiment synthetic-adversary --off-line 0.6:25.0 --criterion critical-line-window --tolerance 0.2
```

The emitted `ExperimentResult` uses `evidence_class=heuristic` and
`method_family=python-synthetic-adversary`. Synthetic evidence is useful for
finding false positives and false negatives in proposed criteria, but it remains
non-rigorous and cannot promote mathematical claims.

## Arb/FLINT Verifier Adapter

The verifier layer now includes an Arb/FLINT capability detector and envelope
producer:

```bash
rhre verifier capability
rhre verifier arb-flint-interval --expression "zeta(2)" --lower 1.64 --upper 1.65
```

If `python-flint` is unavailable, the envelope status is `unknown` and records
that no rigorous interval verification was performed. If the module is present,
the adapter reports capability but still refuses to mark arbitrary expressions
as accepted unless a backend-specific enclosure proof has actually been
performed. In particular, mpmath/numpy output is never relabelled as rigorous
interval evidence.

## RH Research Laboratory, Phase 0 and Phase 1

### One vocabulary, three independent axes

Five disjoint status vocabularies were mapped onto one canonical contract layer
in `contracts/`, along three axes that had been conflated:

- **`Confidence`** — how strongly established. Never lowered because a statement
  is useless, never raised because it is important.
- **`Role`** — what kind of thing it is. A governance rule is not a blocked
  proof attempt; it is not a proof attempt at all.
- **frontier value** — `frontier_relevant` versus `advances_frontier`. A
  classical `A <=> RH` is fully established mathematics *and* worth no progress.

Every legacy value maps explicitly. There is no default branch, so a status
added later fails CI until someone classifies it deliberately. That is not
style: classification by spelling (`status.startswith("exact")`) had promoted 14
of 21 knowledge statuses to rigorous, including one whose name says an external
check has not happened.

`RIGOROUS_NUMERICAL` is deliberately not in `RIGOROUS`. A certified enclosure is
rigorous about a *finite* computation, and reading it as rigorous in general is
the step from "checked to height T" to "true".

### Derived verdicts are never stored

`frontier_relevant`, `advances_frontier`, `property_extractable`, `is_rigorous`,
`usable_as_rule` and friends are computed on read and rejected on write — by
constructors, by `model_validate` from stored JSON, and by legacy migrations
alike. A stored copy of a derived fact is a second thing that can disagree with
the first.

### Discharge authority

An RH-equivalence stops being circular exactly when one direction is genuinely
discharged. Establishing that requires a DRE ruling, not a label:

- an `ObligationDischargeDecision` binding the obligation hash, the evidence
  hashes, the direction, and every stated requirement;
- a `DreReceipt` binding that decision to a specific engine, model pack, input,
  and proof hash;
- authentication of the receipt itself — Ed25519 signature verification is
  implemented, sealed-store and deterministic replay are not.

Verification happens at the point of use, against module-private registries that
no public function can widen. All three registries ship empty, so discharge
authority is deliberately inert.

### Fail-closed authoritative state

An absent research record is not an empty one. Durable memory moved to
`research_state/authoritative/knowledge/` under a sidecar seal, and every
authoritative command now exits nonzero, writes no artifact, and mutates no
state when it cannot read what it depends on. The no-go audit previously caught
every exception from route matching and reported "No no-go rule violations
detected" with exit code 0 — the one state in which the dead-end records cannot
be consulted was also the state that reported a clean audit.

### The closure gate

```bash
python tools/phase1-final-gate.py
```

Seventeen gates in one command, writing `docs/reviews/PHASE1_FINAL_CLOSURE.md`.
CI runs it as a required job alongside the Linux/Windows test matrix and
determinism runs on both platforms. The bounded claim it establishes:

> With the production DRE and verifier trust registries empty, no public
> constructor, public function, CLI command, migration path, or closure path
> can produce a qualifying discharge, accepted rigorous authority,
> RH-equivalent frontier advancement, or a worker-declared proof.

Test count over the build: 181 → 684.
