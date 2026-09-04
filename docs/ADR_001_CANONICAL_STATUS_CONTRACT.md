# ADR-001: Canonical research status contract

- **Status:** Accepted
- **Date:** 2026-08-22
- **Baseline:** `df7016d` (tag `rh-lab-v1-baseline-df7016d`)
- **Supersedes:** nothing
- **Affects:** every subsystem that records what is known about a statement

## Context

The adversarial review of this repository opened on a structural finding: five
disjoint status vocabularies with no mapping between them. `ClaimStatus`,
`EvidenceClass`, `KnowledgeStatus`, `VerificationStatus`, and the DRE pack's
`proof_status` each answered a different question, none of them declared which,
and code moved values between them by substring matching on their spelling.

That produced concrete defects. `_status_from_knowledge` classified by
`status.value.startswith("exact")` and `"derived" in status.value`, which
promoted 14 of 21 knowledge statuses to *rigorous*, including
`derived_symbolic` — which this repository's own documentation defines as "not
yet independently formalized or literature-checked end to end" — and
`derived_symbolic_needs_external_check`, whose name states the check has not
happened.

The split then recurred in new code within a day: `HypothesisState` landed with
six values while the laboratory plan specified ten, overlapping in three.

A second, subtler failure appeared while fixing the first. Making
`rh_equivalent` force `is_rigorous = False` looked like a safety improvement.
It was not: it lowered *mathematical confidence* as a proxy for *research
usefulness*, so the engine reported less certainty than it actually had about a
classical equivalence.

## Decision

**Lifecycle, epistemic confidence, mathematical role, RH equivalence, research
relevance, actual frontier advancement, extractability, and actionability are
independent dimensions.**

No single enum may encode more than one of them. Each axis answers exactly one
question:

| Axis | Question | Representation |
|---|---|---|
| lifecycle | Where is this in the workflow? | `HypothesisLifecycle` |
| epistemic confidence | How strongly established is it? | `EpistemicStatus` |
| mathematical role | What kind of thing is it? | `MathematicalRole` |
| RH equivalence | Does it restate the target? | `rh_equivalent: bool` |
| research relevance | *Would* proving it matter? | `frontier_relevant: bool` |
| frontier advancement | *Has* it moved the frontier? | `advances_frontier: bool` |
| extractability | Does it carry mathematical content? | `property_extractable: bool` |
| actionability | Can work proceed on it now? | `actionable: bool` |

### Consequences of the separation

**A statement may be simultaneously fully established and worth no progress.**
A classical `A ⟺ RH` is `epistemic_status=KNOWN`, `role=EQUIVALENCE`,
`rh_equivalent=true`, `advances_frontier=false`. It remains usable as a
rewriting rule; it earns nothing. The carve-out is `discharged_obligations`: an
equivalence stops being circular exactly when one direction is genuinely
discharged, and naming the discharged obligation is what separates a proof step
from a restatement.

**`frontier_relevant` and `advances_frontier` are not the same field.** The
first asks whether proving something would matter, the second whether it has
been proved. A research target is relevant and not advancing. Only
`advances_frontier` may gate rigorous closure or progress accounting.

**A meta-rule is not a blocked mathematical route.** "Numerical evidence cannot
promote a theorem to proved" is governance, not a proof attempt that failed.
It gets `role=GOVERNANCE`, `epistemic_status=AUTHORITATIVE_POLICY`, and is
excluded from property extraction. `BLOCKED` keeps its narrow meaning: a
mathematical route examined and unable to proceed.

**Lifecycle stays strictly operational.** The laboratory plan's ten proposed
"states" mix workflow position, evidence quality, classification, and proof
progress. Encoding them as one enum would record the same fact twice — there is
no need for a `PROOF_OBLIGATION_IDENTIFIED` lifecycle value when an unresolved
`ProofObligation` reference already says so. Lifecycle is therefore:

```
PROPOSED  TRIAGED  ACTIVE  BLOCKED  RESOLVED  ARCHIVED
```

and everything else is expressed on the orthogonal axes:

| Plan label | Canonical representation |
|---|---|
| `FALSIFIED` | `RESOLVED` + `role=NO_GO` + false-route status + `actionable=false` |
| `EQUIVALENT_REFORMULATION` | `RESOLVED` + `role=EQUIVALENCE` + `rh_equivalent=true` |
| `EXPERIMENTALLY_SUPPORTED` | `ACTIVE` + numerical/synthetic status + evidence refs |
| `PROOF_OBLIGATION_IDENTIFIED` | `ACTIVE` + open `ProofObligation` refs |
| `RIGOROUSLY_ESTABLISHED` | `RESOLVED` + rigorous status; DRE decides advancement |
| `FORMALLY_VERIFIED` | `RESOLVED` + `epistemic_status=FORMALLY_VERIFIED` |

### Rules

1. **No classification by string shape.** No `startswith`, no `in
   status.value`, no name similarity. Every mapping between vocabularies is an
   explicit table, and a test asserts the table is total — a value added later
   fails the suite until someone classifies it deliberately.
2. **One authoritative definition per axis.** Subsystems import from
   `contracts/`; they do not declare local variants.
3. **Fail closed on the unmapped.** An unrecognised value is not rigorous, not
   frontier-advancing, and not extractable.
4. **Confidence is never reduced as a proxy for usefulness**, nor raised as a
   proxy for importance.

## Alternatives considered

**Extend `HypothesisState` with the plan's ten values.** Rejected: it would
encode four different questions in one field, which is the defect being fixed.

**Keep vocabularies separate and translate ad hoc at each call site.** Rejected:
that is the status quo, and it is what allowed substring matching to spread.

**Collapse `frontier_relevant` into `advances_frontier`.** Rejected: a research
target would then be indistinguishable from a refuted route, and prioritisation
would have nothing to sort on.

## Status of enforcement

At the baseline, `tools/inventory-status.py` reports **14 vocabularies, 98
members, 21 models carrying an axis field, and 0 substring classifiers**. The
zero is load-bearing and is verified by reintroducing a classifier and
confirming the detector fires; a detector that silently finds nothing is worse
than none.

Phase 1 unifies these under `contracts/` with exhaustive legacy mappings.
Until then, the inventory is the record of what must be mapped.
