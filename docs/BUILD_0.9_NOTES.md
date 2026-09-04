# Build 0.9 — Research Intelligence Layer

v0.9 completes the remaining symbolic/research-assistance features planned after the v0.8 exact-number boundary.

## Added

- `symbolic/formula_index.py`
  - persistent canonical formula index
  - exact-equivalence lookup by fingerprint
  - conservative structural similarity search
- `symbolic/route_matcher.py`
  - compares new proposals against durable RH memory
  - flags permanent `false_route` entries
  - identifies existing research targets and likely reformulations
- `symbolic/sanity.py`
  - ratio-limit asymptotic checks
  - symbolic power-growth exponent checks
- `symbolic/lean.py`
  - fail-closed Lean export for polynomial identities over `ℚ`
  - emits only mechanically verified `ring` obligations
  - refuses unsupported analytic identities
- `symbolic/certified.py`
  - checks a `MathCertificate` against the intended symbolic expression
  - supports canonical-expression matching when raw text differs
- `mathcert/verifiers.py`
  - independent verifier envelopes for Arb/FLINT/etc.
  - deterministic verifier independence groups
  - validation of accepted certificates before DRE ingestion
- `tests/test_advanced_symbolic.py`
  - regression coverage for the new layer

## Existing v0.7 features now exposed together

The advanced public API also exposes:

- exact decomposition search
- symbolic counterterm bases
- conjecture minimization
- equation/document ingestion
- assumption and proof-gap extraction

## Epistemic rules

1. Structural similarity is a discovery aid, not equivalence.
2. SymPy asymptotic success is symbolic evidence, not an analytic proof.
3. Lean export is intentionally narrow; unsupported mathematics is refused rather than represented as an axiom silently.
4. An accepted external verifier result must name the method family and completed checks.
5. Independent methods receive independent DRE evidence groups; repeated runs of one method family do not.
6. Durable-memory route matches must be reviewed before a proposal can be called novel.

## Intended workflow

```text
paper / LLM proposal / handwritten equation
        |
        v
extract + canonicalize + formula index
        |
        +--> route/no-go matcher
        |
        +--> simplification/decomposition/asymptotic checks
        |
        +--> MathCertificate / independent verifier
        |
        +--> conservative Lean export where supported
        |
        v
DRE evidence + proof/rejection provenance
```

The DRE boundary remains unchanged: DRE is authoritative for epistemic state, while symbolic and numerical workers produce auditable evidence.
