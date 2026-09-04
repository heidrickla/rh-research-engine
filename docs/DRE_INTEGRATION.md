# DRE Integration

## Architecture

```text
DRE (authoritative)
  claim/evidence state
  epistemic status
  proof/rejection graph
  no-go decisions
  replay/hash discipline
        ^
        | YAML evidence envelope + result hash
        |
rh-research-engine (math worker)
  prime generation
  Gamma/binomial filters
  FFT correlation lab
  counterterm discovery
  synthetic zero tests
  synthetic adversarial counterexample tests
  Arb/FLINT verifier envelopes when available
```

The Python worker does **not** decide that RH is proved, that a theorem is rigorous, or that an empirical exponent is established. It emits evidence.

## Independence groups

DRE's evidence-independence semantics are used as mathematical provenance. Re-running one NumPy implementation at 100 values of `X` is one methodological voice, not 100 independent confirmations. A separately implemented Arb/FLINT interval computation can be a second group. The verifier envelope independence group is `math-verifier:<family>:<version>`, distinct from NumPy/mpmath worker families. The adapter fails closed: if rigorous interval verification is unavailable, the envelope status is `unknown` and DRE must not treat it as corroborating proof evidence.

## Numeric boundary

Raw floating point stays outside the DRE decision path. The envelope commits the complete worker result with SHA-256 and optionally sends a selected metric as a scaled integer. This preserves DRE's deterministic/fixed-point posture.

## Workflow

1. Run a worker experiment and store its ordinary JSON result.
2. Export the latest result to a DRE experiment YAML with a claim ID and epistemic class.
3. Copy/symlink `dre/model-packs/riemann-research` into the DRE checkout's `model-packs/` directory.
4. Run `dre validate` on the pack, then `dre run ... --proof` or `--why` on the generated experiment.
5. Archive the DRE decision/proof hashes next to the worker result hash.

The model pack is based on the DRE 0.22.0 model/ontology/rule layout and should be validated against the exact DRE commit used for a research run.
