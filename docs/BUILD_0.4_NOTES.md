# Build 0.4 Notes — DRE-supervised research

## Architectural change

`rh-research-engine` is no longer intended to be the authoritative epistemic engine. It is a deterministic/numerical math worker that produces evidence for DRE.

## Added

- `src/rh_research_engine/dre/contracts.py`
  - typed evidence classes and claim effects
  - canonical SHA-256 result commitment
  - deterministic method/version independence groups
  - integer scaling for selected floating-point metrics
- `src/rh_research_engine/dre/export.py`
  - DRE experiment YAML renderer
- `rhre dre export-latest`
  - selects the latest result globally or by experiment name
  - exports selected metric as a scaled integer
- `dre/model-packs/riemann-research`
  - DRE 0.22-style ontology/rules pack
  - numerical and rigorous-numerical evidence cannot be promoted to analytic proof
  - counterexamples refute candidate branches
  - numerical evidence for an explicitly RH-equivalent statement is blocked from being called proof progress
- `docs/DRE_INTEGRATION.md`
- bridge tests

## Verified

- 16 Python tests pass.
- End-to-end export of `correlation-lab` / claim `C005` succeeds.
- Generated example: `dre/experiments/c005-latest.yaml`.

## Important invariant

Repeated runs from `python-numpy:0.4.0` share one DRE independence group. Parameter sweeps therefore remain one methodological source. A genuinely independent verifier must use a different method family/version identity.

## Next DRE-native work

1. Validate the bundled model pack using the exact DRE 0.22.x checkout.
2. Add a native research-claim model pack to DRE if type-bound rules need stronger relations than the current run-centric pack.
3. Extend DRE proof nodes or model semantics only if required for mathematical provenance; do not weaken determinism or introduce floating point into the DRE decision path.
4. Add an Arb/FLINT worker and a symbolic worker as independent evidence groups.
