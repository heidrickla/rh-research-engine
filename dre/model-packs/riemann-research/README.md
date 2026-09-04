# Riemann Research DRE Model Pack

This pack is the deterministic epistemic supervisor for the external `rh-research-engine` math workers.

The worker is allowed to calculate, fit, search, and propose. DRE owns the classification of what a result *means*.

Key invariant:

> Numerical evidence, including high-precision or repeatable evidence, is not an analytic proof.

Repeated runs from the same algorithm/version use the same DRE `independence_group`, so parameter sweeps do not become independent corroborating witnesses. An independent Arb/FLINT verification, separate symbolic derivation, or formal proof should use a separate method family.

The generated experiment YAML uses only stable summary fields. Full floating-point metrics remain in the worker artifact and are committed by `result_hash`; DRE receives scaled integers for any selected primary metric.
