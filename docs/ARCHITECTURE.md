# Architecture

## Core principle

The harness distinguishes **proposal generation** from **mathematical authority**.

LLMs may propose:

- transformations,
- candidate identities,
- kernels,
- inequalities,
- proof decompositions,
- counterexamples.

Only deterministic components may update a claim to `symbolic` or `proved`.

## Data model

A `Claim` stores:

- statement;
- status;
- assumptions;
- dependencies;
- tags;
- evidence;
- implied upper bound on `Theta = sup Re(rho)` when applicable.

This enables dependency audits and prevents RH-equivalent assumptions from being counted as progress.

## Research loop

1. **Propose** a lemma or transform.
2. **Normalize** it into a machine-readable claim.
3. **Run no-go rules.**
4. **Symbolically test** identities.
5. **Numerically stress-test** against zeta data / synthetic spectra.
6. **Counterexample adversary** attacks the lemma.
7. **Score** by rigor and implied zero-free strip.
8. **Promote** surviving claims to formal proof obligations.

## First optimization objective

If a candidate proves

`S_q(X) << X^theta`

for the localized Gamma filter, record `implied_theta_upper = theta`.

Similarly, if a candidate proves

`corrected_d_k = O(k^-alpha)`

then

`Theta <= 2 - 2 alpha`.

The engine should prefer a rigorous small improvement in `Theta` over a glamorous equivalence to RH.
