# Research objectives

## Objective A — Safe Gamma filter

Study

\[
S_q(X)=q\sum_{n\ge2}(\Lambda(n)-1)(n/X)^q e^{-(n/X)^q}.
\]

Mellin multiplier: `Gamma(1+s/q)`. A zero `rho` contributes `-m_rho Gamma(1+rho/q) X^rho`.

### Score
A rigorous exponent `theta` in `S_q(X)=O(X^theta)` yields `Theta <= theta`.

### Search moves
- exact weighted pair correlation;
- Hardy-Littlewood singular-series counterterm;
- Fourier diagonalization of the sech kernel;
- major/minor arc split;
- large sieve / U^2 estimates;
- optimized q or alternate zero-free Mellin multipliers.

## Objective B — Safe-value binomial sequence

\[
d_k=\sum_{j=0}^k(-1)^j\binom{k}{j}B(2j+2),
\]

\[
\widetilde d_k=d_k+\frac1{2k(k+1)}.
\]

A decay `O(k^-alpha)` implies `Theta <= 2-2 alpha`.

Target endpoint: `alpha = 3/4`.

## Objective C — Screening remainder

Construct the exact filter covariance and subtract the Hardy-Littlewood singular-series model. Search for a proof of a subpower remainder.

## Automatic rejection rules

- boundary unitarity alone;
- generic theta positivity/log-concavity;
- normal convergence of finite Euler products across the strip;
- self-adjoint Hamiltonian => real resonances;
- toroidality alone;
- infinite iterated theta log-concavity.

## Milestones

A proof of any fixed `Theta <= 1-delta` is meaningful progress.

Examples for the binomial exponent:

- `alpha > 1/2` => a nontrivial fixed zero-free strip;
- `alpha = 2/3` => `Theta <= 2/3`;
- `alpha = 3/4` => RH.
