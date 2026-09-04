# Next build sequence

The phase numbering here is the *build sequence*, and is unrelated to the RH
Research Laboratory program's Phase 0/1/2 — that one is tracked in
[`PHASE2_ACTIVATION.md`](PHASE2_ACTIVATION.md). Two different things called
"Phase 2" is a nuisance, so the distinction is stated rather than inferred.

## Phase 1 — deterministic research kernel (complete)

- claim registry
- assumption/dependency graph
- no-go rules
- quantitative scoring
- safe-binomial experiment
- Gamma-filter experiment
- exponent scans

## Phase 2 — proof-search primitives

Implement exact symbolic operators for:

- Mellin transform lookup / validation
- Nörlund-Rice templates
- partial summation
- finite difference transforms
- singular-series model subtraction
- exponent propagation (`bound -> Theta`)

## Phase 3 — counterexample harness

Synthetic families:

- off-line zero quartets
- de Bruijn-Newman deformations
- finite Euler products
- generic all-pass scattering factors
- mock positive theta kernels

A candidate claim must survive all applicable synthetic families.

## Phase 4 — correlation laboratory

Add scalable computation of

`R_q(X) = actual weighted prime-pair covariance - Hardy-Littlewood model`

for logarithmically spaced X and multiple q.

Store fitted exponents with confidence intervals, but never promote them above numerical status.

## Phase 5 — LLM relay

Use external LLMs only as:

- Explorer
- Skeptic
- Literature reviewer
- Proof decomposer

All outputs enter as untrusted claim proposals.

## Phase 6 — formalization

Move surviving finite identities and inequality reductions into Lean. Analytic number-theory theorems may initially be represented as explicitly named axioms, then replaced as library support permits.
