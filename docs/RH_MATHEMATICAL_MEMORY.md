# RH Mathematical Memory

This file is the durable mathematical memory for the RH Research Engine. It is intentionally broader than the currently active proof route. Its purpose is to preserve exact identities, useful equivalences, failed routes, and the current proof frontier so future agents do not need chat history to reconstruct the work.

The machine-readable source is `research_state/authoritative/knowledge/math_knowledge.json`. This document is the human-readable map.

## Epistemic labels

- **exact / exact_algebra / exact_calculus** — identity checked algebraically from the stated definitions.
- **known / known_framework** — established external mathematics; literature citation still belongs in a formal paper.
- **derived_symbolic** — derived in this research but not yet independently formalized or literature-checked end to end.
- **research_target** — a sufficient target, not an established theorem.
- **false_route** — a route explicitly falsified or shown insufficient; it must not be silently revived.
- **conditional_on_RH_standard** — a standard expansion or Hilbert-space statement used under RH.

## Canonical coordinates

Write

\[
\rho=\frac12+\eta+i\gamma,\qquad
\Xi(z)=\xi\left(\frac12+z\right).
\]

Then RH is exactly `eta = 0` for every nontrivial zero. This coordinate is used everywhere in the engine.

A useful squared-energy coordinate is

\[
E_\rho=-(\eta+i\gamma)^2
      =\gamma^2-\eta^2-2i\eta\gamma.
\]

Thus RH becomes the statement that all zero energies are positive real. This connects the problem to Herglotz/Stieltjes functions and positive Hankel moment matrices. An isolated nonreal conjugate energy pair forces a negative 2x2 Hankel determinant, so complex energies are intrinsically detectable once isolated.

## Scattering, Blaschke factors, and passivity

For `omega > 0`,

\[
\Theta_\omega(z)=
\frac{\xi(\frac12-\omega-iz)}
     {\xi(\frac12+\omega-iz)}.
\]

On the real boundary, `|Theta_omega(t)| = 1` unconditionally. This is **not** enough for RH. An off-line zero can live in an all-pass/Blaschke factor without changing the boundary modulus.

A zero `rho = beta+i gamma` gives a scattering pole

\[
z_\rho=-\gamma+i\left(\beta-\frac12-\omega\right).
\]

It becomes an upper-half-plane unstable pole when `beta - 1/2 > omega`.

For `a = eta+i gamma`, the half-plane Blaschke factor

\[
B_a(z)=\frac{z-a}{z+\bar a}
\]

has phase density

\[
-\frac{d}{dt}\arg B_a(it)
=
\frac{2\eta}{\eta^2+(t-\gamma)^2},
\]

with total phase mass `2*pi` per simple right-half-plane zero. Its Green potential is

\[
g_x(a)=\frac12\log
\frac{(x+\eta)^2+\gamma^2}
     {(x-\eta)^2+\gamma^2}.
\]

At `x=1/2`, this is the BSY weight `log|rho/(1-rho)|`.

### Permanent no-go

Boundary unitarity alone cannot prove RH. Any future proposal that depends only on `|Theta|=1` must be rejected unless it adds a genuine interior analyticity/contractivity condition.

## Family inertia

A simple conjugate off-line pair produces a real rank-two residue block of signature `(1,1)`. For `m` distinct simple conjugate pairs, linear independence of their exponential modes gives finite residue signature `(m,m)`.

This is one of the strongest structural results in the project:

> Off-line zeros cannot be hidden by cancellation inside the finite residue sector; they force negative directions in any invariant finite model that sees the pair.

The same horizontal displacement creates exponential mode growth `exp(eta T)` and energy growth `exp(2 eta T)`. The recurring `cosh(eta T)` defect is the symmetric signature of the paired growth/decay modes.

## Xi-gradient disk

For a symmetric off-line pair at `1/2 +- eta + i gamma`, let `x = sigma-1/2`, `u=t-gamma`. Its contribution to

\[
D(\sigma,t)=\partial_\sigma\log|\xi(\sigma+it)|
\]

is

\[
P_\eta(x,u)=
\frac{2x(x^2+u^2-\eta^2)}
{((x-\eta)^2+u^2)((x+\eta)^2+u^2)}.
\]

It is negative exactly when

\[
x^2+u^2<\eta^2.
\]

An off-line zero therefore creates a literal negative-gradient disk centered on the critical line with radius `|eta|`.

## Positive prime Green energy

Let

\[
E(x)=\psi(x)-x.
\]

The core positive Mellin energy is

\[
M(w)=\int_1^\infty E(x)^2x^{-w-1}\,dx.
\]

Its convergence abscissa is

\[
\sigma_c(M)=2\Theta,
\qquad
\Theta=\sup_\rho\Re\rho.
\]

Hence

\[
RH\iff \int_2^X\frac{E(x)^2}{x^2}\,dx=O(\log X).
\]

This is a central positive, noncancelling formulation.

With `b_n=Lambda(n)-1` and `A_N=psi(N)-N`, the discrete energy is

\[
Q_{1,X}
=
\frac{A_X^2}{X}
+
\sum_{N<X}\frac{A_N^2}{N(N+1)}\ge0.
\]

The kernel is

\[
\max(m,n)^{-s}
=(mn)^{-s/2}
 e^{-\frac{s}{2}|\log m-\log n|},
\]

a sampled resolvent of `-D^2+s^2/4`.

## Screening mechanism

The diagonal noise satisfies

\[
\sum_{n\le X}\frac{(\Lambda(n)-1)^2}{n}
=
\frac12\log^2X-\log X+O(1).
\]

Therefore RH-scale total energy requires off-diagonal covariance of leading size

\[
-\frac12\log^2X.
\]

The Hardy--Littlewood singular series has the universal weighted secondary bias

\[
\sum_{h\le H}\left(1-\frac hH\right)
(\mathfrak S(h)-1)
=
-\frac12\log H+O(1).
\]

Inserted into the pair-correlation model, this exactly supplies the leading screening coefficient. The unresolved problem is not the model coefficient; it is the aggregate error between actual prime-pair covariance and that model.

### Current primary proof target

For the localized filter below, define the weighted screening remainder

\[
\mathscr R_q(X)
=
\text{actual weighted pair covariance}
-
\text{Hardy--Littlewood model covariance}.
\]

A rigorous bound

\[
\mathscr R_q(X)=O(X^\theta)
\]

for one fixed `q`, with the deterministic model part controlled, implies

\[
\Theta\le\frac12+\frac\theta2.
\]

Thus

\[
\mathscr R_q(X)=X^{o(1)}
\]

would imply RH. This is the active optimization target. Numerical slopes are diagnostic only.

## Safe Gamma filters

For `q>0`, define

\[
\mathscr L_q(X)
=
\sum_{n\ge2}(\Lambda(n)-1)
 e^{-(n/X)^q}.
\]

Its Mellin multiplier is `Gamma(s/q)/q`, and a zero `rho` contributes

\[
-\frac{m_\rho}{q}\Gamma(\rho/q)X^\rho.
\]

The Gamma factor is zero-free and exponentially damps high ordinates. For fixed `q`, the exact endpoint

\[
\mathscr L_q(X)=O(X^{1/2})
\]

is an RH-equivalent criterion in this logarithmic-derivative smoothing family. The substance is classical Hardy--Littlewood/Grosswald territory; the project uses it as a safe-data spectrometer rather than claiming a novel criterion.

Differentiate to obtain the localized shell

\[
\mathscr S_q(X)
=
q\sum_{n\ge2}
(\Lambda(n)-1)
(n/X)^q e^{-(n/X)^q}.
\]

In log coordinates its kernel is

\[
K_q(u)=q e^{qu}e^{-e^{qu}},
\]

with Fourier transform

\[
\widehat K_q(\omega)=\Gamma(1-i\omega/q).
\]

Hence `q` simultaneously sets prime-space resolution and zero-frequency bandwidth:

\[
\Delta(\log n)\asymp q^{-1},
\qquad
\Delta\gamma\asymp q.
\]

## Exact pair kernel

The global L2 energy of the localized shell can be written exactly as a prime-pair quadratic form with

\[
K_q(m,n)
=
\frac{q\Gamma(2+1/q)}
{2^{2+1/q}\sqrt{mn}}
\operatorname{sech}^{2+1/q}
\left(\frac q2\log\frac mn\right).
\]

For `m,n ~ X`,

\[
K_q(m,n)
\sim
\frac{q}{4X}
\operatorname{sech}^2\left(\frac{q(n-m)}{2X}\right),
\]

so the effective additive shift range is

\[
|n-m|\lesssim X/q.
\]

Adding a regulator `delta` yields a positive susceptibility whose convergence threshold is

\[
\delta_c=2\Theta-1,
\]

independent of `q`. Thus `q` controls spectral resolution while `delta` controls horizontal instability.

## Safe even-zeta binomial spectrometer

Define

\[
c_j
=
-\frac{\zeta'}{\zeta}(2j+2)
-\zeta(2j+2)+1
\]

and

\[
d_k
=
\sum_{j=0}^k(-1)^j\binom{k}{j}c_j.
\]

Then

\[
d_k
=
\sum_{n\ge2}
\frac{\Lambda(n)-1}{n^2}
\left(1-\frac1{n^2}\right)^k.
\]

Nörlund--Rice gives a nontrivial zero contribution

\[
-\frac{m_\rho}{2}
\frac{k!\Gamma(1-\rho/2)}
{\Gamma(k+2-\rho/2)}.
\]

All trivial-zero contributions sum exactly to

\[
-\frac{1}{2k(k+1)}.
\]

Hence define

\[
\widetilde d_k
=d_k+\frac{1}{2k(k+1)}.
\]

The project derived the strong endpoint criterion

\[
RH\stackrel{?}{\iff}
\sup_k k^{3/4}|\widetilde d_k|<\infty.
\]

The question mark is deliberate: the derivation is strong and internally consistent, but this exact endpoint formulation still needs independent literature/formal verification before DRE may mark it `known` or `proved`.

The associated generating function has a single spectral boundary point `z=1` with local exponents `rho/2`; after `tau=-log(1-z)`, RH corresponds to purely oscillatory dilation modes.

## Safe positive moments

At the safe regulator, define

\[
\mu_n
=
\int_1^\infty
\frac{(\log x)^n(\psi(x)-x)^2}{x^3}\,dx.
\]

Every `mu_n` is unconditionally finite. Their factorial root growth detects the rightmost zero:

\[
\limsup_{n\to\infty}
\left(\frac{\mu_n}{n!}\right)^{1/n}
=
\frac{1}{2(1-\Theta)}.
\]

Thus RH is equivalent to this limsup being `1`. This transforms the boundary problem into a uniform factorial-growth problem for positive safe-region moments.

Unit log-window energies

\[
W_k
=
\int_k^{k+1}
 e^{-T}(\psi(e^T)-e^T)^2\,dT
\]

satisfy exponential growth rate

\[
\limsup\frac{\log W_k}{k}=2\Theta-1.
\]

So RH is equivalent to `W_k = exp(o(k))`.

## Hilbert--Pólya reinterpretation

Under RH,

\[
h(T)=e^{-T/2}(\psi(e^T)-e^T)
\]

is a `B^2` almost-periodic state with frequencies `gamma`. Translation is already a unitary group, generated by

\[
P=-i\frac{d}{dT}.
\]

The conceptual shift is:

> We do not necessarily need to discover the self-adjoint operator. The translation generator already exists. RH is the statement that the prime-defined state belongs to its mean-square unitary Hilbert space; an off-line zero produces an exponentially growing nonnormalizable mode.

This is interpretation, not a proof shortcut.

## Theta / Lee--Yang route

The normalized Riemann theta kernel gives a symmetric probability law whose MGF is

\[
\mathbb E[e^{zX}]
=
\frac{\xi(1/2+z)}{\xi(1/2)}.
\]

RH is exactly the Lee--Yang property for this MGF. This is useful, but generic positivity of the theta density is not sufficient.

### Permanent no-go: iterated log-concavity

The proposed infinite curvature hierarchy

\[
q_{r+1}=2q_r-(\log q_r)''
\]

fails at higher level for the actual Riemann kernel. Do not revive this as an all-level positivity proof route.

## SUSY / quantum Hamiltonian route

Theta completion gives an explicit positive SUSY Schrödinger Hamiltonian

\[
H_+=-D^2+\frac14+V_R=A^\dagger A.
\]

But `Xi` appears as an interaction form factor, not as the spectral determinant/Jost function of that Hamiltonian. Positive/self-adjoint/SUSY structure alone therefore does not imply RH.

### Permanent no-go

A future argument of the form

`positive Hamiltonian => all Xi zeros critical`

must be rejected unless it supplies an additional theorem that actually identifies Xi zeros with a self-adjoint spectrum.

## Automorphic / modular route

The modular cusp naturally gives the free operator

\[
-D_x^2+\frac14,
\]

and modular scattering coefficient

\[
\varphi(s)=\frac{\Lambda(2s-1)}{\Lambda(2s)}.
\]

Zeta zeros occur as scattering resonances `s_rho=rho/2`. Off-line zeros split the paired resonance widths from `1/4` to `1/4 +- eta/2`.

A two-mode cusp coherence calculation gives

\[
\mathrm{coh}^2=1-4\eta^2.
\]

This is a clean defect invariant, but self-adjoint scattering can have complex resonances, so it is not sufficient by itself.

For Hecke action on an off-line functional-equation pair, an invariant nondegenerate Hermitian form for nonreal conjugate eigenvalues necessarily has signature `(1,1)`. A positive invariant completion would force reality/temperedness and hence `eta=0`; constructing that positive completion is the missing unitarizability problem.

Toroidality alone selects zeros but does not constrain their real parts. This is a permanent no-go.

## Current research frontier

The project should prioritize, in this order:

1. **Weighted prime-pair screening remainder** `R_q(X)` and a rigorous exponent improvement.
2. **Independent validation** of the exact safe-binomial endpoint criterion.
3. **Certified Arb/FLINT workers** for numerical evidence envelopes.
4. **Symbolic/formal proof workers** for identities currently labeled `derived_symbolic`.
5. Kernel optimization only when it improves provability of the screening remainder, not merely numerical aesthetics.

The most valuable intermediate result is not necessarily RH. Any rigorous bound

\[
\mathscr R_q(X)=O(X^\theta),\qquad \theta<1,
\]

that improves the implied global zero-edge bound would be mathematically meaningful.

## Governance / anti-context-loss rule

Future agents must read `research_state/authoritative/knowledge/math_knowledge.json` before proposing a new RH route. They must:

1. check whether the route is already present;
2. check `false_route` entries and DRE no-go rules;
3. preserve the item's epistemic status;
4. add genuinely new identities or counterexamples back to the knowledge file;
5. never promote numerical agreement or an RH-equivalent reformulation to proof.

This repository, not chat context, is the authoritative memory of the research program.
