# Correlation Laboratory

## Objective

The current narrow target is the weighted screening remainder between the actual centered von-Mangoldt pair covariance and its Hardy--Littlewood singular-series model.

For

\[
g_q(u)=q u^q e^{-u^q},\qquad b_n=\Lambda(n)-1,
\]

define the localized wave

\[
W_q(\log X)=X^{-1/2}\sum_n b_n g_q(n/X).
\]

Then

\[
|W_q|^2
=
X^{-1}\sum_n b_n^2g_q(n/X)^2
+
\frac{2}{X}\sum_{h\ge1}\sum_n b_nb_{n+h}g_q(n/X)g_q((n+h)/X).
\]

The Hardy--Littlewood model replaces

\[
b_nb_{n+h}\rightsquigarrow \mathfrak S(h)-1.
\]

The lab records the residual after this model subtraction.

## Exact integrated kernel

For the globally integrated susceptibility the pair kernel is

\[
K_{q,\delta}(m,n)
=C_{q,\delta}(mn)^{-(1+\delta)/2}
\operatorname{sech}^{2+(1+\delta)/q}
\left(\frac q2\log\frac mn\right),
\]

with

\[
C_{q,\delta}
=\frac{q\,\Gamma(2+(1+\delta)/q)}{2^{2+(1+\delta)/q}}.
\]

At large q the prime-pair range near scale X is \(|m-n|\lesssim X/q\).

## Research acceptance condition

A numerical observation is not a proof. A candidate theorem must provide a uniform analytic estimate for the remainder. The high-value target is

\[
\mathscr R_q(X)=X^{o(1)}
\]

for one fixed q; a stronger natural target is \(O_q(1)\).

Every proposed estimate must be audited against the no-go database and translated into its implied rigorous zero-free strip before receiving a progress score.

## Exponent bookkeeping

If a theorem establishes, for one fixed q,

\[
\mathscr R_q(X)=O(X^\theta)
\]

and the deterministic HL model contribution is bounded, the exact zero response implies

\[
\Theta\le \frac12+\frac\theta2.
\]

The CLI can translate a **proved** exponent:

```bash
rhre score-correlation-bound --theta 0.2
# Theta <= 0.6
```

This is deliberately separate from numerical exponent fitting. Numerical scans are never automatically converted into claims.
