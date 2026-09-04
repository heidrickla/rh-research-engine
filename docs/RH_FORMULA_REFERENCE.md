# RH Formula Reference

Cited from the literature, **not established here**. A citation confers no
status (`symbolic/citations.py`); nothing in this file may be promoted on the
strength of appearing in it.

Columns use this project's canonical axes: `Role`
(`contracts/roles.py`), `Confidence` (`contracts/epistemic.py`). Cite entries by
ID.

**Reading the RH-eq table:** those entries *restate* RH. Per
`contracts/frontier.py` they are `rh_equivalent=true` → `frontier_relevant=false`.
Proving one is proving RH; re-deriving one earns nothing. They are useful as
rewriting rules, not as progress.

Notation: `Θ = sup Re(ρ)` over nontrivial zeros; `σ(n)` sum of divisors; `H_n`
nth harmonic number; `γ` Euler–Mascheroni; `M(x) = Σ_{n≤x} μ(n)`;
`ψ(x) = Σ_{n≤x} Λ(n)`.

## Definitions and unconditional facts

| ID | Statement | Role | Confidence | Source |
|---|---|---|---|---|
| RHF-001 | `ζ(s) = Π_p (1 − p^{−s})^{−1}`, `Re s > 1` | identity | known | [DLMF §25.2](https://dlmf.nist.gov/25.2) |
| RHF-002 | `ζ(s) = 2(2π)^{s−1} sin(πs/2) Γ(1−s) ζ(1−s)` | identity | known | [DLMF 25.4.2](https://dlmf.nist.gov/25.4) |
| RHF-003 | `ξ(s) = ½ s(s−1) π^{−s/2} Γ(s/2) ζ(s)` | definition | known | [DLMF 25.4.4](https://dlmf.nist.gov/25.4) |
| RHF-004 | `ξ(s) = ξ(1−s)` | identity | known | [DLMF 25.4.3](https://dlmf.nist.gov/25.4) |
| RHF-005 | `N(T) = (T/2π)log(T/2π) − T/2π + O(log T)` | bound | known | [Riemann–von Mangoldt](https://en.wikipedia.org/wiki/Riemann%E2%80%93von_Mangoldt_formula) |
| RHF-006 | `Θ ≥ 1/2` — infinitely many zeros lie on `Re s = 1/2` (Hardy 1914) | bound | known | [Riemann hypothesis](https://en.wikipedia.org/wiki/Riemann_hypothesis) |
| RHF-007 | `Λ ≥ 0` (de Bruijn–Newman constant), Rodgers–Tao 2018 | bound | known | [arXiv:1801.05914](https://arxiv.org/abs/1801.05914) |
| RHF-008 | Redheffer: `det R_n = M(n)`, where `R_n[i][j] = 1` iff `j = 1` or `i` divides `j` | identity | known | [Redheffer matrix](https://en.wikipedia.org/wiki/Redheffer_matrix) |
| RHF-009 | Farey count: `#F_n = Σ_{k≤n} φ(k)`, over the reduced fractions in `(0,1]` of order `n` | identity | known | [Farey sequence](https://en.wikipedia.org/wiki/Farey_sequence) |

RHF-006 is enforced in code as `core.bounds.THETA_FLOOR`; a derived `Θ < 1/2`
is rejected, not clamped (`properties/closure.py::theta_is_possible`).

## RH-equivalent (`rh_equivalent=true`, no frontier credit)

| ID | Statement | Source |
|---|---|---|
| RHF-101 | `Θ = 1/2` | [Riemann hypothesis](https://en.wikipedia.org/wiki/Riemann_hypothesis) |
| RHF-102 | Robin: `σ(n) < e^γ n log log n` for all `n > 5040` | [arXiv:math/0604314](https://arxiv.org/abs/math/0604314) |
| RHF-103 | Lagarias: `σ(n) ≤ H_n + e^{H_n} log H_n` for all `n ≥ 1`, equality only at `n = 1` | [arXiv:math/0008177](https://arxiv.org/abs/math/0008177) |
| RHF-104 | Li: `λ_n ≥ 0` for all `n ≥ 1`, where `λ_n = (1/(n−1)!) d^n/ds^n [s^{n−1} log ξ(s)]` at `s = 1` | [Li's criterion](https://en.wikipedia.org/wiki/Li%27s_criterion) |
| RHF-105 | Nyman–Beurling: `χ_(0,1]` lies in the `L²(0,∞)` closure of `span{ρ_a : a ≥ 1}`, `ρ_a(x) = {1/(ax)}` | [arXiv:math/0202141](https://arxiv.org/abs/math/0202141) |
| RHF-106 | Báez-Duarte: RHF-105 holds with `a` restricted to `ℕ` — a strictly smaller subspace | [arXiv:math/0202141](https://arxiv.org/abs/math/0202141) |
| RHF-107 | Mertens: `M(x) = O(x^{1/2+ε})` for every `ε > 0` | [Riemann hypothesis](https://en.wikipedia.org/wiki/Riemann_hypothesis) |
| RHF-108 | von Koch: `π(x) = li(x) + O(√x log x)` | [Riemann hypothesis](https://en.wikipedia.org/wiki/Riemann_hypothesis) |
| RHF-109 | `Λ ≤ 0`; with RHF-007 this makes RH `⟺ Λ = 0` | [arXiv:1801.05914](https://arxiv.org/abs/1801.05914) |
| RHF-110 | Franel–Landau: `Σ_i |F_i − i/\|F_n\|| = O(n^{1/2+ε})`, over the Farey fractions of order `n` | [Farey sequence](https://en.wikipedia.org/wiki/Farey_sequence#Riemann_hypothesis) |

RHF-107 is the trap worth naming: `M(x) = O(x^{1/2})` — no `ε` — is the Mertens
conjecture, **disproved** (Odlyzko–te Riele 1985). The `ε` is load-bearing.

## Proved in the function field case (`rh_equivalent=false`, no credit either way)

Not indexed, and the reason is the same one that keeps Nyman–Beurling's closure
form out of the corpus: these are statements about a different object, and a
formula the corpus cannot define is a formula nothing can check. They live in
`symbolic/finite_field_zeta.py`, where every term is an integer and every
claim is settled by exact arithmetic.

| ID | Statement | Source |
|---|---|---|
| RHF-301 | For a smooth projective curve `C/F_q` of genus `g`: `Z_C(T) = P(T)/((1−T)(1−qT))`, `P ∈ ℤ[T]` of degree `2g` | [Weil conjectures](https://en.wikipedia.org/wiki/Weil_conjectures) |
| RHF-302 | Weil: every reciprocal root of `P` has `\|α\| = √q` — equivalently every zero of `Z_C` has `Re s = 1/2` | [Weil conjectures](https://en.wikipedia.org/wiki/Weil_conjectures) |
| RHF-303 | Hasse: `\|a_q\| ≤ 2√q` for an elliptic curve, `a_q = q + 1 − #E(F_q)` — RHF-302 at `g = 1`, and an **integer** inequality `a_q² ≤ 4q` | [Hasse's theorem](https://en.wikipedia.org/wiki/Hasse%27s_theorem_on_elliptic_curves) |
| RHF-304 | Functional equation: `Z_C(1/(qT)) = q^{1−g} T^{2−2g} Z_C(T)`; for `g = 1`, `Z_C(1/(qT)) = Z_C(T)` | [Weil conjectures](https://en.wikipedia.org/wiki/Weil_conjectures) |

**RHF-302 is a theorem and says nothing whatever about ζ.** It was proved by
intersection theory on `C × C`, a surface with no counterpart over `ℚ`, and the
analogy has stood complete on one side and open on the other since 1948. What
it is good for here is instrumentation: a checker that cannot confirm the
critical line where it provably holds is broken, and until `rhre symbolic
weil-control` there was nothing in this engine that could have said so.

## Consequences of RH (one direction only)

RH implies these. The converse is open or false, so deriving one of these does
**not** yield RH.

| ID | Statement | Note | Source |
|---|---|---|---|
| RHF-201 | Lindelöf: `ζ(1/2 + it) = O(t^ε)` for every `ε > 0` | RH ⟹ Lindelöf; converse open | [Riemann hypothesis](https://en.wikipedia.org/wiki/Riemann_hypothesis) |
| RHF-202 | Schoenfeld: `\|ψ(x) − x\| < (1/8π) √x log²x` for `x ≥ 73.2` | explicit, conditional | [Riemann hypothesis](https://en.wikipedia.org/wiki/Riemann_hypothesis) |
| RHF-203 | Schoenfeld: `\|π(x) − li(x)\| < (1/8π) √x log x` for `x ≥ 2657` | explicit form of RHF-108 | [Riemann hypothesis](https://en.wikipedia.org/wiki/Riemann_hypothesis) |

## Recorded dead ends

Durable memory holds four `false_route` records. Check a proposed route against
them before treating it as new — `rhre symbolic match-route "<statement>"`
matches by content, so rewording does not evade them.

| ID | Route |
|---|---|
| K008 | Boundary unitarity no-go |
| K032 | Iterated theta log-concavity fails |
| K034 | Positive Hamiltonian no-go |
| K038 | Toroidality alone no-go |

## Using this file

- An RH-eq entry may be applied as a rewriting rule (`usable_as_rule`), never as
  progress. Lifting one out of `rh_equivalent` requires a DRE-accepted discharge
  of a named proof obligation — see `docs/EPISTEMIC_BOUNDARIES.md`.
- Nothing here is machine-readable yet. To index it, populate the `formulas`
  field of the matching durable-memory records and run
  `rhre symbolic index-knowledge`; durable memory currently declares none.
  Editing durable memory is a sealed operation (`rhre knowledge seal`) and moves
  the frozen-baseline manifest, so it is deliberate work, not a side effect.
- Statements are transcribed from the cited sources and have not been
  independently verified here. Treat a transcription error as possible: check
  the source before relying on an exact constant or threshold.
