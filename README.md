# RH Research Engine

A deterministic research harness for exploring Riemann Hypothesis proof routes without losing track of assumptions, known equivalences, counterexamples, or quantitative progress.

## Design goals

- **Deterministic core:** claims, dependencies, scores, experiment results, and no-go rules are machine-checkable records.
- **Quantitative progress:** candidate theorems are scored by the zero-free strip they would imply, not by prose plausibility.
- **Automatic skepticism:** known dead ends are encoded as rejection rules.
- **Safe-data experiments:** the first research objective uses only Euler-product-safe values and absolutely convergent prime sums.
- **LLM optionality:** language models may propose or critique ideas, but cannot mark claims as proved.

## Initial RH objective

The first built-in route is the centered von-Mangoldt / Gamma-filter family

\[
\mathscr L_q(X)=\sum_{n\ge2}(\Lambda(n)-1)e^{-(n/X)^q}
\]

and its localized derivative

\[
\mathscr S_q(X)=q\sum_{n\ge2}(\Lambda(n)-1)(n/X)^q e^{-(n/X)^q}.
\]

A zeta zero \(\rho\) contributes \(-m_\rho\Gamma(1+\rho/q)X^\rho\). Therefore a rigorous bound

\[
\mathscr S_q(X)\ll X^\theta
\]

implies a zero-free region \(\Re\rho\le\theta\). The RH endpoint is \(\theta=1/2\).

The second built-in route is the finite safe-value binomial sequence

\[
d_k=\sum_{j=0}^k(-1)^j\binom{k}{j}\left[-\frac{\zeta'}{\zeta}(2j+2)-\zeta(2j+2)+1\right]
\]

with the trivial-zero correction

\[
\widetilde d_k=d_k+\frac1{2k(k+1)}.
\]

Its decay exponent maps directly to an implied bound on \(\Theta=\sup\Re\rho\).

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .[dev]
rhre status
rhre experiment safe-binomial --k-max 80
rhre experiment gamma-filter --x 1000 --q 2
pytest
```

## Status vocabulary

`ClaimStatus` (the research registry):

- `hypothesis` — proposed statement.
- `numerical` — numerically supported only.
- `symbolic` — exact algebra checked, analytic proof incomplete.
- `proved` — proof supplied within the harness dependencies.
- `known` — externally established theorem.
- `equivalent_rh` — restates RH; never counts as progress.
- `false` — counterexample found.

This is one of four distinct vocabularies. `EvidenceClass` describes a single
experimental result, `KnowledgeStatus` describes a durable-memory entry, and
DRE's `proof_status` is what the engine concluded. **`docs/EPISTEMIC_BOUNDARIES.md`
maps between them and names the code that enforces each boundary** — read it
before adding an export path, a status, or a bound.

Two invariants worth stating up front:

- A math worker may never emit `proved` or `known`. `rhre dre export-latest`
  has no `--class` flag; the evidence class travels with the experiment record.
- Θ ≥ 1/2 unconditionally, so any computed bound below 1/2 means an input was
  invalid. Every exponent map raises rather than returning one.

See `docs/ARCHITECTURE.md` and `docs/RESEARCH_OBJECTIVES.md`.

## Correlation laboratory

The second build adds an FFT-based weighted prime-pair laboratory around the localized shell

\[
W_q(\log X)=X^{-1/2}\sum_n(\Lambda(n)-1)g_q(n/X),\qquad
g_q(u)=q u^q e^{-u^q}.
\]

It decomposes the observed shell energy into:

- diagonal prime shot noise,
- actual weighted off-diagonal covariance,
- Hardy--Littlewood singular-series model covariance,
- the residual screening remainder.

Run a single scale:

```bash
rhre experiment correlation-lab --X 20000 --q 4
```

Run a geometric scan and estimate diagnostic log-log slopes:

```bash
rhre experiment correlation-scan --x-min 2000 --x-max 50000 --points 7 --q 4
```

The reported slopes are **experimental diagnostics only**. The harness deliberately does not promote a fitted exponent into a mathematical claim.

## v0.3 active-search experiments

```bash
rhre experiment counterterm-discovery --x-min 2000 --x-max 40000 --points 8 --q 4
rhre experiment local-variance --x 5000 --q 4 --width 1 --samples 33
rhre experiment arc-diagnostics --x 8000 --q 4 --fft-size 65536 --major-width 0.02
rhre experiment synthetic-zero --eta 0.02 --gamma 14.1347251417 --q 20 --t-max 120
```

See `docs/BUILD_0.3_NOTES.md` and `docs/ACTIVE_SEARCH.md`.


## The zeros, and the corpus's statements about them

`mpmath.zetazero` costs about 160 ms per zero, and the cost is Python overhead
rather than precision — asking for five digits instead of fifteen changes
nothing. So anything needing many zeros could not have them.
`symbolic/riemann_siegel.py` evaluates the Hardy function `Z` for a whole array
at once and brackets the roots between Gram points, which is about 0.1 ms per
zero. Every count is checked against `ZeroCount` — Turing's method, which
locates nothing — and a short list raises rather than being returned.

```bash
rhre symbolic verify-line --height 100000
rhre symbolic pair-correlation --height 50000
rhre symbolic level-spacing --height 30000
rhre symbolic spacing-bands
rhre symbolic spacing-decay --zeros zeros6
rhre symbolic explicit-formula --zeros 20000
```

**Whether the zeros are on the line.** `symbolic/argument_principle.py` counts
the zeros in the critical *strip* as `N(T) = theta(T)/pi + 1 + S(T)`, tracking
`arg zeta` along a path that never touches the critical line — so it is
independent of anything computed from `Z`. Compared against the sign changes of
`Z`, agreement below a height means every zero below it is on the line and
simple:

```
T = 100000     138069 /  138069      19.6 s
T = 300000     466659 /  466659     251.2 s
T = 1000000   1747146 / 1747146    1448.5 s
```

This is **evidence, not a proof**, and the records refuse a rigorous confidence
at construction. It says nothing about zeros above the height; `arg` is
reconstructed from samples; and float64 `zeta` carries no error bound at all.

**The same question, certified.** `rhre symbolic certify-line` answers it
through Arb's interval arithmetic, and is filed `rigorous_numerical` — the
first thing in this repository that can be. The two halves of the question cost
wildly different amounts, and saying so is the point:

```
rhre symbolic certify-count --height 100000000    248008025 zeros    0.01 s
rhre symbolic certify-line  --height 10000            10142 zeros      52 s
```

*Counting* the zeros in the strip is free at any height, and it now confirms
every figure in the floating-point table above, including the 1747146 that took
24 minutes to reach. *Placing* them on the line costs milliseconds each, so the
certified verification reaches `T = 10^4` where the floating-point one reaches
`10^6` — two orders of magnitude less reach, for a different kind of claim.

It needs `python-flint`, which is optional and **detected rather than
assumed**: without it `certify-line` refuses rather than falling back, because
a verification that quietly reverts to floating point is one whose confidence
field lies. `rigorous_numerical` is deliberately not in
`contracts.epistemic.RIGOROUS` — it is rigorous about a *finite* computation,
which is not a statement about every zero. And none of it is new mathematics:
zeros have been certified to heights vastly beyond this. What is new here is
that this engine's own verification can be filed at a rigorous confidence
instead of being refused one.

**RH from outside analysis.** Every other criterion in the corpus is
analysis. Two are not, and they were added because the corpus was all one
shape:

```
det R_n = M(x)                          a determinant of a 0/1 matrix
sum_i |F_i - i/|F_n|| = O(n^{1/2+eps})  how evenly the Farey fractions spread
```

`R_n` holds a 1 wherever the column is 1 or the row divides the column, and
nothing else — so RH becomes a statement about how fast the determinant of a
matrix of pure divisibility can grow. `\RedhefferDet` computes it *as a
determinant*, by exact fraction-free elimination, and is not read off `M(n)`:
the whole content of the identity is that a linear-algebra route and a
number-theoretic one agree, and taking the shortcut would check nothing. The
Franel–Landau criterion is combinatorial in the same way — reduced fractions
with a bounded denominator, in order, and RH is whether they sit evenly.

Adding them turned up two defects. `\phi(n)` had been parsing to
`Function('phi')(n)`, a stub that evaluates to nothing, so anything written
with the totient could not have been caught being wrong. And a new function
registered in one of the two name tables and not the other came back from its
own printed form as a stub — `test_every_engine_function_resolves_under_both_policies`
now catches that when the function is added rather than when a formula happens
to use it.

**Whether the corpus records its own formulas correctly.** Two of the indexed
formulas are statements about the zeros, and neither could be evaluated before.
Both are checked in the formula guard, so the tally reads *41 of 41
formulas checked against values*:

- Montgomery's pair correlation, `1 - (sin(pi u)/(pi u))^2` — the index's first
  formula. Mean deviation 0.0165 over 35473 zeros below `T = 30000`, improving
  with height.
- Von Mangoldt's explicit formula, rebuilding `psi(x)` from the zeros —
  residuals below 0.004 at most sampled points over 20000 zeros.

Neither is a test *of* the mathematics. Both are theorems or well-studied
conjectures; the check is whether the corpus **records** them correctly, since
a missing constant parses, indexes, fingerprints and exports exactly as well as
the true statement.


**How the gaps are distributed.** Pair correlation averages over *all* pairs at
a separation. `rhre symbolic level-spacing` measures *consecutive* ones, which
is where level repulsion lives — independent points land arbitrarily close
together and eigenvalues of a random Hermitian matrix do not:

```bash
rhre symbolic level-spacing --height 30000
```

```
P(s < 0.1) = 0.00049   against 0.09516 for independent points   193x rarer
mean deviation from the Wigner surmise: 0.0281   (T = 10^4)
                                        0.0210   (T = 8x10^4)
```

The repulsion figure is the robust half and the reason it is reported
separately: agreement with a fitted curve can survive a wrong normalisation,
and "almost no spacing is small, where chance would make a tenth of them
small" cannot. Deliberately checked — mutating the unfolding to the plausible
wrong one fails the mean-spacing test and leaves the repulsion test passing.

**What the residual is made of.** The curve compared against is the *Wigner
surmise*, which is not the exact GUE law — that is a Fredholm determinant of
the sine kernel, with no elementary closed form. This was first recorded as an
unresolvable ambiguity: part of the residual is the curve, part is a
finite-height correction, and nothing here could separate them. Both halves of
that turned out to be wrong.

`exact_gue_density` evaluates the determinant by Gauss–Legendre Nyström, which
converges super-exponentially for an analytic kernel — five nodes give seven
digits, fifteen give sixteen. With the exact law available the residual splits:

```
the CURVE   mean |exact − surmise|                    0.0018
NOISE       what 59232 spacings would show anyway     0.0045
the ZEROS   everything left                          ~0.014
```

So the surmise accounts for about a twelfth of the floor, not the bulk of it —
the earlier guess was wrong in the direction that made the data look better
explained than it was. Measuring against the exact law in fact gives a slightly
*larger* deviation than against the surmise (0.0223 against 0.0210 at
T = 8×10⁴), because at these heights the finite-height correction happens to
push the zeros the way the surmise is already wrong.

**Is what is left a shape, or the histogram?** Subtracting a noise floor from a
mean absolute deviation assumes something about how the two combine, and a
wrong assumption there manufactures a finding. `rhre symbolic spacing-bands`
asks a question needing no such assumption — noise is independent between
disjoint samples, so a signed residual with the same shape in bands sharing no
zeros is not the noise:

```
(0, 2×10⁴]  (2×10⁴, 5×10⁴]  (5×10⁴, 9×10⁴]     pairwise r = 0.96, 0.96, 0.98
null: 200 samples drawn from the exact law     mean |r| 0.24, never above 0.67
```

The first version of that test used *nested* ranges — the zeros below 30000
against those below 80000, one a third of the other — and got r = 0.98 that was
partly guaranteed by construction. A test for that is kept, asserting the
manufactured correlation, so the reason the bands are disjoint stays visible.

**Checked against zeros this engine did not compute.** All of the above still
leaves two readings open: that the shape is an artefact of this zero-finder,
and that it is permanent rather than a finite-height correction.
`rhre symbolic odlyzko-check` settles both against
[Odlyzko's published tables](https://www-users.cse.umn.edu/~odlyzko/zeta_tables/):

```
our ordinates vs his, 100000 zeros   worst 3.005e-9   (his stated accuracy 3e-9)
residual from his numbers vs ours    r = 1.00000

zero index    residual   noise floor   of the low-height shape
10^12          0.00922       0.01088          +0.043 ± 0.150
10^21          0.01292       0.01088          −0.176 ± 0.150
10^22          0.01120       0.01088          +0.166 ± 0.150
```

So the shape is in the zeros, not the zero-finder — and it is **gone by index
10^12**, sixteen orders of magnitude past anything computed here. Reported as a
bound rather than an absence: ten thousand zeros resolve the surviving
amplitude to about a third and no better, and "it is zero" would claim a
precision the sample does not carry. The tables are not vendored — they are
somebody else's data, and the command says where to fetch them.

Both the floor and the null assume the spacings are independent draws, and they
are not — consecutive gaps are correlated, which is what pair correlation
measures. So both were checked rather than assumed. A moving-block bootstrap,
which preserves local correlation, gives 0.0040–0.0042 against the formula's
0.0045 at block lengths from 1 to 200: the formula is slightly *conservative*,
which is the right direction and not the obvious one — a rigid spectrum
fluctuates less than independent points, so level repulsion suppresses the very
statistic the formula overestimates. And two halves of a *single* band, at
essentially the same height, correlate at 0.98 — closing off the reading that
the residual varies with height rather than being a property of the zeros.

Note the deviation figures are per-bin and depend on the binning; the
decomposition is comparable only at a fixed number of bins, which is why the
default is fixed rather than scaled with the sample.


**And how fast does it die?** `rhre symbolic spacing-decay` fits a power of
`log T` through six disjoint bands of Odlyzko's two million zeros:

```
T ~   2045   amplitude 0.236 ± 0.067        alpha = 1.61 ± 0.17   chi² 1.1 / 4
T ~   6643             0.182 ± 0.030        exponent pinned at 1: chi² 11.9 / 5
T ~  20324             0.179 ± 0.016
T ~  66075             0.143 ± 0.008        steeper than 1/log T,
T ~ 202522             0.122 ± 0.004        at 3.5 sigma over this range
T ~ 710449             0.104 ± 0.002
```

The estimator is **template-free**, and has to be: projecting each band onto a
shape taken from some band biases every overlapping band upward, which is
exactly the direction that fakes a decay. Instead `Σr² − Σσ²`, unbiased by
construction — and the unbiasedness is *tested* on samples drawn from the exact
law rather than argued, because a positive bias would report an amplitude in
every band and a decay from the band sizes alone.

That test earned its keep immediately: it found that a histogram bin is a *mean*
density and the curve was being evaluated at the bin *centre*. The two differ by
about `w²p''/24` — 10⁻³ per bin, and it does not shrink with the sample, so it
was a constant floor under every residual. Bin averages took the null bias from
+0.44 of the noise term to −0.035.

**The exponent is effective, not asymptotic**, and every record says so. `log T`
spans a factor of only 1.8 here, and over that a two-term expansion
`c/log T + d/(log T)²` fits just as well with `d/c ≈ 20` — the "correction"
exceeding the leading term everywhere it was measured. The data does not
separate them, and `DecayFit` refuses to be built without that sentence
attached.

## Moments, where the arithmetic is a separate factor

Every other statistic here measures the zeros against a *universal* law, and
universality means the agreement carries no arithmetic — GUE describes random
matrices and quantum billiards too. The Keating–Snaith moment conjecture is the
exception: its constant factorises into an arithmetic Euler product and a
random-matrix term, written separately.

```
(1/T) ∫₀ᵀ |ζ(½+it)|^{2k} dt  ~  c_k (log T)^{k²},   c_k = a_k · ∏_{j<k} j!/(j+k)!
```

```bash
rhre symbolic moments --k 2 --top 40000
```

The constants are computed, not tabulated, so they can be checked: `g_k` comes
out 1, 2, 42, 24024 — the published values; `a_1 = 1` identically; and
`a_2 = 1/ζ(2) = 6/π²`, which follows in closed form because the inner sum is
`(1+x)/(1−x)³` and the product telescopes. That gives `c_1 = 1` and
`c_2 = 1/(2π²)`, both **theorems** — so the measurement has a known answer
underneath the unknown one.

**The leading term is not the asymptotic.** At `k = 1` the truth is
`log(T/2π) + 2γ − 1`, and comparing against `log T` alone drops a constant of
1.68 — eighteen per cent at T = 2×10⁴. Against the full statement the
integrator agrees to the size of the known error term; against the leading term
it is out by eighteen per cent at the one `k` where the answer is known. So a
discrepancy of that size at `k = 3` would say nothing.

**And the obvious way to do better does not work.** The moment is `T` times a
degree-`k²` polynomial in `log T` whose leading coefficient is `c_k`; fitting it
and reading that off is the natural move. Over reachable heights `log(T/2π)`
spans a factor of about 1.5, and a degree-4 fit across it returns `c_2` out by
hundreds of per cent — at a `k` where the answer is Ingham's theorem.

That is the result, not a failure to produce one. Every `MomentFit` carries
`calibration_error`: the same extraction, run at `k = 2` on the same heights,
every time. No fitted coefficient is ever reported without the demonstrated
error rate of the method that produced it.

The sharper test is against the *full* moment polynomial conjectured by
Conrey–Farmer–Keating–Rubinstein–Snaith rather than one term of it, which needs
those coefficients from a source. That is the next step here, not another fit.

**The sharp test is the whole polynomial, not one term of it.** At `k = 2` the
entire degree-4 polynomial is proven (Ivić, and separately Conrey), and naive
random matrix theory predicts a *different* degree-4 polynomial with the **same
leading coefficient**. So comparing against both isolates exactly the part RMT
does not supply:

```
    T      measured    proven      err       naive RMT    err
  20000    546.372    546.614    4.4e-04     514.897    6.1e-02
  80000    919.734    919.845    1.2e-04     864.677    6.4e-02
 200000   1252.698   1253.264    4.5e-04    1177.726    6.4e-02
```

The data follows the theorem to **10⁻⁴** and sits **6.4%** from the RMT
prediction — and that 6.4% does not shrink with height, so it is not a
finite-height effect but a real failure of RMT below the leading order. The gap
between two polynomials that share a leading coefficient is the arithmetic,
measured.

That also settles what the earlier discrepancy was: against the leading term
alone, in the same variable, the measurement is out by more than 100%. Lower-order
terms, not the conjecture.

**`k = 3` cannot be tested here**, and the reason is the polynomial rather than
the computation. CFKRS conjecture the full `P₃`; Hiary and Odlyzko use those
coefficients without printing them. Until they are transcribed from that source,
the only available comparison at `k = 3` is against a leading term — which the
`k = 1` and `k = 2` calibrations show says nothing.

## Where the Riemann hypothesis is a theorem

Every criterion in the corpus is about one object, and about that object the
question is open — so nothing in the engine had ever been pointed at a case
where the right answer is known. `experiments/synthetic-adversary` supplies
half of what that needs: zeros put off the line on purpose, to see whether a
criterion produces a false positive. Its complement was missing.

A curve over a finite field is the complement. For `y² = x³ + ax + b` over
`F_p`, the zeta function is a rational function with integer coefficients, and
Weil proved every zero lies on `Re s = 1/2`:

```bash
rhre symbolic curve-zeta --a 1 --b 6 --prime 13
rhre symbolic weil-control --limit 30
```

```
#E(F_13) = 13,  a_p = 1,  P(T) = 13T² − T + 1
reciprocal roots  1/2 ± √51 i/2,  |α|² = 13 exactly
ON THE CRITICAL LINE    a_p² = 1 ≤ 4p = 52
```

**Nothing here is approximate.** Point counts are integers, `P` has integer
coefficients, and for genus 1 the entire Riemann hypothesis for the curve is
the integer inequality `a_p² ≤ 4p` — Hasse's bound. That makes it the first
zeta-like object in the engine whose critical-line statement is settled by
arithmetic rather than sampled: `verify-line` is floating point and
`certify-line` needs interval arithmetic, and this needs neither. It is filed
`certified`, not `rigorous_numerical`, because nothing is enclosed — nothing
is approximate.

`weil-control` checks **2260 curves over every field up to `F_29`, all on the
critical line, in 1.5 s**, and holds the integer test against the full
symbolic zeta function at sampled curves so the shortcut is not trusted on its
own. The circle test is separately shown to *reject* fabricated polynomials
whose roots are off it — a control that cannot fail on a bad input is not a
control.

**This is not evidence for the Riemann hypothesis.** Not weak evidence, not
suggestive evidence: none. The function field case was proved by intersection
theory on `C × C`, which has no counterpart over `ℚ`, and the analogy has
stood complete on one side and open on the other since 1948. `CurveZeta`
refuses to be built without that sentence attached — a `verified=True` about a
critical line, in this repository, is exactly the record that would get quoted
without its qualifiers.

## Pattern detection as a research function

```bash
rhre patterns sweep --top 5000
rhre patterns open
```

`rhre patterns sweep` points a blind scan at the corpus's own quantities: every
callable the index uses, derived arithmetic nobody asked for, both sides of
every inequality the corpus asserts, and columns built from the zeros. It
recovers `psi = sum Lambda` and `Mertens = sum mu` — definitions it already
knew, which is the validation — and characterises the primes and the squarefree
numbers from columns nobody thought to compare.

Findings are conjectures by construction: `PatternFinding` refuses any rigorous
confidence at construction, so agreement in every sampled case cannot become a
claim that something is established. Relations nothing explains accumulate in an
open ledger, which a later scan over a wider range can refute and can never
establish — and which reports `INCOMPARABLE`, never a refutation, when the
later scan did not cover the earlier range or measure the same columns.

**The ledger currently stands at zero open findings**, and all five were
settled by being *answered* rather than by widening the range — which a
widening cannot do. Three were `|x| >= x` in disguise. The last two are
elementary inequalities whose equality cases the scan found and could not name:

```
phi(n)     >= Omega(n),  tight exactly on {2, 4, 6}
n - phi(n) >= Omega(n),  tight exactly on the primes, and also 4
```

The second is why `character.py` now names the exceptions. It reported
`closest description: prime (off by 1)` — a number, where the whole point of
characterising a tight set is to produce a description. The one is 4; both
lists were computed and thrown away at the print. Naming them does not close
anything: the finding stayed open until a reason was written, and each reason
in `research_state/pattern_noise.json` carries its argument so a later reader
can check it rather than trust it.

**One quantity is one finding, however many names carry it.** `Mertens` comes
off the sieve and `cum_mu` is accumulated from sympy's `mobius`, kept apart so
that their agreement is a cross-check between two implementations. They hold
the same values on every row, so every relation involving M(x) was found twice
and reached the open ledger twice — same witnesses, same range, same surprise,
same character. The size of the ledger of things nothing explains was set by
how many implementations the sweep happened to carry. Columns that agree on
every examined row are now reported once, under a representative that does not
depend on the order of the list, with the other names recorded on the finding;
the agreement itself is still reported, because that is the cross-check.

Two things must never read as refuting a relation, and both did:

- **Renaming it.** A record filed as `abs_Mertens >= cum_mu` is absent from a
  scan that reports the same relation as `abs_Mertens >= Mertens`, and absent
  is `BROKEN`. Findings carry the names they equally hold under, and the
  ledger looks an entry up by all of them.
- **Explaining it.** A finding retired by a noise rule is absent too, so
  writing down *why* a relation is an artifact recorded "was a regularity of
  the narrower range" about a relation nothing had touched. That is now
  `RETIRED`, carrying the rule's reason — a fifth verdict, and the only one
  that does not come from a range.

## DRE-supervised mode (v0.4)

The Python package is now explicitly a **math worker**, not the authoritative research reasoner. It can export any stored experiment as a DRE evidence envelope:

```bash
rhre experiment correlation-lab --X 20000 --q 4
rhre dre export-latest --claim C005 --primary-metric screening_remainder --out dre/experiments/c005.yaml
```

The bundled DRE pack is at `dre/model-packs/riemann-research`. It enforces the central epistemic rule that numerical evidence cannot be promoted to proof, and it uses DRE independence groups so repeated runs of one implementation do not count as independent corroboration. See `docs/DRE_INTEGRATION.md`.


## Durable mathematical memory (v0.5)

The accumulated RH mathematics is now persisted independently of chat context in:

- `research_state/math_knowledge.json` — machine-readable knowledge graph;
- `docs/RH_MATHEMATICAL_MEMORY.md` — human-readable derivation map.

Future researcher agents should consult this memory before proposing a route. It explicitly distinguishes exact identities, established results, derived-but-unverified statements, active proof obligations, and permanent no-go results.

```bash
rhre knowledge list
rhre knowledge search "screening remainder"
rhre knowledge show K019
rhre knowledge validate
```

The repository is now the authoritative research memory; chat history is not required to reconstruct the mathematical program.

## v0.12 supervisor, adversary, verifier

```bash
rhre supervisor add --id H001 --statement "R_q = O(X^0.49)" --proof-gap "uniformity in q" --falsification finite-window:1:"run synthetic adversary"
rhre supervisor next
rhre experiment synthetic-adversary --eta 0.02 --gamma 14.1347251417
rhre verifier capability
```

The supervisor queue tracks structured hypotheses, proof gaps, cheapest
falsification tests, and next-step selection. RH-equivalent restatements do not
earn progress unless `discharged_obligations` explicitly names what was proved.

Synthetic adversarial systems enforce conjugation and functional-equation
symmetry and can inject critical-line and off-line zeros to test candidate
criteria for false positives and false negatives. Their evidence class is
heuristic and cannot promote claims.

The Arb/FLINT verifier adapter detects `python-flint` when available and emits
`MathCertificate`/`VerifierEnvelope` artifacts. Without rigorous interval
verification it fails closed with `status=unknown`; mpmath/numpy output is never
relabeled as rigorous.
