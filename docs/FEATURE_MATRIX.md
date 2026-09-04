# RH Research Engine Feature Matrix

## Deterministic research control

| Capability | Status | Authority |
|---|---|---|
| Claim/dependency registry | Implemented | DRE-supervised |
| No-go rule tracking | Implemented | DRE-supervised |
| Quantitative `Theta` scoring | Implemented | Deterministic worker + DRE |
| Experiment replay/provenance | Implemented through DRE bridge | DRE |
| Durable RH mathematical memory | Implemented | Repository state, sealed |
| Canonical contract layer (lifecycle / confidence / role / frontier) | Implemented | `contracts/` |
| Total legacy-vocabulary mappings | Implemented | No substring classification |
| Derived frontier verdicts (never stored, never suppliable) | Implemented | `contracts/frontier.py` |
| DRE obligation-discharge authority | Implemented, **inert**: trust registry empty | `contracts/receipts.py` |
| Receipt signature authentication (Ed25519) | Implemented, **inert**: no key registered | Optional `cryptography` |
| Receipt sealed-store authentication | **Not connected.** No decision store exists | — |
| Receipt deterministic-replay authentication | **Not connected.** No engine connected | — |
| Phase 1 closure gate (17 gates, one command) | Implemented | `tools/phase1-final-gate.py` |

## Numerical / analytic workers

| Capability | Status |
|---|---|
| Gamma-filter experiments | Implemented |
| Safe binomial experiments | Implemented |
| Prime-pair correlation laboratory | Implemented |
| Counterterm discovery experiments | Implemented |
| Local-window variance | Implemented |
| Synthetic off-line-zero injection | Implemented |
| Exact arbitrary-precision rational representation | Implemented |
| Certified real/complex interval *representation* | Implemented (representation only; nothing certifies the enclosure) |
| DRE-safe certificate predicates | Implemented |
| Independent verifier envelopes | Implemented |
| Native Arb execution adapter | **Not connected.** Envelope schema only; `ACCEPTED` is refused for this family |
| Native FLINT execution adapter | **Not connected.** Envelope schema only; `ACCEPTED` is refused for this family |

## Symbolic compiler

| Capability | Status |
|---|---|
| Markdown/LaTeX equation extraction | Implemented, conservative |
| Equation AST normalization | Implemented |
| Controlled simplifier with rewrite trace | Implemented |
| Exact equivalence checking | Implemented |
| Formula fingerprints | Implemented |
| Persistent formula index | Implemented |
| Equation-relation fingerprints | Implemented |
| Structural formula similarity | Implemented, discovery-only |
| Hidden-assumption extraction | Implemented |
| Proof-gap extraction | Implemented |
| Transform registry | Implemented |
| Residue analysis | Implemented |
| Asymptotic ratio sanity checks | Implemented |
| Power-growth exponent checks | Implemented |
| Exact decomposition search | Implemented |
| Symbolic counterterm basis | Implemented |
| Conjecture minimizer | Implemented |
| Paper/document equation ingestion | Implemented |
| Automatic indexing on ingestion | Implemented |
| Durable-memory formula indexing | Implemented (durable memory declares no formulas yet) |
| Formula provenance / citation records | Implemented; carries no epistemic status |
| Formal-proof queue | Implemented, fail-closed; emits Lean, never runs it |
| Durable-memory route/no-go matcher | Implemented |
| Certificate-to-symbolic consistency check | Implemented |
| Lean polynomial identity exporter | Implemented, fail-closed |

## Formal proof

`rhre symbolic proof-queue` sorts every indexed equation into one verdict:
`export_ready`, `not_an_identity`, `unsupported_fragment`,
`not_an_equation`, `unrenderable`, or `unparseable`. Refusals stay in the
report rather than being filtered out, because "400 formulas, 3 provable" is the
useful number.

**Export-ready means emitted, not verified.** Nothing in this package runs Lean,
so whether `ring` closes the goal is decided by a compiler that has not been
invoked. `ProofQueueEntry.to_formalization_report()` therefore returns a report
whose `remaining_obligations` names the missing compilation, keeping
`fully_formalized` false and the contract validator refusing
`FORMALLY_VERIFIED`.

The Lean exporter currently handles only mechanically verified polynomial identities over `ℚ` and emits `ring` proofs. Analytic number theory, limits, contour shifts, asymptotic estimates, infinite sums, and zeta-function facts remain explicit proof obligations. The exporter must refuse them until a supported formalization path exists.

## Reference material

[`RH_FORMULA_REFERENCE.md`](RH_FORMULA_REFERENCE.md) — cited RH formulas and
criteria, tagged with this project's axes and separated into unconditional
facts, RH-equivalents (no frontier credit), and one-directional consequences.
Every entry carries a source link. Nothing in it is machine-readable until the
`formulas` field of durable memory is populated.

## Trust posture

Two registries gate every authority path, and both are empty:

| Registry | Entries | Effect while empty |
|---|---|---|
| Trusted DRE engines (fingerprint → pinned model packs) | 0 | No obligation can be discharged |
| Registered verifier adapters | 0 | No envelope reaches a rigorous confidence |
| Registered receipt signing keys | 0 | No signature authenticates |

`rhre` reports this honestly rather than by documentation: `activation_status()`
and `verifier_activation_status()` return counts, never the contents, so reading
them grants nothing. See [`PHASE2_ACTIVATION.md`](PHASE2_ACTIVATION.md).

## Remaining high-value integrations

1. Connect an actual Arb worker that emits `MathCertificate` JSON.
2. Connect an independent FLINT worker using a distinct DRE independence group.
3. Validate the DRE model pack against a pinned DRE commit in CI. (The Python suite, lint, line-ending, determinism, and Phase 1 closure gates now run; the pack is still checked only by inspection.)
4. ~~Add a formal-proof queue~~ — done. `rhre symbolic proof-queue` sorts indexed equations into export-ready and refused, with a verdict for each. Export-ready means Lean was *emitted*, not verified: nothing here runs a compiler, so each export stays an open obligation.
5. ~~Index equations automatically during ingestion~~ — done. `rhre symbolic ingest` indexes by default; `rhre symbolic index-knowledge` reads the `formulas` field of durable memory. Note that the shipped memory declares no formulas, so that half indexes nothing until they are populated — a data gap, not a broken reader.
6. ~~Add literature provenance/citation records~~ — done. `Citation` records source, identifier, theorem label, section, and the surrounding statement, narrowed per equation. It carries no epistemic status: appearing in a paper is not being proved.
7. Connect a DRE decision store, so receipts can authenticate by sealed-store retrieval as well as by signature.
8. Connect a DRE engine, so receipts can authenticate by deterministic replay — the only mechanism needing no trusted third party.
9. ~~Enable branch protection~~ — closed. Gated behind a paid plan for private repositories, and the repository stays private. `tools/githooks/pre-push` is the local substitute.
10. ~~Add pattern detection as a research function~~ — done. `rhre patterns audit` asks whether an assigned quantity has anything in it to fit before fitting it; `rhre patterns scan` looks at every column and every pair, not the one the task named, and ranks what it finds by how much the relation constrains. Unanimity is deliberately *not* the filter: pointed blind at the corpus's own functions, ranking by unanimity kept twenty-four integrality-of-a-counting-column findings and dropped `sigma(n) >= n+1, tight in 12 of 39`, which is the characterisation of the primes. A tight set is tested against ordinary predicates and reported by name, or as UNRECOGNISED — the interesting case, and ranked up rather than dropped.
    Two registries close the loop, and they are counterparts. `rhre patterns dismiss` retires a finding that has been explained, with a required reason; `rhre patterns open` lists the ones nothing has explained, which a later scan over a wider range judges as EXTENDED, CONTRADICTED, BROKEN, INCOMPARABLE or RETIRED. INCOMPARABLE is the one that matters most: a range that does not contain the recorded one refutes nothing, and a comparison that cannot tell "refuted" from "not tested" would quietly retire the findings it never looked at.
    RETIRED is the same distinction found a second time, on the other side of the loop. A finding the noise registry retires is *absent* from the scan the ledger is judged against, and absent is BROKEN — so writing down why a relation is an artifact recorded "was a regularity of the narrower range" about a relation nothing had touched. Explaining and refuting are opposite outcomes and they shared a verdict. RETIRED carries the rule's reason, is read before the range check because a written reason does not depend on what happened to run, and is the only verdict that does not come from a range. `retired` is a required argument to `judge` for the same reason `columns` is: it changes the verdict and cannot be recovered afterwards.
    **One quantity is one finding, however many names carry it.** `Mertens` and `cum_mu` are two implementations of M(x), kept apart so their agreement is a cross-check — and they hold the same values on every row, so every relation involving M(x) was found twice and filed twice, with byte-identical witnesses, universe, surprise and character. The count of open questions was a fact about the sweep's plumbing. Columns agreeing on every examined row are now reported once, under a representative fixed by a rule that does not depend on the order of the list, with the other names on the finding and the agreement itself still reported. The ledger looks entries up by every name a finding answers to; without that, reporting a relation once would REFUTE the copy already filed under the other name.
    Findings are conjectures by construction: `PatternFinding` refuses any confidence in the rigorous set at construction, so agreement in every sampled case cannot become a claim that something is established. Nothing in the ledger promotes anything — widening a range can refute a finding and cannot establish one.
    **The ledger stands at zero open findings**, all five settled by being answered rather than by widening. Three were `|x| >= x` wearing two column names; the last two are elementary inequalities whose equality cases the scan found and could not name — `phi(n) >= Omega(n)` tight on `{2, 4, 6}`, and `n - phi(n) >= Omega(n)` tight on the primes together with 4. The second is why a near miss now names its exceptions: `closest description: prime (off by 1)` is a number, and this package exists to turn numbers into descriptions. Naming them does not set `characterised` — a description says what to go and prove, and closing a question by rewording its headline is the promotion the ledger refuses. `NoiseGround.ELEMENTARY` was added for these two, because `known_identity` claims a citation neither has and `triaged` files a checkable argument under "judged uninteresting by a person"; it is deliberately not spelled `proved`, and a test refuses any ground that collides with a `Confidence` value.
    `rhre patterns sweep` points the whole loop at the corpus's own quantities: every callable the indexed corpus uses, derived arithmetic nobody asked for, **both sides of every inequality the corpus asserts** (Robin, Lagarias, the two Schoenfeld bounds, the Mertens conjecture), and columns built from the zeros. Over `n = 2..4999` it recovers `psi = sum Lambda` and `Mertens = sum mu` — definitions it already knew, which is the validation — and characterises the primes five ways from columns nobody compared, including `Lambda(n) <= log n` tight exactly at the primes, from two corpus columns.
    The scan is run again at double the precision and only findings present in both are believed. That check earns its keep: `zeta(n)` rounds to `1.000…0` past about `n = 100`, so `zeta >= mu` looked tight wherever `mu(n) = 1` and the ledger filed it as an open question about the squarefree numbers.
    **The zero columns produce nothing, and that is a fact about the instrument.** This detector finds *exact* relations; zero ordinates are irrational and satisfy none with an arithmetic function, so no widening will produce one. The structure that is there — pair correlation, the GUE spacing law — is distributional, and a detector built to refuse statistical trends cannot see it by construction.
11. ~~Make the zeros computable in bulk~~ — done. `symbolic/riemann_siegel.py` evaluates the Hardy function `Z` for a whole array at once, by Euler–Maclaurin below the crossover and Riemann–Siegel above it, so a zero costs about 0.1 ms instead of the 160 ms `mpmath.zetazero` spends per root-find. 63519 ordinates below height 50000 in 8.6 s; all 1000 of the first agree with `mpmath.zetazero` to a worst error of 2.0e-12. `ZeroCount` counts by Turing's method rather than by locating anything, lifting its reach from `T = 200` to `10^7`.
12. ~~Count the zeros independently of the critical line~~ — done. `symbolic/argument_principle.py` computes `N(T) = theta(T)/pi + 1 + S(T)` by tracking `arg zeta` along a path that never touches the critical line. Compared against the sign changes of `Z`, agreement below a height says every zero below it is on the line and simple: **1747146 of 1747146 below `T = 10^6`, in 24 minutes** (63519 below `T = 50000` in 21 s; 466659 below `3 x 10^5` in 4 minutes). Both records refuse a rigorous confidence at construction — it is a finite floating-point computation with no enclosure, and RH is a statement about every zero. Item 14 wires those certified enclosures and does reach `rigorous_numerical`, at a lower height; this module stays because it reaches two orders of magnitude further at a weaker confidence.
13. ~~Check the corpus's statements about the zeros against values~~ — done. Two formulas that had sat in the index since ingestion with nothing able to evaluate them, because both need thousands of zeros.
    `rhre symbolic pair-correlation` measures the spacing of the zeros against the index's **first** formula, `1 - (sin(pi u)/(pi u))^2`. Mean deviation **0.0165** over 35473 zeros below `T = 30000`, improving with height (0.0227 at 63k zeros, 0.0168 at 298k) — the direction that matters for an asymptotic conjecture.
    `rhre symbolic explicit-formula` rebuilds `psi(x)` from the zeros by von Mangoldt's formula: residuals below 0.004 at most sampled points over 20000 zeros. Neither is a test *of* the mathematics — both are theorems or well-studied conjectures. They test whether the corpus **records** them correctly, which is a different question and the one this repository keeps getting wrong: a missing constant parses, indexes, fingerprints and exports exactly as well as the true statement.
    Both read the expression **from the index** rather than a retyped copy, so a correction to the corpus moves the check with it and a corpus that stops containing the formula makes the check raise. Both records refuse a rigorous confidence at construction.
14. ~~Reach a rigorous confidence~~ — done, at a lower height, and the height is the finding. `rhre symbolic certify-line` answers exactly the question item 12 answers, through Arb's interval arithmetic instead of float64, and `CertifiedLineVerification` is filed `rigorous_numerical` — the first record in this engine that can be. The claim splits into two halves that cost four orders of magnitude apart: **counting** the zeros in the strip is free (`N(10^8) = 248008025` in 0.01 s, against 24 minutes for the floating-point `N(10^6)`), while **placing** them on the line costs milliseconds each and reaches `T = 10^4` in 52 s. So the cheap half now confirms every figure item 12 has ever produced, and the expensive half reaches two orders of magnitude lower. Item 12 stays; the two are not substitutes.
    Three checks carry the argument, and each is a way the enclosures could be over-read. `Re = 1/2` is tested as *exact equality against 1/2*, never as proximity: a ball around the critical line contains off-line points, so "1/2 to within 1e-30" is a much weaker claim, and Arb marks a zero it has *proved* on the line with radius zero. Every enclosure must lie in `(0, T]` — ask for one zero too many and the extra sits above the height. And they must be pairwise disjoint, or `N` enclosures could describe fewer than `N` zeros and the multiplicity argument giving simplicity fails. Every comparison is on Arb balls, never on floats read out of them, which is sound only because Arb's `<`, `>` and `==` are true only when true of every point — two overlapping intervals answer False in **both** directions.
    `python-flint` is optional and detected, never assumed: without it `certify-line` raises rather than falling back, because a verification that quietly reverts to floating point is one whose `confidence` field lies. It is in `dev` as well as its own extra so CI runs the certified path, on the reasoning that put `cryptography` there. `interval_certificate` still reports UNKNOWN — it is handed endpoints rather than a computation, and installing a backend must not promote it. And `envelope_confidence` maps ACCEPTED to `rigorous_numerical` only because `_registered_adapters` detects the backend independently: before this build nothing was ever ACCEPTED, so that guard was protecting a case that could not arise, and now it can.
    **None of it is new mathematics.** Zeros have been certified rigorously to heights vastly beyond `10^4`. What is new to this repository is that its own verification can be filed at a rigorous confidence rather than refused one — and `rigorous_numerical` is deliberately *not* in `contracts.epistemic.RIGOROUS`, because rigorous about a finite computation is not a statement about every zero.
