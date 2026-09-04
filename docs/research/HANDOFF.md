# Handoff — 2026-08-24

Written at the end of a long session. Read this before starting; it says what
landed, what is in flight, what was found, and where to pick up.

## State

`main` is green. Nine CI jobs pass on every merged commit. The formula guard
reads **41 of 41 formulas checked against values**, and the pattern ledger
holds **zero open findings** — every one was answered rather than refuted.

Nothing is in flight. [#39](https://github.com/heidrickla/rh-research-engine/pull/39)
(moments) and [#40](https://github.com/heidrickla/rh-research-engine/pull/40)
(this page and the pair correlation notes) both merged green; `main` is at
`f1c0e15`.

When there *is* something in flight: **verify CI's conclusion before merging**
(`gh run view <id> --json conclusion -q .conclusion`) and merge with
`gh pr merge <n> --rebase --delete-branch` — squash merges and force pushes are
blocked by the permission classifier here.

## What landed

Merged as #35–#38, in order:

1. **One quantity is one finding.** `Mertens` and `cum_mu` hold the same values,
   so every relation involving M(x) was found and filed twice. Deduplicated —
   and then two things that must never read as refuting a relation were fixed:
   *renaming* it (the copy filed under the other name reads BROKEN) and
   *explaining* it (a finding retired by the noise registry is absent from the
   scan, so writing down why a relation is an artefact recorded it as refuted).
   `RETIRED` is the new verdict.

2. **Certified zeros.** `rhre symbolic certify-line` uses Arb interval
   arithmetic and is the first record here filed `rigorous_numerical`. The two
   halves cost four orders of magnitude apart: counting is free (`N(10^8)` in
   0.01 s), placing on the line reaches `T = 10^4` in 52 s. The count half now
   rigorously confirms every figure this engine has recorded.

3. **Imports from three other fields.** Redheffer (`det R_n = M(n)`) and
   Franel–Landau make RH a statement about a determinant and about
   equidistribution. `finite_field_zeta` supplies the missing *positive
   control* — curves over finite fields, where RH is a theorem and, at genus 1,
   an integer inequality. Level spacing measures consecutive gaps where pair
   correlation averages over all pairs.

4. **The spacing residual, decomposed.** Curve error 0.0018, sampling noise
   0.0045, and about 0.014 that is the zeros — established by correlating the
   signed residual across *disjoint* bands (r = 0.96–0.98 against a measured
   null of 0.24). Checked against Odlyzko's published zeros: our ordinates match
   his to 3.005e-9 against his stated 3e-9, the residual from his numbers
   correlates with ours at 1.00000, and the shape is gone by zero index 10^12.
   Decay measured at `(log T)^-1.61 ± 0.17` — **effective, not asymptotic**.

## Findings worth carrying forward

**The universal part carries no arithmetic.** This is the thread. GUE describes
random matrices and quantum billiards, so agreement with it says the zeros are
a spectrum and nothing about primes. The arithmetic is in the *departures*, and
the moments made that quantitative: two degree-4 polynomials sharing a leading
coefficient, the data following the proven one to 10⁻⁴ and sitting 6.4% from
the naive random-matrix one — a gap that does not shrink with height.

**A positive control is what makes a number readable.** Three times this
session a measurement only became interpretable once a case with a known answer
sat beside it: `weil-control` for the critical line, `k = 1` and `k = 2` for the
moments, and Odlyzko's high tables for the spacing residual. Where a control
was missing, the number was misread.

**Five errors, each of which first looked like a result.** Kept as tests:

- nested ranges correlate by construction (r = 0.98, meaning nothing);
- a histogram bin is a *mean*, compared against a *point* value — a floor under
  every residual that does not shrink with the sample. Fixed in `level_spacing`
  and **still live in `pair_correlation` for another two weeks**, one file away,
  with a docstring in the first file already explaining it. Found only when the
  lower-order terms turned out to be the same size as the bias (2.7e-3);
- a null built from doubly-unfolded data;
- `theta` computed by its asymptotic series, wrong by 2×10⁻² at `t = 1` and
  NaN at `t = 0`, undetected because the lowest zero is at 14.135;
- fitting a moment polynomial to extract its leading coefficient — out by
  hundreds of per cent at the `k` where the answer is a theorem.

The last is why `MomentFit` carries `calibration_error`: no extracted
coefficient is reported without the demonstrated error of the method behind it.

## Where to pick up

**1. Pair correlation with its lower-order terms.** Built, after being filed
here as "started, not built" — `symbolic/conrey_snaith.py` carries the curve and
`check_pair_correlation(..., lower_order=True)` measures the zeros against it.
Both "known issues" recorded on the page were diagnosed wrongly, and the page
now carries what they were: the ~10⁻³ was not noise but the formula's own
lower-order terms, and the float64 warning was right for a reason that was not
the one given.

Against a noise floor **measured** rather than borrowed, Conrey–Snaith sits at
1.08×, 0.87×, 1.07× the noise at `T = 5×10³, 3×10⁴, 2×10⁵` — at the noise,
with no free parameters. Montgomery goes 1.61× → 2.19× → 4.82×, and the growth
is the point: a systematic error stays put while the noise falls, so its ratio
grows; a correct curve's ratio stays flat at one.

The "something remains at 1.7× the noise" first written here was the wrong
ruler — `level_spacing.sampling_noise_floor` is the GUE *spacing* density
assuming independent draws, and this statistic counts ~3N correlated pairs from
N zeros. It understates the noise 1.6×. Nothing remains.

Then the sharper test, `symbolic/height_recovery.py`: Conrey–Snaith is a
one-parameter family in `ℓ = log(t/2π)`, so fit `ℓ` and see whether it lands
where the heights say. Over 1,747,146 zeros below `T = 10⁶` the regression of
fitted on true `ℓ` has **slope 0.94 ± 0.13** — 0.5σ from 1 and 7.2σ from 0,
bias −0.061 ± 0.114. (**This read 0.828 ± 0.119 until 2026-08-25**, when the
band grid was found to include a bin the data does not fill — carrying 38% of
the slope's leverage — and to depend on an arbitrary anchor worth ±0.068.) (It first read 1.058 ± 0.112; review found
the bands were equal-count and so far too wide in `ℓ` at the bottom, and the
regression unweighted across bands of unequal precision. Correcting both moved
the slope two error bars. One band remains a 2.5σ outlier, unexplained.) Unfolding removes the height by
construction, so the only route for `ℓ` into the measured density is the
arithmetic, and Montgomery's curve has no `ℓ` at all.

**The out-of-sample control is not available, and that is measured now rather
than aspirational.** At Odlyzko's tables the curves sit 0.0036 apart (index
10¹²) against a noise floor of 0.0217 at his 10,000-zero sample — 6× below the
noise, and 18× at 10²¹/10²². It would take ~364,000 zeros at that index.
Do not spend a session fetching them.

**2. The consolidation pass, untouched.** The spacing work now spans
`level_spacing`, `spacing_decay`, `odlyzko` and `pair_correlation`, telling one
story four times, plus again in the README. It is all correct and each piece is
checked, but it reads as sediment. Whoever reads it next — including a future
session — would be helped by one pass making it one thing.

**3. `k = 3` moments are blocked on a transcription, not a computation.** CFKRS
conjecture the full `P₃`; Hiary and Odlyzko use those coefficients without
printing them. Until they come from that source, the only comparison available
at `k = 3` is against a leading term, which the `k = 1` and `k = 2` calibrations
show says nothing.

## Practical notes

- **Odlyzko's zeros** are at `https://www-users.cse.umn.edu/~odlyzko/zeta_tables/`
  and `curl` reaches them from the Bash tool. `zeros6` is 2,001,052 ordinates to
  `T = 1.13×10^6`; `zeros3/4/5` are 10,000 each at index 10^12, 10^21, 10^22 and
  give **offsets from a huge base**, so spacings are differences of offsets and
  the base cancels exactly. Not vendored — `rhre symbolic odlyzko-check --data DIR`
  takes a directory and refuses, with the URL, when it is absent.
- **His papers do not extract.** The 1987 Math. Comp. paper and Forrester–Odlyzko
  are scanned images and old TeX encodings. Hiary–Odlyzko (`zeta.moments.pdf`)
  and arXiv PDFs extract cleanly with `pypdf`.
- **`python-flint` is installed in the user's system Python**, not a venv. It is
  optional and detected; `pip uninstall python-flint` reverses it. Both branches
  are tested — with the backend and without.
- The corpus lives in `docs/research/rh-ingestible-algebra.md`. **Re-ingest after
  editing it** (`rhre symbolic ingest <path>`), then run `tools/formula-guard.py`.
