# Lower-order terms in pair correlation

**Status: built.** `symbolic/conrey_snaith.py` carries the curve,
`check_pair_correlation(..., lower_order=True)` measures the zeros against it,
and `tests/test_conrey_snaith.py` holds the controls. This page is kept as the
transcription record — the formulas with their equation numbers — and as the
account of what the two open problems below turned out to be, because both were
diagnosed wrongly here first and the wrong diagnoses are the more useful half.

## Why

`symbolic/moments.py` established the shape of the argument. Montgomery's
`1 − (sin πx/πx)²` is a *universal* law — it describes random matrices and
quantum billiards too — so agreement with it carries no arithmetic. The
arithmetic is in the departure from it, and at `k = 2` the moments made that
quantitative: two degree-4 polynomials sharing a leading coefficient, the data
following the proven one to 10⁻⁴ and sitting 6.4% from the naive random-matrix
one, with the 6.4% not shrinking.

Pair correlation admits exactly the same test, and `symbolic/pair_correlation.py`
currently measures against the leading term alone. Conrey and Snaith derived the
full form with every lower-order term, from the ratios conjecture.

## The formulas

Conrey and Snaith, *Correlations of eigenvalues and Riemann zeros*,
[arXiv:0803.2795](https://arxiv.org/abs/0803.2795), equations (178)–(184) and
(210), (212) — section *"Pair correlation, ζ"*, LaTeX labels `eq:I1ab`,
`eq:Aprime`, `eq:Bprime`, and the restatements `eq:AprimeA`, `eq:BprimeA` in
*"Auxiliary functions"*.

**Transcribed and now VERIFIED against the source.** Not independently derived
— but every expression below was checked character by character against the
LaTeX of the arXiv submission, and all six match: `A`, `B`, `P₁`, `P₂`, `J*`,
and `R₂`. The page said "not independently derived" for as long as it existed,
and everything in this repository that involves a lower-order term rests on it,
so it was worth an hour to stop trusting it. Labels are recorded beside the
numbers because labels survive a revision and printed numbers do not.

Two conventions were checked at the same time, because they are assumptions our
code makes rather than things the formulas state:

- **Dividing by `ℓ²` is right.** The paper gives the random-matrix counterpart
  as `R_{N,2} = N² + J*(iu;−iv) + J*(−iu;iv) = det[[N, S(u−v)],[S(v−u), N]]`,
  so `R₂/N² = 1 − S(u−v)S(v−u)/N²`, which is Montgomery's curve in the scaling
  limit with `ℓ` in the role of `N`.
- **The `2π` in `δ = 2πx/ℓ` is right.** The paper scales "the variables in the
  test function ... by log T/2π", and the zeros have density `ℓ/2π`, so unit
  mean spacing is `x = δℓ/2π`. The `ℓ → ∞` control is the empirical proof: it
  reproduces Montgomery exactly, and would not if the `2π` were misplaced.

```
A(x) = ∏_p (1 − p^-(1+x))(1 − 2/p + p^-(1+x)) / (1 − 1/p)²          (179), (210)

B(x) = Σ_p ( log p / (p^(1+x) − 1) )²                                (180), (212)

P₁(x) = e^(−ℓx) · A(x) · ζ(1+x) · ζ(1−x)                             (182)

P₂(x) = (ζ'/ζ)'(1+x) − B(x)                                          (183)

J*_{ζ,t}(a; b) = P₁(a+b) + P₂(a+b)                                   (184)

R_{ζ,t,2}(u, v) = ℓ² + J*(iu; −iv) + J*(−iu; iv)                     (181)

ℓ = log(t / 2π)
```

With `δ = u − v` the two `J*` arguments are `iδ` and `−iδ`, so

```
R₂ = ℓ² + [P₁ + P₂](iδ) + [P₁ + P₂](−iδ)
```

and the unfolded separation is `x = δℓ/2π`, so `δ = 2πx/ℓ`.

## The control, and it passes

As `ℓ → ∞` with `x` fixed, `R₂/ℓ²` must approach Montgomery's curve. This is
the only check available that does not involve the zeros.

It constrains the leading term and nothing else — deleting `B` from (183)
entirely leaves it passing, as it must, since every lower-order term vanishes
in the limit. See "A third thing, found by breaking the tests" below. This
paragraph originally said this check "is what says the assembly is right rather
than merely plausible", which is more than it can carry.

| x | ℓ = 10 | ℓ = 20 | ℓ = 40 | Montgomery |
|---|---|---|---|---|
| 0.50 | 0.55714 | 0.58111 | 0.59093 | 0.59472 |
| 1.00 | 1.02111 | 1.00408 | 1.00038 | 1.00000 |

At `x = 0.5` the departure runs −0.0376 → −0.0136 → −0.0038, shrinking about
fourfold per doubling of `ℓ`. That is the arithmetic correction vanishing in
the limit, which is what it must do.

## The two open problems, and what they actually were

Both were recorded here as obstacles before being measured. Both measurements
went the other way, and the module docstring in `symbolic/conrey_snaith.py`
carries the numbers.

### "Residual noise at ~10⁻³, source unidentified" — it was not noise

Recorded here as noise at `x = 1.5, 2, 3` that "jumps about", with truncating
`A` and `B` at 10⁵ primes as the stated likely cause. Three measurements:

- **Not the truncation.** 10³ → 10⁶ primes moves `R₂/ℓ²` by at most 5.6×10⁻⁵,
  and 10⁵ → 10⁶ by under 10⁻⁶. The stated hypothesis is refuted.
- **Not the precision.** dps 25, 50 and 100 agree bit for bit.
- **It has an exponent, and noise does not.** `ℓᵏ(R₂/ℓ² − Montgomery)` settles to
  a definite constant: `k = 2` generically (at `x = 0.5`, to −6.2921) and
  `k = 4` at integer `x`, where Montgomery's `sin²` vanishes.

It is this formula's own lower-order terms — the thing it was transcribed for,
filed as an artefact obscuring itself.

**Why it looked like noise, which is the finding worth keeping.** The sampling
here was `ℓ = 10, 20, 40`. Over that range the `1/ℓ²` and `1/ℓ⁴` terms are
still comparable, so the departure changes sign with `ℓ` and does not scale; it
sorts into asymptotic order only past `ℓ ≈ 160`. And `ℓ = log(t/2π)`, so
`ℓ = 10..40` is not an unlucky choice — it is every height a zero has ever been
computed at. `T = 10⁶` is `ℓ = 12.0`; Odlyzko's tables at zero index 10¹² and
10²² are `ℓ = 24.4` and `ℓ = 48.4`. **At every reachable height, no single
term of this expansion dominates** — which is exactly why the full form is
worth carrying, and why the departure must be evaluated and never fitted to a
power of `ℓ`.

### "float64 cannot survive it" — right conclusion, and the reason matters more

The table below is correct and the conclusion is correct. The reason given is
one step short, and the reason is what tells you *where* it fails.

| δ | ζ(1+x)ζ(1−x) | (ζ'/ζ)'(1+x) − B(x) |
|---|---|---|
| 0.5 | 4.190 | −3.981 |
| 0.1 | 100.188 | −101.413 |
| 0.01 | 10000.188 | −10001.571 |

There are **two** cancellations, not one. `P₁ + P₂` cancels the `1/δ²` poles down
to about `−ℓ²`; then `ℓ² + (P₁+P₂)` cancels again down to `R₂`, which vanishes
like `x²` as `x → 0` because the zeros repel. Composing them, `A` needs about
`110 x⁴` of *relative* precision — so the floor depends on `x`, and **no amount
of `dps` fixes it**, because the limit is in `A` and not in ζ. Measured: dps 30,
50 and 80 give the broken small-`x` values identical to the last digit.

Two consequences, both in the module:

- `A` is accumulated as a **sum of logarithms**, not `np.prod`. Sequential
  multiplication over 9592 factors loses `n·ε ≈ 10⁻¹²`; pairwise summation of
  the logarithms loses `log₂(n)·ε ≈ 10⁻¹⁵`. Measured against mpmath:
  1.9×10⁻¹³ against 1.7×10⁻¹⁶. Three orders, for the same arithmetic.
- `MINIMUM_SEPARATION = 10⁻³` refuses below the floor rather than returning the
  plausible shape. At `x = 10⁻⁵` the fast path gives −5.1×10⁻⁸ where the truth
  is +2.8×10⁻¹⁰ — wrong sign, 180× the magnitude.

With that, complex128 carries `A` and `B` fine over the whole histogram, which
bottoms out at `x = 5×10⁻³`, and is 150× faster than mpmath there.

### A third thing, found by breaking the tests

The `ℓ → ∞` control above was called, on this page, "the only check available
that does not involve the zeros … what says the assembly is right rather than
merely plausible". It is necessary and it is not sufficient: **deleting `B`
from (183) entirely leaves it passing.** It has to — every lower-order term
vanishes in the limit, so the limit cannot see any of them. What catches a
dropped lower-order term is level repulsion at `x → 0` and the exponent test,
both at finite `ℓ`.

## What the zeros do

`check_pair_correlation(..., lower_order=True)`, against both curves, with the
Conrey–Snaith curve averaged over the pooled sample's own `ℓ` distribution:

| T | zeros | from Montgomery | from Conrey–Snaith | curves apart | measured noise |
|---|---|---|---|---|---|
| 5×10³ | 4,320 | 0.0462 (1.61×) | 0.0309 (1.08×) | 0.0304 | 0.0287 |
| 3×10⁴ | 35,473 | 0.0262 (2.19×) | 0.0104 (0.87×) | 0.0229 | 0.0120 |
| 2×10⁵ | 298,000 | 0.0171 (4.82×) | 0.0038 (1.07×) | 0.0169 | 0.0036 |

The curves are further apart than the noise at every height, so the data can
tell them apart — though at `T = 5×10³` only just, by a factor 1.06.

**Conrey–Snaith sits at the noise, at every height: 1.08×, 0.87×, 1.07×.** It has
no free parameters and nothing was fitted. Montgomery goes 1.61× → 2.19× →
4.82×, and *that growth is the finding*: a systematic error stays put while the
noise falls with the sample, so its ratio to the noise grows, while a correct
curve's residual falls with the noise and its ratio stays flat at about one.
A curve merely tuned to one sample could not do that.

Shape, at `T = 2×10⁵`: the residual against Montgomery has a −0.045 dip at
`x = 0.45` and a +0.029 bump at `x = 0.85`. Against Conrey–Snaith it is 15 bins
above and 15 below, mean −0.0001.

### Two more things review caught, and neither test could see

**The height weights counted pairs, not zeros.** `density` divides by
`len(unfolded)`, so its expectation is `(1/N) Σₙ R₂(·; ℓₙ)` — every retained
zero weighted once. The `ℓ`-band weights were accumulated one per *pair*
instead, weighting each height by a random quantity that fluctuates with the
very spacings under test, so the curve being compared against was a different,
data-dependent statistic from the density returned. Worth 1e-4 — it moves one
figure in the table above and no conclusion — and wrong regardless. The test
beside it only checked the weights summed to one, which they do either way.

**The quadrature budget lived in a comment.** `BIN_SAMPLES` was cut 21 → 9 to
make CI cheaper, justified against the 9.2e-5 that `ELL_BANDS` contributes. But
the bin-average test derives *both* of its sides from `BIN_SAMPLES`, and the
refinement test varies `ELL_BANDS`, so cutting it to 5 — whose 1.4e-4 blows the
budget — would have left the suite green while these numbers drifted. The
budget is a constant now, and the gate checks the chosen count against a
41-point reference and confirms 5 fails it.

### The noise floor had to be measured, and the first number here was wrong

This section first reported "1.7× the noise" remaining against Conrey–Snaith,
and named tracking it down as the open problem. There was nothing to track
down: the ruler was wrong.

`level_spacing.sampling_noise_floor` is a formula for a *different statistic*.
It evaluates the GUE nearest-neighbour **spacing** density, and it treats bin
counts as multinomial over `N` independent draws. This histogram counts about
`3N` pairs from `N` zeros, every zero appears in several of them, and the zeros
are correlated — which is the thing pair correlation exists to measure. Used
here it understates the noise by about 1.6× at every height, which is exactly
enough to turn "the fit is at the noise" into "something remains".

`pair_sampling_noise_floor` measures the scatter instead: contiguous chunks,
density per chunk, standard error of the pooled estimate read off the scatter
between them. Nothing is assumed about independence. Contiguous rather than
interleaved because pairs are between *nearby* zeros, so taking every k-th
would destroy the correlation being kept; the cost is a 6–12% conservative
inflation at `T = 2×10⁵` from the chunks sitting at different heights, and none
at all at `T = 3×10⁴`.

**Validated against a known answer.** A Poisson process has `R₂ = 1`
identically, so its deviation from 1 is entirely the estimator — the true mean
absolute deviation is directly observable. The chunk estimator predicts it to
1.6% at `N = 35473` and 5.5% at `N = 298000`. The multinomial formula is out by
2.24× at every `N` from 4320 to 298000.

Still untested: any of this out of sample. Odlyzko's tables at `ℓ = 24.4` and
`ℓ = 48.4` are the obvious control and are not vendored here.

## The zeros say what height they are at

The table above compares two curves and finds the data closer to one. A reader
is entitled to ask whether the winner merely has a shape that happens to fit.
This is the test that shape cannot pass.

Conrey–Snaith is a **one-parameter family**: the curve is fully determined by
`ℓ = log(t/2π)` and nothing else in it is free. So don't ask whether it fits —
ask whether it fits at the *right* `ℓ`. Fit `ℓ` to the measured density in
disjoint height bands and compare the answer against `log(t/2π)` for the band
the data came from.

Fitting is the right tool here and not the usual mistake, for one reason: **`ℓ`
has a known answer independent of the fit**, a property of the heights the
zeros sit at. It is a positive control, not a free parameter. (Contrast
`MomentFit.calibration_error`, which exists because a fitted moment coefficient
came out hundreds of per cent wrong at the `k` where the answer is a theorem.)

And unfolding removes the height *by construction* — `w_n = θ(γ_n)/π` has mean
spacing 1 at every height — so the measured density carries no explicit
dependence on `t`. The only route left for `ℓ` into it is the arithmetic.
Montgomery's curve has no `ℓ` at all.

Over **1,747,146 zeros below `T = 10⁶`**, in seven bands each at most 0.5 wide
in `ℓ`, weighted by each band's own measured uncertainty:

| | slope (1 = tracks `ℓ`) | from 1 | from 0 |
|---|---|---|---|
| six full bands, default anchor | 0.970 ± 0.171 | 0.2σ | 5.7σ |
| across five band-grid anchors | 0.94 ± 0.13 | 0.5σ | 7.2σ |
| **twelve continuous bands to `ℓ` = 13.5** | **0.987 ± 0.056** | **0.2σ** | **17.6σ** |

That error is statistical **and** systematic: ±0.044 from the bands' own
scatter, ±0.034 from the band grid's arbitrary phase, combined in quadrature.
Quoting the statistical part alone gives ±0.044 and 22.2σ, and this table did
exactly that until the anchor was measured on twelve bands. On six it was worth
±0.068 — *larger* than the statistical error; the ladder rungs shrink it because
a rung is a band by construction and does not move when the grid shifts.

The last row reaches `ℓ = 12.5, 13.0, 13.5` on rungs built by
`tools/build-ladder.py` at heights `zero_ordinates` cannot reach, and fitted by
`tools/fit-ladder.py`.

**The rungs used to be the weakest bands in the fit and are now the strongest,
and the height never changed.** Built to a fixed count of 200,000 zeros they
were 0.02–0.06 wide in `ℓ` against a 0.5 cap, and carried σ of 0.61, 0.60 and
1.13 — discounted about 36×, so the climb bought +1.5σ and looked like evidence
that height is the wrong axis. The binding constraint was the *count*, which
this repository was choosing, not the height, which it was blaming.

| `ℓ` | zeros at full width | σ before | σ after |
|---|---|---|---|
| 12.5 | 1,697,444 | 0.614 | **0.128** |
| 13.0 | 2,910,374 | 0.598 | **0.158** |
| 13.5 | 4,982,653 | 1.133 | **0.088** |

All three now beat the best band below `T = 10⁶` (0.191), and the slope's error
falls from 0.171 on those bands alone to 0.045.

**The bands are continuous now.** `T < 10⁶` tops out at `ℓ = 11.42` and the
first rung sat at 12.52, so the regression interpolated across 1.10 in `ℓ` —
over two band widths of nothing. Rungs at 11.77, 12.02 and 12.27 fill it, cost
68–122 s each, and came back with σ 0.153, 0.134 and 0.160: better than every
band below `T = 10⁶`.

**Two exponents, and the difference answers "is height worth it".**

| what is held fixed | |
|---|---|
| within ONE band, height fixed | σ ~ `N^−0.399` |
| across six FULL-WIDTH rungs | σ ~ `N^−0.126` (scatter 0.18) |

Those rungs span a 6.6× range in count — 754k to 4.98M — and their σ runs
0.098–0.187 with no trend worth the name. **Count buys precision; height spends
it at nearly the same rate.** `dR₂/dℓ` shrinks as `ℓ` grows, so the same density
error buys a larger `ℓ` error high up, and climbing pays for its own extra
zeros.

That is why *width* was the lever and height was not. Filling a rung to full
width is pure count at fixed height — the `−0.399` regime. Building a higher
rung at the same width is the `−0.126` regime and buys almost nothing per zero.
The earlier realised ≈ `−0.68` flattered even the first of those, partly because
each starting σ was a single draw carrying 25–28% noise and 1.133 at `ℓ = 13.5`
was a high one.

**The band at `ℓ = 11.42` was never anomalous.** It holds the most zeros below
`T = 10⁶` and reported σ 0.315 against 0.191 for its smaller neighbour, which
this file recorded as unexplained. σ is read off the scatter of 8 chunk fits,
and the standard error of a standard deviation from `k` samples is
`≈1/√(2(k−1))` — 27% at `k = 8`. Rotating the chunk boundaries on each band
gives 0.169–0.389 and 0.188–0.339: one distribution, sampled once each.
Averaged over four placements they are 0.253 and 0.200.

**Averaging did not move the slope** — 0.978 → 0.979, same 0.045 — which says
the per-band noise was already averaging out across nine bands. It moves the
individual weights, which enter as `1/σ²`, so a 27% error in σ is 54% in the
weight. The conclusion never rested on that; reading a single band did.

with a bias of −0.061 ± 0.114.

**This read 0.828 ± 0.119 until the banding was fixed**, and the difference is
two arbitrary choices, not mathematics. `T = 10⁶` falls *inside* the seventh
bin, so it held 0.329 of a width — and that one short band carried **38% of the
slope's leverage**, sitting at the extreme of the range where leverage is
highest, while being the 2.5σ outlier. Separately, the band grid is anchored at
whichever zero happens to be lowest; shifting it within one width moves the
slope across 0.875–1.037. That spread, ±0.068, is a systematic and now sits
inside the quoted error.

"Not zero" survives and strengthens, 7.0σ → 7.2σ. "1.4σ from 1" does not: every
full-band anchor lands within 1.1σ of exactly 1.

**This first read 1.058 ± 0.112, and it was biased.** Two review findings:
equal-*count* bands, whose lowest spanned `ℓ = 4.15..9.47` against 0.06 for the
highest and carried the most leverage in the regression; and one pooled
residual spread across bands that are not equally precise. Fitting a synthetic
mixture shows the first is worth −0.104 in `ℓ` at that width and −0.008 at 0.5.
Correcting both moved the slope by two of its own error bars.

**One band is an outlier and it is not explained.** The top band fits 0.466
below its true `ℓ` against its own σ of 0.189 — 2.5σ — and dropping it takes the
slope to 0.970. Not an edge artefact: trimming five units of `t` off the top
leaves it at −0.470. Every reading holds with or without it; the point estimate
does not.

### Three ways this could have been true without any arithmetic

Each is ruled out by construction rather than by argument, and each is on the
record in `HeightRecovery`:

- **An estimator that returns ascending numbers for ascending input.** Shuffling
  which fitted value belongs to which band destroys the slope — the permuted
  null is centred on zero and 52 of 20,000 shuffles reach the measurement.
- **Sampling noise varying across the bands.** The bands hold an equal *count*
  of zeros, not an equal span of height, so the noise floor is identical in
  every one and a trend cannot come from precision improving up the range.
- **Bands sharing data.** They are disjoint. Nested ranges in this repository
  once correlated at r = 0.98 by construction and meant nothing.

The estimator itself is checked against an answer known exactly: fed the curve
at a known `ℓ` it returns it to 0.004, and fed one deliberately between two grid
points it lands between them rather than snapping to the mesh.

### What the out-of-sample test would have cost, and why it is not here

The obvious control — Odlyzko's tables at `ℓ = 24.4` and `ℓ = 48.4`, sixteen
orders of magnitude up — **is not resolvable**, and one calculation says so
before any of it is attempted:

| | curves apart | noise floor at N = 10,000 |
|---|---|---|
| index 10¹² (`ℓ` = 24.5) | 0.00359 | 0.02165 |
| index 10²¹ (`ℓ` = 44.6) | 0.00123 | 0.02165 |
| index 10²² (`ℓ` = 46.8) | 0.00113 | 0.02165 |

The signal is 6× below the noise at 10¹² and 18× below above it. Telling the
curves apart there would take about 364,000 zeros at that index; Odlyzko
published 10,000. The `ℓ` recovery above is the test that *was* available, and
it spans `ℓ` from 8.6 to 11.9 inside data we can compute.

## What building it looked like

1. `A(x)` and `B(x)` with the truncation *measured* rather than chosen — vary
   the prime limit and watch the answer settle, as `ARITHMETIC_PRIME_LIMIT` in
   `moments.py` was fixed against the closed form `a_2 = 1/ζ(2)`.
2. The assembly, with the `ℓ → ∞` control above as a test.
3. Measured pair correlation against both curves — the leading Montgomery term
   and the full Conrey–Snaith form — exactly as `check_fourth_moment` compares
   against the proven and naive-RMT polynomials.
4. The record refusing a rigorous confidence: the ratios conjecture is a
   conjecture, so this is a measurement against an unproved prediction.

All four are done. The prototype this page was written to preserve is gone
from the scratchpad, as predicted; the control table it printed is now a test
(`test_the_page_control_table_is_reproduced`), so the numbers above are a
record rather than a memory.
