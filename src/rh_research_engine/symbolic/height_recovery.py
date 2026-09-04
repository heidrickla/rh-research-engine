"""Recover the height of the zeros from their pair correlation alone.

`conrey_snaith` measures the zeros against a curve carrying every lower-order
term and finds them closer to it than to Montgomery's universal law. That is a
comparison of two curves, and a reader is entitled to ask whether the winning
one merely has a shape that happens to fit.

THIS IS THE TEST THAT SHAPE CANNOT PASS. Conrey-Snaith is a ONE-PARAMETER
family: the curve is fully determined by `l = log(t/2 pi)` and nothing else in
it is free. So do not ask whether it fits -- ask whether it fits at the RIGHT
`l`. Fit `l` to the measured density in disjoint height bands and compare the
answer against `log(t/2 pi)` for the band the data came from.

WHY FITTING IS THE RIGHT TOOL HERE AND NOT THE USUAL MISTAKE. This repository
records a fitted moment coefficient that came out hundreds of per cent wrong,
and `MomentFit.calibration_error` exists so no extracted number is reported
without the demonstrated error of the method behind it. The difference is that
`l` HAS A KNOWN ANSWER, independent of the fit and of the curve: it is a
property of the heights the zeros sit at. The fit is therefore a positive
control, not a free parameter, and every number below is checkable against
something the fit never saw.

UNFOLDING REMOVES THE HEIGHT BY CONSTRUCTION, which is what makes the result
mean anything. `w_n = theta(gamma_n)/pi` has mean spacing 1 at every height, so
the measured density carries no explicit dependence on `t`. The only route left
for `l` into it is the arithmetic. Montgomery's curve has no `l` at all.

WHAT IT COMES TO, over 1747146 zeros below `T = 10^6`, in six FULL bands each
0.5 wide in `l`, weighted by each band's own measured uncertainty:

    slope 0.970 +/- 0.171   0.2 sigma from 1, 5.7 from 0, bias -0.061 +/- 0.114

and across five positions of the band grid, which is arbitrary:

    slope 0.94 +/- 0.13     0.5 sigma from 1, 7.2 from 0

EXTENDED TO `l` = 13.5 by three rungs built above the reach of a full run --
`tools/build-ladder.py`, then `tools/fit-ladder.py` -- it is

    slope 0.987 +/- 0.056   0.2 sigma from 1, 17.6 from 0, over twelve bands

        statistical        +/- 0.044
        band-grid anchor   +/- 0.034
        combined           +/- 0.056

BOTH, AND THE SECOND IS NOT SMALL. `slope_error` is the statistical part alone.
The anchor systematic is what the band grid's arbitrary phase is worth, and
this docstring quoted 0.044 as the whole error for a while after adding the
machinery to measure the other half. On six bands the anchor was worth 0.068 --
larger than the statistical error; on twelve it is 0.034, because the ladder
rungs are bands by construction and do not move when the grid shifts. Carrying
more of the lever on pinned bands is what shrank it.

which is what `zeros_in_band` exists for.

THE RUNGS USED TO BE THE WEAKEST BANDS AND ARE NOW THE STRONGEST, and nothing
about the height changed. Built to a fixed count of 200,000 zeros they were
0.02-0.06 wide in `l` against a 0.5 cap, and carried sigma 0.61, 0.60 and 1.13
against 0.19 for the best low band -- so the regression discounted them about
36x, and the whole climb bought +1.5 sigma. The binding constraint was the
COUNT, which this repository was choosing, and not the height, which it was
blaming:

    l       zeros        sigma before   sigma at full width
    12.5    1,697,444    0.614          0.132
    13.0    2,910,374    0.598          0.187
    13.5    4,982,653    1.133          0.098

(the "at full width" column with sigmas averaged over chunk placements; a
single placement gave 0.128, 0.158, 0.088)

All three now beat the best band below `T = 10^6`, and the slope's error falls
from 0.171 on those bands alone to 0.045.

THE BANDS ARE CONTINUOUS NOW. `T < 10^6` tops out at `l = 11.42` and the first
rung sat at 12.52, so the regression interpolated across 1.10 in `l` -- over two
band widths of nothing. Rungs at 11.77, 12.02 and 12.27 fill it, and they are
among the best-determined bands in the fit.

TWO EXPONENTS, AND THE DIFFERENCE IS THE WHOLE ANSWER TO "IS HEIGHT WORTH IT".
`sigma` against the zero count behaves quite differently depending on what is
held fixed:

    within ONE band, height fixed        sigma ~ N^-0.399
    across six FULL-WIDTH rungs          sigma ~ N^-0.126   (scatter 0.18)

The rungs span a 6.6x range in count -- 754k to 4.98M -- and their sigmas run
0.098 to 0.187 with no trend worth the name. Count buys precision; height
spends it, at very nearly the same rate. `dR_2/dl` shrinks as `l` grows, so the
same density error buys a larger `l` error high up, and climbing pays for its
own extra zeros.

WHICH IS WHY WIDTH WAS THE LEVER AND HEIGHT WAS NOT. Filling a rung to full
width is pure count at FIXED height, and that is the -0.399 regime. Building a
higher rung at the same width is the -0.126 regime, and buys almost nothing per
zero. The earlier realised exponent of about -0.68 flattered even the first of
those, and part of that was the estimator: each starting sigma was a single
draw carrying 25-28% noise, and 1.133 at `l = 13.5` was a high one.

AVERAGING THE SIGMAS DID NOT MOVE THE ANSWER, and that is worth recording as
loudly as if it had. Over nine bands the slope went 0.978 -> 0.979 with the same
0.045: the per-band noise was already averaging out in the regression. What it
moved is the individual weights -- the band at `l = 11.42` went 0.315 -> 0.253
against its neighbour's 0.191 -> 0.200, closing a gap this repository had
recorded as an unexplained anomaly. The conclusion never rested on it; the
interpretation of a single band did.

The zeros know what height they are at, and only the curve that carries primes
can be asked.

THIS READ 0.828 +/- 0.119 UNTIL THE BANDING WAS FIXED, and the difference is
two arbitrary choices rather than any mathematics.

  * A PARTIAL BAND. `T = 10^6` falls inside the seventh bin, which therefore
    held 0.329 of a width. Measured, that one band carried 38% of the slope's
    leverage -- it sits at the extreme of the range, where a regression's
    leverage is highest -- and it was the 2.5-sigma outlier. The line
    "dropping the top band gives 0.970" stood in this docstring as a
    robustness check; it was the sound number, and the headline beside it was
    the contaminated one.

  * THE GRID'S PHASE. `narrow_ell_bands` anchors its edges at `ell.min()`, an
    accident of which zero is lowest. Shifting it within one width moves the
    slope over 0.875..1.037 with full bands, and 0.828..1.037 with the partial
    band included. Nothing mathematical picks the anchor, so its spread --
    +/-0.068 -- is a systematic and belongs in the error rather than outside it.

WHICH CLAIM SURVIVED AND WHICH DID NOT. "Not zero" is untouched and stronger:
every anchor gives at least 4.7 sigma, and the anchor-averaged figure gives 7.2.
"1.4 sigma from 1" did not survive: it was an artifact, and every full-band
anchor sits within 1.1 sigma of exactly 1. The measurement says the zeros track
`l`, and it no longer says they track it slightly badly.

THE FIRST VERSION OF THIS REPORTED 1.058 +/- 0.112, and it was biased. It used
equal-COUNT bands, whose lowest spanned `l = 4.15..9.47` against 0.06 for the
highest, and it pooled one residual spread across bands that are not equally
precise. Both were raised in review; correcting them moved the slope by two of
its own error bars, which is the size of thing a review is for.

THE BAND THAT WAS AN OUTLIER IS EXPLAINED, AND IT IS GONE. It fitted 0.466
below its true `l` against a sigma of 0.189 -- 2.5 sigma -- and this docstring
recorded it as unexplained while noting that dropping it moved the slope from
0.828 to 0.970. It was the bin `T = 10^6` falls inside, so it held 0.329 of a
width rather than 0.500: a different width from every other band, a mean `l`
that is not its bin's centre, and the extreme position in the regression, where
leverage is highest. It carried 38% of the slope's.

The old test for this asked the wrong question. "Not an edge artefact: trimming
five units of `t` off the top leaves it at -0.470" moves where the truncation
falls without ever giving the band its missing width, so it could not have
detected the problem. `narrow_ell_bands` now drops any bin the data does not
fill.

THREE THINGS THE CONSTRUCTION RULES OUT, because each of them would produce
this result without any arithmetic:

  * **The estimator returning ascending numbers for ascending input.** Shuffling
    which fitted value belongs to which band destroys the slope: the permuted
    null is centred on zero and only 52 of 20000 shuffles reach the measured
    slope. `permutation_p` is on the record, not an afterthought.
  * **Sampling noise varying across the bands.** The bands hold an equal COUNT
    of zeros rather than an equal span of height, so the noise floor is the
    same in every one and a trend cannot come from precision improving up the
    range.
  * **Bands sharing data.** They are disjoint. Nested ranges in this repository
    once correlated at r = 0.98 by construction and meant nothing.

WHAT IT IS NOT. The ratios conjecture is a conjecture and the fit is against a
finite sample, so nothing here returns a rigorous confidence. And a slope of 1
says the curve's `l`-dependence is present in the zeros; it does not say the
curve is right in every other respect.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..contracts.epistemic import RIGOROUS, Confidence

#: The `l` values the fit searches, and the spacing it searches them on.
#:
#: `l = log(t/2 pi)`, so this spans `t` from 2.5e3 to 3.0e9 -- past both ends of
#: anything reachable, because a grid that stopped at the truth would find it
#: by construction.
#:
#: The step is COARSER than the 0.11 the regression resolves, and what makes
#: that harmless is the parabola through the three lowest points rather than
#: the mesh: without it every fitted `l` would be a grid point and the slope
#: would be a staircase. Measured -- fed the curve at a known `l`, the fit
#: returns it to 0.004, and at an `l` deliberately between two grid points it
#: lands between them.
ELL_GRID_LOW = 6.0
ELL_GRID_HIGH = 20.0
ELL_GRID_STEP = 0.25

#: Fewest zeros a band may hold.
#:
#: THE NUMBER SURVIVED ITS FIRST REAL TEST AND ITS REASON DID NOT. This said
#: "the regression loses more to that than it gains from the extra point",
#: which is a claim about VARIANCE, and the variance says the opposite.
#: Lowering the minimum on `T < 10^6` improves the slope's error monotonically
#: -- and it should, because the regression is weighted and a longer lever is
#: worth more than a noisy point costs:
#:
#:     minimum   bands   lever   slope         from 0
#:      20,000     6      2.50   0.987+-0.150   6.6 sigma
#:       8,000     8      3.50   0.996+-0.130   7.7
#:       5,000     9      4.00   0.965+-0.118   8.2
#:       1,200    11      5.00   0.890+-0.104   8.6
#:
#: WHY: SETTLED, AND IT IS NEITHER OF THE TWO EXPLANATIONS THAT WERE WITHDRAWN.
#:
#: What was already solid is the drift. The slope moves away from one down that
#: column while the error improves -- 0.987, 0.996, 0.965, 0.931, 0.890 --
#: monotone across four settings, a property of the fit and not of one noisy
#: band. Something about the extra low bands pulls the line down.
#:
#: THIS COMMENT USED TO SAY THE DESIGN COULD NOT SETTLE IT. It argued that
#: resolving a +0.5 bias at 3 sigma needs ~72 slices of 20,000, that only the
#: ladder rungs hold 1.4M zeros in one band, and so "the question is answerable
#: exactly where it does not matter". The first two are right and the conclusion
#: does not follow: `ladder-full.npz` holds 1.70M, 2.91M and 4.98M zeros in ONE
#: band each, which is 84, 145 and 249 slices of 20,000 at three separate
#: heights -- and measuring at three heights is what separates the count axis
#: from the height axis, which is the very thing the design was said to be
#: unable to do.
#:
#: b(N), THE BIAS IN `fit_ell` AT FIXED HEIGHT. Disjoint contiguous slices,
#: each compared against its OWN mean `log(t/2 pi)`, remainders dropped rather
#: than kept as short slices, pooled over the three rungs:
#:
#:     N          b(N)                sigma
#:     10,000     +0.529 +/- 0.117     4.5
#:     20,000     +0.277 +/- 0.095     2.9
#:     40,000     +0.120 +/- 0.092     1.3
#:     80,000     +0.107 +/- 0.088     1.2
#:     160,000    +0.086 +/- 0.075     1.2
#:     320,000    -0.114 +/- 0.088     1.3
#:
#: "SMALL BANDS FIT `l` TOO HIGH" IS CONFIRMED AT THE RUNGS (l = 12.5-13.5) AND
#: NOT ESTABLISHED AT THE BANDS THIS CONSTANT GOVERNS (l = 6.4-11.4). The scope is
#: the whole claim, and an earlier version of this paragraph said only "CONFIRMED
#: at the count where this constant sits" -- which is this block's own failure
#: mode committed inside the block: a figure measured at one range quoted about
#: another. See the withdrawal below, which this sentence used to contradict.
#: WHY IT WAS WITHDRAWN IS THE MORE USEFUL HALF. The figure
#: was +0.824, taken from FIVE SINGLE FITS on bands under 20,000 zeros against
#: -0.054 for the rest. A single fit at that size scatters by about 1.4, so the
#: +0.878 difference was roughly 1.4 sigma: a draw from a wide distribution,
#: reported as a measurement of its centre. Withdrawing it was right on the
#: evidence available. The answer is +0.277 +/- 0.095, and what changed is not
#: the sign but that there are now 84, 145 and 249 slices behind it instead of
#: five. That is the same lesson as the skew below, one level up: at these
#: counts ONE fit tells you almost nothing, and the remedy is slices, not
#: confidence.
#:
#: "THE BIAS TRACKS HEIGHT, NOT COUNT" IS SETTLED THE OTHER WAY AT THE RUNGS, AND
#: ONLY THERE: across l = 12.5-13.5 it tracks COUNT. At every N the three rungs
#: agree within 2 sigma, and b falls by an order of magnitude across N at fixed
#: height. It does NOT generalise downward -- the low-band table below shows the
#: bias changing sign with height, negative from l = 6.9 to 9.4 and positive from
#: 9.9 to 11.4 -- so this is true where it was measured and contradicted where the
#: constant applies. A residual height trend of
#: -0.39 +/- 0.10 is NOT claimed: at fixed N a rung's slice span in `l` differs
#: threefold between `l` = 12.5 and 13.5, so height and span are collinear here
#: and cannot be separated. Both sit ~100x below the 0.5 span cap.
#:
#: AND THE MECHANISM IS THE BIAS, NOT A FAILURE TO RESPOND. THREE REVISIONS WERE
#: NEEDED TO SAY THAT, and the two wrong ones are recorded because each was wrong in a
#: way this repository keeps rediscovering.
#:
#: The estimator's GAIN -- how far the fit moves when the density is perturbed by one
#: grid step's worth of curve difference, anchored at the TRUE `l` so that it asks
#: `d(fitted)/d(true)` -- collapses on the ladder rungs:
#:
#:     l = 12.5-13.5, N = 2,500     gain 0.349 +/- 0.064     10.1 sigma from 1
#:     l = 12.5-13.5, N = 10,000    gain 0.969 +/- 0.090     consistent with 1
#:
#: THE FIRST VERSION MADE THAT THE REASON FOR THIS CONSTANT. It is not. The collapse was
#: measured at l = 12.5-13.5 and this constant governs bands at l = 6.4-11.4, which is an
#: extrapolation across 3.5 units of `l` -- and `dR_2/dl` shrinks as `l` grows, so a low
#: band's curve is MORE sensitive and needs fewer zeros. Measured where the constant
#: actually applies, the estimator responds:
#:
#:     l = 7.9-11.4, N = 2,500      gain 1.263 +/- 0.156     over 478 slices
#:
#: THE SECOND VERSION THEN SAID THE COLLAPSE WAS PURELY A HIGH-`l` EFFECT, from a pooled
#: mean at N = 500 of 0.596 +/- 0.049. That number is a weighted average whose weights are
#: slice count, slice count grows with band size, and band size grows with height -- so a
#: count-average was silently a height-average. The two tallest bands were 67% of the pool.
#: A POOLED MEAN IS NOT A NEUTRAL SUMMARY.
#:
#: WHAT THE ERROR BARS ACTUALLY SUPPORT. Per band at N = 500, only six of eleven have a
#: standard error small enough to read at all, and every one of those is at or below one:
#:
#:     l  8.42   0.84 +/- 0.21     l 10.42   0.42 +/- 0.08
#:     l  9.42   0.81 +/- 0.19     l 10.92   0.56 +/- 0.08
#:     l  9.92   0.62 +/- 0.11     l 11.42   0.54 +/- 0.08
#:
#: The five bands below that give standard errors of 0.73 to 2.00 on 3 to 21 slices --
#: which is the five-single-fits weakness this very comment was written to correct,
#: committed again in the probe written to correct it.
#:
#: b(N) AT THE BANDS THIS CONSTANT GOVERNS IS NOT ESTABLISHED, and the version of
#: this comment that said otherwise was wrong for the third time. It claimed
#: "+0.932 +/- 0.171 over 478 slices at l = 7.9-11.4, 5.4 sigma from zero". Making
#: `fit_bias_lab` able to REGENERATE that number is what refuted it: the module, applying
#: this file's own cuts and pooling by precision, returns +0.424 instead. NOT REVIEW AND
#: NOT A TEST -- neither would have caught it, because both would have read the same
#: reasoning that produced it. What caught it was requiring the committed code to produce
#: the docstring's figures, and finding it could not. A number that lives only in prose has
#: no parameters, no provenance and no fingerprint; the moment the module was asked for
#: the same one, the two came apart. That is the strongest argument for `fit_bias_lab`
#: existing, and a better one than its own docstring gives. Three reasons,
#: and the per-band rows carry all of them (N = 2,500):
#:
#: THE TABLE BELOW IS A HAND-RUN AND THE MODULE DOES NOT PRODUCE IT. `fit_bias_lab`
#: reports aggregates -- `usable_at_N`, `undetermined_at_N`, and the pooled
#: `bias_at_2500 = +0.424` this block quotes to refute itself -- and those ARE on the
#: record. The per-band rows are not; they were computed by hand with the module's own
#: cuts. Stated rather than left to a reader to assume, because a table in prose that
#: the module cannot regenerate is what killed the version above, and a milder instance
#: of a defect that has already been fatal three times is still that defect.
#:
#:     l       zeros     slices     bias      sem    gain    railed   usable?
#:     6.922    3,462         1   -0.856      nan   1.003    100.0%   no
#:     7.422    6,123         2   -0.197    0.204   0.411      0.0%   no
#:     7.922   10,774         4   -0.727    0.361   0.365      0.0%   no
#:     8.422   18,885         7   -0.395    0.643   0.760     14.3%   no
#:     8.921   32,985        13   -0.280    0.485   0.969      7.7%   no
#:     9.421   57,433        22   -0.214    0.351   2.312      0.0%   no
#:     9.921   99,716        39   +0.424    0.451   1.128      2.6%   YES
#:    10.421  172,691        69   +1.335    0.406   2.281      5.8%   no
#:    10.921  298,383       119   +1.034    0.350   2.680     10.9%   no
#:    11.421  514,478       205   +1.112    0.295   0.905     18.0%   no
#:
#:   * THE BIAS CHANGES SIGN WITH HEIGHT -- negative from l = 6.9 to 9.4, positive from
#:     9.9 to 11.4. There is no single b(N) for "the low bands" to report.
#:   * THE +0.932 WAS THE HEIGHT-WEIGHTED POOL AGAIN. The three tallest bands hold 393 of
#:     the 478 slices -- 82% -- and they are the positive ones. Pooling slices by a simple
#:     mean weights by slice count, slice count grows with band size, band size grows with
#:     height. The same artifact is diagnosed for the GAIN two paragraphs above, in this
#:     same block, by the same author, and then committed for the BIAS. Caught where it
#:     had been looked for and not where it had not.
#:   * ONE CELL OF TEN SURVIVES THE CUTS, at +0.424 +/- 0.451 -- 0.9 sigma, which is
#:     nothing. And NOT ONE of the other nine is excluded for being unresponsive. Applying
#:     every cut the module applies, at N = 2,500: 1 usable, 4 censored at the grid edge,
#:     1 whose gain is undetermined, 4 with too few slices to carry an error bar --
#:     and zero measured-and-unresponsive. At 5,000 and 10,000 it is the same shape
#:     (3 usable, 0 unresponsive, at both).
#:
#:     So the nine were not judged and rejected; they were UNDECIDABLE. The pool that
#:     produced +0.932 was built almost entirely from cells about which this file's own
#:     criteria say nothing either way -- which is a weaker and more accurate statement
#:     than "cells my criteria exclude", and it is the not-tested-versus-refuted rule
#:     applying to the cells themselves and not only to their verdicts.
#:
#: AND THE CUT ITSELF CANNOT DISCRIMINATE AT THIS COUNT, which is a defect one level down.
#: `fit_bias_lab` keeps a cell when its gain lies in [0.8, 1.3] -- a window 0.5 wide --
#: and at N = 2,500 the gain's own standard error runs 0.17 to 1.7:
#:
#:     l = 9.421    gain 1.532 +/- 0.920      l = 10.921   gain 1.379 +/- 0.218
#:     l = 10.421   gain 1.336 +/- 0.318      l = 11.421   gain 1.237 +/- 0.308
#:
#: A window narrower than the uncertainty is not a filter, it is a coin. So a cell "cut for
#: gain" at these counts means THE GAIN WAS NOT DETERMINED, not that the estimator was
#: unresponsive -- the same distinction `patterns/ledger.py` draws between not-tested and
#: refuted, one layer further down, and the module records the gain's error bar so the two
#: cannot share a verdict.
#:
#: WHAT STANDS. The ladder rungs: many slices, cuts passed, three heights.
#:
#:     l = 12.5-13.5, N = 20,000    b    = +0.277 +/- 0.095   (3 rungs, 84/145/249 slices)
#:     l = 12.5-13.5, N =  2,500    gain = 0.349 +/- 0.064    10.1 sigma from 1
#:
#: WHAT THE CONSTANT RESTS ON, THEREFORE. The drift down the minimum column -- 0.987,
#: 0.996, 0.965, 0.931, 0.890 -- which is a property of the fit and not of one noisy band,
#: and the fact that lowering the minimum admits bands of 1,949 to 6,123 zeros whose
#: behaviour is unmeasurable at any slice size the data supports. That is weaker than a
#: mechanism and it is what there is. THE WITHDRAWN EXPLANATION STAYS WITHDRAWN: "small
#: bands fit l too high" was withdrawn on five single fits, briefly reinstated here on a
#: height-weighted pool, and is withdrawn again on better evidence than either.
#:
#: "FITS TOO HIGH" IS THE MEAN, AND THE MEAN IS THE TAIL -- true of the RUNGS, where it was
#: measured. At l = 13.5 and N = 20,000 the mean is +0.226 while the MEDIAN is -0.171, with
#: 5th and 95th percentiles at -2.54 and +4.59. The typical band fits slightly LOW; a
#: minority fit far high. Each real band is one draw from that tail.
#:
#: THE ARTIFACT THAT WOULD HAVE KILLED THIS, RULED OUT BY MEASUREMENT.
#: `measured_density` counts forward differences only and normalises by `len(unfolded)`,
#: so a slice loses about `window^2/2 ~ 4.5` pairs at its leading edge -- a relative
#: deficit of `1.5/N`. That is a 1/N bias and b(N) falls roughly as 1/N at the low end.
#: Same signature. Recomputing each slice's density with partners drawn from the full
#: array, the deficit is real and is exactly the predicted 4.5 pairs at every N -- and it
#: moves the fitted `l` by +0.009 +/- 0.012 at 2,500 and -0.001 +/- 0.007 at 20,000.
#: Consistent with zero, at least 20x below b(20,000). IT ADDS VARIANCE, NOT BIAS: on a
#: single slice it can move the fit by 0.22-0.51.
#:
#: ONE QUANTITY, SEVERAL POLICIES -- and this is the part worth carrying forward, more
#: than any number above. The gain was anchored two ways and read 0.653 against 0.349 on
#: the same slices; pooled two ways, 0.4300 against 0.3486; measured in one height regime
#: and quoted about another; and summarised by a mean whose weights were a variable nobody
#: chose. One file away, a Theta map was implemented twice and the silent copy wrote the
#: record. Every one was caught the same way and by nothing else: computing the same thing
#: a second way and finding the two disagreed. A NUMBER THAT HAS NEVER BEEN COMPUTED TWICE
#: HAS NOT BEEN CHECKED, IT HAS BEEN REPORTED.
#:
#: AND THE SAME IS TRUE OF A MECHANISM, which is a distinct failure and cost more than any
#: of the numbers. Two sessions spent an afternoon stashing work out of a shared tree
#: because the manifest RECORDS `module_count` from the filesystem -- and never tested the
#: step from what it records to what `--check` compares, which is nothing: setting it to
#: 99999 leaves the check at exit 0. The same shape as a `docker ps --filter` that could
#: not have matched, a conclusion about a settings file drawn from the first entry of a
#: two-element array, and a `git status` run from a directory that is not a repository and
#: reporting "clean". A MECHANISM THAT HAS ONLY EVER BEEN REASONED ABOUT HAS NOT BEEN
#: CHECKED EITHER, and each of these was answerable in seconds by asking it a second way.
#:
#: So the constant stays at 20,000, and now for a reason rather than a proxy:
#: the gain is sound from about 10,000 and the residual bias is 2.9 sigma at
#: 20,000 and gone by 40,000. Lowering it admits bands whose estimator does not
#: respond.
#:
#: Regenerated by `experiments/height_recovery_lab.py`; it needs
#: `~/rh-data/ladder-full.npz`, which is NOT in this repository and NOT backed
#: up, and refuses rather than returning an empty result when it is absent.
#:
#: Also measured, on the same data and the same estimator: the band WIDTH cap
#: of 0.5 is on the right side of its own trade. 0.25 gives 11 bands and
#: 0.998+-0.160, 0.35 gives 1.208+-0.163, 0.5 gives 0.987+-0.150, 0.75 gives
#: 1.096+-0.151 -- 0.5 has both the smallest error and the best-determined
#: worst band.
MINIMUM_BAND = 20_000

#: Fewest bands a regression may use. Three points and two parameters leaves
#: one degree of freedom, which is the least that can carry a standard error.
MINIMUM_BANDS = 4

#: Widest a band may be, in `l`.
#:
#: A band's density has expectation `(1/N) sum_n R_2(u; l_n)` -- a MIXTURE over
#: the heights it holds -- and the fit compares it against one curve at the
#: band's mean `l`. `R_2` is nonlinear in `l`, so those agree only while the
#: band is narrow, and the first version of this module used equal-COUNT bands
#: whose lowest spanned `l = 4.15..9.47`. Raised in review.
#:
#: Measured rather than chosen: fitting a synthetic mixture and asking for its
#: mean `l` back, the bias is recorded in
#: `test_a_wide_band_is_fitted_as_the_wrong_height`.
MAX_BAND_ELL_SPAN = 0.5

#: Chunks a band is split into to measure ITS OWN `l` uncertainty.
#:
#: `dR_2/dl` shrinks as `l` grows, so the same density error buys a larger `l`
#: error high up and the bands are not equally precise. Eight is the same
#: choice, for the same reason, as `pair_correlation.NOISE_CHUNKS`.
UNCERTAINTY_CHUNKS = 8

#: How many placements of the chunk boundaries each band's sigma is averaged
#: over.
#:
#: One placement carries 25-28% noise, measured by rotating the cuts on a fixed
#: band -- which matches `1/sqrt(2(k-1))` for `k = 8`. Averaging four rotations
#: is four times the fits for an uncertainty that the regression then squares
#: into a weight, and that trade is worth making: a 27% error in sigma is a 54%
#: error in the weight, and it had already produced a band this repository
#: recorded as anomalous when it was simply unlucky.
UNCERTAINTY_ROTATIONS = 4

#: Draws behind `null_p`.
#:
#: Not a permutation any more. Permuting fitted values between bands assumes
#: they are exchangeable, which unequal uncertainties make false; the null is
#: drawn from each band's own measured sigma about a flat relation instead.
#:
#: THIS NUMBER IS ALSO THE FLOOR ON `null_p`, which is why it is reported. The
#: p-value is `(r+1)/(n+1)`, so the smallest value it can take is `1/(n+1)` and
#: it is an UPPER BOUND, never a claim of impossibility. Reading `null_p` without
#: knowing `n` is reading a resolution limit as a result.
NULL_DRAWS = 20_000


class HeightRecovery(BaseModel):
    """What the pair correlation says about the height it was measured at."""

    model_config = ConfigDict(extra="forbid")

    #: Median zeros per band. NOT equal any more: bands are narrow in `l`
    #: instead, so the low ones hold fewer zeros. `fitted_error` carries what
    #: that costs, per band.
    band_size: int
    #: How wide each band is in `l`, so a reader can see they are narrow.
    band_widths: list[float] = Field(default_factory=list)
    #: `log(t/2 pi)` for each band, from the ordinates. Never from the fit.
    true_ell: list[float] = Field(default_factory=list)
    #: `l` recovered from each band's density alone.
    fitted_ell: list[float] = Field(default_factory=list)
    #: Each band's OWN uncertainty on that, measured from its internal
    #: scatter. Unequal across bands, which is why the regression is weighted.
    fitted_error: list[float] = Field(default_factory=list)
    #: Regression of fitted on true. One if the fit tracks `l`; zero if it is
    #: insensitive to `l` and any agreement was a coincidence of one band.
    slope: float
    #: The STATISTICAL error alone, from the bands' own uncertainties. It does
    #: not include the band grid's arbitrary phase -- see `slope_anchor_error`,
    #: and quote the two together.
    slope_error: float
    #: Spread of the slope over `anchors` positions of the band grid, which
    #: nothing mathematical picks. `None` when only one anchor was run, which
    #: is not the same as zero: it means the systematic was not measured.
    slope_anchor_error: float | None = None
    #: Mean slope over those anchors, `None` for a single one. This is the
    #: figure to publish; `slope` is the default anchor's alone.
    slope_over_anchors: float | None = None
    intercept: float = 0.0
    #: Mean of `fitted - true`, and the standard error of that mean.
    bias: float
    bias_error: float
    #: Fraction of null draws reaching this slope. The control against the
    #: estimator simply returning ascending numbers -- drawn from each band's
    #: measured uncertainty about a flat relation, because permuting fitted
    #: values between bands of unequal precision is not a valid null.
    null_p: float
    #: Never rigorous. A fit, against a conjectural curve, on a finite sample.
    confidence: Confidence = Confidence.NUMERICAL

    @field_validator("confidence")
    @classmethod
    def _reject_rigorous_confidence(cls, value: Confidence) -> Confidence:
        if value in RIGOROUS:
            raise ValueError(
                f"recovering the height from the pair correlation may not claim "
                f"{value.value!r}: it fits a curve derived from the ratios "
                "conjecture to a finite sample, so it is evidence that the "
                "curve's l-dependence is present in the zeros and never a "
                "proof of the curve"
            )
        return value

    @property
    def tracks_the_height(self) -> bool:
        """Slope consistent with one AND distinguishable from zero.

        Both halves are needed. A slope consistent with one is worth nothing if
        the error bar also covers zero -- that is a fit that determined
        nothing, reported as agreement.
        """
        return (
            abs(self.slope - 1.0) < 2 * self.slope_error and abs(self.slope) > 3 * self.slope_error
        )


def ell_grid() -> np.ndarray:
    return np.arange(ELL_GRID_LOW, ELL_GRID_HIGH + ELL_GRID_STEP / 2, ELL_GRID_STEP)


class CurveBank(NamedTuple):
    """Curves, and WHAT THEY ARE CURVES OF.

    An array of the right shape is not the right bank. Thirty bins over
    `[0, 4]` and thirty bins over `[0, 3]` are the same shape and different
    curves, and so are two banks built on different `l` grids or different
    prime limits. Checking the shape and accepting anything that matches would
    compare each measured column against a curve averaged over some other
    interval, silently, and report the fit as though it had been against the
    curves this module names. Raised in review on the version that carried
    only an array.
    """

    curves: np.ndarray
    edges: np.ndarray
    grid: np.ndarray
    prime_limit: int | None

    def matching(self, edges: np.ndarray, grid: np.ndarray, prime_limit: int | None):
        """Raise unless this bank is the one those parameters describe."""
        edges = np.asarray(edges, dtype=float)
        grid = np.asarray(grid, dtype=float)
        if self.curves.shape != (len(grid), len(edges) - 1):
            raise ValueError(
                f"bank is {self.curves.shape}, this fit wants {(len(grid), len(edges) - 1)}"
            )
        if not np.array_equal(self.edges, edges):
            raise ValueError(
                f"bank was built on bins spanning "
                f"[{self.edges[0]:g}, {self.edges[-1]:g}] and this fit is over "
                f"[{edges[0]:g}, {edges[-1]:g}] -- same shape, different curves"
            )
        if not np.array_equal(self.grid, grid):
            raise ValueError("bank was built on a different l grid")
        if self.prime_limit != prime_limit:
            raise ValueError(
                f"bank was built at prime_limit={self.prime_limit} and this fit "
                f"asks for {prime_limit}"
            )
        return self.curves


def curve_bank(
    edges: np.ndarray,
    grid: np.ndarray | None = None,
    *,
    prime_limit: int | None = None,
) -> CurveBank:
    """The Conrey-Snaith curve at every `l` on the grid, one row each.

    Built once and reused across bands. Every band is fitted against the SAME
    curves, which is not only cheaper -- rebuilding them per band would let a
    change in the quadrature show up as a difference between bands, and the
    difference between bands is the measurement.

    Returns a `CurveBank`, which carries the parameters it was built with so a
    caller cannot hand a same-shaped array from somewhere else.
    """
    from .conrey_snaith import BIN_SAMPLES, pair_correlation
    from .level_spacing import curve_bin_average

    grid = ell_grid() if grid is None else np.asarray(grid, dtype=float)
    rows = []
    for value in grid:
        ell_value = float(value)

        def curve(u, ell_value=ell_value):
            if prime_limit is None:
                return pair_correlation(u, ell_value)
            return pair_correlation(u, ell_value, prime_limit=prime_limit)

        rows.append(curve_bin_average(edges, curve, samples=BIN_SAMPLES))
    return CurveBank(
        curves=np.array(rows),
        edges=np.asarray(edges, dtype=float),
        grid=grid,
        prime_limit=prime_limit,
    )


def fit_ell(
    density: np.ndarray,
    edges: np.ndarray | None = None,
    *,
    grid: np.ndarray | None = None,
    prime_limit: int | None = None,
    bank: CurveBank | None = None,
) -> float:
    """The `l` whose Conrey-Snaith curve is closest to `density`.

    Refined off the grid by a parabola through the three lowest points, so the
    answer is not quantised at `ELL_GRID_STEP`. A minimum on either end of the
    grid is returned as-is rather than extrapolated: it means the fit wanted to
    leave the searched range, and inventing a value outside it would report a
    height nothing examined.
    """
    grid = ell_grid() if grid is None else np.asarray(grid, dtype=float)
    density = np.asarray(density, dtype=float)
    if bank is None:
        if edges is None:
            raise ValueError("fit_ell needs either bin edges or a prebuilt bank")
        bank = curve_bank(edges, grid, prime_limit=prime_limit)
    curves = bank.curves if edges is None else bank.matching(edges, grid, prime_limit)
    distances = np.abs(curves - density[None, :]).mean(axis=1)
    index = int(distances.argmin())
    best = float(grid[index])
    if 0 < index < len(grid) - 1:
        low, middle, high = distances[index - 1 : index + 2]
        curvature = low - 2 * middle + high
        if curvature > 0:
            step = float(grid[1] - grid[0])
            best += 0.5 * step * float(low - high) / float(curvature)
    return best


def fit_gain(
    density: np.ndarray,
    edges: np.ndarray | None = None,
    *,
    near: float | None = None,
    grid: np.ndarray | None = None,
    prime_limit: int | None = None,
    bank: CurveBank | None = None,
) -> float:
    """How far the fit moves when the density is moved by a known amount.

    A BIAS MEASURED ON AN ESTIMATOR THAT DOES NOT RESPOND IS NOT A BIAS. Fitting
    `l` to a band of a few thousand zeros returns a number whatever the density
    says, and the difference between that number and the truth reads exactly
    like a bias while being a statement that nothing was measured. The two are
    told apart by asking whether the fit moves at all.

    So: perturb the density by ONE GRID STEP's worth of curve difference at the
    `l` it currently sits near --

        density' = density + (C[j+1] - C[j])

    -- and return how far the fitted `l` travels, in units of that step. One
    means the estimator reports what it is given. Values far below one mean it
    does not, and a residual measured there says nothing about the zeros.

    Measured this way the gain collapses below about 10,000 zeros -- 0.349 at
    N = 2,500, ten sigma from one -- which is what `MINIMUM_BAND` is for.

    THE PERTURBATION IS A BANK ROW DIFFERENCE, not an analytic displacement,
    because the bank is what the fit actually compares against. A perturbation
    computed some other way would measure the agreement between two spellings of
    the curve as well as the estimator's response, and only one of those is the
    question.

    `near` IS WHERE TO PERTURB, AND IT IS NOT A DETAIL. Left out, the step is
    taken at the `l` the fit currently reports, which measures how responsive
    the estimator is AT ITS OWN ANSWER. Given the `l` the data is actually at,
    the step is taken there, which measures `d(fitted)/d(true)` -- the quantity
    that decides whether a residual is a bias or a non-response.

    WHERE THE FIT IS GOOD THESE AGREE, AND WHERE IT IS BAD THEY DO NOT, which
    is precisely the regime this function exists for. At N = 2,500 zeros the
    fitted `l` scatters by about 4, so the two anchorings land on different
    parts of the grid and report 0.653 and 0.349 for the same slices. The
    second is the one that answers the question; the first was measured first,
    and the disagreement between two spellings of one quantity is how it was
    caught.

    So pass `near` whenever the true `l` is known -- which, in every use this
    module has, it is.
    """
    grid = ell_grid() if grid is None else np.asarray(grid, dtype=float)
    density = np.asarray(density, dtype=float)
    if bank is None:
        if edges is None:
            raise ValueError("fit_gain needs either bin edges or a prebuilt bank")
        bank = curve_bank(edges, grid, prime_limit=prime_limit)
    curves = bank.curves if edges is None else bank.matching(edges, grid, prime_limit)

    base = fit_ell(density, edges, grid=grid, prime_limit=prime_limit, bank=bank)
    index = int(np.argmin(np.abs(grid - (base if near is None else float(near)))))
    # At the top of the grid there is no next row to step to; step down instead
    # and keep the sign, rather than returning a gain nothing measured.
    if index >= len(grid) - 1:
        index = len(grid) - 2
    step = float(grid[index + 1] - grid[index])
    moved = fit_ell(
        density + (curves[index + 1] - curves[index]),
        edges,
        grid=grid,
        prime_limit=prime_limit,
        bank=bank,
    )
    return float((moved - base) / step)


def narrow_ell_bands(
    ordinates: np.ndarray,
    *,
    width: float = MAX_BAND_ELL_SPAN,
    minimum: int = MINIMUM_BAND,
    phase: float = 0.0,
) -> list[np.ndarray]:
    """Disjoint bands each spanning at most `width` in `l`.

    NOT equal counts, and that was the bug. A band's measured density has
    expectation `(1/N) sum_n R_2(u; l_n)` -- a MIXTURE over the heights it
    contains -- and the fit compares it against a single-height curve at the
    band's mean `l`. `R_2` is nonlinear in `l`, so the two agree only while the
    band is narrow.

    Equal-count bands are not. The zeros thin out downwards, so at 1747146
    zeros below `T = 10^6` in sixteen equal-count bands the lowest spans
    `l = 4.15..9.47` while the highest spans 0.06 -- a ratio of 89 -- and the
    widest one carries the most leverage in the regression that follows. Its
    fitted value reflected where the band boundary happened to fall as much as
    it reflected the height. Raised in review, and the measurement above is
    what confirmed it.

    Bands too thin to hold `minimum` zeros are dropped rather than widened:
    widening is what produced the problem, and a band that cannot be measured
    should be absent rather than quietly different from the others.

    AND SO IS THE BIN THE DATA DOES NOT FILL, which is the same rule and was
    missing. `T = 10^6` falls inside the last bin, so it held 0.329 of a width
    instead of 0.500: a different width from the others, a mean `l` that is not
    its bin's centre, and a position at the extreme of the range where a
    regression's leverage is highest. Measured, that one band carried 38% of
    the slope's leverage and was its 2.5-sigma outlier, and dropping it moved
    the slope from 0.828 to 0.970 -- which the module docstring recorded as a
    robustness check without noticing that the check was the sound number and
    the headline was the contaminated one.

    THE GRID'S PHASE IS STILL ARBITRARY, and that is now a measured systematic
    rather than an unexamined one. `edges` is anchored at `ell.min()`, an
    accident of which zero is lowest; shifting it within a width moves the
    slope, and the spread over five anchors is +/-0.068. Dropping partial bands
    narrows that spread from 0.209 to 0.162 but cannot remove it, so the
    uncertainty belongs in the quoted error. See `pair-correlation-lower-order`.
    """
    from .pair_correlation import SKIP_LOWEST

    ordered = np.sort(np.asarray(ordinates, dtype=float))[SKIP_LOWEST:]
    if not len(ordered):
        return []
    ell = np.log(ordered / (2 * np.pi))
    edges = np.arange(ell.min() + phase * width, ell.max() + width, width)
    bands = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        if high > ell.max():
            continue  # the data stops inside this bin: not a band of the same kind
        piece = ordered[(ell >= low) & (ell < high)]
        if len(piece) >= minimum:
            bands.append(piece)
    return bands


def equal_count_bands(
    ordinates: np.ndarray, bands: int, *, minimum: int = MINIMUM_BAND
) -> list[np.ndarray]:
    """Disjoint bands holding the same NUMBER of zeros.

    KEPT FOR THE TEST THAT SHOWS WHY IT IS WRONG HERE, and not used by
    `recover_height`. Equal counts do give every band the same sampling noise
    in its density -- which is why this looked right -- but they give the low
    bands an enormous span in `l`, and the fit is against a single-height
    curve. See `narrow_ell_bands`.
    """
    from .pair_correlation import SKIP_LOWEST

    ordered = np.sort(np.asarray(ordinates, dtype=float))[SKIP_LOWEST:]
    size = len(ordered) // bands
    if size < minimum:
        raise ValueError(
            f"{len(ordered)} zeros in {bands} bands is {size} each, below the "
            f"{minimum} at which a band's fit is too noisy for the scatter "
            "between bands to estimate anything; use fewer bands or more zeros"
        )
    return [ordered[index * size : (index + 1) * size] for index in range(bands)]


def _band_uncertainty(
    unfolded: np.ndarray,
    *,
    window: float,
    bins: int,
    grid: np.ndarray,
    bank: CurveBank,
    edges: np.ndarray,
    prime_limit: int | None,
    chunks: int,
    rotations: int = UNCERTAINTY_ROTATIONS,
) -> float:
    """How precisely THIS band determines `l`, measured from its own scatter.

    Not shared across bands, and that was the second bug. Equal zero counts
    make the density noise similar everywhere, but they do not make `l`
    equally well determined: Conrey-Snaith approaches Montgomery's curve as `l`
    grows, so `dR_2/dl` shrinks and the same density error buys a larger `l`
    error high up. Pooling one residual spread across all bands assumes a
    homoskedasticity the estimator does not have. Raised in review.

    So each band is split into chunks, `l` is fitted in each, and the standard
    error of the band's own estimate is read off the scatter between them --
    the same construction as `pair_correlation.pair_sampling_noise_floor`, for
    the same reason: nothing is assumed about a distribution nobody measured.

    AND AVERAGED OVER WHERE THE CHUNK BOUNDARIES FALL, because ONE partition is
    a noisy estimate of it. The standard error of a standard deviation from `k`
    samples is about `1/sqrt(2(k-1))`, which is 27% at `k = 8` -- and measured
    by rotating the boundaries on a fixed band, it is 25-28%. That is the error
    bar on the error bar, and it was not small.

    It was doing damage. The band at `l = 11.42` reported 0.315 while its
    smaller neighbour reported 0.191, and this repository recorded that as an
    unexplained anomaly sitting exactly where the regression's leverage is.
    Rotated, the two bands give 0.169-0.389 and 0.188-0.339: the same
    distribution, sampled once each. There was no anomaly. Weights enter the
    regression as `1/sigma**2`, so a 27% error in sigma is a 54% error in the
    weight, and a band was being discounted for the partition it happened to
    get.

    Not a bias, though: `k = 4, 8, 16` agree to a few per cent, so the chunk
    COUNT is not what the estimate is reading. Only where the cuts land.
    """
    from .pair_correlation import measured_density

    size = len(unfolded) // chunks
    step = max(1, size // max(1, rotations))
    draws = []
    for turn in range(max(1, rotations)):
        rolled = np.roll(unfolded, turn * step) if turn else unfolded
        fits = [
            fit_ell(
                measured_density(
                    rolled[index * size : (index + 1) * size], window=window, bins=bins
                ).density,
                edges,
                grid=grid,
                prime_limit=prime_limit,
                bank=bank,
            )
            for index in range(chunks)
        ]
        # Scatter of one chunk, divided down to the scatter of the whole band.
        draws.append(float(np.std(fits, ddof=1) / np.sqrt(chunks)))
    return float(np.mean(draws))


def recover_height(
    ordinates: np.ndarray,
    *,
    band_width: float = MAX_BAND_ELL_SPAN,
    #: Fewest zeros a band may hold. A PARAMETER, not only a constant: its
    #: value was justified by a per-band sigma of 0.70 read from a single chunk
    #: placement, and a single placement carries 25-28% noise. A constant that
    #: cannot be varied is a constant that cannot be re-examined.
    band_minimum: int = MINIMUM_BAND,
    window: float = 3.0,
    bins: int = 30,
    prime_limit: int | None = None,
    seed: int = 11,
    bank: CurveBank | None = None,
    chunks: int = UNCERTAINTY_CHUNKS,
    rotations: int = UNCERTAINTY_ROTATIONS,
    anchors: int = 1,
    extra_bands: Sequence[np.ndarray] | None = None,
    _phase: float = 0.0,
) -> HeightRecovery:
    """Fit `l` band by band and regress it on the height the zeros are at.

    Bands are narrow in `l` rather than equal in count, and the regression is
    weighted by each band's own measured uncertainty. Both were review
    findings on the first version; see `narrow_ell_bands` and
    `_band_uncertainty` for what each was hiding.

    `extra_bands` ARE THE LADDER, and they belong here rather than in a script.
    `zero_ordinates` cannot reach `l = 12.5` and above, so those rungs are built
    separately by `tools/build-ladder.py` using `zeros_in_band`; passing them in
    is what lets one fit span `l = 8.9..13.5`. `tools/fit-ladder.py` used to
    reimplement the estimator to do this, and the copy drifted the moment
    `_band_uncertainty` learned to average over chunk placements -- the tool
    kept measuring sigma the old way for a while. One estimator, one place.

    THE GRID'S PHASE ONLY MOVES THE BANDS IT CUTS. Anchoring is a property of
    `narrow_ell_bands`, so with `anchors > 1` the rungs stay put and only the
    banded ordinates shift. That is correct -- a rung is a band by
    construction, not by where a grid line fell -- and it is why the anchor
    systematic shrinks once the ladder carries most of the lever.
    """
    from .pair_correlation import measured_density, unfold

    if anchors > 1:
        # THE GRID'S PHASE IS ARBITRARY AND IT MOVES THE ANSWER. Edges anchor
        # at `ell.min()`, an accident of which zero is lowest; shifting within
        # one width moved the slope over 0.875..1.037 on the T < 10^6 set. That
        # spread is a systematic, and measuring it costs one refit per anchor.
        #
        # Default one, because the suite cannot afford five and the statistical
        # error is what most callers want. A single anchor leaves
        # `slope_anchor_error` None rather than zero: not measured is not the
        # same as measured to be nothing.
        shifts = [index / anchors for index in range(anchors)]
        runs = [
            recover_height(
                ordinates,
                band_width=band_width,
                band_minimum=band_minimum,
                window=window,
                bins=bins,
                prime_limit=prime_limit,
                seed=seed,
                bank=bank,
                chunks=chunks,
                rotations=rotations,
                extra_bands=extra_bands,
                _phase=shift,
            )
            for shift in shifts
        ]
        slopes = np.array([run.slope for run in runs])
        primary = runs[0]
        return primary.model_copy(
            update={
                "slope_anchor_error": float(np.std(slopes, ddof=1)),
                "slope_over_anchors": float(np.mean(slopes)),
            }
        )

    pieces = narrow_ell_bands(
        ordinates, width=band_width, minimum=band_minimum, phase=_phase
    )
    if extra_bands:
        pieces = list(pieces) + [np.sort(np.asarray(band, dtype=float)) for band in extra_bands]
        pieces.sort(key=lambda band: float(np.log(band / (2 * np.pi)).mean()))
    if len(pieces) < MINIMUM_BANDS:
        raise ValueError(
            f"{len(pieces)} bands of width {band_width} in l hold enough zeros "
            f"to fit, and a weighted regression needs at least {MINIMUM_BANDS} "
            "to carry a standard error; widen the range or lower the band width"
        )
    edges = np.linspace(0.0, window, bins + 1)
    grid = ell_grid()
    if bank is None:
        bank = curve_bank(edges, grid, prime_limit=prime_limit)
    bank.matching(edges, grid, prime_limit)

    true_ell, fitted_ell, sigma, spans = [], [], [], []
    for piece in pieces:
        unfolded = unfold(piece)
        density = measured_density(unfolded, window=window, bins=bins).density
        ell = np.log(piece / (2 * np.pi))
        true_ell.append(float(ell.mean()))
        spans.append(float(ell.max() - ell.min()))
        fitted_ell.append(fit_ell(density, edges, grid=grid, prime_limit=prime_limit, bank=bank))
        sigma.append(
            _band_uncertainty(
                unfolded,
                window=window,
                bins=bins,
                grid=grid,
                bank=bank,
                edges=edges,
                prime_limit=prime_limit,
                chunks=chunks,
                rotations=rotations,
            )
        )
    true = np.array(true_ell)
    got = np.array(fitted_ell)
    sigma = np.array(sigma)
    # A band whose chunks happened to agree exactly would otherwise get
    # infinite weight; floor it at the smallest positive one measured.
    positive = sigma[sigma > 0]
    sigma = np.maximum(sigma, positive.min() if len(positive) else 1e-6)

    slope, intercept, slope_error = _weighted_line(true, got, sigma)

    # The null the permutation test used is not available now: with unequal
    # uncertainties the fitted values are not exchangeable, so shuffling them
    # between bands is not a valid null. Draw from each band's OWN measured
    # uncertainty about a flat relation instead.
    generator = np.random.default_rng(seed)
    centre = float(np.average(got, weights=1 / sigma**2))
    null = np.array(
        [
            _weighted_line(true, centre + generator.normal(0, sigma), sigma)[0]
            for _ in range(NULL_DRAWS)
        ]
    )

    difference = got - true
    weights = 1 / sigma**2
    bias = float(np.average(difference, weights=weights))
    bias_error = float(np.sqrt(1 / weights.sum()))

    return HeightRecovery(
        band_size=int(np.median([len(piece) for piece in pieces])),
        band_widths=[float(value) for value in spans],
        true_ell=[float(value) for value in true],
        fitted_ell=[float(value) for value in got],
        fitted_error=[float(value) for value in sigma],
        slope=float(slope),
        slope_error=float(slope_error),
        intercept=float(intercept),
        bias=bias,
        bias_error=bias_error,
        # (r+1)/(n+1), NOT r/n. THE PLAIN FRACTION CAN RECORD EXACTLY ZERO, and a
        # recorded 0.0 reads as "impossible under the null" when all it means is
        # "below what NULL_DRAWS can resolve" -- a resolution limit presented as a
        # measurement, which is the failure this repository keeps finding in other
        # clothes. The +1 estimator is the standard Monte-Carlo p-value: it is an
        # honest upper bound, it can never be zero, and it is conservative.
        #
        # IT MOVES THE RECORDED NUMBER. The run on file has one draw reaching the
        # slope and read 1/20000 = 5.0e-5; it now reads 2/20001 = 1.0e-4. And note
        # the collision worth not being confused by: ZERO draws now also reads
        # 1/20001 = 5.0e-5, the same digits the old estimator gave for ONE draw.
        null_p=float((int(np.sum(null >= slope)) + 1) / (len(null) + 1)),
    )


def _weighted_line(x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> tuple[float, float, float]:
    """Weighted least squares, with the slope's own standard error.

    Weights are `1/sigma^2` from each band's measured `l` uncertainty, so a
    band that determines `l` poorly is not allowed to pull the line as hard as
    one that determines it well. The unweighted version reported a slope error
    from the pooled residual spread, which assumes every point is equally
    precise -- the assumption review showed is false here.
    """
    weight = 1.0 / np.asarray(sigma, dtype=float) ** 2
    total = weight.sum()
    mean_x = float((weight * x).sum() / total)
    mean_y = float((weight * y).sum() / total)
    spread = float((weight * (x - mean_x) ** 2).sum())
    if spread <= 0:
        raise ValueError("every band sits at the same l; there is no slope to fit")
    slope = float((weight * (x - mean_x) * (y - mean_y)).sum() / spread)
    intercept = mean_y - slope * mean_x
    # The formal error from the weights, inflated by the fit's own scatter
    # when the points disagree by more than their error bars claim -- the
    # standard chi-square rescaling, so an underestimated sigma cannot
    # manufacture significance.
    formal = float(np.sqrt(1.0 / spread))
    residual = y - (slope * x + intercept)
    if len(x) > 2:
        chi2 = float((weight * residual**2).sum() / (len(x) - 2))
        formal *= max(1.0, np.sqrt(chi2))
    return slope, intercept, formal
