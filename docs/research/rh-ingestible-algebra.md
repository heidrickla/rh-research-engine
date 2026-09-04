# RH-adjacent source material for ingestion

Source material for `rhre symbolic ingest`. Every entry in the
[formula reference](../RH_FORMULA_REFERENCE.md) with an algebraic statement is
ingestible: function application is represented, so `\xi(s)` is xi applied to s
rather than xi times s; binders carry their bounds; `\log n` is application, not
juxtaposition; and a bare `\log n`, `\sqrt x` or `\left|x\right|` all read as
written.

Statements are transcribed from the cited sources and are not independently
verified here. Parsing a formula says nothing about whether it is true — the
proof queue reports the ξ functional equation as `not_an_identity`, because
SymPy cannot verify a theorem, and the ζ functional equation as
`unsupported_fragment`. Only genuine polynomial identities over ℚ reach Lean.

## Exponent and Θ relations

The engine's own arithmetic, written so it can be indexed and checked.
`Theta = sup Re(rho)` over nontrivial zeros; `theta` is a remainder exponent.

The implied bound from a remainder exponent, as `core.bounds` computes it:

$$\Theta = 1/2 + \theta/2$$

The RH endpoint of that relation, reached only when the remainder exponent is
zero:

$$\Theta = 1/2$$

The unconditional floor. `Theta >= 1/2` holds because zeros lie on the critical
line, so a derived value below it means the premise is wrong, not that the
result is stronger — enforced as `core.bounds.THETA_FLOOR` and re-checked in
`properties.closure.theta_is_possible`.

$$\Theta \ge 1/2$$

$$2\Theta - 1 = \theta$$

Source: [DLMF §25.4](https://dlmf.nist.gov/25.4) for the zeta background;
the Θ/θ transfer is this project's own arithmetic (`docs/ARCHITECTURE.md`).

## de Bruijn–Newman

RH is equivalent to `\Lambda <= 0`, and `\Lambda >= 0` is proved, so RH is
equivalent to:

$$\Lambda \ge 0$$

$$\Lambda = 0$$

Source: Rodgers–Tao, [arXiv:1801.05914](https://arxiv.org/abs/1801.05914).

## Definitions and functional equations

$$\forall s, \Re(s) > 1: \zeta(s) = \prod_p (1-p^{-s})^{-1}$$

$$\forall s, \Re(s) > 0: \Gamma(s) = \int_0^{\infty} t^{s-1}e^{-t} dt$$

$$\forall s, \Re(s) > 1: \zeta(s) = \sum_{n=1}^{\infty} n^{-s}$$

$$\zeta(s) = 2(2\pi)^{s-1}\sin(\pi s/2)\Gamma(1-s)\zeta(1-s)$$

$$\xi(s) = \frac{1}{2}s(s-1)\pi^{-s/2}\Gamma(s/2)\zeta(s)$$

$$\xi(s) = \xi(1-s)$$

Source: [DLMF §25.2, §25.4](https://dlmf.nist.gov/25.4). The `sin(\pi s/2)`
factor is load-bearing: it supplies the trivial zeros at the negative even
integers, and the equation is simply false without it.

The Euler product is indexed over POSITION: the product over all primes is
the product over `k` of a term in the k-th prime, written with `NthPrime`,
which evaluates to 2, 3, 5, ... so a truncation can be checked against the
actual primes.

## Zero counting

$$N(T) = (T/(2\pi))\log(T/(2\pi)) - T/(2\pi) + 7/8 + O(\log T)$$

Source: [Riemann–von Mangoldt](https://en.wikipedia.org/wiki/Riemann%E2%80%93von_Mangoldt_formula).
The `7/8` is the sharp form and is not decoration. Without it the residual
grows systematically instead of oscillating, so the statement is true but
weaker than it looks -- the leftover is a constant absorbed into `O(\log T)`,
not the fluctuation `S(T)`. Verified against located zeros: N(100) = 29
against a main term of 28.13, which fixes the sign as +7/8.

The `O(\log T)` term is part of the theorem. Dropped, the line asserts an exact
count that is not true of any `T`.

## RH-equivalent criteria

Each carries its hypothesis. Robin's inequality is FALSE at n = 5040 and the
claim is about n above it, so a criterion indexed without its threshold is a
different -- and false -- statement, and nothing could check it.

None is an identity SymPy can verify — the proof queue reports each as `not_an_identity`, which says `ring`
cannot discharge it and nothing about whether it is true.

$$\forall n, n > 5040: \sigma(n) < e^{\EulerGamma} n \log\log n$$

$$\forall n, n \ge 1: \sigma(n) \le H(n) + e^{H(n)}\log H(n)$$

$$\forall n, n \ge 1: \lambda_n \ge 0$$

$$\lambda_n = \frac{1}{(n-1)!}\left.\frac{d^n}{ds^n}(s^{n-1}\log\xi(s))\right|_{s=1}$$

$$M(x) = O(x^{1/2+\epsilon})$$

$$\pi(x) = li(x) + O(\sqrt{x}\log x)$$

`\EulerGamma` is Euler's constant. The sources write `\gamma`, which
also names the ordinate of a zero in `\rho = 1/2 + i\gamma` -- both stand
alone, so the spelling is made explicit here rather than guessed at.

Sources: Robin via [arXiv:math/0604314](https://arxiv.org/abs/math/0604314);
Lagarias, [arXiv:math/0008177](https://arxiv.org/abs/math/0008177);
[Li's criterion](https://en.wikipedia.org/wiki/Li%27s_criterion);
Mertens and von Koch via
[Riemann hypothesis](https://en.wikipedia.org/wiki/Riemann_hypothesis);
Rodgers–Tao, [arXiv:1801.05914](https://arxiv.org/abs/1801.05914).

`O(...)` indexes as an uninterpreted function, not as an asymptotic
comparison: this engine does not implement one. SymPy's `Order` would be
actively wrong here — it is a germ at zero, and it absorbs the terms it
dominates, which deletes `li(x)` from von Koch's theorem.

**`M(x) = O(x^{1/2+\epsilon})` is equivalent to RH; `M(x) = O(x^{1/2})` is the
Mertens conjecture and was disproved.** The epsilon is load-bearing and the two
differ by one character.

## Dirichlet series

Each carries the half-plane it converges on. A series stated without one is
not the identity -- outside it the sum diverges, and a checker evaluating
there reports the definition of zeta as false. All are identities, not
criteria: they hold whether or not RH does.

`\sigma` here is the sum of divisors, so its series needs `\Re s > 2`.

$$\forall s, \Re(s) > 1: \sum_{n=1}^{\infty} \mu(n) n^{-s} = 1/\zeta(s)$$

$$\forall s, \Re(s) > 1: \sum_{n=1}^{\infty} \Lambda(n) n^{-s} = -\zeta'(s)/\zeta(s)$$

$$\forall s, \Re(s) > 2: \sum_{n=1}^{\infty} \sigma(n) n^{-s} = \zeta(s)\zeta(s-1)$$

$$\forall s, \Re(s) > 1: \zeta'(s) = -\sum_{n=2}^{\infty} (\log n) n^{-s}$$

$$\forall s, \Re(s) > 0: \zeta(s) = (1/(1-2^{1-s}))\sum_{n=1}^{\infty} (-1)^{n-1} n^{-s}$$

Sources: [DLMF 27.4.5, 27.4.11, 27.4.12](https://dlmf.nist.gov/27.4);
[DLMF 25.2.6, 25.2.3](https://dlmf.nist.gov/25.2). The last converges for
`\Re s > 0`, which is why it, and not the defining series, reaches the
critical strip.

## The critical line

$$Z(t) = e^{i\theta(t)}\zeta(1/2+it)$$

$$\theta(t) = \arg(\Gamma(1/4+it/2)) - (t/2)\log(\pi)$$

Source: [DLMF 25.10.1, 25.10.2](https://dlmf.nist.gov/25.10). `Z` is real on
the real axis and `|Z(t)| = |\zeta(1/2+it)|`, so its sign changes locate
zeros on the line. Checked: `Z` vanishes at `t = 14.1347` and `t = 21.0220`
with imaginary part below `1e-22`.

## Zeros

$$\psi(x) = x - \sum_{k=1}^{\infty} 2\Re(x^{\rho(k)}/\rho(k)) - \log(2\pi) - (1/2)\log(1-x^{-2})$$

Source: von Mangoldt 1895, via
[explicit formulae](https://en.wikipedia.org/wiki/Explicit_formulae_for_L-functions).
The sum is conditionally convergent and only means anything taken in order of
`|\Im \rho|`; `\rho(k)` is the k-th zero in that order, so the ordering is
carried rather than assumed. Each term is paired with its conjugate, which is
what `2\Re` does. Truncated at 200 zeros it gives 19.2624 against
`\psi(20) = 19.2657`.

$$1 - (\sin(\pi u)/(\pi u))^2$$

Source: [Montgomery's pair correlation
conjecture](https://en.wikipedia.org/wiki/Montgomery%27s_pair_correlation_conjecture),
1973. The conjectured density of normalised gaps between zeros, assuming RH.
It is the GUE density from random matrix theory -- a conjecture, not a
theorem, and not known to imply RH.

$$\lim_{x \to \infty} \pi(x)\log(x)/x = 1$$

Source: the prime number theorem, [DLMF 27.12](https://dlmf.nist.gov/27.12).
Unconditional, and weaker than RH: RH is equivalent to a much sharper error
term -- see von Koch above.


## Nyman-Beurling and Baez-Duarte

RH holds exactly when the indicator of `(0,1]` lies in the `L^2` closure of
the dilations `rho_a(x) = {1/ax}`. That form is stated as a distance `d_n`
vanishing, and `d_n` is an infimum over a span -- it is not written here,
because a formula whose terms this corpus cannot define is a formula
nothing can check.

Baez-Duarte's own computable form of the criterion is indexed instead:

$$c_k = \sum_{j=0}^{k} (-1)^j \binom{k}{j} / \zeta(2j+2)$$

$$c_k = O(k^{-3/4+\epsilon})$$

Sources: Baez-Duarte,
[arXiv:math/0202141](https://arxiv.org/abs/math/0202141) for the closure
form; the coefficients and the `-3/4` exponent via Maslanka,
[arXiv:math/0603713](https://arxiv.org/abs/math/0603713).

`c_0 = 1/\zeta(2)` exactly, which is the cheapest check there is that the
sum is being read right. The alternating binomial sum cancels hard -- terms
reach `binom(k, k/2)` while the total stays under one -- so it needs working
precision far above the answer: at k = 32, twenty digits already lose the
tenth decimal.

## Chebyshev functions

$$\psi(x) = \sum_{n=1}^{x} \Lambda(n)$$

Source: [DLMF §27.2](https://dlmf.nist.gov/27.2).

## Consequences of RH, not equivalences

Indexed separately because the direction matters. RH implies each of these;
none of them implies RH, so a proof of one advances nothing on its own.

$$\zeta(1/2 + i t) = O(t^{\epsilon})$$

$$\forall x, x \ge 74: \left|\psi(x) - x\right| < (1/(8\pi))\sqrt{x}(\log x)^2$$

$$\forall x, x \ge 2657: \left|\pi(x) - li(x)\right| < (1/(8\pi))\sqrt{x}\log x$$

Lindelöf holds for every `\epsilon > 0`; the converse is open. Schoenfeld's
bounds are conditional on RH. The thresholds are part of the statements and
are carried in the formulas; 74 is used for the first, being the least
integer above Schoenfeld's 73.2.

Sources: [Riemann hypothesis](https://en.wikipedia.org/wiki/Riemann_hypothesis),
Schoenfeld 1976 via the same.

## Polynomial identities

Genuine identities, included because they are the only class the Lean exporter
can currently discharge (`ring` over ℚ). They are algebra, not number theory —
their value here is as a live end-to-end path through
ingest → index → proof queue → Lean, not as RH content.

$$(x+1)^2 = x^2 + 2x + 1$$

$$(x-1)(x+1) = x^2 - 1$$

$$(x+y)^2 - (x-y)^2 = 4xy$$

Source: elementary algebra; no citation is meaningful and none is claimed.

## Redheffer and Farey -- RH outside analysis

Every other criterion here is analysis. These two are not, and that is the
reason for having them: one turns RH into the growth of a determinant of a
0/1 matrix, the other into how evenly a set of fractions is spread.

`R_n` is the `n x n` matrix with a 1 wherever the column is 1 or the row
divides the column, and 0 elsewhere -- nothing in it but divisibility. Its
determinant is the summatory Mobius function:

$$\RedhefferDet(n) = M(n)$$

An exact identity, and evaluable on both sides, so it is checked against
values rather than only parsed. `\RedhefferDet` is computed AS a determinant,
by fraction-free elimination, and not read off `M(n)` -- the whole content of
the statement is that a linear-algebra route and a number-theoretic one agree,
and computing one from the other would check nothing. The same discipline that
makes `M = \sum \mu` a real rediscovery in the pattern sweep rather than a
vacuous one.

With `M(x) = O(x^{1/2+\epsilon})` already indexed above, RH is equivalent to
`\det R_n` growing no faster than `n^{1/2+\epsilon}`.

`F_n` is the set of reduced fractions in `(0, 1]` with denominator at most
`n`, in increasing order. How many there are is an identity:

$$\FareyCount(n) = \sum_{k=1}^{n} \phi(k)$$

and how far the `i`-th sits from where an evenly spread set would put it is
the Franel-Landau criterion, equivalent to RH:

$$\FareyDeviation(n) = O(n^{1/2+\epsilon})$$

`\FareyDeviation(n)` is `\sum_i |F_i - i/\FareyCount(n)|`, summed exactly
over rationals: the deviations are around `1e-5` by `n = 1000` and alternate
in sign, so through floats the sum would lose its leading digits to
cancellation. `\FareyCount` likewise enumerates the fractions rather than
summing `\phi`, since that sum is the statement being checked.

Both carry an evaluation cap, and both caps bound the CHECK rather than the
mathematics: the determinant is cubic (0.23 s at `n = 200`, 15.9 s at 800)
and the fractions number about `3n^2/\pi^2`. Above the cap each refuses
rather than returning something cheaper.

Sources: [Redheffer matrix](https://en.wikipedia.org/wiki/Redheffer_matrix);
Franel-Landau via [Farey sequence](https://en.wikipedia.org/wiki/Farey_sequence#Riemann_hypothesis)
and Edwards, *Riemann's Zeta Function*, ch. 12.

## Coverage

Every formula in `RH_FORMULA_REFERENCE.md` with an algebraic statement is
indexed. Binders carry their bounds, `O(...)` stays uninterpreted rather than
becoming SymPy's `Order`, derivatives are operators, and a binder binds
tighter than addition -- the reading every source assumes.

