from pathlib import Path

import sympy as sp

from rh_research_engine.symbolic import (
    equivalent,
    extract_assumptions,
    extract_equations,
    fingerprint,
    parse_math,
    residue,
    simplify_with_trace,
)
from rh_research_engine.symbolic.functions import SREPR_NAMESPACE
from rh_research_engine.symbolic.models import ProofStep
from rh_research_engine.symbolic.proof_gap import extract_proof_gaps


def test_extract_markdown_equations():
    items = extract_equations("A $x^2+2*x+1=(x+1)^2$ and $$y=1/x$$")
    assert len(items) == 2
    assert items[0].lhs is not None
    assert items[1].rhs == "1/x"


def test_equivalence_and_fingerprint():
    assert equivalent("(x+1)**2", "x**2+2*x+1").equivalent is True
    assert fingerprint("(x+1)**2").sha256 == fingerprint("x**2+2*x+1").sha256


def test_simplifier_records_steps():
    result = simplify_with_trace("(x**2-1)/(x-1)")
    assert result.simplified in {"x + 1", "x + 1.0"}
    assert result.steps


def test_assumption_extractor_denominator_and_log():
    assumptions = extract_assumptions("log(x)/(x-1)")
    conditions = {a.condition for a in assumptions}
    assert "x - 1 != 0" in conditions or "1 - x != 0" in conditions
    assert any("x > 0" in c for c in conditions)


def test_residue():
    assert residue("1/(s-rho)", "s", "rho").residue == "1"


def test_proof_gap_extracts_unproved_chain():
    steps = [ProofStep(id="A", statement="known", status="known"), ProofStep(id="B", statement="numerical bridge", status="numerical", depends_on=["A"]), ProofStep(id="C", statement="conclusion", status="proved", depends_on=["B"])]
    gaps = extract_proof_gaps(steps)
    assert {gap.step_id for gap in gaps} == {"B", "C"}


def test_asymptotic_bound_is_inert_not_sympy_order():
    """`O` must not become SymPy's `Order`.

    `Order` is a germ at a point, and at ZERO by default, so it states the
    opposite regime to every asymptotic bound in the literature. It also
    absorbs: `Add.flatten` folds dominated terms into it, which deleted `li(x)`
    from von Koch's theorem outright.
    """
    parsed = parse_math(r"M(x) = O(x^{1/2+\epsilon})")
    assert parsed.parse_error is None
    expr = sp.sympify(parsed.sympy_srepr)
    assert not expr.rhs.has(sp.Order)
    assert isinstance(expr.rhs, sp.core.function.AppliedUndef)
    # epsilon stays a free symbol; Order had taken it for a limit variable.
    assert sp.Symbol("epsilon") in expr.rhs.free_symbols


def test_fingerprint_reads_names_the_way_the_extractor_did():
    """The fingerprint must describe the formula that was validated.

    This path used to re-read the extractor's own output with a bare
    `parse_expr`, so `M(x)` came back as `M*x` and the stored hash identified
    an expression nobody had written. A clean re-parse of the wrong thing is
    indistinguishable from success, which is why this asserts structure.
    """
    canonical = fingerprint("Eq(M(x), O(x**(epsilon + 1/2)))").canonical
    assert "Mertens(Symbol('x'))" in canonical
    assert "Order" not in canonical
    assert "Mul(Symbol('M')" not in canonical
    # And it is the real summatory Mobius function, not a stub:
    assert sp.sympify("Mertens(100)", locals=dict(SREPR_NAMESPACE)) == 1
    # And it agrees with the object the extractor built.
    parsed = parse_math(r"M(x) = O(x^{1/2+\epsilon})")
    assert canonical == fingerprint(
        sp.sympify(parsed.sympy_srepr, locals=dict(SREPR_NAMESPACE))
    ).canonical


def test_von_koch_formula_indexes():
    """Regression: `li` has no leading-term routine.

    Nothing was wrong with `li`. `Order` absorption asked for the leading term
    while trying to swallow it, and the failure surfaced under the wrong name.
    """
    parsed = parse_math(r"\pi(x) = li(x) + O(\sqrt{x}\log x)")
    assert parsed.parse_error is None
    canonical = fingerprint(parsed.normalized).canonical
    assert "li(Symbol('x'))" in canonical
    assert "primepi(Symbol('x'))" in canonical


def test_inequality_srepr_keeps_the_operator():
    """Strict and non-strict are different claims.

    The srepr was hand-built as `Relation(lhs, rhs)`, which named neither the
    operator nor a real SymPy class. Robin's criterion is the STRICT
    inequality; the non-strict form is not the same statement, and both used
    to reduce to the same bytes.
    """
    strict = parse_math(r"\sigma(n) < 2n")
    loose = parse_math(r"\sigma(n) \le 2n")
    assert isinstance(sp.sympify(strict.sympy_srepr), sp.StrictLessThan)
    assert isinstance(sp.sympify(loose.sympy_srepr), sp.LessThan)
    assert fingerprint(strict.normalized).sha256 != fingerprint(loose.normalized).sha256


def test_printed_form_and_srepr_fingerprint_alike():
    """The two ways of reading an extracted formula must agree.

    `normalized` is display text and `sympy_srepr` is an exact round-trip. When
    they fingerprint differently, one of them is wrong about the formula and
    the stored hash depends on which path the caller happened to take. Every
    misreading fixed here -- `Order`, `M(x)` as a product, the dropped
    inequality operator -- shows up as a divergence in this check.
    """
    corpus = Path("docs/research/rh-ingestible-algebra.md")
    equations = [
        item
        for item in extract_equations(corpus.read_text(encoding="utf-8"))
        if item.parse_error is None and item.sympy_srepr
    ]
    assert len(equations) >= 15, "corpus shrank; this check needs real formulas"
    for item in equations:
        from_text = fingerprint(item.normalized).canonical
        from_srepr = fingerprint(
            sp.sympify(item.sympy_srepr, locals=dict(SREPR_NAMESPACE))
        ).canonical
        assert from_text == from_srepr, f"{item.source}\n  text : {from_text}\n  srepr: {from_srepr}"


def test_bars_are_read_by_position():
    r"""`|x|` is the one notation whose delimiters are identical at both ends.

    Position tells them apart. A bar where an operand is still expected can
    only open; one in operator position closes what it is inside, and if there
    is nothing to close it is the divisibility sign -- `p | n` is "p divides
    n", a statement about the operands rather than a delimiter at all. So
    `|a| + |b|`, nested bars, and `p | n` all read correctly.
    """
    bounded = parse_math(r"\left|\pi(x) - li(x)\right| < (1/(8\pi))\sqrt{x}\log x")
    assert bounded.parse_error is None
    assert "Abs(" in bounded.normalized

    a, b, c = sp.symbols("a b c")
    nested = parse_math(r"\left|a + \left|b\right|\right| = c")
    assert sp.sympify(nested.sympy_srepr) == sp.Eq(sp.Abs(a + sp.Abs(b)), c)

    summed = parse_math(r"|a| + |b| = c")
    assert sp.sympify(summed.sympy_srepr) == sp.Eq(sp.Abs(a) + sp.Abs(b), c)

    divides = parse_math(r"p | n")
    n, q = sp.symbols("n p")
    assert sp.sympify(divides.sympy_srepr) == sp.Eq(sp.Mod(n, q), 0)


def test_derivative_is_an_operator_not_a_quotient():
    r"""`\frac{d^n}{ds^n}` is not a fraction with `d` on top.

    Read as one it becomes a product of symbols named `d` and `ds`, which
    parses without complaint and means nothing. Li's criterion is defined by
    exactly this notation, so it has to be carried, and SymPy's `Derivative`
    carries it exactly.
    """
    parsed = parse_math(r"\lambda_n = \frac{1}{(n-1)!}\frac{d^n}{ds^n}[s^{n-1}\log\xi(s)]")
    assert parsed.parse_error is None, parsed.parse_error
    expression = sp.sympify(parsed.sympy_srepr)
    derivative = next(
        node for node in sp.preorder_traversal(expression)
        if isinstance(node, sp.Derivative)
    )
    s, n = sp.symbols("s n")
    assert derivative.variable_count == ((s, n),)
    assert not expression.has(sp.Symbol("ds"))

    # The order is carried, so a second derivative is not the same object.
    first = parse_math(r"f = \frac{d}{dx}(g(x))")
    second = parse_math(r"f = \frac{d^2}{dx^2}(g(x))")
    assert first.sympy_srepr != second.sympy_srepr


def _parsed(source: str) -> sp.Basic:
    result = parse_math(source)
    assert result.parse_error is None, result.parse_error
    return sp.sympify(result.sympy_srepr, locals=dict(SREPR_NAMESPACE))


def test_indexed_formulas_agree_with_what_is_known():
    """The point of using the real functions rather than stubs of their names.

    `Function('sigma')(5040)` is nothing at all, so Robin's inequality could be
    stored and never checked -- a formula that cannot be evaluated cannot be
    caught being wrong. With `divisor_sigma` it evaluates, and it has to fail
    at exactly the value the literature says it fails at.
    """
    n = sp.Symbol("n")
    robin = _parsed(r"\sigma(n) < e^{\EulerGamma} n \log\log n")

    # 5040 is the largest known exception; RH is the claim there are no others.
    assert robin.subs(n, 5040) == sp.false
    for value in (5041, 10080, 100000):
        assert robin.subs(n, value) == sp.true, value

    lagarias = _parsed(r"\sigma(n) \le H(n) + e^{H(n)}\log H(n)")
    for value in (1, 2, 5040, 10080):
        assert bool(lagarias.subs(n, value)) is True, value


def test_the_defined_functions_are_the_functions_they_are_named_after():
    """von Mangoldt, Mertens and Chebyshev psi, checked against their values."""
    assert _parsed(r"\Lambda(8)") == sp.log(2)
    assert _parsed(r"\Lambda(12)") == 0
    assert _parsed(r"\Lambda(9)") == sp.log(3)

    # M(x) = sum of mu(n); the Mertens conjecture that |M(x)| < sqrt(x) is false.
    assert _parsed(r"M(10)") == -1
    assert _parsed(r"M(100)") == 1

    # psi(10) = log(2**3 * 3**2 * 5 * 7) = log(2520).
    assert sp.simplify(_parsed(r"\psi(10)") - sp.log(2520)) == 0

    # The Euler product's index really is the k-th prime.
    assert [_parsed(f"NthPrime({k})") for k in range(1, 5)] == [2, 3, 5, 7]


def test_the_xi_functional_equation_actually_holds():
    """`xi(s) = xi(1-s)` is stored AND verifiable.

    An undefined `Function('xi')` makes the statement unfalsifiable: both
    sides are opaque and nothing can disagree. `RiemannXi` evaluates, so the
    identity is checked here rather than asserted -- on the real axis and off
    it, since a symmetry that only held on the reals would not be this one.
    """
    equation = _parsed(r"\xi(s) = \xi(1-s)")
    s = sp.Symbol("s")
    for point in (2, -1, sp.Rational(1, 3), sp.Rational(3, 10) + 2 * sp.I):
        difference = sp.Abs(
            equation.lhs.subs(s, point) - equation.rhs.subs(s, point)
        ).evalf(30)
        assert difference < sp.Float("1e-25"), (point, difference)


def test_the_zero_count_is_the_count_and_not_its_estimate():
    """`N(T)` counts zeros; Riemann-von Mangoldt only approximates it.

    Keeping them distinct is the point -- a formula that estimates N(T) is
    not checkable against a definition that is the same estimate. The first
    three ordinates are 14.13, 21.02 and 25.01, so N(30) is 3.
    """
    assert _parsed(r"N(14)") == 0
    assert _parsed(r"N(15)") == 1
    assert _parsed(r"N(30)") == 3
    assert _parsed(r"N(100)") == 29

    # And the estimate is close to it without being it.
    estimate = _parsed(r"(T/(2\pi))\log(T/(2\pi)) - T/(2\pi)")
    approximated = float(estimate.subs(sp.Symbol("T"), 100).evalf())
    assert 0 < abs(approximated - 29) < 3


def test_an_asymptotic_bound_records_which_regime_it_is_in():
    """`O(...)` is at infinity here, and says so.

    SymPy's `Order` defaults to a germ at ZERO and absorbs what it dominates,
    which would have deleted `li(x)` from von Koch's theorem. This one states
    its limit point, so two claims about different regimes cannot collide.
    """
    koch = _parsed(r"\pi(x) = li(x) + O(\sqrt{x}\log x)")
    bound = next(
        node for node in sp.preorder_traversal(koch)
        if type(node).__name__ == "BigO"
    )
    assert bound.args[1] == sp.oo
    # li(x) survived: nothing absorbed it.
    assert koch.has(sp.li)


def test_every_module_that_reads_a_formula_uses_one_policy():
    """A formula read under two policies is two different formulas.

    `equivalence` had its own, and so did the Lean exporter, the simplifier,
    the assumption extractor and the decomposer -- each a bare `parse_expr`
    with no `local_dict`, resolving names differently from the parser that
    produced the string it was handed. The exporter reported Lindelof as
    UNPARSEABLE for that reason, which reads as "this formula is malformed"
    when the formula was fine and the reader was not.
    """
    import re

    source_root = Path("src/rh_research_engine/symbolic")
    offenders = []
    for path in sorted(source_root.glob("*.py")):
        if path.name == "parser.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\bparse_expr\(", line) and "local_dict" not in line:
                # A continuation line carries the argument on the next line.
                offenders.append(f"{path.name}:{number}")
    remaining = [
        entry
        for entry in offenders
        if not _call_supplies_locals(source_root, entry)
    ]
    assert not remaining, f"parse without the shared policy: {remaining}"


def _call_supplies_locals(root: Path, entry: str) -> bool:
    name, number = entry.rsplit(":", 1)
    lines = (root / name).read_text(encoding="utf-8").splitlines()
    window = "\n".join(lines[int(number) - 1 : int(number) + 4])
    return "local_dict" in window


def test_no_corpus_formula_is_unreadable_by_any_consumer():
    """The proof queue may refuse to PROVE anything; it may not fail to read.

    `unparseable` is a claim about the formula. Every other verdict is a claim
    about what can be discharged, which is the honest kind.
    """
    from rh_research_engine.symbolic.lean import export_polynomial_identity

    corpus = Path("docs/research/rh-ingestible-algebra.md")
    for item in extract_equations(corpus.read_text(encoding="utf-8")):
        if item.parse_error is not None or item.lhs is None or item.rhs is None:
            continue
        export = export_polynomial_identity(item.lhs, item.rhs)
        assert not (export.reason or "").startswith("parse failed"), (
            f"{item.source}: {export.reason}"
        )


def test_i_is_the_imaginary_unit_unless_a_binder_bound_it():
    """`1/2 + it` is a point on the critical line.

    Read as a free symbol, `i` made it a product of two unknowns, and every
    formula about the critical line was stored that way. But `i` is also the
    commonest index letter there is, so a binder that binds it still wins --
    that is decidable from the binder rather than guessed at.
    """
    t_, n, f = sp.Symbol("t"), sp.Symbol("n"), sp.Function("f")

    on_the_line = _parsed(r"\zeta(1/2 + it)")
    assert on_the_line == sp.zeta(sp.Rational(1, 2) + sp.I * t_)

    # Glued or spaced, it is the same point.
    assert _parsed(r"\zeta(1/2 + i t)") == on_the_line

    # Bound as an index, it is a variable again.
    summed = _parsed(r"\sum_{i=1}^{n} f(i)")
    index = summed.limits[0][0]
    assert index == sp.Symbol("i")
    assert summed == sp.Sum(f(sp.Symbol("i")), (sp.Symbol("i"), 1, n))


def test_prime_notation_is_a_derivative():
    r"""`\zeta'(s)` is Lagrange's notation, and a quote is not a Python operator.

    Left alone it opens a string literal and the whole formula fails to
    tokenize, which is how `-\zeta'(s)/\zeta(s)` stayed out of the index.
    """
    s = sp.Symbol("s")
    assert _parsed(r"\zeta'(s)") == sp.Derivative(sp.zeta(s), (s, 1))
    assert _parsed(r"f''(x)") == sp.Derivative(sp.Function("f")(sp.Symbol("x")),
                                               (sp.Symbol("x"), 2))

    # A compound argument means the derivative evaluated AT that point, which
    # is a substitution and not a derivative with respect to `1 - s`.
    at_a_point = _parsed(r"\zeta'(1-s)")
    assert isinstance(at_a_point, sp.Subs)


def test_the_dirichlet_series_sum_to_what_they_claim():
    """Truncations must approach the closed form, or the formula is decoration."""
    series = _parsed(r"\sum_{n=1}^{\infty} \mu(n) n^{-s} = 1/\zeta(s)")
    partial = sum(int(sp.mobius(m)) / m**2.0 for m in range(1, 4001))
    assert abs(partial - 1 / float(sp.zeta(2))) < 1e-3
    assert series.lhs.function.has(sp.mobius)

    divisors = _parsed(r"\sum_{n=1}^{\infty} \sigma(n) n^{-s} = \zeta(s)\zeta(s-1)")
    partial = sum(int(sp.divisor_sigma(m)) / m**4.0 for m in range(1, 4001))
    assert abs(partial - float(sp.zeta(4) * sp.zeta(3))) < 1e-3
    assert divisors.lhs.function.has(sp.divisor_sigma)


def test_hardy_z_vanishes_at_the_first_zeros_and_is_real():
    """`|Z(t)| = |zeta(1/2+it)|` and Z is real, so its sign changes find zeros."""
    definition = _parsed(r"Z(t) = e^{i\theta(t)}\zeta(1/2+it)")
    assert definition.rhs.has(sp.zeta)

    def z(height):
        angle = sp.arg(sp.gamma(sp.Rational(1, 4) + sp.I * height / 2))
        angle -= height * sp.log(sp.pi) / 2
        return complex(
            (sp.exp(sp.I * angle) * sp.zeta(sp.Rational(1, 2) + sp.I * height)).evalf(25)
        )

    for ordinate in (sp.Float("14.134725141734693"), sp.Float("21.022039638771555")):
        value = z(ordinate)
        assert abs(value.real) < 1e-6, ordinate
        assert abs(value.imag) < 1e-15, ordinate


def test_the_zeros_are_located_not_assumed():
    """`NthZetaZero` finds the zero; it does not place it on the line.

    A formula summing over zeros is only checkable because these are real
    values. That the first ones have real part 1/2 is a RESULT here, not an
    assumption baked into the function.
    """
    from rh_research_engine.symbolic.functions import NthZetaZero

    first = NthZetaZero(1)
    assert abs(complex(first).real - 0.5) < 1e-12
    assert abs(complex(first).imag - 14.134725141734693) < 1e-9

    ordinates = [complex(NthZetaZero(k)).imag for k in range(1, 6)]
    assert ordinates == sorted(ordinates), "zeros must come in order of height"


def test_the_evaluation_bar_says_where_a_derivative_is_taken():
    r"""`\left. X \right|_{s=1}` is a substitution, and Li needs it.

    Without the point, the right side of Li's definition is a function of s
    while the left is a number -- so it parsed, indexed, and defined nothing.
    """
    parsed = parse_math(
        r"\lambda_n = \frac{1}{(n-1)!}\left.\frac{d^n}{ds^n}(s^{n-1}\log\xi(s))\right|_{s=1}"
    )
    assert parsed.parse_error is None, parsed.parse_error
    expression = sp.sympify(parsed.sympy_srepr, locals=dict(SREPR_NAMESPACE))
    substitution = next(
        node for node in sp.preorder_traversal(expression) if isinstance(node, sp.Subs)
    )
    assert substitution.point == (sp.Integer(1),)
    assert substitution.variables == (sp.Symbol("s"),)
    # The right side no longer depends on s.
    assert sp.Symbol("s") not in expression.rhs.free_symbols


def test_the_li_definition_agrees_with_an_independent_route():
    """Two representations of lambda_n, and one exact closed form.

    No typed constants. An earlier version of this test asserted against
    sixteen-digit values written from memory, and they were wrong from the
    tenth digit -- so it passed only because its tolerance was 1e-9. The
    computation was right the whole time; the REFERENCE was the imprecise
    part, which is the same shape as evaluating through a double and asking
    for thirty digits.

    lambda_1 has the closed form 1 + gamma/2 - log(4 pi)/2. For higher n the
    corpus's derivative definition is checked against the Taylor route,
    log xi(z/(z-1)) = -log 2 + sum (lambda_n / n) z^n, which is a different
    representation of the same quantity.
    """
    import mpmath

    mpmath.mp.dps = 40

    def xi(s):
        shifted = mpmath.mpf(1) if s == 1 else (s - 1) * mpmath.zeta(s)
        return mpmath.pi ** (-s / 2) * mpmath.gamma(s / 2 + 1) * shifted

    def by_derivative(n):
        def integrand(s):
            return s ** (n - 1) * mpmath.log(xi(s))

        return mpmath.diff(integrand, 1, n) / mpmath.factorial(n - 1)

    def by_taylor(n):
        return n * mpmath.taylor(lambda z: mpmath.log(xi(z / (z - 1))), 0, n)[n]

    closed_form = 1 + mpmath.euler / 2 - mpmath.log(4 * mpmath.pi) / 2
    assert abs(by_derivative(1) - closed_form) < mpmath.mpf("1e-30")

    for n in range(1, 6):
        derivative, taylor = by_derivative(n), by_taylor(n)
        relative = abs(derivative - taylor) / abs(taylor)
        assert relative < mpmath.mpf("1e-30"), (n, relative)
        # Li's criterion, as far as it is checked here.
        assert derivative > 0, n



def test_xi_is_entire():
    """xi(0) = xi(1) = 1/2.

    Both are points where the FACTORED form has a cancelling pole -- Gamma's
    at 0, zeta's at 1 -- and evaluating the factors separately raises there.
    That is a fact about the factorisation, not about xi, and a function that
    claims to be xi has to give the value.
    """
    from rh_research_engine.symbolic.functions import RiemannXi

    assert RiemannXi(0) == sp.Rational(1, 2)
    assert RiemannXi(1) == sp.Rational(1, 2)
    assert abs(complex(RiemannXi(sp.Rational(1, 2)).evalf(20)).real - 0.497120778) < 1e-8

    # And the functional equation still holds away from those points.
    point = sp.Rational(3, 10) + 2 * sp.I
    difference = sp.Abs(RiemannXi(point) - RiemannXi(1 - point)).evalf(30)
    assert difference < sp.Float("1e-25")


def test_a_digit_before_j_is_multiplication_not_an_imaginary_literal():
    """`2j` is one token to Python's lexer, and it means 2i.

    `\\zeta(2j+2)` in Baez-Duarte's coefficients came out as `zeta(2 + 2*I)`:
    the summation index read as the imaginary unit, a complex argument, and a
    clean parse of a different formula. Implicit multiplication cannot reach
    it because the damage is done in the tokenizer.
    """
    parsed = parse_math(r"c_k = \sum_{j=0}^{k} (-1)^j \binom{k}{j} / \zeta(2j+2)")
    assert parsed.parse_error is None, parsed.parse_error
    expression = sp.sympify(parsed.sympy_srepr, locals=dict(SREPR_NAMESPACE))
    assert not expression.has(sp.I)
    assert expression.rhs.function.has(sp.binomial)

    # c_0 is exactly 1/zeta(2), the cheapest check that the sum reads right.
    at_zero = expression.rhs.subs(sp.Symbol("k"), 0).doit()
    assert sp.simplify(at_zero - 1 / sp.zeta(2)) == 0

    # And `i` is still the imaginary unit.
    assert parse_math(r"\zeta(1/2 + it)").normalized == "zeta(I*t + 1/2)"

def test_the_zero_count_agrees_with_locating_every_zero():
    """The fast count is the same count, not a cheaper approximation of it.

    `ZeroCount` used to answer by finding every zero below T -- one root-find
    per zero, repeated for every T, which is quadratic across a column and is
    the reason its reach sat at T = 200. It now reads the count off the
    argument of zeta by Turing's method, in constant time.

    That is only allowed if the answers are identical, so they are compared
    here at every integer and half-integer height up to 300: 598 comparisons
    against 139 zeros actually located. A swap justified by "it should agree"
    is the kind this repository exists to prevent.
    """
    from bisect import bisect_right

    import mpmath

    from rh_research_engine.symbolic.functions import ZeroCount, _locate_zero

    ordinates: list[float] = []
    while not ordinates or ordinates[-1] <= 300:
        ordinates.append(float(mpmath.im(_locate_zero(len(ordinates) + 1))))

    heights = [sp.Rational(k, 2) for k in range(2, 601)]
    disagreements = [
        (height, bisect_right(ordinates, float(height)), int(ZeroCount(height)))
        for height in heights
        if bisect_right(ordinates, float(height)) != int(ZeroCount(height))
    ]
    assert not disagreements, disagreements[:5]
    # The comparison is only worth anything if it had zeros to compare over.
    assert len(ordinates) >= 139


def test_the_zero_count_reaches_past_where_locating_them_could():
    """The limit was the cost, not the mathematics.

    N(10^6) needs 1747146 zeros located to be answered the old way, and is a
    single constant-time evaluation now. This is the check that the limit was
    actually lifted rather than merely re-documented.
    """
    from rh_research_engine.symbolic.functions import ZeroCount

    assert ZeroCount(sp.Integer(10**4)) == 10142
    assert ZeroCount(sp.Integer(10**6)) == 1747146
    # Still symbolic beyond the limit: a formula naming N(10^100) must not
    # hang a parse.
    assert ZeroCount(sp.Integer(10**9)).func.__name__ == "ZeroCount"


# --- RH outside analysis: Redheffer and Farey -----------------------------


def test_every_engine_function_resolves_under_both_policies():
    """A function in one name table and not the other loses its identity.

    There are two registries and they exist for different jobs: one resolves a
    LaTeX name, the other an `srepr` name. A class in the first alone parses
    correctly and then comes back from its OWN printed form as
    `Function('name')` -- a stub, which evaluates to nothing, so a formula
    written with it could never be caught being wrong.

    That is exactly what happened to `RedhefferDet`. The formula guard caught
    it, but only because a corpus formula happened to use it; this catches it
    when the function is added, which is when it is cheap.
    """
    from rh_research_engine.symbolic.functions import (
        APPLIED_FUNCTIONS,
        ENGINE_FUNCTIONS,
        SREPR_NAMESPACE,
    )

    for function in ENGINE_FUNCTIONS:
        name = function.__name__
        assert SREPR_NAMESPACE.get(name) is function, f"{name} missing from srepr"
        assert APPLIED_FUNCTIONS.get(name) is function, f"{name} missing from applied"


def test_the_redheffer_determinant_is_the_mertens_function():
    """RH as a statement about the determinant of a matrix of divisibility.

    Held to the DETERMINANT, computed by elimination, not to a shortcut
    through `Mertens` -- which would make the identity a tautology and check
    nothing, the mistake `cum_mu` exists in the pattern sweep to avoid.
    """
    import sympy as sp

    from rh_research_engine.symbolic.functions import Mertens, RedhefferDet

    for n in (1, 2, 3, 7, 15, 60, 119, 200):
        determinant = RedhefferDet(sp.Integer(n))
        assert determinant == Mertens(sp.Integer(n)), f"disagree at n = {n}"
    # A value nothing could have guessed from the small cases.
    assert RedhefferDet(sp.Integer(200)) == sp.Integer(-8)


def test_the_farey_count_is_the_totient_summatory():
    """`|F_n| = sum phi(k)`, with the left side enumerated rather than summed."""
    import sympy as sp

    from rh_research_engine.symbolic.functions import FareyCount

    for n in (1, 2, 5, 10, 40, 100):
        expected = sum(sp.totient(k) for k in range(1, n + 1))
        assert FareyCount(sp.Integer(n)) == expected, f"disagree at n = {n}"


def test_the_farey_deviation_is_exact_and_small():
    """Summed over rationals, because the deviations cancel.

    Each is around 1e-5 by n = 1000 and they alternate in sign; through floats
    the total loses its leading digits. A `Rational` coming back is the whole
    claim -- a float would be a different, weaker statement about the same
    quantity.
    """
    import sympy as sp

    from rh_research_engine.symbolic.functions import FareyCount, FareyDeviation

    deviation = FareyDeviation(sp.Integer(100))
    assert deviation.is_Rational, f"not exact: {deviation!r}"
    assert 2.0 < float(deviation) < 2.1

    # It never exceeds the trivial bound: every term is under 1, and there are
    # |F_n| of them.
    for n in (10, 50, 100):
        assert FareyDeviation(sp.Integer(n)) < FareyCount(sp.Integer(n))


def test_both_refuse_above_their_cap_rather_than_returning_something_cheaper():
    """A cap bounds the CHECK, and must not quietly bound the answer.

    Returning `Mertens(n)` above the Redheffer cap would be the tempting
    shortcut and would make the identity unfalsifiable exactly where it stops
    being verified. Unevaluated is the honest result, and the pattern scan
    already drops a column it cannot parse rather than guessing at it.
    """
    import sympy as sp

    from rh_research_engine.symbolic.functions import (
        _FAREY_LIMIT,
        _REDHEFFER_LIMIT,
        FareyCount,
        FareyDeviation,
        RedhefferDet,
    )

    beyond = sp.Integer(_REDHEFFER_LIMIT + 1)
    # Unevaluated, not a number. `is_number` is TRUE of an unevaluated
    # function applied to a number, so it is the wrong question -- what
    # matters is that no value came back.
    assert RedhefferDet(beyond).func is RedhefferDet
    assert not RedhefferDet(beyond).is_Integer

    beyond_farey = sp.Integer(_FAREY_LIMIT + 1)
    assert FareyCount(beyond_farey).func is FareyCount
    assert not FareyCount(beyond_farey).is_Integer
    assert FareyDeviation(beyond_farey).func is FareyDeviation
    assert not FareyDeviation(beyond_farey).is_Rational

    # And below it, all three do return one.
    assert RedhefferDet(sp.Integer(_REDHEFFER_LIMIT)).is_Integer
    assert FareyCount(sp.Integer(_FAREY_LIMIT)).is_Integer


def test_the_totient_is_a_real_function_and_not_a_stub():
    r"""`\phi(n)` parsed to `Function('phi')(n)`, which evaluates to nothing.

    The Farey count is written with it, and a formula whose terms cannot be
    evaluated cannot be caught being wrong -- the exact failure that let
    Robin's inequality sit in the index for months with `Function('sigma')`
    standing in for the divisor sum.
    """
    from rh_research_engine.symbolic.parser import parse_math

    parsed = parse_math(r"\phi(12) = \phi(12)")
    assert parsed.parse_error is None
    assert "Integer(4)" in parsed.sympy_srepr, parsed.sympy_srepr
