"""The guard that keeps a formula from being left broken.

A gate that cannot fail is decoration, so these break each invariant and
confirm the guard notices. Two of these checks originally did NOT notice: the
name-resolution check read a five-line window and caught the NEIGHBOURING
call's `local_dict`, and the stub check looked for a SymPy attribute spelled
the same as the LaTeX name, which `\\sigma` -> `divisor_sigma` is not.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GUARD = REPO / "tools" / "formula-guard.py"


def _guard_module():
    spec = importlib.util.spec_from_file_location("formula_guard", GUARD)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registered before execution: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, which is absent for a module loaded by path.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_repository_currently_passes():
    """Integration: the checked-in tree reads every formula."""
    result = subprocess.run(
        [sys.executable, str(GUARD)], capture_output=True, text=True, encoding="utf-8", cwd=REPO
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EVERY FORMULA READS" in result.stdout


def test_a_reader_with_its_own_name_resolution_is_caught(tmp_path, monkeypatch):
    """Per call, not per line.

    The two calls here are adjacent and only the first has lost its
    `local_dict`. A window-based check reads the second one's and passes.
    """
    module = _guard_module()
    source = tmp_path / "reader.py"
    source.write_text(
        "from sympy.parsing.sympy_parser import parse_expr\n"
        "def read(a, b):\n"
        "    lhs = parse_expr(a, transformations=())\n"
        "    rhs = parse_expr(b, transformations=(), local_dict={})\n"
        "    return lhs, rhs\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "SYMBOLIC", tmp_path)
    # minimum=1: this test points SYMBOLIC at a one-file directory to exercise
    # the per-call logic, so it opts out of the enumeration floor explicitly.
    check = module.one_name_resolution_policy(minimum=1)
    assert not check.passed
    assert len(check.failures) == 1
    assert "reader.py:3" in check.failures[0]


def test_a_reader_that_supplies_locals_is_accepted(tmp_path, monkeypatch):
    module = _guard_module()
    source = tmp_path / "reader.py"
    source.write_text(
        "from sympy.parsing.sympy_parser import parse_expr\n"
        "def read(a):\n"
        "    return parse_expr(a, transformations=(), local_dict={})\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "SYMBOLIC", tmp_path)
    # minimum=1 for the same reason as above: a one-file SYMBOLIC.
    assert module.one_name_resolution_policy(minimum=1).passed


@pytest.mark.parametrize("name", ["sigma", "zeta", "pi", "li", "H", "M", "xi", "N"])
def test_the_names_that_must_denote_functions_are_listed(name):
    """The mapping is what is being protected, so it is stated, not inferred.

    `\\sigma` is `divisor_sigma`: the spellings differ, so nothing can derive
    this list from SymPy's namespace.
    """
    assert name in _guard_module().MUST_DENOTE_A_FUNCTION


def test_a_formula_that_does_not_parse_is_caught(tmp_path, monkeypatch):
    module = _guard_module()
    corpus = tmp_path / "corpus.md"
    corpus.write_text(r"$$\xi(s) = \bogus{1-s}$$" + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "CORPUS", corpus)
    check = module.every_formula_parses()
    assert not check.passed
    assert "bogus" in check.failures[0]


def test_an_undeclared_free_symbol_is_caught(tmp_path, monkeypatch):
    """A free symbol nobody declared is a name that failed to resolve.

    This is what makes `i` knowable. Left as a Symbol it is indistinguishable
    from a variable BY ITS SHAPE -- but not by its name, because the corpus
    knows which names are variables and `i` is not one of them.
    """
    module = _guard_module()
    corpus = tmp_path / "corpus.md"
    corpus.write_text(r"$$\zeta(1/2 + i q) = 0$$" + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "CORPUS", corpus)
    monkeypatch.setitem(module.DECLARED_SYMBOLS, "q", "a test variable")

    # With `i` resolving, only the declared `q` is free.
    assert module.every_symbol_is_declared().passed

    monkeypatch.delitem(module.DECLARED_SYMBOLS, "q")
    check = module.every_symbol_is_declared()
    assert not check.passed
    assert "`q` is a free symbol" in check.failures[0]


def test_an_identity_that_is_false_is_caught(tmp_path, monkeypatch):
    """The functional equation without `sin(pi*s/2)` gives +1/12 at s = -1.

    zeta(-1) is -1/12. The formula parsed, indexed and fingerprinted for weeks
    in that state; evaluating it takes microseconds and settles it.
    """
    module = _guard_module()
    corpus = tmp_path / "corpus.md"
    corpus.write_text(
        r"$$\zeta(s) = 2(2\pi)^{s-1}\Gamma(1-s)\zeta(1-s)$$" + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "CORPUS", corpus)
    check = module.an_identity_holds_numerically()
    assert not check.passed
    assert "left =" in check.failures[0] and "right =" in check.failures[0]


def test_the_real_functional_equation_passes(tmp_path, monkeypatch):
    module = _guard_module()
    corpus = tmp_path / "corpus.md"
    corpus.write_text(
        r"$$\zeta(s) = 2(2\pi)^{s-1}\sin(\pi s/2)\Gamma(1-s)\zeta(1-s)$$" + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "CORPUS", corpus)
    assert module.an_identity_holds_numerically().passed


def test_a_definition_is_not_treated_as_an_identity(tmp_path, monkeypatch):
    """`Theta = 1/2 + theta/2` introduces a symbol on one side.

    It is a definition, true of the value it defines rather than of every
    value, so substituting into it would manufacture a failure. The filter is
    that an identity's two sides involve the SAME variables.
    """
    module = _guard_module()
    corpus = tmp_path / "corpus.md"
    corpus.write_text(r"$$\Theta = 1/2 + \theta/2$$" + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "CORPUS", corpus)
    assert module.an_identity_holds_numerically().passed


def test_the_numeric_checks_cover_the_formulas_that_matter():
    """A check that quietly stops covering something looks like one that passes.

    The previous version of this test reimplemented the guard's selection
    logic and drifted out of step with it, which is the failure mode it was
    supposed to prevent. It now reads what the checks REPORT having evaluated,
    so the two cannot disagree.
    """
    module = _guard_module()
    covered: set[str] = set()
    for check in module.CHECKS:
        if check.__name__ in module.NUMERIC_CHECKS:
            covered.update(check().covered)

    def present(fragment: str) -> bool:
        return any(fragment in source for source in covered)

    # One from each numeric check, named rather than counted, so a drop is
    # attributable rather than just a smaller number.
    assert present(r"\sin(\pi s/2)"), "the functional equation is unchecked"
    assert present(r"\xi(s) = \xi(1-s)"), "the xi functional equation is unchecked"
    assert present(r"\prod_p"), "the Euler product is unchecked"
    assert present(r"\mu(n) n^{-s}"), "the Mobius series is unchecked"
    assert present("5040"), "Robin's inequality is unchecked"
    assert present("2657"), "Schoenfeld's bound is unchecked"
    assert present(r"\binom{k}{j}"), "Baez-Duarte is unchecked"
    assert present(r"\Lambda = 0"), "the de Bruijn-Newman statement is unchecked"
    assert present(r"\lambda_n \ge 0"), "Li's criterion is unchecked"

    assert len(covered) >= 38, f"coverage fell to {len(covered)}"


def test_the_fast_subset_is_a_latency_split_not_a_strength_split():
    """`--fast` may skip checks; it may not skip them in CI.

    Every name in NUMERIC_CHECKS has to BE a check, or the skip list is
    silently skipping nothing -- or worse, a renamed check stops being run
    anywhere.
    """
    module = _guard_module()
    names = {check.__name__ for check in module.CHECKS}
    assert module.NUMERIC_CHECKS <= names, module.NUMERIC_CHECKS - names
    assert len(names - module.NUMERIC_CHECKS) >= 5


def test_a_series_with_the_wrong_closed_form_is_caught(tmp_path, monkeypatch):
    """A truncation must APPROACH the value the formula claims.

    Equality is the wrong test for an infinite series -- any truncation is
    wrong by something -- so the check is convergence: truncate twice, and the
    error must shrink. A series converging to a different value sits at a
    fixed distance from the claim.
    """
    module = _guard_module()
    corpus = tmp_path / "corpus.md"
    corpus.write_text(
        r"$$\forall s, \Re(s) > 1: \sum_{n=1}^{\infty} \mu(n) n^{-s} = 2/\zeta(s)$$"
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "CORPUS", corpus)
    check = module.a_series_converges_to_its_closed_form()
    assert not check.passed
    assert check.covered, "the check skipped it rather than failing it"


def test_the_real_dirichlet_series_converges(tmp_path, monkeypatch):
    module = _guard_module()
    corpus = tmp_path / "corpus.md"
    corpus.write_text(
        r"$$\forall s, \Re(s) > 1: \sum_{n=1}^{\infty} \mu(n) n^{-s} = 1/\zeta(s)$$"
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "CORPUS", corpus)
    check = module.a_series_converges_to_its_closed_form()
    assert check.passed, check.failures
    assert check.covered


def test_a_criterion_claimed_below_its_threshold_is_caught(tmp_path, monkeypatch):
    """Robin's inequality is FALSE at n = 5040.

    That is what makes the hypothesis testable rather than decorative: move
    the threshold one below the known exception and the claim becomes false at
    a point the check probes.
    """
    module = _guard_module()
    corpus = tmp_path / "corpus.md"
    corpus.write_text(
        r"$$\forall n, n > 5039: \sigma(n) < e^{\EulerGamma} n \log\log n$$" + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "CORPUS", corpus)
    check = module.a_stated_relation_holds()
    assert not check.passed
    assert "5040" in check.failures[0]


def test_the_real_threshold_passes(tmp_path, monkeypatch):
    module = _guard_module()
    corpus = tmp_path / "corpus.md"
    corpus.write_text(
        r"$$\forall n, n > 5040: \sigma(n) < e^{\EulerGamma} n \log\log n$$" + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "CORPUS", corpus)
    check = module.a_stated_relation_holds()
    assert check.passed, check.failures
    assert check.covered


def test_a_conditioned_identity_is_not_probed_outside_its_domain(tmp_path, monkeypatch):
    """Gamma's integral diverges for Re s <= 0.

    A checker that ignores the condition reports the definition of Gamma as
    false -- which it did, at s = -1.3, before the domain was carried in the
    formula.
    """
    module = _guard_module()
    corpus = tmp_path / "corpus.md"
    corpus.write_text(
        r"$$\forall s, \Re(s) > 0: \Gamma(s) = \int_0^{\infty} t^{s-1}e^{-t} dt$$"
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "CORPUS", corpus)
    check = module.an_identity_holds_numerically()
    assert check.passed, check.failures


def _corpus(tmp_path, monkeypatch, module, body: str):
    corpus = tmp_path / "corpus.md"
    corpus.write_text(body + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "CORPUS", corpus)
    return corpus


def test_a_bound_with_the_wrong_exponent_is_caught(tmp_path, monkeypatch):
    """`pi(x) = li(x) + O(log x)` is false, and only a SLOPE sees it.

    Over two decades that bound is about three times worse than the true one,
    which is inside any endpoint threshold loose enough to allow the honest
    wandering of O(). Judged as a slope in log-log it is unmistakable.
    """
    module = _guard_module()
    _corpus(tmp_path, monkeypatch, module, r"$$\pi(x) = li(x) + O(\log x)$$")
    check = module.an_asymptotic_bound_stays_bounded()
    assert not check.passed
    assert "grows like" in check.failures[0]


def test_the_real_von_koch_bound_passes(tmp_path, monkeypatch):
    module = _guard_module()
    _corpus(tmp_path, monkeypatch, module, r"$$\pi(x) = li(x) + O(\sqrt{x}\log x)$$")
    check = module.an_asymptotic_bound_stays_bounded()
    assert check.passed, check.failures
    assert check.covered


def test_a_residual_that_dips_near_zero_is_not_a_missing_power(tmp_path, monkeypatch):
    """The zero-counting formula, sampled over four decades, is not growth.

    Its ratio runs 0.0005, 0.1048, 0.0356 -- rising and then FALLING. The first
    sample is tiny because S(100) happens to be about 0.002, and an endpoint
    slope reads that lucky near-zero as x^0.46 and calls a true bound a missing
    power.

    This could not be seen while `ZeroCount` was computed by locating every
    zero below T, which capped the sample at (20, 60, 180). Making the count
    cheap made the check wrong, which is the useful kind of breakage.
    """
    module = _guard_module()
    _corpus(
        tmp_path,
        monkeypatch,
        module,
        r"$$N(T) = (T/(2\pi))\log(T/(2\pi)) - T/(2\pi) + 7/8 + O(\log T)$$",
    )
    check = module.an_asymptotic_bound_stays_bounded()
    assert check.passed, check.failures
    assert check.covered, "and it must actually have been sampled"


def test_growth_that_never_falls_back_is_still_caught(tmp_path, monkeypatch):
    """The looser criterion must not become no criterion.

    `pi(x) = li(x) + O(log x)` rises at every step -- 1.1, 1.85, 9.4 -- so
    tolerating a residual that falls back does not let it through. Requiring
    every STEP to clear the threshold would have: its first step is 0.11.
    """
    module = _guard_module()
    _corpus(tmp_path, monkeypatch, module, r"$$\pi(x) = li(x) + O(\log x)$$")
    check = module.an_asymptotic_bound_stays_bounded()
    assert not check.passed
    assert "without falling back" in check.failures[0]


def test_contradictory_statements_about_one_subject_are_caught(tmp_path, monkeypatch):
    """`Theta = 1/2` against `Theta >= 1` cannot both hold.

    This is how the RH statements become checkable: they cannot be verified,
    but they can be held against the proved bounds, and a corpus that
    contradicts itself is broken whatever the answer turns out to be.
    """
    module = _guard_module()
    _corpus(
        tmp_path, monkeypatch, module,
        "$$\\Theta = 1/2$$\n\n$$\\Theta \\ge 1$$",
    )
    check = module.definitions_are_consistent()
    assert not check.passed
    assert "contradicts" in check.failures[0]


def test_the_real_theta_statements_agree(tmp_path, monkeypatch):
    module = _guard_module()
    _corpus(
        tmp_path, monkeypatch, module,
        "$$\\Theta = 1/2$$\n\n$$\\Theta \\ge 1/2$$\n\n$$\\Theta = 1/2 + \\theta/2$$",
    )
    check = module.definitions_are_consistent()
    assert check.passed, check.failures
    assert check.covered


def test_a_definition_with_no_value_is_caught(tmp_path, monkeypatch):
    """Li's definition without its evaluation point defines nothing.

    The right side stays a function of s while the left is a number. That is
    not a weaker definition; it is not one.
    """
    module = _guard_module()
    _corpus(
        tmp_path, monkeypatch, module,
        r"$$\lambda_n = \frac{1}{(n-1)!}\frac{d^n}{ds^n}(s^{n-1}\log\xi(s))$$",
    )
    check = module.a_definition_is_computable()
    assert not check.passed
    assert "produces no value" in check.failures[0]


def test_a_limit_moving_away_from_its_value_is_caught(tmp_path, monkeypatch):
    module = _guard_module()
    _corpus(tmp_path, monkeypatch, module, r"$$\lim_{x \to \infty} \pi(x)\log(x)/x = 2$$")
    check = module.a_limit_approaches_its_value()
    assert not check.passed
    assert "moves AWAY" in check.failures[0]


def test_the_prime_number_theorem_limit_approaches_one(tmp_path, monkeypatch):
    module = _guard_module()
    _corpus(tmp_path, monkeypatch, module, r"$$\lim_{x \to \infty} \pi(x)\log(x)/x = 1$$")
    check = module.a_limit_approaches_its_value()
    assert check.passed, check.failures
    assert check.covered

def test_the_statements_about_the_zeros_are_covered_and_hold():
    """The last two formulas nothing could evaluate.

    Both need thousands of zeros, and a zero cost 160 ms until
    `riemann_siegel`. With them checkable the guard reaches every formula in
    the corpus rather than all but two.
    """
    module = _guard_module()
    check = module.a_statement_about_the_zeros_holds()
    assert check.passed, check.failures
    assert len(check.covered) == 2, check.covered
    assert any("rho(k)" in source for source in check.covered)
    assert any("sin(\\pi u)" in source for source in check.covered)


def test_a_dropped_constant_in_the_explicit_formula_is_caught(tmp_path, monkeypatch):
    """The failure the check exists for.

    A missing `log(2 pi)` parses, indexes, fingerprints and exports exactly as
    well as the true statement. Only a value tells them apart, and it does so by
    a mile: the residual becomes the size of the dropped term, 1.84, against a
    truncation error of 0.024.
    """
    import json

    from rh_research_engine.symbolic import explicit_formula

    crippled = tmp_path / "index.json"
    crippled.write_text(
        json.dumps(
            [
                {
                    "canonical": (
                        "Eq(ChebyshevPsi(Symbol('x')), Add(Symbol('x'), "
                        "Mul(Integer(-1), Rational(1, 2), "
                        "log(Add(Integer(1), Mul(Integer(-1), "
                        "Pow(Symbol('x'), Integer(-2)))))), "
                        "Mul(Integer(-1), Integer(2), "
                        "Sum(re(Mul(Pow(Symbol('x'), NthZetaZero(Symbol('k'))), "
                        "Pow(NthZetaZero(Symbol('k')), Integer(-1)))), "
                        "Tuple(Symbol('k'), Integer(1), oo)))))"
                    )
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(explicit_formula, "INDEX_PATH", crippled)

    module = _guard_module()
    check = module.a_statement_about_the_zeros_holds()
    assert not check.passed
    assert any("missing" in failure for failure in check.failures)
    assert any(
        "constant_is_minus_log_two_pi" in failure for failure in check.failures
    )


def test_a_corpus_without_the_formulas_fails_rather_than_passing_vacuously(
    tmp_path, monkeypatch
):
    """Nothing to check is not the same as everything checking out.

    A check that quietly covers zero formulas when the corpus loses them is a
    check that stops guarding on the day it matters.
    """
    import json

    from rh_research_engine.symbolic import explicit_formula, pair_correlation

    empty = tmp_path / "index.json"
    empty.write_text(json.dumps([{"canonical": "Symbol('x')"}]), encoding="utf-8")
    monkeypatch.setattr(explicit_formula, "INDEX_PATH", empty)
    monkeypatch.setattr(pair_correlation, "INDEX_PATH", empty)

    module = _guard_module()
    check = module.a_statement_about_the_zeros_holds()
    assert not check.passed
    assert check.covered == []


def test_an_enumeration_that_sees_nothing_is_not_a_pass(tmp_path, monkeypatch):
    """A check that iterates can succeed at iterating over nothing.

    `one_name_resolution_policy` was already fooled once -- a five-line window
    read the NEIGHBOURING call's `local_dict` -- and the repair hardened what it
    does with each file while leaving "did it see any files" unasked. Point
    SYMBOLIC at a directory with no modules and the loop body never runs,
    `failures` stays empty, and it reports that one name-resolution policy holds
    across the package having examined zero files.

    Breaking the checked thing does not find this: an injected missing
    `local_dict` still fails. The hole is not that the gate cannot fail today,
    it is that it can succeed at scanning nothing after a rename or a moved
    directory. Verified by construction before the floor existed: PASS on zero
    files.
    """
    module = _guard_module()
    empty = tmp_path / "no_modules"
    empty.mkdir()
    monkeypatch.setattr(module, "SYMBOLIC", empty)

    # minimum=0 is the guard as it was before the floor: no lower bound at all.
    # It passes, which is the defect reproduced rather than described.
    assert module.one_name_resolution_policy(minimum=0).passed, (
        "without a floor an empty enumeration must pass -- that is the defect"
    )
    check = module.one_name_resolution_policy()
    assert not check.passed
    assert "below the floor" in check.failures[0]
    assert module.MINIMUM_SYMBOLIC_MODULES >= 1


def test_a_missing_index_is_not_a_passing_check(monkeypatch, tmp_path):
    """`return check` on an absent file returned a check with no failures.

    So a missing `formula_index.json` reported that the index holds nothing the
    corpus dropped, on the strength of never having opened it. It is regenerable
    by `rhre symbolic ingest`, so absent is plausible rather than exotic -- which
    makes a silent pass more likely to be met, not less.
    """
    module = _guard_module()
    monkeypatch.setattr(module, "REPO", tmp_path)
    (tmp_path / "research_state").mkdir()

    check = module.the_index_holds_no_superseded_record()
    assert not check.passed
    assert "does not exist" in check.failures[0]
    assert "symbolic ingest" in check.failures[0]

