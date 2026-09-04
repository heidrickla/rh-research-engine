"""Boundary tests: every assertion here is a defect that was exploitable.

These are the negative tests from the adversarial review. Each one failed
against the pre-fix tree. They exist to keep the epistemic boundaries closed,
so treat a failure here as a regression in what the system is *allowed to
conclude*, not merely in what it computes.
"""

import pytest

from rh_research_engine.core.bounds import correlation_remainder_to_theta
from rh_research_engine.core.models import Claim, ClaimStatus, EvidenceClass, ExperimentResult
from rh_research_engine.core.nogo import violations
from rh_research_engine.core.promote import evaluate_export
from rh_research_engine.core.scoring import score_claim
from rh_research_engine.dre import ClaimEffect, DreEvidenceEnvelope, WorkerClassError
from rh_research_engine.mathcert import (
    ImpossibleIntervalError,
    MathCertificate,
    RealInterval,
    VerifierMetadata,
    certificate_predicates,
    screening_exponent_to_theta,
)
from rh_research_engine.symbolic import (
    check_certificate_against_expression,
    equivalent,
    export_polynomial_identity,
    fingerprint,
    minimize_conjecture,
    parse_math,
    safe_binomial_decay_to_theta,
    screening_remainder_to_theta,
    simplify_with_trace,
)
from rh_research_engine.symbolic.exponents import ImpossibleBoundError
from rh_research_engine.symbolic.models import EquationKind


def _numerical_run() -> ExperimentResult:
    return ExperimentResult(
        name="correlation-lab",
        parameters={"X": 3000, "q": 4.0},
        metrics={"screening_remainder": 1.4301704821488714},
        observations=["No rigorous bound is inferred from a numerical value."],
    )


def _envelope(**kw) -> DreEvidenceEnvelope:
    return DreEvidenceEnvelope.from_experiment(
        _numerical_run(), claim_id="C005", primary_metric_name="screening_remainder", **kw
    )


# --- F-01: numerical evidence cannot be relabelled as proof ------------------


@pytest.mark.parametrize("forbidden", [EvidenceClass.PROVED, EvidenceClass.KNOWN])
def test_worker_cannot_assert_proof_or_known(forbidden):
    # Pydantic wraps a validator failure in ValidationError, which subclasses
    # ValueError; the message is what callers act on.
    with pytest.raises(ValueError, match="may not assert evidence_class"):
        DreEvidenceEnvelope(
            experiment_name="correlation-lab",
            claim_id="C005",
            claim_effect=ClaimEffect.SUPPORTS,
            evidence_class=forbidden,
            method_family="python-numpy",
            worker_version="0.9.0",
            metrics={"screening_remainder": 1.43},
        )


def test_evidence_class_cannot_be_overridden_at_export_time():
    with pytest.raises(WorkerClassError):
        DreEvidenceEnvelope.from_experiment(
            _numerical_run(), claim_id="C005", evidence_class=EvidenceClass.SYMBOLIC
        )


def test_export_takes_provenance_from_the_experiment_record():
    env = _envelope()
    assert env.evidence_class is EvidenceClass.NUMERICAL
    assert env.method_family == "python-numpy"


def test_numerical_evidence_cannot_assert_a_theta_bound():
    with pytest.raises(ValueError, match="no deductive force"):
        _envelope(theta_upper=0.5)


def test_theta_below_one_half_is_impossible():
    result = ExperimentResult(
        name="x", parameters={}, metrics={}, evidence_class=EvidenceClass.SYMBOLIC
    )
    with pytest.raises(ValueError, match="below 1/2"):
        DreEvidenceEnvelope.from_experiment(result, claim_id="C005", theta_upper=0.25)


def test_independent_verification_must_be_earned():
    with pytest.raises(ValueError, match="independently_verified"):
        _envelope(independently_verified=True)


# --- F-05: independence groups cannot be manufactured -----------------------


def test_relabelling_cannot_manufacture_a_second_witness():
    """One numpy run used to become three 'independent' corroborating sources."""
    a = _envelope()
    b = _envelope()
    assert a.independence_group == b.independence_group
    assert a.payload_hash == b.payload_hash


def test_payload_hash_ignores_provenance_labels():
    base = _numerical_run()
    relabelled = base.model_copy(update={"method_family": "arb-interval"})
    a = DreEvidenceEnvelope.from_experiment(base, claim_id="C005")
    b = DreEvidenceEnvelope.from_experiment(relabelled, claim_id="C005")
    assert a.payload_hash == b.payload_hash, "dedup must survive relabelling"
    assert a.provenance_hash != b.provenance_hash, "provenance must remain distinguishable"


def test_payload_hash_absorbs_cross_platform_float_drift():
    """Same experiment, different machine, last-ULP difference: one identity."""
    windows = _numerical_run()
    linux = windows.model_copy(
        update={"metrics": {"screening_remainder": 1.4301704821488714 + 1.2e-17}}
    )
    a = DreEvidenceEnvelope.from_experiment(windows, claim_id="C005")
    b = DreEvidenceEnvelope.from_experiment(linux, claim_id="C005")
    assert a.payload_hash == b.payload_hash


# --- F-02: assumptions survive the DRE boundary -----------------------------


def test_certificate_assumptions_reach_dre():
    cert = MathCertificate(
        expression="R_q(X)",
        value=RealInterval.from_decimals("0.0", "0.0"),
        verifier=VerifierMetadata(method="arb", precision_bits=256),
        assumptions=["assumes RH", "valid only for X < 10**6"],
    )
    facts = certificate_predicates(cert)
    assert facts["assumption_count"] == 2
    assert facts["assumptions_present"] is True
    assert facts["unconditional"] is False
    assert "assumes RH" in str(facts)


def test_unconditional_certificate_says_so():
    cert = MathCertificate(
        expression="R_q(X)",
        value=RealInterval.from_decimals("0.1", "0.2"),
        verifier=VerifierMetadata(method="arb", precision_bits=256),
    )
    facts = certificate_predicates(cert)
    assert facts["unconditional"] is True
    assert facts["assumption_count"] == 0


def test_dre_payload_still_excludes_high_precision_endpoints():
    cert = MathCertificate(
        expression="x",
        value=RealInterval.from_decimals("0.0000000000000000001", "0.0000000000000000002"),
        verifier=VerifierMetadata(method="arb", precision_bits=512),
    )
    assert "0.0000000000000000001" not in str(certificate_predicates(cert))


# --- F-10 / F-11: bound maps reject impossible outputs ----------------------


def test_negative_remainder_exponent_is_rejected_not_clamped():
    with pytest.raises(ValueError):
        correlation_remainder_to_theta(-0.3)


@pytest.mark.parametrize("value", [-0.5, -1.0])
def test_screening_map_rejects_negative_exponents(value):
    with pytest.raises(ImpossibleBoundError):
        screening_remainder_to_theta(value)


@pytest.mark.parametrize("alpha", [1.0, 5.0])
def test_binomial_map_rejects_impossible_theta(alpha):
    with pytest.raises(ImpossibleBoundError):
        safe_binomial_decay_to_theta(alpha)


def test_binomial_map_accepts_the_rh_endpoint():
    assert safe_binomial_decay_to_theta(0.75).theta_upper == pytest.approx(0.5)


def test_certified_interval_map_rejects_negative_enclosure():
    with pytest.raises(ImpossibleIntervalError):
        screening_exponent_to_theta(RealInterval.from_decimals("-1.0", "-0.9"))


def test_minimizer_refuses_to_narrate_an_impossible_target():
    result = minimize_conjecture("prove screening remainder R_q(X) = O(X^-0.5) for one fixed q")
    assert result.rule_id == "MIN-REJECT-IMPOSSIBLE-EXPONENT"
    assert not result.changed
    assert any("impossible" in r for r in result.rationale)
    assert not any("useful partial progress" in r for r in result.rationale)


# --- F-06: domain information survives canonicalization ---------------------


def test_cancellation_records_the_nonzero_denominator_assumption():
    result = simplify_with_trace("(x**2-1)/(x-1)")
    assert result.simplified in {"x + 1", "x + 1.0"}
    assert "x - 1 != 0" in result.assumptions


@pytest.mark.parametrize("left,right", [("(x**2-1)/(x-1)", "x+1"), ("x/x", "1"), ("x*(1/x)", "1")])
def test_equality_across_a_removable_singularity_is_conditional(left, right):
    result = equivalent(left, right)
    assert result.assumptions, "domain gap must be reported"
    assert result.method.endswith("_conditional")
    assert fingerprint(left).sha256 != fingerprint(right).sha256


def test_genuinely_equal_expressions_still_match():
    assert fingerprint("(x+1)**2").sha256 == fingerprint("x**2+2*x+1").sha256
    result = equivalent("(x+1)**2", "x**2+2*x+1")
    assert result.equivalent is True
    assert result.assumptions == []


def test_certificate_is_not_substitutable_across_a_pole():
    cert = MathCertificate(
        expression="x+1",
        value=RealInterval.from_decimals("2.0", "2.0000001"),
        verifier=VerifierMetadata(method="arb", precision_bits=256),
    )
    checked = check_certificate_against_expression(cert, "(x**2-1)/(x-1)")
    assert checked.usable is False
    assert checked.domain_gap == ["x - 1 != 0"]
    assert checked.warnings


# --- F-07: the LaTeX parser fails closed ------------------------------------


@pytest.mark.parametrize("src", [r"\frac{1}{}", r"\frac{1}", r"\frac{1}{2"])
def test_malformed_fraction_fails_closed(src):
    """`\\frac{1}` used to parse as the literal constant 0."""
    assert parse_math(src).parse_error is not None


@pytest.mark.parametrize(
    "src, head",
    [
        (r"\lim_{X\to\infty} R_q(X) = 0", "Limit"),
        (r"\sum_{n\ge2}(\Psi(n)-1)e^{-(n/X)^q} = 0", "Sum"),
        (r"\int_0^\infty f(u)\,du = 1", "Integral"),
        (r"\prod_p (1-p^{-s})^{-1}", "Product"),
    ],
)
def test_binders_keep_their_bounds(src, head):
    r"""A binder's bounds separate a finite check from an asymptotic theorem.

    Each of these is now carried exactly rather than refused, which makes the
    thing to guard against different: not that the binder is missing, but that
    it survives with its range intact. A `\sum` flattened into ordinary
    algebra, or one whose bounds were quietly dropped, would be the failure.
    """
    import sympy as sp

    from rh_research_engine.symbolic.functions import SREPR_NAMESPACE

    parsed = parse_math(src)
    assert parsed.parse_error is None, parsed.parse_error
    expression = sp.sympify(parsed.sympy_srepr, locals=dict(SREPR_NAMESPACE))
    binders = [node for node in sp.preorder_traversal(expression) if type(node).__name__ == head]
    assert binders, f"{head} vanished from {expression}"
    assert binders[0].args[1:], f"{head} kept no range in {expression}"


def test_a_quantified_statement_keeps_its_guard():
    r"""`\forall s, \Re(s)>1: P` is an implication, not a bare `P`.

    Dropping the quantifier loses nothing -- the variable is free either way.
    Dropping the guard would lose the domain the claim is restricted to, and
    turn a true statement about a half-plane into a false one about all of C.
    """
    import sympy as sp

    parsed = parse_math(r"\forall s, \Re(s)>1: \zeta(s)=1")
    assert parsed.parse_error is None, parsed.parse_error
    expression = sp.sympify(parsed.sympy_srepr)
    assert isinstance(expression, sp.Implies)
    assert expression.args[0].has(sp.re)


def test_stray_comma_producing_a_tuple_is_rejected():
    assert parse_math("a, b").parse_error is not None


def test_unknown_control_sequence_is_rejected():
    assert parse_math(r"\bogus{x}").parse_error is not None


@pytest.mark.parametrize(
    "src", [r"\frac{1}{s-1}", "x^2+2*x+1=(x+1)^2", r"\frac{2\pi}{s-1}", r"(x+1)^{2}"]
)
def test_well_formed_algebra_still_parses(src):
    assert parse_math(src).parse_error is None


# --- function application is represented, not multiplied --------------------
#
# This parser forces Greek names to plain Symbols so that a beta/gamma/zeta
# command does not resolve to the SymPy *function* and raise on a missing
# argument. The cost was that implicit multiplication then read an applied
# symbol as a product and reported a CLEAN parse -- the worst outcome, because
# nothing downstream had any way to notice.
#
# It was refused outright for a while, which was safe but lost the formulas.
# Applied names now become Functions, decided from the syntax rather than from
# a list of known names, so `f(x)` means f applied to x.


@pytest.mark.parametrize(
    "src,applied",
    [
        (r"\xi(s) = \xi(1-s)", "RiemannXi(Symbol('s'))"),
        # `\Gamma` is SymPy's gamma, not a stub of the same name.
        (r"\frac{\Gamma(s)}{2\pi}", "gamma(Symbol('s'))"),
        # A name with no counterpart stays an undefined function.
        (r"F(T) = T", "Function('F')"),
    ],
)
def test_an_applied_name_parses_as_a_function(src, applied):
    result = parse_math(src)
    assert result.parse_error is None, result.parse_error
    assert applied in result.sympy_srepr, result.sympy_srepr


def test_the_xi_functional_equation_is_not_turned_into_a_false_statement():
    """It used to parse as Eq(s*xi, xi*(1 - s)) -- true statement, false parse."""
    result = parse_math(r"\xi(s) = \xi(1-s)")
    assert result.parse_error is None
    srepr = result.sympy_srepr

    # Applied on both sides ...
    assert srepr.count("RiemannXi(") == 2
    # ... and never multiplied by its own argument.
    assert "Mul(Symbol('s'), Symbol('xi'))" not in srepr
    assert "Symbol('xi')" not in srepr


def test_two_applications_do_not_collapse_into_a_squared_factor():
    """`Gamma(1-s) zeta(1-s)` produced ... *(1 - s)*(1 - s) -- an invented square."""
    import sympy as sp

    result = parse_math(r"\Gamma(1-s)\zeta(1-s)")
    assert result.parse_error is None
    expr = sp.sympify(result.sympy_srepr)

    # Two distinct applications of one argument, not that argument squared.
    s = sp.Symbol("s")
    assert expr == sp.gamma(1 - s) * sp.zeta(1 - s)
    assert not expr.has(sp.Pow(1 - s, 2))


@pytest.mark.parametrize("src", ["2(x+1)", "2*(x+1)", "(x+1)(x-1)"])
def test_implicit_multiplication_by_a_number_or_group_still_parses(src):
    """The Function decision keys on an identifier, so arithmetic is untouched."""
    assert parse_math(src).parse_error is None


def test_a_digit_before_a_parenthesis_is_still_multiplication():
    """`2(x+1) = 2x+2` holds, and SymPy collapses it to True on sight.

    If the digit had been treated as an applied name the two sides would be
    unrelated and no such collapse could happen.
    """
    import sympy as sp

    result = parse_math("2(x+1) = 2x+2")
    assert result.parse_error is None
    assert sp.sympify(result.sympy_srepr) == sp.true


@pytest.mark.parametrize(
    "src,expect",
    [
        (r"\sigma(n) \le H(n)", "<="),
        (r"\lambda_n \ge 0", ">="),
    ],
)
def test_relation_commands_are_understood(src, expect):
    r"""`\le` and `\ge` were unmapped, so half the criteria were unreadable."""
    result = parse_math(src)
    assert result.parse_error is None, result.parse_error
    assert result.kind is EquationKind.INEQUALITY


def test_log_and_sqrt_reach_their_real_sympy_meaning():
    r"""Mapped to SymPy's own functions, not to opaque placeholders.

    Safe only because applied names are Functions now -- as Symbols, `\log(x)`
    would have parsed as `log*(x)`.
    """
    import sympy as sp

    result = parse_math(r"\sqrt{x}")
    assert result.parse_error is None
    assert sp.sympify(result.sympy_srepr) == sp.sqrt(sp.Symbol("x"))

    result = parse_math(r"\log(x)")
    assert result.parse_error is None
    assert sp.sympify(result.sympy_srepr) == sp.log(sp.Symbol("x"))


def test_an_integral_without_a_differential_infers_its_variable():
    r"""`\int_0^1 x` is over x, and every source that writes it means that.

    The differential is how an integrand names its variable, but an integrand
    that names exactly one identifier has already said it. What stays out of
    reach is an integrand naming several, where the `dx` carried information
    nothing else does -- and that says so.
    """
    import sympy as sp

    x = sp.Symbol("x")
    parsed = parse_math(r"\int_0^1 x")
    assert parsed.parse_error is None, parsed.parse_error
    assert sp.sympify(parsed.sympy_srepr) == sp.Integral(x, (x, 0, 1))

    ambiguous = parse_math(r"\int_0^1 x y")
    assert ambiguous.parse_error is not None
    assert "which one it runs over" in ambiguous.parse_error.lower()


def test_a_symbol_command_does_not_fuse_with_the_letter_before_it():
    r"""`i\gamma` became the single symbol `igamma_`, losing the imaginary unit.

    One name where the source wrote two. Nothing downstream could tell, because
    the parse succeeded and produced a perfectly ordinary symbol.

    `i` also has to BE the imaginary unit. Left as a Symbol it was a second
    unknown, so `1/2 + i*gamma` -- a point on the critical line -- was a
    product of two free variables that happened to print correctly.
    """
    import sympy as sp

    result = parse_math(r"\rho = 1/2 + i\gamma")
    assert result.parse_error is None
    assert "igamma" not in result.normalized
    expression = sp.sympify(result.sympy_srepr)
    assert expression.rhs == sp.Rational(1, 2) + sp.I * sp.Symbol("gamma")


# --- F-12: scoring cannot rank restatement above proof ----------------------


def test_rh_equivalent_claim_cannot_outrank_a_proved_claim():
    circular = Claim(
        id="E", statement="RH restated", status=ClaimStatus.EQUIVALENT_RH, implied_theta_upper=0.0
    )
    proved = Claim(id="P", statement="genuine partial result", status=ClaimStatus.PROVED)
    assert score_claim(circular).total < score_claim(proved).total
    assert score_claim(circular).total <= 0.0


def test_seed_rh_equivalence_tag_incurs_the_circularity_penalty():
    claim = Claim(
        id="C002",
        statement="RH is equivalent to ...",
        status=ClaimStatus.SYMBOLIC,
        tags={"rh_equivalence"},
    )
    assert score_claim(claim).assumption_penalty >= 10.0
    assert score_claim(claim).total <= 0.0


def test_unproved_claim_cannot_assert_theta_progress():
    claim = Claim(
        id="H",
        statement="fit suggests Theta<=0.5",
        status=ClaimStatus.HYPOTHESIS,
        implied_theta_upper=0.5,
    )
    assert score_claim(claim).progress == 0.0
    assert "ignored" in score_claim(claim).explanation


def test_refuted_claim_never_scores_positive():
    claim = Claim(id="F", statement="s", status=ClaimStatus.FALSE, implied_theta_upper=0.0)
    assert score_claim(claim).total <= 0.0


def test_seed_registry_scores_c002_as_circular():
    from rh_research_engine.core.bootstrap import seed_claims

    c002 = next(c for c in seed_claims() if c.id == "C002")
    assert c002.status is ClaimStatus.EQUIVALENT_RH
    assert score_claim(c002).total <= 0.0


# --- F-13: a reworded dead end is still a dead end --------------------------


def test_renaming_a_tag_does_not_resurrect_a_no_go_route():
    reworded = Claim(
        id="N2",
        statement=(
            "A novel all-pass phase-rigidity criterion for the zeta scattering quotient "
            "forces every zero onto the critical line"
        ),
        tags={"unitary_boundary_route_v2"},
    )
    assert any(v.id == "boundary-unitarity" for v in violations(reworded))


def test_tag_matching_still_works():
    claim = Claim(id="x", statement="bad", tags={"boundary_unitarity_only"})
    assert any(v.id == "boundary-unitarity" for v in violations(claim))


def test_unrelated_claim_trips_nothing():
    claim = Claim(id="ok", statement="The screening remainder is O(X^{1/4}) for fixed q.")
    assert violations(claim) == []


# --- F-14: serialization is byte-reproducible -------------------------------


def test_claim_tags_serialize_in_a_stable_order():
    claim = Claim(id="C", statement="s", tags={"b", "a", "c"})
    assert claim.model_dump(mode="json")["tags"] == ["a", "b", "c"]


# --- F-09: the gate actually stands in the path -----------------------------


def test_gate_blocks_a_theta_claim_from_numerical_evidence():
    result = _numerical_run().model_copy(update={"evidence_class": EvidenceClass.SYMBOLIC})
    env = DreEvidenceEnvelope.from_experiment(result, claim_id="C005", theta_upper=0.5)
    decision = evaluate_export(env, claim=Claim(id="C005", statement="ok"))
    assert decision.allowed


def test_gate_blocks_support_for_a_refuted_claim():
    env = _envelope()
    refuted = Claim(
        id="C003",
        statement="Boundary unitarity of the zeta scattering ratio alone proves RH.",
        status=ClaimStatus.FALSE,
        tags={"boundary_unitarity_only"},
    )
    decision = evaluate_export(env, claim=refuted)
    assert not decision.allowed
    assert any(f.gate == "no-go" for f in decision.blocks)


def test_gate_blocks_corrupt_durable_memory(tmp_path):
    bad = tmp_path / "k.json"
    bad.write_text(
        '[{"id":"K999","title":"t","status":"proved","domain":"d","statement":"s"}]',
        encoding="utf-8",
    )
    decision = evaluate_export(_envelope(), knowledge_path=bad)
    assert not decision.allowed
    assert any(f.gate == "durable-memory" for f in decision.blocks)


def test_gate_allows_an_honest_numerical_export():
    decision = evaluate_export(
        _envelope(), claim=Claim(id="C005", statement="screening remainder bound")
    )
    assert decision.allowed


def test_gate_warns_on_rh_equivalent_support():
    decision = evaluate_export(_envelope(rh_equivalent=True))
    assert decision.allowed
    assert any(f.gate == "rh-equivalence" for f in decision.warnings)


# --- F-19: the Lean exporter stays fail-closed and stops raising ------------


def test_lean_export_handles_float_coefficients():
    out = export_polynomial_identity("0.5*x", "x/2", "half")
    assert out.supported is True


@pytest.mark.parametrize(
    "left,right", [("(x**2-1)/(x-1)", "x+1"), ("x/x", "1"), ("sin(x)", "x"), ("1", "1")]
)
def test_lean_export_refuses_non_polynomial_identities(left, right):
    assert export_polynomial_identity(left, right, "t").supported is False


def test_lean_export_still_emits_verified_polynomial_identities():
    out = export_polynomial_identity("(x+1)**2", "x**2+2*x+1", "square_identity")
    assert out.supported is True
    assert "ring" in (out.lean or "")


# --- missing durable memory blocks the export end to end --------------------


def test_no_export_is_written_when_durable_memory_is_missing(tmp_path):
    """The whole point of the gate, exercised through the CLI.

    `dre export-latest` passes an explicit knowledge path. That used to bypass
    the resolver's fail-closed behaviour and return an empty list, so every
    no-go and route check passed vacuously and the artifact was written anyway.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "research_state").mkdir()
    # An experiment to export, but no durable memory anywhere.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rh_research_engine.cli",
            "experiment",
            "gamma-filter",
            "--x",
            "100",
            "--q",
            "2",
        ],
        cwd=workspace,
        capture_output=True,
        text=True, encoding="utf-8",
        env={"PYTHONPATH": str(repo / "src"), "PATH": ""},
        check=True,
    )
    out = workspace / "dre" / "experiments" / "blocked.yaml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rh_research_engine.cli",
            "dre",
            "export-latest",
            "--claim",
            "C005",
            "--out",
            str(out),
        ],
        cwd=workspace,
        capture_output=True,
        text=True, encoding="utf-8",
        env={"PYTHONPATH": str(repo / "src"), "PATH": ""},
    )
    assert result.returncode != 0, result.stdout
    assert not out.exists(), "an artifact was written with no durable memory present"
    combined = result.stdout + result.stderr
    assert "durable memory" in combined


def test_a_function_without_parentheses_is_applied_not_multiplied():
    r"""`\log n` is application in mathematics; it now parses as one.

    It was refused for a while. Refusing beat misparsing, but `\log n` is
    ordinary notation and asking a document to write `\log(n)` fits the
    mathematics to the parser. A function command applies to the atom that
    follows it.
    """
    import sympy as sp

    result = parse_math(r"\log n")
    assert result.parse_error is None, result.parse_error
    assert sp.sympify(result.sympy_srepr) == sp.log(sp.Symbol("n"))

    result = parse_math(r"\sqrt x")
    assert result.parse_error is None, result.parse_error
    assert sp.sympify(result.sympy_srepr) == sp.sqrt(sp.Symbol("x"))


def test_robin_inequality_is_represented_correctly():
    r"""The whole statement, checked as mathematics rather than as a parse.

    Three separate defects had to be fixed before this one formula was right:

      * `e^{\gamma}` left `e` an undefined Symbol, so the bound was some
        unknown raised to another -- a clean parse meaning nothing;
      * `\log\log` fused into the single undefined name `loglog` during
        command substitution, so the statement became a product;
      * wrapping the bare applications consumed both names at once and gave
        `log(log)` with the argument stranded outside.

    Each looked fine in isolation. Only checking the assembled meaning caught
    them, which is why this asserts the shape of the expression.
    """
    import sympy as sp

    result = parse_math(r"\sigma(n) < e^{\gamma} n \log\log n")
    assert result.parse_error is None, result.parse_error

    expr = sp.sympify(result.sympy_srepr)
    n = sp.Symbol("n")

    # sigma applied to n, not multiplied by it.
    assert "divisor_sigma(Symbol('n'))" in result.sympy_srepr
    # Euler's number, via SymPy's own exp -- not a variable named e.
    assert any(isinstance(a, sp.exp) for a in sp.preorder_traversal(expr))
    assert "Symbol('e')" not in result.sympy_srepr
    # log(log(n)), nested -- not `loglog` and not `log(log)` with n outside.
    right = expr.args[1]
    assert sp.log(sp.log(n)) in right.args
    assert "loglog" not in result.sympy_srepr


def test_lagarias_inequality_is_represented_correctly():
    r"""`\log H(n)` must take all of `H(n)`, not just the name `H`.

    An earlier atom rule stopped at the identifier and produced `log(H)(n)`.
    """
    import sympy as sp

    result = parse_math(r"\sigma(n) \le H(n) + e^{H(n)}\log H(n)")
    assert result.parse_error is None, result.parse_error

    expr = sp.sympify(result.sympy_srepr)
    H, n = sp.harmonic, sp.Symbol("n")
    right = expr.args[1]
    assert sp.log(H(n)) in right.atoms(sp.log)
    assert sp.exp(H(n)) in right.atoms(sp.exp)


def test_the_xi_definition_parses_as_a_product_not_an_application():
    r"""`s(s-1)` is multiplication; `\xi(s)` is application. Same syntax.

    Resolved by looking at how each name is used across the whole side: a name
    that ALSO appears standalone is a variable, so `s(s-1)` is a product, while
    `xi`, `Gamma` and `zeta` are only ever applied and are functions. Deciding
    on the applied form alone made `s` a function and the definition failed to
    parse at all.
    """
    import sympy as sp

    result = parse_math(r"\xi(s) = \frac{1}{2}s(s-1)\pi^{-s/2}\Gamma(s/2)\zeta(s)")
    assert result.parse_error is None, result.parse_error

    srepr = result.sympy_srepr
    # xi has no SymPy counterpart; Gamma and zeta do, and get the real ones.
    assert "RiemannXi(" in srepr
    for applied in ("gamma(", "zeta("):
        assert applied in srepr, applied
    # s is a variable here, not a function.
    assert "Function('s')" not in srepr
    assert "Symbol('s')" in srepr

    expr = sp.sympify(srepr)
    s = sp.Symbol("s")
    right = expr.args[1]
    assert (s - 1) in right.args or any((s - 1) in a.args for a in right.args)


# --- binders: bounded ones are exact, unbounded ones are not inventable ------


def test_a_bounded_sum_is_represented_exactly():
    r"""`\sum_{n=1}^{\infty} n^{-s}` is the Dirichlet series, a core formula.

    A bound written in the source is exact and SymPy's Sum carries it
    faithfully, so refusing it was refusing something representable. The whole
    class was rejected under one blanket rule.
    """
    import sympy as sp

    n, s = sp.Symbol("n"), sp.Symbol("s")
    for src in (r"\zeta(s) = \sum_{n=1}^{\infty} n^{-s}", r"\zeta(s) = \sum_{n=1}^\infty n^{-s}"):
        result = parse_math(src)
        assert result.parse_error is None, result.parse_error
        total = sp.sympify(result.sympy_srepr).args[1]
        # Compare the parts: the parsed tree is built with evaluate=False, so it
        # prints identically to an evaluated one without being equal to it.
        assert isinstance(total, sp.Sum)
        assert total.limits == ((n, 1, sp.oo),)
        assert sp.simplify(total.function - n ** (-s)) == 0


def test_a_bounded_product_is_represented_exactly():
    import sympy as sp

    k = sp.Symbol("k")
    result = parse_math(r"\prod_{k=1}^{3} k")
    assert result.parse_error is None, result.parse_error
    product = sp.sympify(result.sympy_srepr)
    assert isinstance(product, sp.Product)
    assert product.limits == ((k, 1, 3),)
    assert product.function == k


def test_the_euler_product_is_indexed_by_position_and_is_checkable():
    r"""`\prod_p` runs over the primes, which is an index SymPy cannot hold.

    The statement is the same one written over POSITION: the product over all
    primes is the product over k of a term in the k-th prime. `NthPrime` is a
    real function rather than a placeholder, so the result can be evaluated --
    a truncation of the Euler product must equal the product over the actual
    first primes, which is what this checks.
    """
    import sympy as sp

    from rh_research_engine.symbolic.functions import SREPR_NAMESPACE, NthPrime

    parsed = parse_math(r"\zeta(s) = \prod_p (1-p^{-s})^{-1}")
    assert parsed.parse_error is None, parsed.parse_error
    expression = sp.sympify(parsed.sympy_srepr, locals=dict(SREPR_NAMESPACE))

    product = next(
        node for node in sp.preorder_traversal(expression) if isinstance(node, sp.Product)
    )
    index, lower, upper = product.limits[0]
    assert (lower, upper) == (1, sp.oo)

    # Truncated at four factors it is the product over 2, 3, 5, 7 -- not over
    # 1, 2, 3, 4, which is what indexing by position would mean if `NthPrime`
    # were an opaque token.
    truncated = sp.Product(product.function, (index, 1, 4)).doit()
    expected = sp.prod([(1 - sp.Integer(q) ** (-sp.Symbol("s"))) ** -1 for q in (2, 3, 5, 7)])
    assert sp.simplify(truncated - expected) == 0
    assert [NthPrime(i) for i in range(1, 5)] == [2, 3, 5, 7]


def test_a_binder_binds_tighter_than_addition():
    r"""`\sum_{n=1}^{5} n + 1` is `(\sum n) + 1`.

    This is the precedence every source assumes when it writes a binder
    without brackets, so reading it any other way would be reading something
    the author did not write. 15 + 1, not the sum of (n+1) which is 20.
    """
    import sympy as sp

    parsed = parse_math(r"\sum_{n=1}^{5} n + 1")
    assert parsed.parse_error is None, parsed.parse_error
    assert sp.sympify(parsed.sympy_srepr).doit() == 16


def test_parenthesising_the_body_resolves_the_ambiguity():
    import sympy as sp

    n = sp.Symbol("n")
    result = parse_math(r"\sum_{n=1}^{5} (n + 1)")
    assert result.parse_error is None, result.parse_error
    total = sp.sympify(result.sympy_srepr)
    assert isinstance(total, sp.Sum)
    assert total.limits == ((n, 1, 5),)
    assert sp.simplify(total.function - (n + 1)) == 0


def test_constants_reach_sympys_own_objects():
    r"""A Symbol named `pi` is not pi.

    `\pi` sat in the forced-Symbol set with every other Greek letter, so it
    parsed as an undefined variable that merely happened to be spelled "pi".
    Every formula containing it was quietly wrong -- `N(T) = (T/2\pi)\log(T/2\pi)
    - T/2\pi` is not the Riemann-von Mangoldt formula when pi is free -- and
    nothing downstream could tell, because the parse was clean and the printed
    form identical.

    Found only by asserting on structure rather than on `parse_error is None`.
    """
    import sympy as sp

    assert sp.sympify(parse_math(r"\pi").sympy_srepr) == sp.pi
    assert sp.sympify(parse_math(r"\infty").sympy_srepr) == sp.oo

    result = parse_math(r"N(T) = (T/(2\pi))\log(T/(2\pi)) - T/(2\pi)")
    assert result.parse_error is None, result.parse_error
    expr = sp.sympify(result.sympy_srepr)
    assert sp.pi in expr.atoms(sp.NumberSymbol)
    assert sp.Symbol("pi") not in expr.free_symbols


def test_the_binder_rewrite_produces_a_real_sum_not_a_lookalike():
    r"""`Sum` and `oo` must not be shadowed either.

    The rewrite emitted the text `Sum(..., (n, 1, oo))`, which then parsed as an
    opaque `Function('Sum')` over `Symbol('oo')`. It printed exactly like a real
    summation and was not one.
    """
    import sympy as sp

    result = parse_math(r"\zeta(s) = \sum_{n=1}^{\infty} n^{-s}")
    assert result.parse_error is None, result.parse_error
    total = sp.sympify(result.sympy_srepr).args[1]

    assert isinstance(total, sp.Sum), f"got {type(total).__name__}"
    assert "Function('Sum')" not in result.sympy_srepr
    assert "Symbol('oo')" not in result.sympy_srepr


def test_pi_is_the_constant_or_the_prime_counting_function_by_use():
    r"""`\pi` is genuinely both, and the applied form is what separates them.

    Reserving the name so it always reached SymPy's number made von Koch's
    theorem parse as `pi * x` -- pi TIMES x, rather than pi APPLIED to x. The
    same applied/standalone rule that distinguishes `s(s-1)` from `\xi(s)`
    settles this one.
    """
    import sympy as sp

    counting = sp.sympify(parse_math(r"\pi(x) = li(x) + O(\sqrt{x}\log x)").sympy_srepr)
    assert "primepi(" in sp.srepr(counting)
    assert sp.pi not in counting.atoms(sp.NumberSymbol)

    constant = sp.sympify(parse_math(r"N(T) = (T/(2\pi))\log(T/(2\pi)) - T/(2\pi)").sympy_srepr)
    assert sp.pi in constant.atoms(sp.NumberSymbol)
    assert "primepi(" not in sp.srepr(constant)
