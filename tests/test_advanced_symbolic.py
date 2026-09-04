from pathlib import Path

from rh_research_engine.mathcert import (
    MathCertificate,
    RealInterval,
    VerificationStatus,
    VerifierEnvelope,
    VerifierMetadata,
    validate_external_envelope,
)
from rh_research_engine.symbolic import (
    FormulaIndex,
    check_asymptotic,
    check_certificate_against_expression,
    export_polynomial_identity,
    growth_exponent,
    ingest_text,
    match_route,
)


def test_formula_index_exact_and_structural(tmp_path: Path):
    index = FormulaIndex(tmp_path / "formulas.json")
    index.add("(x+1)**2", record_id="F1", source="paper.md", line=10)
    index.add("sin(x)+cos(x)", record_id="F2")
    matches = index.query("x**2+2*x+1")
    assert matches[0].record.id == "F1"
    assert matches[0].exact is True


def test_ingestion_can_populate_formula_index(tmp_path: Path):
    ingestion = ingest_text("# Lemma\nWe use $x^2+2*x+1=(x+1)^2$.", source="paper.md")
    index = FormulaIndex(tmp_path / "formulas.json")
    added = index.add_ingestion(ingestion, tags=["paper"])
    assert len(added) == 1
    assert added[0].source == "paper.md"
    assert added[0].tags == ["paper"]
    matches = index.query_equation("x**2+2*x+1", "(x+1)**2")
    assert matches[0].exact is True


def test_asymptotic_ratio_and_growth():
    result = check_asymptotic("X**2 + X", "X**2", "X")
    assert result.verdict == "confirmed_ratio_limit"
    growth = growth_exponent("X**(3/2)", "X")
    assert growth.exponent == "3/2"


def test_lean_export_is_fail_closed():
    ok = export_polynomial_identity("(x+1)**2", "x**2+2*x+1", "square_identity")
    assert ok.supported is True
    assert "ring" in (ok.lean or "")
    bad = export_polynomial_identity("sin(x)", "x", "bad")
    assert bad.supported is False


def test_certificate_expression_matching():
    cert = MathCertificate(
        expression="x**2",
        value=RealInterval.from_decimals("1.0", "1.1"),
        verifier=VerifierMetadata(method="arb", precision_bits=256),
    )
    checked = check_certificate_against_expression(cert, "x**2")
    assert checked.expression_match is True
    assert checked.predicates["definitely_positive"] == "true"


def test_unbacked_arb_acceptance_is_refused():
    """No Arb adapter exists in this build, so ACCEPTED cannot be substantiated.

    The certificate is a self-declaration: `method` is a free string and
    `certificate_hash` recomputes over whatever the metadata was edited to.
    """
    cert = MathCertificate(
        expression="R_q(1000)",
        value=RealInterval.from_decimals("0.1", "0.2"),
        verifier=VerifierMetadata(method="arb", precision_bits=256),
    )
    envelope = VerifierEnvelope(
        verifier_family="arb",
        verifier_version="2.23.0",
        certificate=cert,
        status=VerificationStatus.ACCEPTED,
        checks=["interval enclosure verified"],
    )
    assert envelope.independence_group == "math-verifier:arb:2.23.0"
    errors = validate_external_envelope(envelope, allowed_families={"arb", "flint"})
    assert any("no registered execution adapter" in e for e in errors)
    assert any("worker_hash" in e for e in errors)


def test_unconnected_verifier_may_report_unknown():
    """The honest envelope for a disconnected backend is a verdict of 'unknown'."""
    cert = MathCertificate(
        expression="R_q(1000)",
        value=RealInterval.from_decimals("0.1", "0.2"),
        verifier=VerifierMetadata(method="arb", precision_bits=256),
    )
    envelope = VerifierEnvelope(
        verifier_family="arb",
        verifier_version="2.23.0",
        certificate=cert,
        status=VerificationStatus.UNKNOWN,
    )
    assert validate_external_envelope(envelope, allowed_families={"arb", "flint"}) == []


def test_unapproved_family_is_refused():
    cert = MathCertificate(
        expression="R_q(1000)",
        value=RealInterval.from_decimals("0.1", "0.2"),
        verifier=VerifierMetadata(method="totally-legit"),
    )
    envelope = VerifierEnvelope(
        verifier_family="totally-legit",
        verifier_version="1.0",
        certificate=cert,
        status=VerificationStatus.UNKNOWN,
    )
    errors = validate_external_envelope(envelope, allowed_families={"arb", "flint"})
    assert any("unapproved verifier family" in e for e in errors)


def test_route_matcher_finds_known_no_go():
    matches = match_route("boundary unitarity scattering ratio proves RH")
    assert any(item.status == "false_route" for item in matches)


def test_correcting_a_formula_removes_the_wrong_one(tmp_path: Path):
    """A corrected document must not leave its error behind.

    Record ids are content hashes, so a fix files a second record beside the
    first instead of replacing it. The functional equation sat in the index
    twice, with and without its `sin(pi*s/2)` factor, and a structural search
    would have returned the false one as indexed knowledge.
    """
    index = FormulaIndex(tmp_path / "formulas.json")
    index.add_ingestion(ingest_text("$$f(s) = 2g(1-s)$$", source="paper.md"))

    skipped: list[str] = []
    index.add_ingestion(
        ingest_text("$$f(s) = 2g(1-s)h(s)$$", source="paper.md"), skipped=skipped
    )

    expressions = [record.expression for record in index.load()]
    assert len(expressions) == 1
    assert "h(s)" in expressions[0]
    assert any("superseded" in line for line in skipped)


def test_other_documents_survive_an_ingestion(tmp_path: Path):
    """Pruning is scoped to the document that was re-read.

    The index is a projection of many sources. Re-reading one of them says
    nothing about what the others contain.
    """
    index = FormulaIndex(tmp_path / "formulas.json")
    index.add_ingestion(ingest_text("$$a = b + 1$$", source="other.md"))
    index.add_ingestion(ingest_text("$$f(s) = 2g(1-s)$$", source="paper.md"))
    index.add_ingestion(ingest_text("$$f(s) = 3g(1-s)$$", source="paper.md"))

    sources = sorted(record.source for record in index.load())
    assert sources == ["other.md", "paper.md"]
