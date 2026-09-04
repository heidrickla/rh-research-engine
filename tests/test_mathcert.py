import json

import pytest

from rh_research_engine.mathcert import (
    BigRational,
    ComplexInterval,
    MathCertificate,
    RealInterval,
    ScientificInteger,
    TruthValue,
    VerifierMetadata,
    certificate_predicates,
    contains_zero,
    definitely_positive,
    screening_exponent_to_theta,
    strictly_less_than,
)


def test_rational_canonicalization_and_decimal():
    r = BigRational(numerator=2, denominator=4)
    assert (r.numerator, r.denominator) == (1, 2)
    d = BigRational.from_decimal("1.25e-3")
    assert (d.numerator, d.denominator) == (1, 800)


def test_scientific_normalizes():
    x = ScientificInteger(mantissa=12000, decimal_exponent=-8)
    assert x.mantissa == 12
    assert x.decimal_exponent == -5


def test_interval_predicates_three_valued():
    pos = RealInterval.from_decimals("0.001", "0.002")
    crossing = RealInterval.from_decimals("-0.001", "0.002")
    assert definitely_positive(pos) == TruthValue.TRUE
    assert definitely_positive(crossing) == TruthValue.UNKNOWN
    assert contains_zero(crossing) == TruthValue.TRUE


def test_interval_comparison():
    a = RealInterval.from_decimals("1.0", "1.1")
    b = RealInterval.from_decimals("1.2", "1.3")
    c = RealInterval.from_decimals("1.05", "1.25")
    assert strictly_less_than(a, b) == TruthValue.TRUE
    assert strictly_less_than(a, c) == TruthValue.UNKNOWN


def test_complex_interval():
    z = ComplexInterval(
        real=RealInterval.from_decimals("0.4999", "0.5001"),
        imag=RealInterval.from_decimals("14.1347", "14.1348"),
    )
    assert z.real.lower.as_fraction() < z.real.upper.as_fraction()


def test_certificate_hash_stable():
    cert = MathCertificate(
        expression="screening_remainder(X=100000,q=4)",
        value=RealInterval.from_decimals("1.381", "1.382"),
        verifier=VerifierMetadata(method="arb", precision_bits=256, worker_version="0.8.0"),
    )
    h1 = cert.certificate_hash()
    h2 = MathCertificate.model_validate_json(cert.model_dump_json()).certificate_hash()
    assert h1 == h2
    assert len(h1) == 64


def test_dre_payload_contains_only_predicates_not_endpoints():
    cert = MathCertificate(
        expression="x",
        value=RealInterval.from_decimals("0.0000000000000000001", "0.0000000000000000002"),
        verifier=VerifierMetadata(method="arb", precision_bits=512),
    )
    facts = certificate_predicates(cert)
    assert facts["definitely_positive"] == "true"
    text = json.dumps(facts)
    assert "0.0000000000000000001" not in text


def test_screening_exponent_propagation_exact():
    theta = RealInterval.from_decimals("0.013472820", "0.013472840")
    bound = screening_exponent_to_theta(theta)
    assert bound.lower.as_fraction() == BigRational.from_decimal("0.506736410").as_fraction()
    assert bound.upper.as_fraction() == BigRational.from_decimal("0.506736420").as_fraction()


def test_invalid_interval_rejected():
    with pytest.raises(ValueError):
        RealInterval.from_decimals("2", "1")
