from rh_research_engine.core.models import EvidenceClass
from rh_research_engine.experiments.adversarial_synthetic import (
    SyntheticSystem,
    SyntheticZero,
    evaluate_criteria,
    run,
)


def test_synthetic_system_adds_functional_equation_partner():
    system = SyntheticSystem(off_line_zeros=[SyntheticZero(beta=0.6, gamma=25.0)])
    zeros = {(round(z.beta, 1), z.gamma) for z in system.symmetric_zeros()}
    assert (0.6, 25.0) in zeros
    assert (0.4, 25.0) in zeros
    assert system.functional_equation_symmetric() is True
    assert system.expected_rh is False


def test_candidate_criterion_false_positive_is_reported():
    system = SyntheticSystem(off_line_zeros=[SyntheticZero(beta=0.51, gamma=10.0)])
    [result] = evaluate_criteria(system, tolerance=0.02, criteria=["critical-line-window"])
    assert result.predicted_rh is True
    assert result.false_positive is True
    assert result.false_negative is False


def test_synthetic_experiment_is_non_rigorous_and_cannot_promote():
    result = run(off_line=[SyntheticZero(beta=0.6, gamma=10.0)])
    assert result.evidence_class is EvidenceClass.HEURISTIC
    assert result.method_family == "python-synthetic-adversary"
    assert result.metrics["functional_equation_symmetric"] is True
    assert result.metrics["false_positive_count"] >= 0
    assert any("cannot promote" in item for item in result.observations)
