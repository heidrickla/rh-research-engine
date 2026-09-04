"""The already-landed subsystems, read through the canonical contracts.

Adaptation only: no behaviour was added. These assert that what each subsystem
produces lands where the contract layer says it should, and in particular that
neither can reach frontier advancement.
"""

import pytest

from rh_research_engine.contracts.artifacts import CounterexampleVerdict
from rh_research_engine.contracts.epistemic import RIGOROUS, Confidence
from rh_research_engine.contracts.roles import Role
from rh_research_engine.experiments.adversarial_synthetic import (
    SyntheticSystem,
    SyntheticZero,
    evaluate_criteria,
)
from rh_research_engine.experiments.synthetic_contracts import (
    counterexample_artifacts,
    model_family_spec,
    synthetic_model,
    verdict_for,
)
from rh_research_engine.mathcert.arb_flint import (
    MAX_CONFIDENCE,
    envelope_confidence,
    interval_certificate,
)
from rh_research_engine.mathcert.verifiers import VerificationStatus

# --- synthetic adversary ----------------------------------------------------


def _family():
    return model_family_spec(artifact_id="fam:1", method_version="0.11.0")


def test_family_declares_both_what_it_satisfies_and_what_it_violates():
    family = _family()
    assert family.satisfied_properties
    assert family.violated_properties
    assert not set(family.satisfied_properties) & set(family.violated_properties)


def test_family_states_it_is_not_zeta():
    violated = " ".join(_family().violated_properties)
    assert "Euler product" in violated
    assert "Riemann zeta function" in violated


def test_synthetic_artifacts_never_advance_the_frontier():
    family = _family()
    system = SyntheticSystem(
        critical_line_zeros=[14.13], off_line_zeros=[SyntheticZero(beta=0.52, gamma=14.13)]
    )
    model = synthetic_model(
        system, artifact_id="mod:1", family_ref=family.artifact_id, method_version="0.11.0"
    )
    artifacts = counterexample_artifacts(
        evaluate_criteria(system), model_ref=model.artifact_id, method_version="0.11.0"
    )
    assert artifacts
    for artifact in (family, model, *artifacts):
        assert artifact.epistemic_status is Confidence.SYNTHETIC
        assert artifact.epistemic_status not in RIGOROUS
        assert artifact.advances_frontier is False


def test_off_line_system_is_recorded_as_expecting_rh_false():
    system = SyntheticSystem(off_line_zeros=[SyntheticZero(beta=0.6, gamma=1.0)])
    model = synthetic_model(
        system, artifact_id="mod:2", family_ref="fam:1", method_version="0.11.0"
    )
    assert model.instance_parameters["expected_rh"] is False


def test_a_single_correct_prediction_is_inconclusive_not_a_separator():
    """One data point cannot establish that a criterion separates the classes."""
    clean = SyntheticSystem(critical_line_zeros=[14.13])
    for result in evaluate_criteria(clean):
        assert result.false_positive is False and result.false_negative is False
        assert verdict_for(result) is CounterexampleVerdict.INCONCLUSIVE


def test_a_criterion_that_misses_an_off_line_zero_is_a_false_positive():
    system = SyntheticSystem(off_line_zeros=[SyntheticZero(beta=0.6, gamma=1.0)])
    results = evaluate_criteria(system, tolerance=0.5)  # tolerance swallows the deviation
    assert any(verdict_for(r) is CounterexampleVerdict.FALSE_POSITIVE for r in results)


def test_counterexample_artifacts_are_criteria_carrying_the_model_dependency():
    artifacts = counterexample_artifacts(
        evaluate_criteria(SyntheticSystem(critical_line_zeros=[14.13])),
        model_ref="mod:9",
        method_version="0.11.0",
    )
    for artifact in artifacts:
        assert artifact.mathematical_role is Role.CRITERION
        assert artifact.dependencies == ["mod:9"]
        assert artifact.assumptions


# --- Arb/FLINT adapter ------------------------------------------------------


def test_disconnected_backend_maps_to_unknown_not_something_weaker_but_positive():
    envelope = interval_certificate(expression="R_q(1000)", lower="0.1", upper="0.2")
    assert envelope.status is VerificationStatus.UNKNOWN
    assert envelope_confidence(envelope) is Confidence.UNKNOWN


def test_the_adapter_ceiling_is_not_rigorous():
    """A certified enclosure is rigorous about a *finite* computation."""
    assert MAX_CONFIDENCE is Confidence.RIGOROUS_NUMERICAL
    assert MAX_CONFIDENCE not in RIGOROUS


@pytest.mark.parametrize(
    "status,expected",
    [
        (VerificationStatus.ACCEPTED, Confidence.RIGOROUS_NUMERICAL),
        (VerificationStatus.REJECTED, Confidence.REFUTED),
        (VerificationStatus.UNKNOWN, Confidence.UNKNOWN),
    ],
)
def test_every_verification_status_maps(status, expected):
    from rh_research_engine.contracts.mappings import confidence_from_verification_status

    assert confidence_from_verification_status(status) is expected


def test_adapter_still_fails_closed_on_the_certificate_itself():
    envelope = interval_certificate(expression="R_q(1000)", lower="0.1", upper="0.2")
    assert envelope.certificate.assumptions or envelope.notes
    assert not envelope.checks
