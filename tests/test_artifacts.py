"""Artifact contract tests.

Phase 1 ships schemas, not engines. What must hold now is that the schemas
refuse the shapes the adversarial review showed are dangerous: worker-asserted
proof, an undeclared synthetic model, an unverified solver certificate, an
uncitable literature match, and a Lean file that "compiles" on an axiom.
"""

import pytest

from rh_research_engine.contracts import Confidence, Role
from rh_research_engine.contracts.artifacts import (
    ARTIFACT_MODELS,
    Artifact,
    ArtifactError,
    ArtifactType,
    CounterexampleArtifact,
    CounterexampleVerdict,
    FormalizationReport,
    InequalityCertificate,
    LiteratureMatch,
    LiteratureVerdict,
    ModelFamilySpec,
    ObligationStatus,
    OperatorExperiment,
    ProofObligation,
    PropertyAssertion,
    ResearchRunManifest,
)


# Pydantic converts a ValueError raised inside a validator into ValidationError,
# so `pytest.raises(ArtifactError)` would not match even though ArtifactError is
# what the validator raised. ValidationError subclasses ValueError, so asserting
# on ValueError plus the message covers both and is what the rest of the suite
# does. See test_contract_errors_surface_as_validation_errors below.
def _envelope(**overrides):
    base = dict(
        artifact_id="art:1",
        created_by="rh-math-worker",
        method_family="python-numpy",
        method_version="0.11.0",
    )
    base.update(overrides)
    return base


# --- the shared envelope ----------------------------------------------------


def test_every_artifact_type_with_a_model_is_registered():
    for artifact_type, model in ARTIFACT_MODELS.items():
        assert issubclass(model, Artifact)
        instance_type = model.model_fields["artifact_type"].default
        assert instance_type is artifact_type


def test_the_fourteen_missing_contracts_now_exist():
    expected = {
        ArtifactType.PROOF_OBLIGATION,
        ArtifactType.PROPERTY_ASSERTION,
        ArtifactType.MATHEMATICAL_OBJECT,
        ArtifactType.MODEL_FAMILY_SPEC,
        ArtifactType.SYNTHETIC_MODEL,
        ArtifactType.COUNTEREXAMPLE,
        ArtifactType.TRANSFORM_DERIVATION,
        ArtifactType.KERNEL_ANALYSIS,
        ArtifactType.OPERATOR_EXPERIMENT,
        ArtifactType.INEQUALITY_CERTIFICATE,
        ArtifactType.LITERATURE_MATCH,
        ArtifactType.FORMALIZATION_REPORT,
        ArtifactType.SUPERVISOR_DECISION,
        ArtifactType.RESEARCH_RUN_MANIFEST,
    }
    assert expected <= set(ARTIFACT_MODELS)
    assert len(expected) == 14


def test_artifact_hash_is_stable_and_ignores_metadata():
    a = PropertyAssertion(**_envelope(object_id="o", property_kind="bound", value="X^0.2"))
    b = PropertyAssertion(
        **_envelope(object_id="o", property_kind="bound", value="X^0.2", metadata={"note": "x"})
    )
    assert a.artifact_hash() == b.artifact_hash()


def test_worker_cannot_assert_proof_on_any_artifact():
    for status in (Confidence.PROVED, Confidence.KNOWN, Confidence.FORMALLY_VERIFIED):
        with pytest.raises(ValueError, match="may not assert"):
            PropertyAssertion(
                **_envelope(
                    object_id="o",
                    property_kind="bound",
                    value="X^0.2",
                    epistemic_status=status,
                )
            )


def test_external_verifier_may_supply_proof_status():
    artifact = PropertyAssertion(
        **_envelope(
            created_by="external-verifier",
            object_id="o",
            property_kind="bound",
            value="X^0.2",
            epistemic_status=Confidence.PROVED,
        )
    )
    assert artifact.advances_frontier is True


def test_frontier_axes_are_derived_not_stored():
    """Stored copies of a derived fact are two things that can disagree."""
    artifact = PropertyAssertion(
        **_envelope(
            created_by="external-verifier",
            object_id="o",
            property_kind="bound",
            value="X^0.2",
            epistemic_status=Confidence.KNOWN,
            mathematical_role=Role.EQUIVALENCE,
            rh_equivalent=True,
        )
    )
    assert "frontier_relevant" not in PropertyAssertion.model_fields
    assert artifact.frontier_relevant is False
    assert "restates RH" in artifact.frontier.explain()


def test_open_qualifiers_merge_assumptions_and_conditions():
    artifact = PropertyAssertion(
        **_envelope(
            object_id="o",
            property_kind="bound",
            value="X^0.2",
            assumptions=["assumes RH"],
            conditions=["X < 10**6"],
        )
    )
    assert artifact.open_qualifiers == ["X < 10**6", "assumes RH"]
    assert artifact.advances_frontier is False


# --- proof obligations ------------------------------------------------------


def test_discharged_obligation_must_name_its_evidence():
    with pytest.raises(ValueError, match="names no evidence"):
        ProofObligation(
            **_envelope(statement="prove the remainder is X^o(1)"),
            status=ObligationStatus.DISCHARGED,
        )


def test_open_obligation_reports_itself_open():
    obligation = ProofObligation(**_envelope(statement="prove it"))
    assert obligation.open is True


# --- synthetic models -------------------------------------------------------


def test_model_family_must_declare_its_relationship_to_zeta():
    with pytest.raises(ValueError, match="declares no relationship"):
        ModelFamilySpec(**_envelope(family_name="toy"))


def test_model_family_cannot_both_satisfy_and_violate():
    with pytest.raises(ValueError, match="both satisfied"):
        ModelFamilySpec(
            **_envelope(
                family_name="toy",
                satisfied_properties=["functional_equation"],
                violated_properties=["functional_equation"],
            )
        )


def test_synthetic_separator_cannot_be_relabelled_into_a_theorem():
    with pytest.raises(ValueError, match="cannot become a theorem"):
        CounterexampleArtifact(
            **_envelope(
                created_by="external-verifier",
                criterion_ref="c",
                model_ref="m",
                verdict=CounterexampleVerdict.SEPARATES_SYNTHETIC_CLASSES,
                epistemic_status=Confidence.PROVED,
            )
        )


def test_synthetic_evidence_never_advances_the_frontier():
    artifact = CounterexampleArtifact(
        **_envelope(
            criterion_ref="c",
            model_ref="m",
            verdict=CounterexampleVerdict.SEPARATES_SYNTHETIC_CLASSES,
        )
    )
    assert artifact.advances_frontier is False


# --- analytic compiler ------------------------------------------------------


def test_operator_experiment_stays_numerical():
    with pytest.raises(ValueError, match="evidence about the discretization"):
        OperatorExperiment(
            **_envelope(
                created_by="external-verifier",
                operator_name="H",
                discretization="finite-difference",
                dimension=512,
                epistemic_status=Confidence.PROVED,
            )
        )


def test_transform_gaps_must_be_recorded_as_conditions():
    from rh_research_engine.contracts.artifacts import TransformDerivation

    with pytest.raises(ValueError, match="not among its conditions"):
        TransformDerivation(
            **_envelope(
                transform="mellin",
                input_expression="exp(-u**q)",
                output_expression="gamma(s/q)/q",
                justification_gaps=["interchange of limit and sum unjustified"],
            )
        )


# --- inequality certificates ------------------------------------------------


def _cert(**overrides):
    base = dict(
        statement="R_q(X) >= 0",
        input_hash="a" * 64,
        solver_family="z3",
        solver_version="4.13",
        solver_options_hash="b" * 64,
        certificate_format="smt2-proof",
        certificate_hash="c" * 64,
    )
    base.update(overrides)
    return InequalityCertificate(**_envelope(**base))


def test_unverified_solver_output_cannot_claim_strength():
    with pytest.raises(ValueError, match="counts for nothing"):
        _cert(created_by="external-verifier", epistemic_status=Confidence.CERTIFIED)


def test_accepted_certificate_must_name_its_verifier():
    with pytest.raises(ValueError, match="names no verifier"):
        _cert(verification_status="accepted")


def test_verified_certificate_is_accepted():
    cert = _cert(
        verification_status="accepted",
        verifier_family="arb-flint",
        verifier_version="0.6.0",
        epistemic_status=Confidence.RIGOROUS_NUMERICAL,
    )
    assert cert.epistemic_status is Confidence.RIGOROUS_NUMERICAL
    # Rigorous about a finite computation, so still not frontier-advancing.
    assert cert.advances_frontier is False


def test_certificate_records_replay_identity_not_search_order():
    """Replay verifies the certificate; it never reproduces the solver search."""
    cert = _cert(random_seed=7)
    for field in ("input_hash", "solver_options_hash", "certificate_hash", "random_seed"):
        assert field in InequalityCertificate.model_fields
    assert cert.random_seed == 7
    assert cert.certificate_hash == "c" * 64
    # Two runs of the same solver that find different certificates are different
    # artifacts, and the hash says so.
    assert _cert(certificate_hash="f" * 64).artifact_hash() != cert.artifact_hash()


# --- literature -------------------------------------------------------------


def _lit(**overrides):
    base = dict(query="screening remainder bound", retrieval_provider="manual", retrieved_at_tick=1)
    base.update(overrides)
    return LiteratureMatch(**_envelope(**base))


def test_claimed_equivalence_must_cite_a_source():
    with pytest.raises(ValueError, match="cannot be cited"):
        _lit(verdict=LiteratureVerdict.KNOWN_EQUIVALENT)


def test_claimed_equivalence_must_record_theorem_assumptions():
    with pytest.raises(ValueError, match="hypotheses are compared"):
        _lit(verdict=LiteratureVerdict.KNOWN_EQUIVALENT, source_identifiers=["arXiv:1234.5678"])


def test_inconclusive_match_needs_nothing():
    assert _lit().verdict is LiteratureVerdict.INSUFFICIENT_EVIDENCE


def test_literature_match_uses_a_logical_tick_not_wall_clock():
    assert "retrieved_at_tick" in LiteratureMatch.model_fields
    assert "retrieved_at" not in LiteratureMatch.model_fields


# --- formalization ----------------------------------------------------------


def test_axiom_backed_lean_cannot_claim_formal_verification():
    with pytest.raises(ValueError, match="not a proof"):
        FormalizationReport(
            **_envelope(
                created_by="external-verifier",
                target="screening_bound",
                axioms_introduced=["axiom explicit_formula_for_kernel_K"],
                epistemic_status=Confidence.FORMALLY_VERIFIED,
            )
        )


def test_clean_formalization_may_claim_verification():
    report = FormalizationReport(
        **_envelope(
            created_by="external-verifier",
            target="square_identity",
            steps_formalized=["ring"],
            epistemic_status=Confidence.FORMALLY_VERIFIED,
        )
    )
    assert report.fully_formalized is True


def test_report_with_open_obligations_is_not_fully_formalized():
    report = FormalizationReport(
        **_envelope(target="t", remaining_obligations=["uniform bound"])
    )
    assert report.fully_formalized is False


# --- run manifest -----------------------------------------------------------


def test_run_manifest_is_procedural_and_serializes_stably():
    manifest = ResearchRunManifest(
        **_envelope(
            run_id="run-1",
            started_at_tick=1,
            inputs_hash="d" * 64,
            engine_fingerprint="e" * 64,
            produced_artifacts=["z", "a", "m"],
        )
    )
    assert manifest.mathematical_role is Role.PROCEDURAL
    assert manifest.property_extractable is False
    assert manifest.model_dump(mode="json")["produced_artifacts"] == ["a", "m", "z"]


def test_contract_errors_surface_as_validation_errors():
    """Document the wrapping, so it is not rediscovered a fourth time.

    ArtifactError is raised by the validator; pydantic re-raises it inside a
    ValidationError. Both are ValueError, and the message survives intact, so
    callers match on the message.
    """
    import pydantic

    with pytest.raises(pydantic.ValidationError) as excinfo:
        ModelFamilySpec(**_envelope(family_name="toy"))
    assert isinstance(excinfo.value, ValueError)
    assert "declares no relationship" in str(excinfo.value)
    assert issubclass(ArtifactError, ValueError)
