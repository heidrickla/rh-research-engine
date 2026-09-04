"""One module that attacks every public entry point.

This is the Phase 1 acceptance artifact. Everything else in the suite checks
that a component behaves; this checks that no *composition* of public
constructors, public functions, stores, migrations, or closure paths can
manufacture mathematical authority while the production trust registries are
empty.

The attacks are written the way a motivated user would write them: construct
the record that says what you want it to say, label it convincingly, and hand
it to the thing that reads labels. Each one has to fail, and fail for a stated
reason -- a refusal that does not say which check tripped sends a researcher
looking at the mathematics when the actual problem is a disconnected engine.

Read with `docs/reviews/PHASE1_FINAL_CLOSURE.md`, which cites these by name.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pydantic
import pytest

import helpers_discharge
from rh_research_engine.contracts import activation_status
from rh_research_engine.contracts.artifacts import (
    ArtifactError,
    DreDecisionStatus,
    FormalizationReport,
    ObligationDischargeDecision,
    ProofObligation,
    PropertyAssertion,
)
from rh_research_engine.contracts.discharge import resolve_discharges
from rh_research_engine.contracts.epistemic import RIGOROUS, Confidence
from rh_research_engine.contracts.frontier import DERIVED_FIELD_NAMES, assess
from rh_research_engine.contracts.roles import Role
from rh_research_engine.core.knowledge import KnowledgeBase, KnowledgeIntegrityError
from rh_research_engine.mathcert import (
    VerificationStatus,
    interval_certificate,
    validate_external_envelope,
    verifier_activation_status,
)
from rh_research_engine.mathcert.verifiers import _test_only_registered_adapters
from rh_research_engine.properties.models import (
    ClosureMode,
    EpistemicStatus,
    MathematicalRole,
    PropertyGraph,
    PropertyKind,
    PropertyRecord,
    Provenance,
)
from rh_research_engine.properties.store import (
    PropertyGraphIntegrityError,
    PropertyGraphStore,
)
from rh_research_engine.supervisor.models import QueueSchemaError
from rh_research_engine.supervisor.store import HypothesisQueueStore

# ---------------------------------------------------------------------------
# 9.1 Inventory: what counts as a public entry point
# ---------------------------------------------------------------------------

#: Every module whose public surface an attacker can reach by importing it.
#: Listed rather than discovered so that adding a module is a deliberate act
#: with a review attached, not something that quietly widens the attack surface.
PUBLIC_MODULES = [
    "rh_research_engine.contracts",
    "rh_research_engine.contracts.artifacts",
    "rh_research_engine.contracts.discharge",
    "rh_research_engine.contracts.epistemic",
    "rh_research_engine.contracts.frontier",
    "rh_research_engine.contracts.hashing",
    "rh_research_engine.contracts.lifecycle",
    "rh_research_engine.contracts.mappings",
    "rh_research_engine.contracts.migrations",
    "rh_research_engine.contracts.receipts",
    "rh_research_engine.contracts.roles",
    "rh_research_engine.mathcert",
    "rh_research_engine.mathcert.verifiers",
    "rh_research_engine.properties.closure",
    "rh_research_engine.properties.store",
    "rh_research_engine.supervisor.store",
    "rh_research_engine.symbolic.route_matcher",
]

#: Parameter names that would hand a caller the authority decision. Any public
#: function taking one of these is a public trust-set override by another name.
TRUST_OVERRIDE_PARAMETERS = {
    "trusted_engines",
    "trusted_dre_engines",
    "registered_adapters",
    "trust_set",
    "trusted",
    "allowed_engines",
    # Authenticator-shaped overrides. A caller that can supply the thing which
    # decides whether a receipt is genuine has authenticated itself, exactly as
    # a caller supplying a fingerprint has authorized itself -- so the seam
    # added for receipt work-product certification needs the same guard, not a
    # second one written later after the first bypass is found.
    "authenticated_receipts",
    "authenticator",
    "verifier_key",
    "public_key",
    "signing_key",
}


def _public_functions(module):
    for name, obj in vars(module).items():
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            yield name, obj


def test_a_contract_refusal_surfaces_as_a_validation_error():
    """Pydantic wraps a validator's ValueError, and the message survives.

    Recorded once here so the next reader does not rediscover it: every
    `ArtifactError` raised inside a model validator reaches the caller as a
    `ValidationError`. Both are `ValueError`, so a caller catching that catches
    either, and the assertions below match on the message rather than the type.
    """
    assert issubclass(ArtifactError, ValueError)
    assert issubclass(pydantic.ValidationError, ValueError)


def test_the_public_surface_is_enumerable():
    """Every listed module imports and exposes something. A typo here is silent."""
    import importlib

    for name in PUBLIC_MODULES:
        module = importlib.import_module(name)
        assert vars(module), name


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_no_public_function_takes_a_trust_set(module_name):
    """Attack: supply a caller-owned trusted-engine set. -> No public override."""
    import importlib

    module = importlib.import_module(module_name)
    for name, function in _public_functions(module):
        params = set(inspect.signature(function).parameters)
        offending = params & TRUST_OVERRIDE_PARAMETERS
        assert not offending, f"{module_name}.{name} accepts {sorted(offending)}"


def test_both_trust_registries_report_inert():
    """The Phase 1 posture, stated by the code rather than by the documentation.

    The verifier half is asserted against a FORCED empty registry. Read off
    the machine it said "python-flint is not installed here", which is a fact
    about the checkout and not about the posture -- and it duly failed the day
    one was installed, having never once tested what it was named for.
    """
    dre = activation_status()
    with _test_only_registered_adapters(set()):
        verifier = verifier_activation_status()
    assert dre["discharge_authority_active"] is False
    assert dre["trusted_engine_count"] == 0
    assert verifier["rigorous_verification_active"] is False
    assert verifier["registered_adapter_count"] == 0


# ---------------------------------------------------------------------------
# 9.2 Attack cases: DRE discharge authority
# ---------------------------------------------------------------------------


def _resolve(registry):
    """Resolve against the *production* trust registry. No test widening."""
    obligations, evidence, decisions, receipts = registry
    return resolve_discharges(
        ["OBL-1"],
        obligations=obligations,
        evidence=evidence,
        decisions=decisions,
        receipts=receipts,
    )


def test_directly_constructed_accepted_decision_confers_no_authority():
    """Attack: construct an accepted DRE decision. -> No authority.

    The model validates: `dre_decision_status` is an ordinary caller-supplied
    field, and refusing to let anyone construct one would make it impossible to
    represent a real decision. Constructing it is simply not the same as it
    being believed.
    """
    obligations, evidence, decisions, _ = helpers_discharge.accepted_registry()
    decision = decisions["OBL-1"]
    assert decision.dre_decision_status is DreDecisionStatus.ACCEPTED

    # Handed over with no receipt at all -- the shape a worker can produce.
    resolution = resolve_discharges(
        ["OBL-1"], obligations=obligations, evidence=evidence, decisions=decisions
    )
    assert resolution.qualifying == []
    assert "caller-supplied" in resolution.explain()


def test_the_verified_wrapper_cannot_be_constructed():
    """Attack: construct a "verified" decision wrapper. -> Impossible."""
    from rh_research_engine.contracts import receipts

    assert not hasattr(receipts, "VerifiedDecision")
    with pytest.raises(receipts.ReceiptError, match="cannot be constructed directly"):
        receipts._VerifiedDecision(
            object(),
            decision=None,
            receipt=None,
            receipt_hash="x",
            engine_fingerprint="y",
        )


@pytest.mark.parametrize("claimed_author", ["dre", "external-verifier", "rh-math-worker"])
def test_spoofing_created_by_confers_no_authority(claimed_author):
    """Attack: spoof `created_by`. -> No authority.

    `created_by` is a free string. "external-verifier" is a claim about who
    produced a record, made by whoever produced the record.
    """
    obligations, evidence, decisions, receipts = helpers_discharge.accepted_registry()
    decisions["OBL-1"] = decisions["OBL-1"].model_copy(update={"created_by": claimed_author})
    resolution = resolve_discharges(
        ["OBL-1"],
        obligations=obligations,
        evidence=evidence,
        decisions=decisions,
        receipts=receipts,
    )
    assert resolution.qualifying == []


def test_an_invented_receipt_reference_is_rejected():
    """Attack: invent a DRE receipt reference. -> Rejected."""
    obligations, evidence, decisions, receipts = helpers_discharge.accepted_registry()
    receipt = receipts["DEC-1"]
    receipts["DEC-1"] = receipt.model_copy(update={"decision_ref": "DEC-SOMETHING-ELSE"})
    with helpers_discharge.trusting_test_engine(receipts.values()):
        resolution = resolve_discharges(
            ["OBL-1"],
            obligations=obligations,
            evidence=evidence,
            decisions=decisions,
            receipts=receipts,
        )
    assert resolution.qualifying == []
    assert "receipt covers" in resolution.explain()


def test_changing_the_decision_after_the_receipt_is_rejected():
    """Attack: change the decision after the receipt. -> Rejected."""
    resolution = _resolve(helpers_discharge.stale_receipt_registry())
    assert resolution.qualifying == []


def test_changing_the_evidence_after_the_receipt_is_rejected():
    """Attack: change the evidence after the receipt. -> Rejected."""
    registry = helpers_discharge.swapped_evidence_registry()
    with helpers_discharge.trusting_registry(registry):
        resolution = _resolve(registry)
    assert resolution.qualifying == []
    assert "changed since the decision" in resolution.explain()


def test_an_rh_equivalent_premise_with_a_prose_discharge_does_not_advance():
    """Attack: RH-equivalent premise + prose discharge. -> No advancement.

    `discharged_obligations` holds references. A sentence describing a discharge
    resolves to no ProofObligation artifact, so it discharges nothing.
    """
    verdict = assess(
        role=Role.EQUIVALENCE,
        confidence=Confidence.KNOWN,
        rh_equivalent=True,
        qualifying_discharges=[],
    )
    assert verdict.advances_frontier is False
    assert verdict.frontier_relevant is False

    resolution = resolve_discharges(["we proved the forward direction last week"])
    assert resolution.qualifying == []
    assert "prose description of a discharge is an assertion" in resolution.explain()


def test_an_rh_equivalent_premise_with_a_raw_decision_does_not_advance():
    """Attack: RH-equivalent premise + raw decision. -> No advancement."""
    obligations, evidence, decisions, _ = helpers_discharge.accepted_registry()
    record = PropertyAssertion(
        artifact_id="P-1",
        created_by="external-verifier",
        method_family="lean",
        method_version="4.0",
        object_id="obj",
        property_kind="equivalence",
        value="A <=> RH",
        epistemic_status=Confidence.KNOWN,
        mathematical_role=Role.EQUIVALENCE,
        rh_equivalent=True,
        discharged_obligations=["OBL-1"],
    )
    verdict = record.frontier_with(
        obligations=obligations, evidence=evidence, decisions=decisions
    )
    assert verdict.advances_frontier is False
    assert verdict.frontier_relevant is False


# ---------------------------------------------------------------------------
# 9.2 Attack cases: derived verdicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", sorted(DERIVED_FIELD_NAMES))
def test_supplying_a_derived_field_is_rejected_by_every_entry_point(field):
    """Attack: supply derived fields. -> Rejected.

    Constructor, `model_validate` from stored JSON, and a legacy migration all
    funnel through the same before-validator, so a record cannot acquire
    frontier advancement by being written down anywhere.
    """
    base = dict(
        artifact_id="A-1",
        created_by="rh-math-worker",
        method_family="python",
        method_version="0.11.0",
        object_id="obj",
        property_kind="bound",
        value="X^0.4",
    )
    with pytest.raises(Exception, match="cannot accept"):
        PropertyAssertion(**base, **{field: True})
    with pytest.raises(Exception, match="cannot accept"):
        PropertyAssertion.model_validate({**base, field: True})

    graph_record = dict(
        id="P-1",
        object_id="obj",
        kind=PropertyKind.GROWTH_BOUND,
        value="X^0.4",
        status=EpistemicStatus.SYMBOLIC_DERIVED,
        provenance=[Provenance(source_type="symbolic", source_id="s", method="m")],
    )
    with pytest.raises(Exception, match="cannot accept"):
        PropertyRecord(**graph_record, **{field: True})
    with pytest.raises(Exception, match="cannot accept"):
        PropertyRecord.model_validate(
            {
                **graph_record,
                "kind": "growth_bound",
                "status": "symbolic_derived",
                "provenance": [
                    {"source_type": "symbolic", "source_id": "s", "method": "m"}
                ],
                field: True,
            }
        )


def test_derived_verdicts_are_absent_from_every_serialized_record():
    """A stored copy of a derived fact is a second thing that can disagree."""
    record = PropertyAssertion(
        artifact_id="A-1",
        created_by="rh-math-worker",
        method_family="python",
        method_version="0.11.0",
        object_id="obj",
        property_kind="bound",
        value="X^0.4",
    )
    payload = json.loads(json.dumps(record.model_dump(mode="json")))
    assert DERIVED_FIELD_NAMES.isdisjoint(payload)
    assert DERIVED_FIELD_NAMES.isdisjoint(json.loads(record.canonical_json()))


def test_closure_reads_the_registries_not_a_stored_snapshot():
    """Attack: pre-compute a favourable verdict into the graph. -> Rejected.

    Closure resolves discharges itself on every run. There is no cached verdict
    to poison because there is no cached verdict.
    """
    from rh_research_engine.properties.closure import implication_closure

    premise = PropertyRecord(
        id="P-EQ",
        object_id="obj",
        kind=PropertyKind.GROWTH_BOUND,
        value="X^0.4",
        status=EpistemicStatus.KNOWN,
        provenance=[Provenance(source_type="knowledge", source_id="K", method="m")],
        role=MathematicalRole.EQUIVALENCE,
        rh_equivalent=True,
        discharged_obligations=["OBL-1"],
        metadata={"advances_frontier": True, "frontier_relevant": True},
    )
    graph = implication_closure(
        PropertyGraph(properties=[premise]), mode=ClosureMode.RIGOROUS
    )
    assert [p.id for p in graph.properties] == ["P-EQ"], "no conclusion was derived"


# ---------------------------------------------------------------------------
# 9.2 Attack cases: worker self-promotion and evidence class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "confidence", [Confidence.PROVED, Confidence.KNOWN, Confidence.FORMALLY_VERIFIED]
)
def test_a_worker_cannot_export_evidence_as_proved_known_or_formal(confidence):
    """Attack: worker declares its own output proved. -> Refused at construction."""
    with pytest.raises(pydantic.ValidationError, match="may not assert"):
        PropertyAssertion(
            artifact_id="A-1",
            created_by="rh-math-worker",
            method_family="python-numpy",
            method_version="0.11.0",
            object_id="obj",
            property_kind="bound",
            value="X^0.4",
            epistemic_status=confidence,
        )


def test_a_worker_labelling_a_numerical_result_rigorous_earns_no_frontier_credit():
    """Attack: worker labels a numerical result rigorous. -> No frontier credit.

    RIGOROUS_NUMERICAL is deliberately not in RIGOROUS. A certified enclosure is
    rigorous about a *finite* computation; reading it as rigorous in general is
    the step from "checked to height T" to "true".
    """
    assert Confidence.RIGOROUS_NUMERICAL not in RIGOROUS
    verdict = assess(role=Role.BOUND, confidence=Confidence.RIGOROUS_NUMERICAL)
    assert verdict.advances_frontier is False
    assert verdict.frontier_relevant is True, "worth proving, just not proved"
    assert "not rigorous" in verdict.explain()


def test_rigorous_numerical_evidence_cannot_discharge_an_obligation():
    """The same boundary on the discharge path rather than the frontier path."""
    registry = helpers_discharge.rigorous_numerical_evidence_registry()
    with helpers_discharge.trusting_registry(registry):
        resolution = _resolve(registry)
    assert resolution.qualifying == []
    assert "no deductive force" in resolution.explain()


def test_a_lean_file_that_compiles_on_an_axiom_is_not_formally_verified():
    """Attack: compile Lean with axioms. -> Not formally verified."""
    with pytest.raises(
        pydantic.ValidationError, match="compiles with an axiom is not a proof"
    ):
        FormalizationReport(
            artifact_id="F-1",
            created_by="external-verifier",
            method_family="lean",
            method_version="4.0",
            target="theta_bound",
            steps_formalized=["step-1"],
            axioms_introduced=["axiom rh_holds : RiemannHypothesis"],
            epistemic_status=Confidence.FORMALLY_VERIFIED,
        )


# ---------------------------------------------------------------------------
# 9.2 Attack cases: external verifiers
# ---------------------------------------------------------------------------


def test_a_missing_verifier_backend_reports_unknown_not_accepted(monkeypatch):
    """Attack: no backend present. -> Unknown/unavailable.

    The absence is forced. `interval_certificate` reads the real capability,
    so on a machine with python-flint installed the version is the real one --
    and the status is STILL unknown, which is the durable half of the claim
    and is asserted separately below.
    """
    from rh_research_engine.mathcert import arb_flint

    monkeypatch.setattr(
        arb_flint,
        "detect_arb_flint",
        lambda: arb_flint.ArbFlintCapability(available=False, reason="forced absent"),
    )
    envelope = arb_flint.interval_certificate(
        expression="zeta(2)", lower="1.64", upper="1.65"
    )
    assert envelope.status is VerificationStatus.UNKNOWN
    assert envelope.verifier_version == "unavailable"


def test_supplied_endpoints_are_unknown_whether_or_not_a_backend_is_present():
    """The durable half: a certificate for a computation nobody ran.

    `interval_certificate` is handed endpoints. Installing a backend gives it
    no way to check they enclose anything, so it must keep saying UNKNOWN --
    and the test above, which forces the backend absent, could no longer say
    so on its own.
    """
    envelope = interval_certificate(expression="zeta(2)", lower="1.64", upper="1.65")
    assert envelope.status is VerificationStatus.UNKNOWN


@pytest.mark.parametrize("family", ["arb", "flint", "arb-flint", "pari", "lean"])
def test_a_fake_verifier_family_string_is_rejected(family):
    """Attack: name a real external verifier that is not connected. -> Rejected."""
    envelope = interval_certificate(expression="zeta(2)", lower="1.64", upper="1.65")
    envelope.verifier_family = family
    envelope.certificate.verifier.method = family
    envelope.status = VerificationStatus.ACCEPTED
    envelope.checks = ["enclosure verified"]
    errors = validate_external_envelope(envelope, allowed_families={family})
    assert errors, f"{family} was accepted with no adapter behind it"


def test_repeating_one_method_under_new_labels_is_not_independent_corroboration():
    """Attack: rerun the same method under different labels. -> Same group."""
    first = interval_certificate(expression="zeta(2)", lower="1.64", upper="1.65")
    relabelled = interval_certificate(expression="zeta(2)", lower="1.64", upper="1.65")
    relabelled.notes = ["a second, totally independent check"]
    assert first.independence_group == relabelled.independence_group


# ---------------------------------------------------------------------------
# 9.2 Attack cases: durable state
# ---------------------------------------------------------------------------


def test_removing_durable_memory_blocks_rather_than_reads_as_empty(tmp_path):
    """Attack: remove durable memory. -> Blocked, not an empty knowledge base."""
    with pytest.raises(KnowledgeIntegrityError, match="not an empty one"):
        KnowledgeBase(tmp_path / "absent.json").load()


def test_removing_the_seal_from_the_canonical_copy_is_fatal(tmp_path, monkeypatch):
    """Attack: delete the sidecar seal. -> Blocked.

    An unsealed authoritative copy is the one state in which truncation and
    tampering go undetected, so it fails closed rather than loading.
    """
    from rh_research_engine.core import knowledge as knowledge_module

    canonical = tmp_path / "math_knowledge.json"
    canonical.write_text("[]", encoding="utf-8", newline="")
    monkeypatch.setattr(knowledge_module, "CANONICAL_KNOWLEDGE_PATH", canonical)
    with pytest.raises(KnowledgeIntegrityError, match="has no seal"):
        KnowledgeBase(canonical).load()


def test_replacing_memory_and_resealing_is_caught_by_the_frozen_manifest():
    """Attack: replace memory and re-seal. -> Manifest comparison catches it.

    Re-sealing makes the file self-consistent again, which is exactly why the
    seal alone is not enough: it proves the bytes match a digest written by
    whoever wrote the bytes. The frozen df7016d manifest was written before the
    program started and is never regenerated, so the record IDs, semantic
    hashes, dependency edges, and the four no-go records are checked against
    something the attacker cannot also update.
    """
    baseline = json.loads(
        Path("docs/contracts/frozen-baseline-df7016d.json").read_text(encoding="utf-8")
    )
    invariants = baseline["knowledge_invariants"]
    live = KnowledgeBase()
    assert invariants["record_count"] == len(live.load())
    assert invariants["semantic_hashes"] == live.semantic_hashes()
    assert sorted(invariants["no_go_ids"]) == ["K008", "K032", "K034", "K038"]


def test_a_reworded_no_go_route_is_still_matched():
    """Attack: reword a known dead end. -> Matched, by content not by name."""
    from rh_research_engine.symbolic import match_route

    matches = match_route(
        "Consider imposing unitarity purely on the boundary of the region.",
        limit=5,
    )
    no_go = [m for m in matches if m.is_no_go]
    assert no_go, [m.knowledge_id for m in matches]
    assert no_go[0].action == "reject_or_require_new_distinguishing_assumption"


def test_injecting_proved_into_the_property_graph_is_rejected(tmp_path):
    """Attack: hand-edit `"status": "proved"` into the graph file. -> Rejected."""
    path = tmp_path / "property_graph.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "objects": [],
                "properties": [
                    {
                        "id": "P-1",
                        "object_id": "obj",
                        "kind": "theta_bound",
                        "value": "1/2",
                        "status": "proved",
                        "provenance": [
                            {
                                "source_type": "manual",
                                "source_id": "hand-edit",
                                "method": "typing",
                            }
                        ],
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
        newline="",
    )
    store = PropertyGraphStore(path)
    store.seal()
    with pytest.raises(PropertyGraphIntegrityError, match="failed validation"):
        store.load()


@pytest.mark.parametrize(
    "document,expected",
    [
        ({"hypotheses": []}, "declares no schema_version"),
        ({"schema_version": "99", "hypotheses": []}, "99"),
    ],
    ids=["missing", "unknown"],
)
def test_a_queue_with_a_missing_or_unknown_schema_version_is_rejected(
    tmp_path, document, expected
):
    """Attack: missing / unknown queue schema. -> Rejected, never guessed."""
    path = tmp_path / "hypotheses.json"
    path.write_text(json.dumps(document), encoding="utf-8", newline="")
    with pytest.raises((QueueSchemaError, ValueError), match=expected):
        HypothesisQueueStore(path).load()


# ---------------------------------------------------------------------------
# 9.3 Global negative invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("registry_name", helpers_discharge.ALL_REGISTRIES)
def test_no_public_composition_produces_a_qualifying_discharge(registry_name):
    """The central Phase 1 acceptance invariant, one registry at a time.

    `accepted_registry` is in the list on purpose: the combination that works
    under the scoped test trust set has to fail without it, or the test set is
    not what is doing the work.
    """
    assert _resolve(getattr(helpers_discharge, registry_name)()).qualifying == []


@pytest.mark.parametrize("registry_name", helpers_discharge.ALL_REGISTRIES)
def test_no_public_composition_advances_an_rh_equivalent_premise(registry_name):
    """The same invariant, one level up: no registry lifts a restatement of RH."""
    obligations, evidence, decisions, receipts = getattr(
        helpers_discharge, registry_name
    )()
    premise = PropertyAssertion(
        artifact_id="P-EQ",
        created_by="external-verifier",
        method_family="lean",
        method_version="4.0",
        object_id="obj",
        property_kind="equivalence",
        value="A <=> RH",
        epistemic_status=Confidence.KNOWN,
        mathematical_role=Role.EQUIVALENCE,
        rh_equivalent=True,
        discharged_obligations=["OBL-1"],
    )
    verdict = premise.frontier_with(
        obligations=obligations,
        evidence=evidence,
        decisions=decisions,
        receipts=receipts,
    )
    assert verdict.advances_frontier is False
    assert verdict.frontier_relevant is False


@pytest.mark.parametrize("registry_name", helpers_discharge.ALL_REGISTRIES)
def test_no_registry_produces_a_rigorous_closure_from_a_circular_premise(registry_name):
    """The same invariant on the closure path, which is where a graph is written."""
    from rh_research_engine.properties.closure import implication_closure

    obligations, evidence, decisions, receipts = getattr(
        helpers_discharge, registry_name
    )()
    premise = PropertyRecord(
        id="P-EQ",
        object_id="obj",
        kind=PropertyKind.GROWTH_BOUND,
        value="X^0.4",
        status=EpistemicStatus.KNOWN,
        provenance=[Provenance(source_type="knowledge", source_id="K", method="m")],
        role=MathematicalRole.EQUIVALENCE,
        rh_equivalent=True,
        discharged_obligations=["OBL-1"],
    )
    graph = implication_closure(
        PropertyGraph(properties=[premise]),
        mode=ClosureMode.RIGOROUS,
        obligations=obligations,
        evidence=evidence,
        decisions=decisions,
        receipts=receipts,
    )
    assert [p.id for p in graph.properties] == ["P-EQ"]


def test_no_public_path_produces_an_accepted_rigorous_verifier_result():
    """No envelope reaches a RIGOROUS confidence while no backend is connected."""
    from rh_research_engine.mathcert.arb_flint import envelope_confidence

    envelope = interval_certificate(expression="zeta(2)", lower="1.64", upper="1.65")
    for status in VerificationStatus:
        envelope.status = status
        assert envelope_confidence(envelope) not in RIGOROUS


def test_an_obligation_marked_discharged_must_still_name_its_evidence():
    """Closing an obligation without saying what closed it is deleting it."""
    with pytest.raises(pydantic.ValidationError, match="names no evidence"):
        ProofObligation(
            artifact_id="OBL-1",
            created_by="rh-math-worker",
            method_family="python",
            method_version="0.11.0",
            statement="prove the forward direction",
            status="discharged",
        )


def test_an_accepted_decision_must_bind_evidence_one_to_one():
    """A decision whose refs and hashes disagree in length pins nothing."""
    with pytest.raises(pydantic.ValidationError, match="one-to-one"):
        ObligationDischargeDecision(
            artifact_id="DEC-1",
            created_by="dre",
            method_family="dre",
            method_version="0.22.0",
            obligation_ref="OBL-1",
            obligation_hash="0" * 64,
            evidence_refs=["EV-1", "EV-2"],
            evidence_hashes=["1" * 64],
            discharged_direction="A => RH",
            dre_decision_ref="dre-run-0001",
            dre_decision_status=DreDecisionStatus.ACCEPTED,
            dre_pack_version="0.1.0",
        )
