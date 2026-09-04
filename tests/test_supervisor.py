import pytest

from rh_research_engine.contracts.epistemic import Confidence
from rh_research_engine.contracts.lifecycle import HypothesisLifecycle
from rh_research_engine.contracts.roles import Role
from rh_research_engine.properties import EpistemicStatus
from rh_research_engine.supervisor import (
    FalsificationTest,
    Hypothesis,
    HypothesisQueue,
    ProofGap,
    extract_from_hypothesis,
)


def _test(cost: int = 1) -> FalsificationTest:
    return FalsificationTest(id="cheap", cost=cost, description="try to break it")


def test_rh_equivalent_restatement_is_not_frontier_progress():
    hypothesis = Hypothesis(
        id="H-RH",
        statement="Criterion A is equivalent to RH.",
        rh_equivalent=True,
        lifecycle=HypothesisLifecycle.TRIAGED,
        mathematical_role=Role.EQUIVALENCE,
        falsification_tests=[_test()],
    )
    assert hypothesis.frontier_relevant is False
    assert hypothesis.actionable is False
    assert hypothesis.advances_frontier is False
    assert "restates RH" in hypothesis.explain_actionability()


def test_rh_equivalence_can_never_advance_without_discharge():
    """Structurally impossible, not merely rejected.

    The previous model raised on `state=ADVANCED` because that field was an
    explicit assertion of progress that could contradict the facts. Progress is
    now derived, so no combination of inputs can express the contradiction --
    which this checks across the whole product rather than on one case.
    """
    for lifecycle in HypothesisLifecycle:
        for confidence in Confidence:
            for role in Role:
                hypothesis = Hypothesis(
                    id="H",
                    statement="Another RH equivalent reformulation.",
                    rh_equivalent=True,
                    lifecycle=lifecycle,
                    epistemic_status=confidence,
                    mathematical_role=role,
                )
                assert hypothesis.advances_frontier is False, (
                    f"{lifecycle.value}/{confidence.value}/{role.value} advanced an "
                    "undischarged RH-equivalence"
                )
                assert hypothesis.frontier.reasons


def test_a_prose_discharge_never_unlocks_advancement():
    base = dict(
        id="H",
        statement="One direction discharged.",
        rh_equivalent=True,
        lifecycle=HypothesisLifecycle.RESOLVED,
        epistemic_status=Confidence.RIGOROUS_DERIVED,
    )
    assert Hypothesis(**base).advances_frontier is False
    prose = Hypothesis(**base, discharged_obligations=["forward direction"])
    assert prose.advances_frontier is False
    assert "ProofObligation" in prose.frontier.explain()


def test_a_resolved_obligation_with_qualifying_evidence_allows_frontier_credit():
    from helpers_discharge import accepted_registry, trusting_test_engine

    hypothesis = Hypothesis(
        id="H-GOOD",
        statement="One direction of an equivalence is discharged.",
        rh_equivalent=True,
        discharged_obligations=["OBL-1"],
        lifecycle=HypothesisLifecycle.RESOLVED,
        epistemic_status=Confidence.RIGOROUS_DERIVED,
    )
    obligations, evidence, decisions, receipts = accepted_registry()
    with trusting_test_engine(receipts.values()):
        assessed = hypothesis.frontier_with(
            obligations=obligations,
            evidence=evidence,
            decisions=decisions,
            receipts=receipts,
        )
        assert assessed.frontier_relevant is True
        assert assessed.advances_frontier is True
    # Outside the scoped trust set -- and with no registry at all -- it stays circular.
    assert hypothesis.advances_frontier is False


def test_conjectural_hypothesis_is_relevant_but_advances_nothing():
    """Lifecycle and confidence are independent: active work, no result yet."""
    hypothesis = Hypothesis(
        id="H-WIP",
        statement="Unconditional screening bound target.",
        lifecycle=HypothesisLifecycle.ACTIVE,
        epistemic_status=Confidence.CONJECTURAL,
        falsification_tests=[_test()],
    )
    assert hypothesis.frontier_relevant is True
    assert hypothesis.advances_frontier is False
    assert hypothesis.actionable is True


def test_open_proof_gaps_block_advancement_and_are_named():
    hypothesis = Hypothesis(
        id="H-GAP",
        statement="Screening bound.",
        lifecycle=HypothesisLifecycle.RESOLVED,
        epistemic_status=Confidence.RIGOROUS_DERIVED,
        proof_gaps=[ProofGap(id="g", description="uniformity in q unproven")],
    )
    assert hypothesis.advances_frontier is False
    assert "uniformity in q unproven" in hypothesis.frontier.explain()


def test_blocked_hypothesis_is_not_actionable_and_says_why():
    hypothesis = Hypothesis(
        id="H-BLOCKED",
        statement="Needs the kernel workbench first.",
        lifecycle=HypothesisLifecycle.BLOCKED,
        blocker_refs=["HYP-002"],
        falsification_tests=[_test()],
    )
    assert hypothesis.actionable is False
    assert "HYP-002" in hypothesis.explain_actionability()


def test_resolved_hypothesis_is_not_actionable():
    hypothesis = Hypothesis(
        id="H-DONE",
        statement="Settled.",
        lifecycle=HypothesisLifecycle.RESOLVED,
        falsification_tests=[_test()],
    )
    assert hypothesis.actionable is False
    assert "work has concluded" in hypothesis.explain_actionability()


def test_hypothesis_without_a_falsification_test_says_so():
    hypothesis = Hypothesis(id="H-NOTEST", statement="Untestable as written.")
    assert hypothesis.actionable is False
    assert "no falsification test" in hypothesis.explain_actionability()


def test_queue_selects_cheapest_actionable_frontier_relevant_step():
    circular = Hypothesis(
        id="H1",
        statement="RH iff C.",
        rh_equivalent=True,
        mathematical_role=Role.EQUIVALENCE,
        falsification_tests=[_test(0)],
    )
    useful = Hypothesis(
        id="H2",
        statement="Unconditional screening bound target.",
        proof_gaps=[ProofGap(id="g", description="missing uniformity")],
        falsification_tests=[_test(3)],
    )
    queue = HypothesisQueue(hypotheses=[circular, useful])
    step = queue.next_step()
    assert step is not None
    assert step.hypothesis_id == "H2"


def test_queue_reports_open_hypotheses():
    queue = HypothesisQueue(
        hypotheses=[
            Hypothesis(id="A", statement="a", lifecycle=HypothesisLifecycle.ACTIVE),
            Hypothesis(id="B", statement="b", lifecycle=HypothesisLifecycle.ARCHIVED),
        ]
    )
    assert [h.id for h in queue.open_hypotheses()] == ["A"]


def test_hypothesis_property_graph_record_remains_symbolic():
    hypothesis = Hypothesis(
        id="H3",
        statement="R_q = O(X^0.49)",
        assumptions=["model assumption"],
        proof_gaps=[ProofGap(id="g", description="uniform in q")],
        falsification_tests=[_test()],
    )
    [prop] = extract_from_hypothesis(hypothesis)
    assert prop.status is EpistemicStatus.SYMBOLIC_DERIVED
    assert prop.is_rigorous is False
    # Derived verdicts are no longer copied into metadata: a stored snapshot of
    # a derived fact is free to drift from the facts it derives from.
    for derived in ("actionable", "advances_frontier", "frontier_relevant"):
        assert derived not in prop.metadata
    assert prop.metadata["lifecycle"] == "proposed"
    assert prop.metadata["epistemic_status"] == "conjectural"


def test_refuted_hypothesis_extracts_as_blocked():
    """The old code read a lifecycle field to decide an epistemic question."""
    hypothesis = Hypothesis(
        id="H4",
        statement="Boundary unitarity implies RH.",
        lifecycle=HypothesisLifecycle.RESOLVED,
        epistemic_status=Confidence.REFUTED,
    )
    [prop] = extract_from_hypothesis(hypothesis)
    assert prop.status is EpistemicStatus.BLOCKED


# --- migration from the deprecated HypothesisState --------------------------


def test_legacy_state_still_loads_and_recovers_both_facts():
    """The old field carried a workflow position and a verdict at once."""
    hypothesis = Hypothesis.model_validate(
        {"id": "H-OLD", "statement": "legacy record", "state": "falsified"}
    )
    assert hypothesis.lifecycle is HypothesisLifecycle.RESOLVED
    assert hypothesis.epistemic_status is Confidence.REFUTED
    assert hypothesis.metadata["migrated_from_state"] == "falsified"


@pytest.mark.parametrize(
    "state,lifecycle,confidence",
    [
        ("proposed", HypothesisLifecycle.PROPOSED, Confidence.CONJECTURAL),
        ("actionable", HypothesisLifecycle.TRIAGED, Confidence.CONJECTURAL),
        ("testing", HypothesisLifecycle.ACTIVE, Confidence.CONJECTURAL),
        ("blocked", HypothesisLifecycle.BLOCKED, Confidence.CONJECTURAL),
        ("falsified", HypothesisLifecycle.RESOLVED, Confidence.REFUTED),
        ("advanced", HypothesisLifecycle.RESOLVED, Confidence.RIGOROUS_DERIVED),
    ],
)
def test_every_legacy_state_migrates(state, lifecycle, confidence):
    hypothesis = Hypothesis.model_validate(
        {"id": "H", "statement": "s", "state": state}
    )
    assert hypothesis.lifecycle is lifecycle
    assert hypothesis.epistemic_status is confidence


def test_explicit_axes_win_over_a_stale_legacy_state():
    hypothesis = Hypothesis.model_validate(
        {
            "id": "H",
            "statement": "s",
            "state": "proposed",
            "lifecycle": "active",
            "epistemic_status": "symbolic_derived",
        }
    )
    assert hypothesis.lifecycle is HypothesisLifecycle.ACTIVE
    assert hypothesis.epistemic_status is Confidence.SYMBOLIC_DERIVED


def test_hypothesis_no_longer_has_a_state_field():
    assert "state" not in Hypothesis.model_fields
    assert "lifecycle" in Hypothesis.model_fields
    assert "epistemic_status" in Hypothesis.model_fields


def test_loading_a_v1_queue_upgrades_the_declared_schema_version(tmp_path):
    """A file that claims v1 while holding v2 records misleads the next migration."""
    import json

    from rh_research_engine.supervisor.store import HypothesisQueueStore

    path = tmp_path / "hypotheses.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "hypotheses": [{"id": "HYP-1", "statement": "legacy", "state": "advanced"}],
            }
        ),
        encoding="utf-8",
    )
    store = HypothesisQueueStore(path)
    queue = store.load()
    assert queue.schema_version == "2"
    assert queue.migration is not None
    assert queue.migration.source_schema_version == "1"

    store.save(queue)
    reread = json.loads(path.read_text(encoding="utf-8"))
    assert reread["schema_version"] == "2"
    assert "state" not in reread["hypotheses"][0]
    assert reread["hypotheses"][0]["lifecycle"] == "resolved"


def test_a_current_queue_is_not_marked_as_migrated():
    queue = HypothesisQueue()
    assert queue.schema_version == "2"
    assert queue.migration is None


# --- derived verdicts cannot be supplied through any route ------------------


DERIVED = ["advances_frontier", "frontier_relevant", "property_extractable", "actionable"]


@pytest.mark.parametrize("field", DERIVED)
def test_constructor_refuses_a_derived_verdict(field):
    """Silently dropping it is worse than refusing: the caller believes it worked."""
    with pytest.raises(ValueError, match="derived from"):
        Hypothesis(id="H", statement="s", **{field: True})


@pytest.mark.parametrize("field", DERIVED)
def test_model_validate_refuses_a_derived_verdict(field):
    with pytest.raises(ValueError, match="derived from"):
        Hypothesis.model_validate({"id": "H", "statement": "s", field: True})


@pytest.mark.parametrize("field", DERIVED)
def test_legacy_migration_refuses_a_derived_verdict(field):
    """A legacy payload must not smuggle one in alongside `state`."""
    with pytest.raises(ValueError, match="derived from"):
        Hypothesis.model_validate(
            {"id": "H", "statement": "s", "state": "advanced", field: True}
        )


def test_legacy_advanced_state_does_not_confer_advancement_directly():
    """`advanced` maps to an epistemic status, never to the verdict."""
    hypothesis = Hypothesis.model_validate(
        {"id": "H", "statement": "s", "state": "advanced", "rh_equivalent": True}
    )
    assert hypothesis.epistemic_status is Confidence.RIGOROUS_DERIVED
    assert hypothesis.advances_frontier is False  # circular, nothing discharged


def test_unknown_keys_are_refused_rather_than_dropped():
    with pytest.raises(ValueError):
        Hypothesis(id="H", statement="s", not_a_field=1)


def test_derived_verdicts_are_absent_from_the_serialized_form():
    hypothesis = Hypothesis(id="H", statement="s")
    dumped = hypothesis.model_dump(mode="json")
    for field in DERIVED:
        assert field not in dumped
    # ...and the round trip therefore cannot reintroduce one.
    assert Hypothesis.model_validate(dumped).advances_frontier is False


# --- the full product invariant ---------------------------------------------


def test_rh_equivalence_never_advances_across_the_full_input_product():
    """lifecycle x confidence x role x discharged-obligations.

    The critical rule: an RH-equivalent premise cannot produce frontier
    advancement without BOTH a named discharged obligation and qualifying
    rigorous evidence.
    """

    for lifecycle in HypothesisLifecycle:
        for confidence in Confidence:
            for role in Role:
                for discharged in ([], ["forward direction proved"]):
                    hypothesis = Hypothesis(
                        id="H",
                        statement="An RH equivalent reformulation.",
                        rh_equivalent=True,
                        lifecycle=lifecycle,
                        epistemic_status=confidence,
                        mathematical_role=role,
                        discharged_obligations=list(discharged),
                    )
                    # Prose never resolves, so nothing in this product can
                    # advance. Unlocking requires a registry, tested separately.
                    expected = False
                    assert hypothesis.advances_frontier is expected, (
                        f"{lifecycle.value}/{confidence.value}/{role.value}/"
                        f"discharged={bool(discharged)}"
                    )
                    if not expected:
                        assert hypothesis.frontier.reasons


def test_rewriting_rules_may_still_use_an_rh_equivalence():
    """Frontier rules are gated on advancement; rewriting rules only on rigor."""
    from rh_research_engine.contracts.frontier import usable_as_rule

    assert usable_as_rule(role=Role.EQUIVALENCE, confidence=Confidence.KNOWN) is True
    hypothesis = Hypothesis(
        id="H",
        statement="A <=> RH, classical.",
        rh_equivalent=True,
        mathematical_role=Role.EQUIVALENCE,
        epistemic_status=Confidence.KNOWN,
    )
    assert hypothesis.advances_frontier is False
    assert usable_as_rule(
        role=hypothesis.mathematical_role, confidence=hypothesis.epistemic_status
    ) is True


# --- queue migration: idempotence and non-mutation --------------------------


def _v1_file(tmp_path):
    import json

    path = tmp_path / "hypotheses.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "hypotheses": [
                    {"id": "HYP-1", "statement": "legacy", "state": "advanced"},
                    {"id": "HYP-2", "statement": "dead", "state": "falsified"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_loading_alone_never_mutates_the_source_file(tmp_path):
    import hashlib

    from rh_research_engine.supervisor.store import HypothesisQueueStore

    path = _v1_file(tmp_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    HypothesisQueueStore(path).load()
    HypothesisQueueStore(path).load()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_migration_record_separates_file_and_semantic_hashes(tmp_path):
    from rh_research_engine.supervisor.store import HypothesisQueueStore

    queue = HypothesisQueueStore(_v1_file(tmp_path)).load()
    record = queue.migration
    assert record is not None
    assert record.source_schema_version == "1"
    assert record.target_schema_version == "2"
    assert record.migration_id == "hypothesis-contract-v1"
    assert record.mapping_version == 1
    # Three distinct hashes: the bytes, the content before, the content after.
    assert record.source_file_sha256
    assert record.source_semantic_hash
    assert record.normalized_semantic_hash
    assert record.source_file_sha256 != record.source_semantic_hash
    assert record.source_semantic_hash != record.normalized_semantic_hash


def test_load_v1_save_v2_load_v2_is_idempotent(tmp_path):
    import hashlib

    from rh_research_engine.supervisor.store import HypothesisQueueStore

    path = _v1_file(tmp_path)
    store = HypothesisQueueStore(path)

    first = store.load()
    store.save(first)
    after_first_save = hashlib.sha256(path.read_bytes()).hexdigest()

    second = store.load()
    assert second.schema_version == "2"
    # The migration record is provenance *of the file*, not of the load: it
    # records that these contents came from a v1 payload with that hash, so it
    # persists unchanged rather than being re-derived or dropped.
    assert second.migration == first.migration
    assert second.migration.source_schema_version == "1"
    assert [h.id for h in second.hypotheses] == [h.id for h in first.hypotheses]
    assert [h.lifecycle for h in second.hypotheses] == [h.lifecycle for h in first.hypotheses]
    assert [h.epistemic_status for h in second.hypotheses] == [
        h.epistemic_status for h in first.hypotheses
    ]

    store.save(second)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == after_first_save


def test_a_stored_queue_without_a_schema_version_is_rejected(tmp_path):
    """Unreadable, not assumable.

    Guessing "probably current" is how a v1 file gets read as v2 and its
    `state` fields dropped as unknown keys -- a silent downgrade of every
    record in the plan.
    """
    import json

    from rh_research_engine.supervisor.models import QueueSchemaError
    from rh_research_engine.supervisor.store import HypothesisQueueStore

    path = tmp_path / "hypotheses.json"
    path.write_text(
        json.dumps({"hypotheses": [{"id": "H", "statement": "s", "state": "proposed"}]}),
        encoding="utf-8",
    )
    with pytest.raises(QueueSchemaError, match="declares no schema_version"):
        HypothesisQueueStore(path).load()


def test_in_memory_construction_may_omit_the_version():
    """Only a *stored* queue must declare it; the model has a current default."""
    assert HypothesisQueue(hypotheses=[]).schema_version == "2"


def test_the_file_hash_differs_from_the_semantic_hash(tmp_path):
    """Reformatting changes the bytes and not the content; both are recorded."""
    import hashlib
    import json

    from rh_research_engine.supervisor.store import HypothesisQueueStore

    compact = tmp_path / "compact.json"
    spaced = tmp_path / "spaced.json"
    document = {
        "schema_version": "1",
        "hypotheses": [{"id": "H", "statement": "s", "state": "proposed"}],
    }
    compact.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    spaced.write_text(json.dumps(document, indent=4), encoding="utf-8")

    a = HypothesisQueueStore(compact).load().migration
    b = HypothesisQueueStore(spaced).load().migration
    assert hashlib.sha256(compact.read_bytes()).hexdigest() != (
        hashlib.sha256(spaced.read_bytes()).hexdigest()
    )
    assert a.source_file_sha256 != b.source_file_sha256, "different bytes"
    assert a.source_semantic_hash == b.source_semantic_hash, "same content"
