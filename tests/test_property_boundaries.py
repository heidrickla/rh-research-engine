"""Promotion-boundary tests for the property graph.

Same threat model as tests/test_promotion_boundaries.py, applied to the
properties subsystem: can an unverified, conditional, circular, or impossible
statement acquire a stronger status by passing through extraction, closure, or
the graph store?

Every assertion here corresponds to a defect that was exploitable.
"""

import json

import pytest

from rh_research_engine.core.knowledge import KnowledgeBase, KnowledgeItem, KnowledgeStatus
from rh_research_engine.properties import (
    META_ROLES,
    RIGOROUS_STATUSES,
    ClosureMode,
    EpistemicStatus,
    ImplicationRule,
    MathematicalRole,
    PropertyGraph,
    PropertyGraphIntegrityError,
    PropertyGraphStore,
    PropertyKind,
    PropertyRecord,
    Provenance,
    extract_from_knowledge,
    implication_closure,
    theta_is_possible,
)
from rh_research_engine.properties.extract import (
    _ROLE_MAP,
    _STATUS_MAP,
    _role_from_knowledge,
    _status_from_knowledge,
    is_property_extractable,
)


def _item(status: KnowledgeStatus, statement: str) -> KnowledgeItem:
    return KnowledgeItem(
        id="K1", title="t", status=status, domain="bounds", statement=statement
    )


def _prov() -> Provenance:
    return Provenance(source_type="symbolic", source_id="s", method="test")


def _theta_bounds(graph: PropertyGraph) -> list[PropertyRecord]:
    return [p for p in graph.properties if p.kind is PropertyKind.THETA_BOUND]


# --- P-01: unverified statuses must not read as rigorous --------------------


@pytest.mark.parametrize(
    "status",
    [
        KnowledgeStatus.DERIVED_SYMBOLIC,
        KnowledgeStatus.DERIVED_SYMBOLIC_NEEDS_EXTERNAL_CHECK,
        KnowledgeStatus.EXACT_ALGEBRA_IN_MODEL,
        KnowledgeStatus.KNOWN_MODEL_INPUT,
        KnowledgeStatus.RESEARCH_TARGET,
        KnowledgeStatus.ASYMPTOTIC_DERIVED,
    ],
)
def test_unverified_statuses_are_not_rigorous(status):
    """docs/RH_MATHEMATICAL_MEMORY.md: derived_symbolic is "not yet
    independently formalized or literature-checked end to end"."""
    assert _status_from_knowledge(status) not in RIGOROUS_STATUSES


@pytest.mark.parametrize(
    "status",
    [KnowledgeStatus.EXACT, KnowledgeStatus.EXACT_ALGEBRA, KnowledgeStatus.KNOWN],
)
def test_genuinely_established_statuses_stay_rigorous(status):
    assert _status_from_knowledge(status) in RIGOROUS_STATUSES


def test_status_map_covers_every_knowledge_status():
    """A status added later must be classified deliberately, not by spelling.

    The mapping was substring-based, so any new status containing "derived" or
    starting with "exact" would have been promoted to rigorous by accident.
    """
    missing = [s.value for s in KnowledgeStatus if s not in _STATUS_MAP]
    assert missing == [], f"unmapped statuses default silently: {missing}"


def test_false_route_is_a_blocked_mathematical_route():
    """BLOCKED keeps its narrow meaning: a route examined and unable to proceed."""
    assert _status_from_knowledge(KnowledgeStatus.FALSE_ROUTE) is EpistemicStatus.BLOCKED
    assert _role_from_knowledge(KnowledgeStatus.FALSE_ROUTE) is MathematicalRole.NO_GO


def test_governance_is_meta_not_a_blocked_route():
    """A rule about how the engine reasons is not a failed proof attempt.

    Filing it as BLOCKED would invite later code to read "numerical evidence
    cannot promote a theorem to proved" as a mathematical route that was tried.
    """
    assert _status_from_knowledge(KnowledgeStatus.GOVERNANCE) is EpistemicStatus.AUTHORITATIVE_POLICY
    assert _role_from_knowledge(KnowledgeStatus.GOVERNANCE) is MathematicalRole.GOVERNANCE
    assert _role_from_knowledge(KnowledgeStatus.GOVERNANCE) in META_ROLES


def test_governance_items_are_not_property_extractable():
    item = _item(KnowledgeStatus.GOVERNANCE, "Numerical evidence cannot promote a theorem.")
    assert is_property_extractable(item) is False
    assert extract_from_knowledge(item) == []


def test_shipped_governance_item_yields_no_properties():
    """K042 is the engine's own epistemic rule, not a statement about zeta."""
    governance = [i for i in KnowledgeBase().load() if i.status is KnowledgeStatus.GOVERNANCE]
    assert governance
    for item in governance:
        assert extract_from_knowledge(item) == []


def test_role_map_covers_every_knowledge_status():
    missing = [s.value for s in KnowledgeStatus if s not in _ROLE_MAP]
    assert missing == [], f"unmapped roles default silently: {missing}"


def test_shipped_memory_rigorous_count_matches_the_documented_legend():
    """Only the exact-identity and known-literature families are rigorous."""
    items = KnowledgeBase().load()
    rigorous = [i for i in items if _status_from_knowledge(i.status) in RIGOROUS_STATUSES]
    for item in rigorous:
        assert item.status.value.startswith(("exact", "known")), item.status.value
        assert item.status is not KnowledgeStatus.EXACT_ALGEBRA_IN_MODEL


# --- P-02: conditions on a status reach every extracted property ------------


def test_conditional_on_rh_theta_bound_keeps_its_condition():
    """An RH-conditional bound arriving as unconditional is the circular step."""
    props = extract_from_knowledge(
        _item(KnowledgeStatus.CONDITIONAL_ON_RH_STANDARD, "Under RH, Theta <= 1/2 here.")
    )
    assert props
    for prop in props:
        assert "conditional on RH" in prop.assumptions
        assert prop.is_rigorous is False


def test_conditional_on_rh_growth_bound_keeps_its_condition():
    props = extract_from_knowledge(
        _item(KnowledgeStatus.CONDITIONAL_ON_RH_STANDARD, "Under RH, R_q = O(X^0.2).")
    )
    assert props
    assert all("conditional on RH" in p.assumptions for p in props)


def test_in_model_result_carries_the_model_assumption():
    props = extract_from_knowledge(
        _item(KnowledgeStatus.EXACT_ALGEBRA_IN_MODEL, "R_q = O(X^0.2)")
    )
    assert props
    assert all(p.assumptions for p in props)


# --- P-03: the closure never emits an impossible Theta bound ----------------


@pytest.mark.parametrize("exponent", ["-0.5", "-1", "-0.0001"])
def test_negative_exponent_yields_no_theta_bound(exponent):
    """Theta >= 1/2 unconditionally, so Theta <= 0.25 is provably false."""
    graph = implication_closure(
        PropertyGraph(
            properties=extract_from_knowledge(
                _item(KnowledgeStatus.KNOWN, f"R_q = O(X^{exponent})")
            )
        ),
        mode=ClosureMode.RIGOROUS,
    )
    assert _theta_bounds(graph) == []


@pytest.mark.parametrize("exponent", ["0", "0.2", "1"])
def test_valid_exponent_still_derives_a_theta_bound(exponent):
    graph = implication_closure(
        PropertyGraph(
            properties=extract_from_knowledge(
                _item(KnowledgeStatus.KNOWN, f"R_q = O(X^{exponent})")
            )
        ),
        mode=ClosureMode.RIGOROUS,
    )
    assert _theta_bounds(graph)


def test_theta_is_possible_predicate():
    assert theta_is_possible("0.2")
    assert theta_is_possible("0")
    assert not theta_is_possible("-0.5")
    assert theta_is_possible("theta"), "a symbolic exponent is not yet a number to check"


def test_exploratory_mode_also_refuses_impossible_bounds():
    graph = implication_closure(
        PropertyGraph(
            properties=extract_from_knowledge(_item(KnowledgeStatus.KNOWN, "R_q = O(X^-0.5)"))
        ),
        mode=ClosureMode.EXPLORATORY,
    )
    assert _theta_bounds(graph) == []


# --- P-04: conditions block rigor and survive the closure -------------------


def test_conditions_block_rigor():
    prop = PropertyRecord(
        id="p1",
        object_id="o",
        kind=PropertyKind.GROWTH_BOUND,
        value="X^0.2",
        status=EpistemicStatus.CERTIFIED,
        provenance=[_prov()],
        conditions=["valid only for X < 10**6"],
    )
    assert prop.is_rigorous is False
    assert prop.open_qualifiers == ["valid only for X < 10**6"]


def test_conditioned_property_does_not_feed_the_rigorous_closure():
    prop = PropertyRecord(
        id="p1",
        object_id="o",
        kind=PropertyKind.GROWTH_BOUND,
        value="X^0.2",
        status=EpistemicStatus.CERTIFIED,
        provenance=[_prov()],
        conditions=["s - 1 != 0"],
    )
    graph = implication_closure(PropertyGraph(properties=[prop]), mode=ClosureMode.RIGOROUS)
    assert _theta_bounds(graph) == []


def test_conclusions_inherit_both_assumptions_and_conditions():
    prop = PropertyRecord(
        id="p1",
        object_id="o",
        kind=PropertyKind.GROWTH_BOUND,
        value="X^0.2",
        status=EpistemicStatus.CERTIFIED,
        provenance=[_prov()],
        conditions=["s - 1 != 0"],
        assumptions=["assumes the model"],
    )
    graph = implication_closure(PropertyGraph(properties=[prop]), mode=ClosureMode.EXPLORATORY)
    derived = _theta_bounds(graph)
    assert derived
    assert derived[0].conditions == ["s - 1 != 0"]
    assert derived[0].assumptions == ["assumes the model"]
    assert derived[0].is_rigorous is False


# --- P-05: the graph store refuses unearnable and tampered records ----------


def test_store_refuses_an_injected_proved_record(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "objects": [],
                "edges": [],
                "properties": [
                    {
                        "id": "prop:INJECTED",
                        "object_id": "o",
                        "kind": "theta_bound",
                        "value": "1/2",
                        "status": "proved",
                        "provenance": [
                            {"source_type": "manual", "source_id": "x", "method": "by-hand"}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PropertyGraphIntegrityError):
        PropertyGraphStore(path).load()


def test_proved_status_cannot_be_constructed():
    # Pydantic wraps a validator failure in ValidationError, which subclasses
    # ValueError (as does ForbiddenStatusError); the message is what callers act on.
    with pytest.raises(ValueError, match="cannot be produced by this package"):
        PropertyRecord(
            id="p",
            object_id="o",
            kind=PropertyKind.THETA_BOUND,
            value="1/2",
            status=EpistemicStatus.PROVED,
            provenance=[_prov()],
        )


def test_store_seal_detects_tampering(tmp_path):
    path = tmp_path / "graph.json"
    store = PropertyGraphStore(path)
    store.save(PropertyGraph(properties=extract_from_knowledge(_item(KnowledgeStatus.KNOWN, "R_q = O(X^0.2)"))))
    assert store.seal_path.exists()
    assert store.load().properties

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["properties"][0]["status"] = "certified"
    path.write_text(json.dumps(payload), encoding="utf-8", newline="")
    with pytest.raises(PropertyGraphIntegrityError, match="does not match its seal"):
        store.load()


# --- P-06: RH-equivalence is representable and never implies progress -------


def test_rh_equivalent_stays_rigorous_but_earns_no_frontier_progress():
    """The three axes are independent, and this is the case that proves it.

    A classical `A <=> RH` is established mathematics -- reporting it as
    non-rigorous would understate our actual confidence. What it must not do is
    count as progress: proving yet another reformulation of `A` moves nothing.
    """
    props = extract_from_knowledge(
        _item(KnowledgeStatus.KNOWN_EQUIVALENCE_FRAMEWORK, "R_q = O(X^0.2)")
    )
    assert props
    for prop in props:
        assert prop.status is EpistemicStatus.KNOWN, "axis 1: fully established"
        assert prop.role is MathematicalRole.EQUIVALENCE, "axis 2: an equivalence"
        assert prop.rh_equivalent is True, "axis 3: restates the target"
        assert prop.is_rigorous is True, "confidence is not reduced by uselessness"
        assert prop.advances_frontier is False, "but it earns no progress credit"
        assert prop.is_usable_as_rule is True, "and remains a valid rewriting rule"


def _equivalence(**overrides) -> PropertyRecord:
    base = dict(
        id="p",
        object_id="o",
        kind=PropertyKind.GROWTH_BOUND,
        value="X^0.2",
        status=EpistemicStatus.KNOWN,
        role=MathematicalRole.EQUIVALENCE,
        provenance=[_prov()],
        rh_equivalent=True,
    )
    base.update(overrides)
    return PropertyRecord(**base)


def test_a_prose_discharge_does_not_restore_frontier_progress():
    """A sentence is an assertion, not a discharge."""
    prop = _equivalence(discharged_obligations=["forward direction proved in Lemma 4"])
    assert prop.advances_frontier is False
    assert "ProofObligation" in prop.frontier.explain()


def test_a_resolved_obligation_restores_frontier_progress():
    from helpers_discharge import accepted_registry, trusting_test_engine

    prop = _equivalence(discharged_obligations=["OBL-1"])
    obligations, evidence, decisions, receipts = accepted_registry()
    with trusting_test_engine(receipts.values()):
        assert prop.frontier_with(
            obligations=obligations,
            evidence=evidence,
            decisions=decisions,
            receipts=receipts,
        ).advances_frontier
    # ...and without the registry it stays circular.
    assert prop.advances_frontier is False


def test_meta_records_never_advance_the_frontier():
    prop = PropertyRecord(
        id="p",
        object_id="o",
        kind=PropertyKind.GROWTH_BOUND,
        value="X^0.2",
        status=EpistemicStatus.KNOWN,
        role=MathematicalRole.GOVERNANCE,
        provenance=[_prov()],
    )
    assert prop.is_rigorous is True
    assert prop.advances_frontier is False
    assert prop.is_usable_as_rule is False


def test_meta_record_implies_nothing_in_the_closure():
    prop = PropertyRecord(
        id="p",
        object_id="o",
        kind=PropertyKind.GROWTH_BOUND,
        value="X^0.2",
        status=EpistemicStatus.KNOWN,
        role=MathematicalRole.GOVERNANCE,
        provenance=[_prov()],
    )
    graph = implication_closure(PropertyGraph(properties=[prop]), mode=ClosureMode.RIGOROUS)
    assert _theta_bounds(graph) == []


def test_transformation_rules_may_use_an_rh_equivalence():
    """Frontier rules are gated on progress; rewriting rules only on rigor."""
    prop = PropertyRecord(
        id="p",
        object_id="o",
        kind=PropertyKind.GROWTH_BOUND,
        value="X^0.2",
        status=EpistemicStatus.KNOWN,
        role=MathematicalRole.EQUIVALENCE,
        provenance=[_prov()],
        rh_equivalent=True,
    )
    rewriting_rule = ImplicationRule(
        id="RH-PROP-001",
        premise_kind=PropertyKind.GROWTH_BOUND,
        conclusion_kind=PropertyKind.THETA_BOUND,
        description="treated as a pure rewriting step",
        produces_frontier_claim=False,
    )
    frontier = implication_closure(PropertyGraph(properties=[prop]), mode=ClosureMode.RIGOROUS)
    rewritten = implication_closure(
        PropertyGraph(properties=[prop]), mode=ClosureMode.RIGOROUS, rules=[rewriting_rule]
    )
    assert _theta_bounds(frontier) == [], "a frontier rule must refuse it"
    assert _theta_bounds(rewritten), "a rewriting rule may use it"


@pytest.mark.parametrize("mode", [ClosureMode.RIGOROUS, ClosureMode.EXPLORATORY])
def test_rh_equivalent_premise_never_implies_a_theta_bound(mode):
    """Restating the target is not progress toward it, in either mode."""
    graph = implication_closure(
        PropertyGraph(
            properties=extract_from_knowledge(
                _item(KnowledgeStatus.KNOWN_EQUIVALENCE_FRAMEWORK, "R_q = O(X^0.2)")
            )
        ),
        mode=mode,
    )
    assert _theta_bounds(graph) == []


# --- axis 3 has two distinct readings, and both are needed ------------------


def test_research_target_is_frontier_relevant_but_has_not_advanced_anything():
    """"Worth proving" and "has been proved" are different questions.

    A research target scores frontier_relevant=True precisely because proving it
    would matter, and advances_frontier=False because it has not been proved.
    Only the latter may drive a closure.
    """
    props = extract_from_knowledge(_item(KnowledgeStatus.RESEARCH_TARGET, "R_q = O(X^0.2)"))
    assert props
    for prop in props:
        assert prop.role is MathematicalRole.BOUND
        assert prop.rh_equivalent is False
        assert prop.frontier_relevant is True, "worth proving"
        assert prop.advances_frontier is False, "but not yet proved"


def test_rh_equivalence_is_not_frontier_relevant_even_when_established():
    props = extract_from_knowledge(
        _item(KnowledgeStatus.KNOWN_EQUIVALENCE_FRAMEWORK, "R_q = O(X^0.2)")
    )
    assert props
    for prop in props:
        assert prop.is_rigorous is True
        assert prop.frontier_relevant is False, "another reformulation earns nothing"
        assert prop.advances_frontier is False


def test_meta_records_are_neither_relevant_nor_advancing():
    prop = PropertyRecord(
        id="p",
        object_id="o",
        kind=PropertyKind.GROWTH_BOUND,
        value="X^0.2",
        status=EpistemicStatus.KNOWN,
        role=MathematicalRole.GOVERNANCE,
        provenance=[_prov()],
    )
    assert prop.frontier_relevant is False
    assert prop.advances_frontier is False


def test_established_non_circular_bound_both_relevant_and_advancing():
    props = extract_from_knowledge(_item(KnowledgeStatus.KNOWN, "R_q = O(X^0.2)"))
    assert props
    for prop in props:
        assert prop.frontier_relevant is True
        assert prop.advances_frontier is True


def test_no_go_route_is_retained_but_not_worth_pursuing():
    """A refuted route stays in the record; proving it is not a path forward."""
    props = extract_from_knowledge(_item(KnowledgeStatus.FALSE_ROUTE, "R_q = O(X^0.2)"))
    assert props, "no-go routes are retained, not dropped"
    for prop in props:
        assert prop.role is MathematicalRole.NO_GO
        assert prop.status is EpistemicStatus.BLOCKED
        assert prop.frontier_relevant is False
        assert prop.advances_frontier is False


# --- the closure actually receives the registries ---------------------------


def test_closure_without_registries_cannot_advance_a_discharged_equivalence():
    """The gate must be strict, not stuck.

    `implication_closure` called `prop.frontier` with nothing, so no discharge
    could ever resolve and a legitimately discharged equivalence was as
    permanently blocked as an undischarged one.
    """
    prop = _equivalence(
        kind=PropertyKind.GROWTH_BOUND, value="X^0.2", discharged_obligations=["OBL-1"]
    )
    graph = implication_closure(PropertyGraph(properties=[prop]), mode=ClosureMode.RIGOROUS)
    assert _theta_bounds(graph) == []


def test_closure_with_an_accepted_decision_advances():
    from helpers_discharge import accepted_registry, trusting_test_engine

    obligations, evidence, decisions, receipts = accepted_registry()
    prop = _equivalence(
        kind=PropertyKind.GROWTH_BOUND, value="X^0.2", discharged_obligations=["OBL-1"]
    )
    with trusting_test_engine(receipts.values()):
        graph = implication_closure(
            PropertyGraph(properties=[prop]),
            mode=ClosureMode.RIGOROUS,
            obligations=obligations,
            evidence=evidence,
            decisions=decisions,
            receipts=receipts,
        )
    assert _theta_bounds(graph), "an accepted discharge must be able to advance"


def test_closure_with_an_unaccepted_decision_still_refuses():
    from helpers_discharge import pending_decision_registry

    obligations, evidence, decisions, receipts = pending_decision_registry()
    prop = _equivalence(
        kind=PropertyKind.GROWTH_BOUND, value="X^0.2", discharged_obligations=["OBL-1"]
    )
    graph = implication_closure(
        PropertyGraph(properties=[prop]),
        mode=ClosureMode.RIGOROUS,
        obligations=obligations,
        evidence=evidence,
        decisions=decisions,
        receipts=receipts,
    )
    assert _theta_bounds(graph) == []


def test_exploratory_mode_relaxes_rigor_but_never_circularity():
    """Approved behaviour: exploratory may relax rigour, not circularity.

    An RH-equivalent premise stays out of frontier-producing rules in both
    modes; it remains available through rewriting rules.
    """
    from helpers_discharge import no_decision_registry

    obligations, evidence, decisions, receipts = no_decision_registry()
    circular = _equivalence(
        kind=PropertyKind.GROWTH_BOUND, value="X^0.2", discharged_obligations=["OBL-1"]
    )
    graph = implication_closure(
        PropertyGraph(properties=[circular]),
        mode=ClosureMode.EXPLORATORY,
        obligations=obligations,
        evidence=evidence,
        decisions=decisions,
        receipts=receipts,
    )
    assert _theta_bounds(graph) == []

    # A non-circular but non-rigorous premise is admitted in exploratory mode.
    speculative = PropertyRecord(
        id="p-spec",
        object_id="o",
        kind=PropertyKind.GROWTH_BOUND,
        value="X^0.2",
        status=EpistemicStatus.SYMBOLIC_DERIVED,
        role=MathematicalRole.BOUND,
        provenance=[_prov()],
    )
    relaxed = implication_closure(
        PropertyGraph(properties=[speculative]), mode=ClosureMode.EXPLORATORY
    )
    assert _theta_bounds(relaxed)


def test_closure_imports_meta_roles_from_the_contract_layer():
    import rh_research_engine.properties.closure as closure_module
    from rh_research_engine.contracts.roles import META_ROLES as canonical

    assert closure_module.META_ROLES is canonical


def test_a_successful_discharge_is_recorded_in_the_derived_provenance():
    """The registry must not change the result invisibly.

    Without this the graph showed a rigorous derived bound with nothing
    explaining why a circular premise had been allowed to drive it.
    """
    from helpers_discharge import TEST_ENGINE, accepted_registry, trusting_test_engine

    obligations, evidence, decisions, receipts = accepted_registry()
    prop = _equivalence(
        kind=PropertyKind.GROWTH_BOUND, value="X^0.2", discharged_obligations=["OBL-1"]
    )
    with trusting_test_engine(receipts.values()):
        graph = implication_closure(
            PropertyGraph(properties=[prop]),
            mode=ClosureMode.RIGOROUS,
            obligations=obligations,
            evidence=evidence,
            decisions=decisions,
            receipts=receipts,
        )
    [derived] = _theta_bounds(graph)
    receipts = derived.metadata["discharge_receipts"]
    assert set(receipts) == {"OBL-1"}
    assert len(receipts["OBL-1"]) == 64

    attribution = " ".join(derived.provenance[0].assumptions)
    assert TEST_ENGINE in attribution
    assert "authorized by DRE engine" in attribution

    # ...and it survives serialization, so the stored graph is auditable alone.
    import json as _json

    dumped = _json.loads(_json.dumps(derived.model_dump(mode="json")))
    assert dumped["metadata"]["discharge_receipts"]["OBL-1"] == receipts["OBL-1"]


def test_a_derivation_without_a_discharge_carries_no_receipt_key():
    graph = implication_closure(
        PropertyGraph(
            properties=extract_from_knowledge(_item(KnowledgeStatus.KNOWN, "R_q = O(X^0.2)"))
        ),
        mode=ClosureMode.RIGOROUS,
    )
    [derived] = _theta_bounds(graph)
    assert "discharge_receipts" not in derived.metadata


# --- the graph is built from formulas, not from prose -----------------------


def test_object_inventory_does_not_mint_objects_from_prose():
    """A word in a sentence is not a mathematical object.

    The symbol regex used to run over `title` and `statement`, so every English
    word in durable memory became a MathObject of kind CLAIM whose "expression"
    was the sentence it came from. A real build produced 558 objects from 42
    records -- 'rather', 'width', and 'Schr' (a truncated "Schrodinger") among
    them -- and every downstream analysis then ran on the fabrications.
    """
    from rh_research_engine.core.knowledge import KnowledgeItem, KnowledgeStatus
    from rh_research_engine.properties.inventory import object_inventory

    item = KnowledgeItem(
        id="K900",
        title="Schrodinger operators and rather wide windows",
        status=KnowledgeStatus.EXACT_ALGEBRA,
        domain="analysis",
        statement="Use z = s - 1/2 and consider the width of the window.",
        formulas=["Theta + theta"],
    )
    names = {obj.name for obj in object_inventory(knowledge=[item])}

    assert names == {"Theta", "theta"}
    for prose_word in ("rather", "width", "Schrodinger", "window", "consider"):
        assert prose_word not in names


def test_a_failed_domain_analysis_yields_no_property():
    """"Unknown" asserts nothing, so it is not a property record.

    These used to be emitted with `status=BLOCKED` and `value="unknown"`,
    identical in every field but the object they pointed at. 471 of 481
    properties in a real graph were that record, burying the ten real ones.
    """
    from rh_research_engine.properties.models import MathObject, ObjectKind
    from rh_research_engine.properties.singularity import propagate_singularities

    unanalysable = MathObject(
        id="obj:prose",
        name="width",
        kind=ObjectKind.CLAIM,
        expression="Use z = s - 1/2 and consider the width of the window.",
    )
    failures: list[str] = []
    records = propagate_singularities([unanalysable], unanalysed=failures)

    assert records == []
    # Reported, not silently dropped: a caller that wants the count can have it.
    assert failures and failures[0].startswith("width:")


def test_a_real_expression_still_yields_its_singularity():
    """The refusal is scoped to failures, not to the feature."""
    from rh_research_engine.properties.models import MathObject, ObjectKind
    from rh_research_engine.properties.singularity import propagate_singularities

    genuine = MathObject(
        id="obj:frac", name="f", kind=ObjectKind.EXPRESSION, expression="1/(s-1)"
    )
    records = propagate_singularities([genuine])
    assert records, "a genuine pole must still be recorded"
    assert all(r.value != "unknown" for r in records)
