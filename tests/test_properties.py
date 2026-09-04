from rh_research_engine.core.knowledge import KnowledgeItem, KnowledgeStatus
from rh_research_engine.properties import (
    ClosureMode,
    EpistemicStatus,
    PropertyGraph,
    PropertyGraphStore,
    PropertyKind,
    analyze_discriminators,
    extract_from_knowledge,
    implication_closure,
    object_inventory,
)
from rh_research_engine.symbolic.formula_index import FormulaRecord


def test_inventory_extracts_objects_with_provenance():
    record = FormulaRecord(
        id="F1",
        expression="R_q(X) = O(X^theta)",
        canonical_hash="h",
        canonical="c",
        kind="equation",
        lhs="R_q(X)",
        rhs="O(X^theta)",
        source="doc",
    )
    objects = object_inventory([record], [])
    assert any(obj.name == "R_q" for obj in objects)
    assert all(obj.provenance for obj in objects)


def test_rigorous_closure_only_uses_rigorous_statuses():
    item = KnowledgeItem(
        id="K1",
        title="remainder bound",
        status=KnowledgeStatus.KNOWN_OR_STANDARD_CONSEQUENCE,
        domain="bounds",
        statement="R_q = O(X^theta)",
    )
    graph = PropertyGraph(properties=extract_from_knowledge(item))
    closed = implication_closure(graph, mode=ClosureMode.RIGOROUS)
    assert any(prop.kind is PropertyKind.THETA_BOUND for prop in closed.properties)
    assert all(prop.status is not EpistemicStatus.HEURISTIC for prop in closed.properties)


def test_exploratory_closure_does_not_promote_synthetic_rh_claims():
    item = KnowledgeItem(
        id="K2",
        title="fit",
        status=KnowledgeStatus.RESEARCH_TARGET,
        domain="experiment",
        statement="R_q = O(X^theta)",
    )
    graph = implication_closure(
        PropertyGraph(properties=extract_from_knowledge(item)), mode=ClosureMode.EXPLORATORY
    )
    theta = [prop for prop in graph.properties if prop.kind is PropertyKind.THETA_BOUND]
    assert theta
    assert all(prop.status is EpistemicStatus.HEURISTIC for prop in theta)


def test_discriminator_is_fail_closed_for_synthetic_evidence():
    item = KnowledgeItem(
        id="K3",
        title="fit",
        status=KnowledgeStatus.RESEARCH_TARGET,
        domain="experiment",
        statement="R_q = O(X^theta)",
    )
    graph = PropertyGraph(properties=extract_from_knowledge(item))
    results = analyze_discriminators(graph)
    assert results
    assert all(not result.promoted_to_proof for result in results)
    assert all("cannot prove" in result.reason for result in results)


def test_property_graph_store_query_round_trip(tmp_path):
    item = KnowledgeItem(
        id="K4",
        title="remainder bound",
        status=KnowledgeStatus.KNOWN_OR_STANDARD_CONSEQUENCE,
        domain="bounds",
        statement="R_q = O(X^theta)",
    )
    store = PropertyGraphStore(tmp_path / "property_graph.json")
    graph = PropertyGraph(properties=extract_from_knowledge(item))
    store.save(graph)
    assert store.load().graph_hash() == graph.graph_hash()
    assert store.query(kind=PropertyKind.GROWTH_BOUND, rigorous_only=True)
