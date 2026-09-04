"""The contract layer's own tests.

The mappings are only a contract if they are *total*. A table with a hole in it
is a table that silently classifies whatever falls through, which is the defect
the layer exists to remove.
"""

import pytest

from rh_research_engine.contracts import (
    MATHEMATICAL_ROLES,
    META_ROLES,
    NON_DEDUCTIVE,
    RIGOROUS,
    WORKER_FORBIDDEN,
    Confidence,
    HypothesisLifecycle,
    Role,
    assess,
    at_least,
    is_rigorous,
    usable_as_rule,
)
from rh_research_engine.contracts.epistemic import CONFIDENCE_RANK, NON_MATHEMATICAL
from rh_research_engine.contracts.lifecycle import (
    CLOSED_LIFECYCLES,
    OPEN_LIFECYCLES,
    STALLED_LIFECYCLES,
)
from rh_research_engine.contracts.mappings import (
    KNOWLEDGE_STATUS_RH_EQUIVALENT,
    LEGACY_TABLES,
    UnmappedValueError,
    confidence_from_knowledge_status,
    role_from_knowledge_status,
)

# --- totality: every legacy value is classified deliberately ----------------


@pytest.mark.parametrize("table_name", sorted(LEGACY_TABLES))
def test_every_legacy_value_has_an_explicit_mapping(table_name):
    """A value added later must fail here until someone classifies it."""
    enum_type, table = LEGACY_TABLES[table_name]
    missing = sorted(str(member.value) for member in enum_type if member not in table)
    assert missing == [], (
        f"{table_name} is missing {missing}. Add explicit entries; do not add a "
        "default branch, which would classify an unreviewed value as something."
    )


@pytest.mark.parametrize("table_name", sorted(LEGACY_TABLES))
def test_no_mapping_table_has_stray_keys(table_name):
    enum_type, table = LEGACY_TABLES[table_name]
    members = set(enum_type)
    stray = sorted(str(key) for key in table if key not in members)
    assert stray == [], f"{table_name} maps values that are not members: {stray}"


@pytest.mark.parametrize("table_name", sorted(LEGACY_TABLES))
def test_mapping_targets_are_canonical_members(table_name):
    _, table = LEGACY_TABLES[table_name]
    for value in table.values():
        assert isinstance(value, (Confidence, Role, HypothesisLifecycle))


def test_unmapped_value_raises_and_names_the_table():
    class Rogue(str):
        pass

    with pytest.raises(UnmappedValueError) as excinfo:
        confidence_from_knowledge_status(Rogue("invented_status"))
    message = str(excinfo.value)
    assert "invented_status" in message
    assert "KNOWLEDGE_STATUS_TO_CONFIDENCE" in message
    assert "deliberately" in message


# --- the axes stay independent ----------------------------------------------


def test_every_confidence_has_a_rank():
    missing = [c.value for c in Confidence if c not in CONFIDENCE_RANK]
    assert missing == []


def test_rigorous_and_non_deductive_are_disjoint():
    assert not (RIGOROUS & NON_DEDUCTIVE)


def test_rigorous_numerical_is_not_rigorous():
    """A certified enclosure is rigorous about a *finite* computation.

    Treating it as rigorous in general is the step from "checked to height T"
    to "true".
    """
    assert Confidence.RIGOROUS_NUMERICAL not in RIGOROUS
    assert Confidence.RIGOROUS_NUMERICAL in NON_DEDUCTIVE


def test_worker_forbidden_are_all_rigorous():
    assert WORKER_FORBIDDEN <= RIGOROUS


def test_mathematical_and_meta_roles_partition_the_enum():
    assert MATHEMATICAL_ROLES | META_ROLES == set(Role)
    assert not (MATHEMATICAL_ROLES & META_ROLES)


def test_lifecycles_partition_into_open_stalled_closed():
    assert OPEN_LIFECYCLES | STALLED_LIFECYCLES | CLOSED_LIFECYCLES == set(
        HypothesisLifecycle
    )
    assert not (OPEN_LIFECYCLES & CLOSED_LIFECYCLES)


def test_confidence_comparison_rejects_category_errors():
    """Policy and refutation are not points on a confidence scale."""
    for bad in (Confidence.AUTHORITATIVE_POLICY, Confidence.REFUTED):
        with pytest.raises(ValueError, match="category error"):
            at_least(bad, Confidence.KNOWN)
        with pytest.raises(ValueError, match="category error"):
            at_least(Confidence.KNOWN, bad)


def test_confidence_ordering_is_transitive_over_mathematical_values():
    scale = [c for c in Confidence if c not in NON_MATHEMATICAL and c is not Confidence.REFUTED]
    for a in scale:
        for b in scale:
            for c in scale:
                if at_least(a, b) and at_least(b, c):
                    assert at_least(a, c), f"{a} >= {b} >= {c} but not {a} >= {c}"


# --- frontier: relevance and advancement stay distinct ----------------------


def test_research_target_is_relevant_but_not_advancing():
    result = assess(role=Role.BOUND, confidence=Confidence.CONJECTURAL)
    assert result.frontier_relevant is True
    assert result.advances_frontier is False
    assert "not rigorous" in result.explain()


def test_rh_equivalence_is_rigorous_yet_neither_relevant_nor_advancing():
    """The case that proves the axes are independent."""
    result = assess(role=Role.EQUIVALENCE, confidence=Confidence.KNOWN, rh_equivalent=True)
    assert is_rigorous(Confidence.KNOWN)
    assert result.frontier_relevant is False
    assert result.advances_frontier is False
    assert "restates RH" in result.explain()
    # ...and it remains a usable rewriting rule.
    assert usable_as_rule(role=Role.EQUIVALENCE, confidence=Confidence.KNOWN)


def test_only_a_resolved_discharge_restores_advancement():
    """`assess` takes *resolved* references; prose never reaches it."""
    unresolved = assess(
        role=Role.EQUIVALENCE, confidence=Confidence.KNOWN, rh_equivalent=True
    )
    assert unresolved.advances_frontier is False

    resolved = assess(
        role=Role.EQUIVALENCE,
        confidence=Confidence.KNOWN,
        rh_equivalent=True,
        qualifying_discharges=["OBL-1"],
    )
    assert resolved.frontier_relevant is True
    assert resolved.advances_frontier is True


def test_assess_signature_names_what_it_requires():
    import inspect

    params = inspect.signature(assess).parameters
    assert "qualifying_discharges" in params
    assert "discharged_obligations" not in params


def test_open_qualifiers_block_advancement_and_are_named():
    result = assess(
        role=Role.BOUND,
        confidence=Confidence.KNOWN,
        open_qualifiers=["valid only for X < 10**6"],
    )
    assert result.frontier_relevant is True
    assert result.advances_frontier is False
    assert "valid only for X < 10**6" in result.explain()


def test_meta_roles_yield_nothing_and_say_why():
    result = assess(role=Role.GOVERNANCE, confidence=Confidence.AUTHORITATIVE_POLICY)
    assert result.property_extractable is False
    assert result.frontier_relevant is False
    assert result.advances_frontier is False
    assert "meta, not mathematics" in result.explain()


@pytest.mark.parametrize("role", [Role.DEFINITION, Role.NO_GO])
def test_definitions_and_no_go_routes_never_advance(role):
    result = assess(role=role, confidence=Confidence.KNOWN)
    assert result.frontier_relevant is False
    assert result.advances_frontier is False


def test_established_non_circular_bound_advances():
    result = assess(role=Role.BOUND, confidence=Confidence.KNOWN)
    assert result.frontier_relevant is True
    assert result.advances_frontier is True
    assert result.reasons == []


def test_assessment_always_explains_a_negative_verdict():
    """A bare False tells a researcher nothing about what would change it."""
    for role in Role:
        for confidence in Confidence:
            result = assess(role=role, confidence=confidence)
            if not result.advances_frontier:
                assert result.reasons, f"{role.value}/{confidence.value} denied without a reason"


# --- the shipped durable memory maps cleanly --------------------------------


def test_shipped_knowledge_maps_without_gaps():
    from rh_research_engine.core.knowledge import KnowledgeBase

    for item in KnowledgeBase().load():
        confidence = confidence_from_knowledge_status(item.status)
        role = role_from_knowledge_status(item.status)
        assert isinstance(confidence, Confidence)
        assert isinstance(role, Role)


def test_governance_records_map_to_policy_and_meta():
    from rh_research_engine.core.knowledge import KnowledgeBase, KnowledgeStatus

    governance = [i for i in KnowledgeBase().load() if i.status is KnowledgeStatus.GOVERNANCE]
    assert governance, "the shipped memory has a governance record (K042)"
    for item in governance:
        assert confidence_from_knowledge_status(item.status) is Confidence.AUTHORITATIVE_POLICY
        assert role_from_knowledge_status(item.status) in META_ROLES


def test_rh_equivalent_knowledge_statuses_are_flagged_not_downgraded():
    for status in KNOWLEDGE_STATUS_RH_EQUIVALENT:
        role = role_from_knowledge_status(status)
        assert role is Role.EQUIVALENCE
        # Confidence is unaffected by circularity; only the frontier axis is.
        assert confidence_from_knowledge_status(status) is not Confidence.REFUTED


# --- the deprecation stays contained ----------------------------------------


def test_deprecated_hypothesis_state_is_confined_to_compat_and_mappings():
    """A deprecation that spreads is not a deprecation.

    `HypothesisState` conflated lifecycle with epistemic verdict. It survives
    only so stored records keep loading; anything else importing it re-couples
    the two axes the contract layer just separated.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "rh_research_engine"
    allowed = {
        "supervisor/compat.py",  # defines it
        "contracts/mappings.py",  # owns the translation
    }
    offenders: list[str] = []
    for path in sorted(src.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(src)).replace("\\", "/")
        if rel in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "HypothesisState":
                        offenders.append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.Name) and node.id == "HypothesisState":
                offenders.append(f"{rel}:{node.lineno}")
    assert offenders == [], (
        f"HypothesisState is used outside {sorted(allowed)}: {offenders}. "
        "Use Hypothesis.lifecycle plus Hypothesis.epistemic_status instead."
    )


def test_supervisor_package_no_longer_exports_the_deprecated_enum():
    import rh_research_engine.supervisor as supervisor

    assert "HypothesisState" not in supervisor.__all__
    assert "HypothesisLifecycle" in supervisor.__all__


def test_compat_module_records_when_the_deprecation_expires():
    from rh_research_engine.supervisor import compat

    assert compat.REMOVE_AFTER
    assert "HypothesisLifecycle" in compat.HypothesisState.__doc__


# --- discharge authority: DRE rules, labels do not -------------------------


import pytest as _pytest  # noqa: E402

import helpers_discharge  # noqa: E402


def _resolve(registry, *, trusting: bool = True):
    """Resolve inside the scoped test trust set, unless asked not to."""
    from contextlib import nullcontext

    from helpers_discharge import trusting_test_engine
    from rh_research_engine.contracts.discharge import resolve_discharges

    obligations, evidence, decisions, receipts = registry
    with (trusting_test_engine(receipts.values()) if trusting else nullcontext()):
        return resolve_discharges(
            ["OBL-1"],
            obligations=obligations,
            evidence=evidence,
            decisions=decisions,
            receipts=receipts,
        )


def test_a_verified_dre_decision_discharges():
    from helpers_discharge import TEST_ENGINE, accepted_registry

    resolution = _resolve(accepted_registry())
    assert resolution.qualifying == ["OBL-1"]
    assert "verified against DRE engine" in resolution.explain()
    assert resolution.receipts["OBL-1"]
    assert TEST_ENGINE in resolution.provenance_note()[0]


@_pytest.mark.parametrize(
    "registry_name,expected", sorted(helpers_discharge.REFUSAL_REASONS.items())
)
def test_every_authority_gap_is_refused_with_its_reason(registry_name, expected):
    """Every gate names the check that failed, not just that something did.

    A refusal that says only "verification failed" sends a researcher looking
    at the mathematics when the actual problem is a disconnected engine, a
    stale receipt, or a decision edited after the ruling.
    """
    resolution = _resolve(getattr(helpers_discharge, registry_name)())
    assert resolution.qualifying == []
    assert expected in resolution.explain()


def test_the_refusal_table_covers_every_non_qualifying_registry():
    """A registry added without a reason would be silently untested."""
    expected = set(helpers_discharge.ALL_REGISTRIES) - {"accepted_registry"}
    assert set(helpers_discharge.REFUSAL_REASONS) == expected


# --- the production invariant ----------------------------------------------


@_pytest.mark.parametrize("registry_name", helpers_discharge.ALL_REGISTRIES)
def test_no_public_path_yields_a_discharge_with_the_real_trust_registry(registry_name):
    """The invariant: with the production trust set empty, nothing discharges.

    `accepted_registry` is included deliberately -- the combination that works
    under the scoped test trust set must fail without it, or the test set is
    not what is doing the work.
    """
    resolution = _resolve(getattr(helpers_discharge, registry_name)(), trusting=False)
    assert resolution.qualifying == []


def test_the_verified_type_is_neither_exported_nor_constructible():
    """A "verified" value any caller can build is a label, not a verification."""
    import rh_research_engine.contracts as contracts
    from rh_research_engine.contracts import receipts

    assert "VerifiedDecision" not in contracts.__all__
    assert "verify_decision" not in contracts.__all__
    assert "TRUSTED_DRE_ENGINES" not in contracts.__all__
    assert not hasattr(contracts, "VerifiedDecision")
    assert set(receipts.__all__) == {
        "DreReceipt",
        "ReceiptAuthentication",
        "ReceiptError",
        "activation_status",
    }

    with _pytest.raises(receipts.ReceiptError, match="cannot be constructed directly"):
        receipts._VerifiedDecision(
            object(),
            decision=None,
            receipt=None,
            receipt_hash="x",
            engine_fingerprint="y",
        )


def test_no_public_function_accepts_a_trust_set_override():
    """A caller that can pass its own fingerprint has authorized itself."""
    import inspect

    from rh_research_engine.contracts import discharge, receipts

    for module in (discharge, receipts):
        for name, obj in vars(module).items():
            if name.startswith("_") or not inspect.isfunction(obj):
                continue
            params = inspect.signature(obj).parameters
            assert "trusted_engines" not in params, f"{module.__name__}.{name}"


def test_activation_status_reports_inert_without_exposing_the_set():
    from rh_research_engine.contracts import activation_status

    status = activation_status()
    assert status["discharge_authority_active"] is False
    assert status["trusted_engine_count"] == 0
    assert any("signature" in r for r in status["activation_requires"])


def test_a_decision_without_a_receipt_cannot_be_verified():
    """The record saying 'accepted' establishes nothing about who authorized it."""
    from helpers_discharge import unreceipted_decision_registry

    resolution = _resolve(unreceipted_decision_registry())
    assert resolution.qualifying == []


def test_a_receipt_from_an_untrusted_engine_is_refused():
    from helpers_discharge import untrusted_engine_registry

    resolution = _resolve(untrusted_engine_registry())
    assert resolution.qualifying == []
    assert "not trusted by this build" in resolution.explain()


def test_a_receipt_bound_to_a_stale_decision_is_refused():
    from helpers_discharge import stale_receipt_registry

    resolution = _resolve(stale_receipt_registry())
    assert resolution.qualifying == []
    assert "edited after the ruling" in resolution.explain()


def test_a_worker_can_construct_certified_evidence_but_it_confers_nothing():
    """CERTIFIED is not worker-forbidden, which is exactly why labels cannot rule."""
    from helpers_discharge import worker_minted_evidence_registry

    _, evidence, _, _ = worker_minted_evidence_registry()
    artifact = evidence["EV-1"]
    assert artifact.created_by == "rh-math-worker"
    assert artifact.epistemic_status is Confidence.CERTIFIED  # constructed fine
    assert _resolve(worker_minted_evidence_registry()).qualifying == []
