import json
from pathlib import Path

import pytest

from rh_research_engine.core.knowledge import (
    KnowledgeBase,
    KnowledgeIntegrityError,
    KnowledgeStatus,
)

# Durable memory now lives at the canonical authoritative path.
MEMORY = Path("research_state/authoritative/knowledge/math_knowledge.json")


def test_math_knowledge_loads_and_is_substantial() -> None:
    kb = KnowledgeBase(MEMORY)
    items = kb.load()
    assert len(items) >= 40
    assert kb.get("K019") is not None
    assert kb.get("K019").status == KnowledgeStatus.RESEARCH_TARGET


def test_math_knowledge_dependencies_are_valid() -> None:
    assert KnowledgeBase(MEMORY).validate_dependencies() == []


def test_math_knowledge_preserves_no_go_routes() -> None:
    kb = KnowledgeBase(MEMORY)
    false_routes = {i.id for i in kb.load() if i.status == KnowledgeStatus.FALSE_ROUTE}
    assert {"K008", "K032", "K034", "K038"} <= false_routes


def test_math_knowledge_search() -> None:
    hits = KnowledgeBase(MEMORY).search("screening remainder")
    assert any(item.id == "K019" for item in hits)


def test_shipped_memory_is_strict_json_and_sealed() -> None:
    """The shipped file used to end `]}]}]}` and load only via a recovery path."""
    raw = MEMORY.read_bytes()
    assert isinstance(json.loads(raw.decode("utf-8")), list)
    assert b"\r\n" not in raw
    kb = KnowledgeBase(MEMORY)
    assert kb.seal_path.exists()
    assert kb.audit() == []


def test_trailing_bytes_are_a_hard_error(tmp_path: Path) -> None:
    p = tmp_path / "k.json"
    p.write_text('[{"id":"K","title":"t","status":"exact","domain":"d","statement":"s"}]]}', encoding="utf-8")
    with pytest.raises(KnowledgeIntegrityError):
        KnowledgeBase(p).load()


def test_truncated_array_with_closing_suffix_is_rejected(tmp_path: Path) -> None:
    """A prefix parse used to load 5 of 43 items silently, losing every no-go."""
    full = [
        {"id": f"K{i:03d}", "title": "t", "status": "exact", "domain": "d", "statement": "s"}
        for i in range(1, 44)
    ]
    p = tmp_path / "k.json"
    p.write_text(json.dumps(full[:5])[:-1] + "]}", encoding="utf-8")
    with pytest.raises(KnowledgeIntegrityError):
        KnowledgeBase(p).load()


def test_status_vocabulary_is_closed_against_proof_claims(tmp_path: Path) -> None:
    p = tmp_path / "k.json"
    p.write_text(
        json.dumps(
            [
                {
                    "id": "K999",
                    "title": "RH resolved",
                    "status": "proved",
                    "domain": "spectral",
                    "statement": "The Riemann Hypothesis is proved.",
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(KnowledgeIntegrityError):
        KnowledgeBase(p).load()
    items, quarantined = KnowledgeBase(p).load_with_quarantine()
    assert items == []
    assert quarantined[0].id == "K999"
    assert "closed vocabulary" in quarantined[0].reason
    assert "proved" not in set(KnowledgeStatus)


def test_seal_detects_tampering(tmp_path: Path) -> None:
    p = tmp_path / "k.json"
    p.write_text('[{"id":"K","title":"t","status":"exact","domain":"d","statement":"s"}]', encoding="utf-8")
    kb = KnowledgeBase(p)
    kb.seal()
    assert len(kb.load()) == 1
    p.write_text('[{"id":"K","title":"t","status":"exact","domain":"d","statement":"EDITED"}]', encoding="utf-8")
    with pytest.raises(KnowledgeIntegrityError, match="does not match its seal"):
        kb.load()


# --- staged path relocation --------------------------------------------------


import hashlib  # noqa: E402

from rh_research_engine.core.knowledge import (  # noqa: E402
    CANONICAL_KNOWLEDGE_PATH,
    LEGACY_KNOWLEDGE_PATH,
    resolve_knowledge_path,
)

CANONICAL = Path(CANONICAL_KNOWLEDGE_PATH)
LEGACY = Path(LEGACY_KNOWLEDGE_PATH)


def _write(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="")
    return path


ONE = '[{"id":"K1","title":"t","status":"exact","domain":"d","statement":"s"}]'
OTHER = '[{"id":"K2","title":"t","status":"exact","domain":"d","statement":"s"}]'


def test_resolve_prefers_canonical_when_only_it_exists(tmp_path):
    canonical = _write(tmp_path / "new.json", ONE)
    assert resolve_knowledge_path(canonical=canonical, legacy=tmp_path / "old.json") == canonical


def test_there_is_no_default_legacy_fallback():
    """An ordinary read must not reach for a path that is supposed to be gone."""
    import inspect

    assert inspect.signature(resolve_knowledge_path).parameters["legacy"].default is None


def test_an_explicit_legacy_path_still_resolves_for_rollback(tmp_path):
    """Deliberate comparison against an archived copy stays possible."""
    legacy = _write(tmp_path / "old.json", ONE)
    assert resolve_knowledge_path(canonical=tmp_path / "new.json", legacy=legacy) == legacy


def test_resolve_prefers_canonical_when_both_are_identical(tmp_path):
    canonical = _write(tmp_path / "new.json", ONE)
    legacy = _write(tmp_path / "old.json", ONE)
    assert resolve_knowledge_path(canonical=canonical, legacy=legacy) == canonical


def test_resolve_refuses_to_choose_when_the_copies_diverge(tmp_path):
    canonical = _write(tmp_path / "new.json", ONE)
    legacy = _write(tmp_path / "old.json", OTHER)
    with pytest.raises(KnowledgeIntegrityError, match="two paths with different contents"):
        resolve_knowledge_path(canonical=canonical, legacy=legacy)


def test_resolve_reports_both_hashes_when_diverged(tmp_path):
    """The message has to be actionable: which file, and what does each hash?"""
    canonical = _write(tmp_path / "new.json", ONE)
    legacy = _write(tmp_path / "old.json", OTHER)
    with pytest.raises(KnowledgeIntegrityError) as excinfo:
        resolve_knowledge_path(canonical=canonical, legacy=legacy)
    message = str(excinfo.value)
    assert hashlib.sha256(ONE.encode()).hexdigest() in message
    assert hashlib.sha256(OTHER.encode()).hexdigest() in message


def test_missing_durable_memory_is_fatal(tmp_path):
    """An absent research record is not an empty one.

    Returning a non-existent path let every caller degrade to zero records,
    which reads as a clean knowledge base and silently satisfies every no-go
    check.
    """
    with pytest.raises(KnowledgeIntegrityError, match="durable memory is missing"):
        resolve_knowledge_path(canonical=tmp_path / "new.json", legacy=tmp_path / "old.json")


def test_invalid_seal_on_the_canonical_copy_is_fatal(tmp_path):
    canonical = _write(tmp_path / "new.json", ONE)
    kb = KnowledgeBase(canonical)
    kb.seal()
    canonical.write_text(OTHER, encoding="utf-8", newline="")
    with pytest.raises(KnowledgeIntegrityError, match="does not match its seal"):
        kb.load()


def test_repository_is_in_the_retired_state():
    """Canonical only, sealed; the legacy copy and its seal are gone."""
    assert CANONICAL.exists(), "canonical copy missing"
    assert CANONICAL.with_suffix(".json.sha256").exists(), "canonical seal missing"
    assert not LEGACY.exists(), "legacy copy still present"
    assert not LEGACY.with_suffix(".json.sha256").exists(), "legacy seal still present"


def test_the_migration_manifest_is_retained_after_retirement():
    """It is the only record of what the relocation was meant to preserve.

    Post-retirement integrity checks compare against it, so deleting it would
    leave the canonical copy with nothing to be checked against.
    """
    manifest = Path("docs/contracts/knowledge-path-migration.json")
    assert manifest.exists()
    import json as _json

    doc = _json.loads(manifest.read_text(encoding="utf-8"))
    assert doc["source_path"] == "research_state/math_knowledge.json"
    assert doc["source_hash"] == doc["target_hash"]


def test_migration_preserved_the_named_records():
    kb = KnowledgeBase(CANONICAL)
    items = kb.load()
    assert len(items) == 42
    assert len(kb.semantic_hashes()) == 42
    assert len(kb.dependency_edges()) == 45
    by_status = {i.id: i.status for i in items}
    for no_go in ("K008", "K032", "K034", "K038"):
        assert by_status[no_go] is KnowledgeStatus.FALSE_ROUTE
    assert by_status["K019"] is KnowledgeStatus.RESEARCH_TARGET
    assert kb.validate_dependencies() == []


def test_migration_is_idempotent():
    """Re-running the staged copy changes nothing and still verifies."""
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    before = hashlib.sha256(CANONICAL.read_bytes()).hexdigest()
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(repo / "tools" / "migrate-knowledge-path.py")],
            capture_output=True, text=True, encoding="utf-8", cwd=repo,
        )
        assert result.returncode == 0, result.stderr
    assert hashlib.sha256(CANONICAL.read_bytes()).hexdigest() == before


def test_migration_manifest_records_byte_identity_and_invariants():
    import json as _json

    manifest = _json.loads(
        (Path("docs/contracts/knowledge-path-migration.json")).read_text(encoding="utf-8")
    )
    assert manifest["source_hash"] == manifest["target_hash"]
    assert manifest["record_count"] == 42
    assert manifest["invariants"]["no_go_ids"] == ["K008", "K032", "K034", "K038"]
    assert manifest["invariants"]["research_target_ids"] == ["K019"]
    assert len(manifest["invariants"]["semantic_hashes"]) == 42
    assert manifest["ambiguous_records"] == []


def test_no_module_hardcodes_the_retired_legacy_path():
    """A hardcoded old path silently matches against nothing after the move.

    `match_route` did exactly that, so every no-go route read as novel until the
    route-matcher test caught it.
    """
    import ast

    src = Path("src/rh_research_engine")
    offenders: list[str] = []
    for path in sorted(src.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "knowledge.py":
            continue  # knowledge.py defines LEGACY_KNOWLEDGE_PATH deliberately
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value == "research_state/math_knowledge.json"
            ):
                offenders.append(f"{path.relative_to(src)}:{node.lineno}")
    assert offenders == [], f"retired path hardcoded at {offenders}"


def test_route_matcher_resolves_rather_than_hardcoding():
    import inspect

    from rh_research_engine.symbolic import match_route

    assert inspect.signature(match_route).parameters["knowledge_path"].default is None


# --- explicit paths fail closed too -----------------------------------------


def test_an_explicit_missing_path_is_fatal(tmp_path):
    """`dre export-latest` passes an explicit path, which bypassed the resolver."""
    with pytest.raises(KnowledgeIntegrityError, match="durable memory is missing"):
        KnowledgeBase(tmp_path / "absent.json").load()


def test_allow_missing_is_the_only_way_to_get_an_empty_base(tmp_path):
    assert KnowledgeBase(tmp_path / "absent.json", allow_missing=True).load() == []


def test_route_matcher_actions_come_from_the_canonical_mapping():
    """`"equivalent" in item.status` missed conditional_on_RH_standard.

    That status is RH-equivalent and contains no "equivalent" substring, so an
    RH-conditional route was reported as ordinary prior work.
    """
    from rh_research_engine.symbolic.route_matcher import _action_for

    assert _action_for(KnowledgeStatus.CONDITIONAL_ON_RH_STANDARD).startswith(
        "classify_as_reformulation"
    )
    assert _action_for(KnowledgeStatus.KNOWN_EQUIVALENCE_FRAMEWORK).startswith(
        "classify_as_reformulation"
    )
    assert _action_for(KnowledgeStatus.FALSE_ROUTE).startswith("reject_or_require")
    assert _action_for(KnowledgeStatus.RESEARCH_TARGET) == "link_to_existing_target"
    assert _action_for(KnowledgeStatus.GOVERNANCE).startswith("governance_record")


def test_every_knowledge_status_gets_an_action():
    from rh_research_engine.symbolic.route_matcher import _action_for

    for status in KnowledgeStatus:
        assert _action_for(status)
