"""Guard the contract surface captured at the pre-laboratory baseline.

Phase 1 of the laboratory program unifies artifact schemas and migrates stored
state, which is exactly the kind of change that can quietly drop a field or
widen a vocabulary. These tests make an unintended change fail loudly; an
intended one is a one-line regeneration plus a reviewable diff.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "docs" / "contracts" / "schema-snapshot.json"
TOOL = REPO / "tools" / "snapshot-schemas.py"

sys.path.insert(0, str(REPO / "tools"))


def _snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_schema_surface_matches_the_baseline():
    """Any schema change must be a deliberate, committed regeneration."""
    result = subprocess.run(
        [sys.executable, str(TOOL), "--check"],
        capture_output=True,
        text=True, encoding="utf-8",
        cwd=REPO,
    )
    assert result.returncode == 0, (
        "schema surface drifted from docs/contracts/schema-snapshot.json\n"
        f"{result.stdout}\n{result.stderr}\n"
        "If intended: python tools/snapshot-schemas.py, then commit."
    )


def test_snapshot_is_sealed_and_lf():
    raw = SNAPSHOT.read_bytes()
    assert b"\r\n" not in raw, "CRLF would change the seal on the other platform"
    import hashlib

    seal = SNAPSHOT.with_suffix(SNAPSHOT.suffix + ".sha256")
    assert seal.read_text(encoding="utf-8").split()[0] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    "enum_name,expected_members",
    [
        # 8 since `inferred` was added: reasoning that is neither measured nor
        # cited had nowhere to go, so it went to `known` -- the status that
        # promises literature and had no field to hold a citation.
        ("rh_research_engine.core.models.ClaimStatus", 8),
        ("rh_research_engine.core.models.EvidenceClass", 7),
        ("rh_research_engine.core.knowledge.KnowledgeStatus", 21),
        ("rh_research_engine.properties.models.EpistemicStatus", 9),
        ("rh_research_engine.properties.models.MathematicalRole", 8),
        # Relocated to supervisor/compat.py when the axes were split; the
        # canonical replacement is HypothesisLifecycle.
        ("rh_research_engine.supervisor.compat.HypothesisState", 6),
        ("rh_research_engine.contracts.lifecycle.HypothesisLifecycle", 6),
        ("rh_research_engine.contracts.epistemic.Confidence", 14),
        ("rh_research_engine.contracts.roles.Role", 11),
    ],
)
def test_every_epistemic_vocabulary_is_captured(enum_name, expected_members):
    """The vocabularies are the whole point of the snapshot.

    `pkgutil.walk_packages` silently skipped `core/` and `math/` -- they have no
    `__init__.py` and are implicit namespace packages -- so the first snapshot
    omitted ClaimStatus, EvidenceClass and KnowledgeStatus entirely and would
    have diffed clean while they changed underneath.
    """
    enums = _snapshot()["enums"]
    assert enum_name in enums, f"{enum_name} is missing from the snapshot"
    assert len(enums[enum_name]) == expected_members


def _load_tool():
    """Import the snapshot tool by path (its filename is not an identifier)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("snapshot_schemas", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_namespace_package_modules_are_reachable():
    """Regression guard for the discovery bug itself.

    `core/` and `math/` ship without `__init__.py`. Module discovery must not
    depend on that, or the vocabularies living there vanish from the snapshot.
    """
    modules = _load_tool()._iter_modules()
    for expected in (
        "rh_research_engine.core.models",
        "rh_research_engine.core.knowledge",
        "rh_research_engine.math.correlation",
    ):
        assert expected in modules, f"{expected} not discovered"


# --- failure reporting: a gate that says only "it changed" is not a gate -----


def _run(tool: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "tools" / tool), *args],
        capture_output=True,
        text=True, encoding="utf-8",
        cwd=REPO,
    )


def test_status_inventory_matches_baseline():
    result = _run("inventory-status.py", "--check")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def _requires_git_checkout():
    """Skip when history is absent.

    The gate itself exits nonzero without git, and should -- "cannot verify" is
    not "verified". A *test* asserting that gate is a different question: on an
    exported copy the environment cannot answer, and reporting that as a defect
    is noise that trains people to ignore the suite.
    """
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout; the baseline is reconstructed from history")


def test_authoritative_state_matches_the_current_state_manifest():
    _requires_git_checkout()
    result = _run("snapshot-baseline.py", "--check")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_schema_drift_report_names_the_member_and_field():
    """The report must be actionable, not merely truthful."""
    tool = _load_tool()
    before = {
        "enums": {"pkg.Status": ["a", "b"]},
        "models": {"pkg.Thing": {"properties": {"x": {"type": "integer"}}, "required": ["x"]}},
    }
    after = {
        "enums": {"pkg.Status": ["a", "c"]},
        "models": {"pkg.Thing": {"properties": {"y": {"type": "string"}}, "required": []}},
    }
    report = "\n".join(tool.describe_drift(before, after))
    assert "removed members: ['b']" in report
    assert "added members:   ['c']" in report
    assert "removed field: x" in report
    assert "added field:   y" in report


def test_knowledge_invariant_failure_names_the_record_and_the_reason():
    import importlib.util

    spec = importlib.util.spec_from_file_location("bm", REPO / "tools" / "snapshot-baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    problems = module._describe_invariant_drift(
        {"no_go_ids": ["K008", "K032"], "record_count": 42},
        {"no_go_ids": ["K008"], "record_count": 41},
    )
    report = "\n".join(problems)
    assert "K032" in report, "must name the lost record"
    assert "refuted line of attack" in report, "must say why it matters"
    assert "42 -> 41" in report


def test_substring_classifier_report_names_file_and_line():
    import importlib.util

    spec = importlib.util.spec_from_file_location("si", REPO / "tools" / "inventory-status.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = "\n".join(
        module.describe_drift(
            {"substring_classifiers": []},
            {
                "substring_classifiers": [
                    {
                        "file": "src/x.py",
                        "line": 42,
                        "reason": "startswith on an enum value",
                        "source": 'if status.value.startswith("exact"):',
                    }
                ]
            },
        )
    )
    assert "src/x.py:42" in report
    assert "ADR-001" in report


def test_frozen_baseline_is_separate_and_immutable():
    """A frozen reference that drifts is not a reference.

    The two manifests were one file, which made "baseline" a lie: it moved with
    every commit and could not answer "what changed since the program started".
    """
    import json as _json

    frozen = REPO / "docs" / "contracts" / "frozen-baseline-df7016d.json"
    current = REPO / "docs" / "contracts" / "current-state-manifest.json"
    assert frozen.exists() and current.exists()

    frozen_doc = _json.loads(frozen.read_text(encoding="utf-8"))
    current_doc = _json.loads(current.read_text(encoding="utf-8"))
    assert frozen_doc["manifest_kind"] == "frozen-baseline"
    assert frozen_doc["immutable"] is True
    assert current_doc["manifest_kind"] == "current-state"
    assert frozen_doc["baseline_commit"].startswith("df7016d")

    # Regenerating the frozen baseline is refused without --force.
    result = _run("snapshot-baseline.py", "--frozen")
    assert result.returncode == 1
    assert "immutable" in result.stderr


def test_both_manifests_use_the_same_invariant_shape():
    """A field on only one side reads as a change and buries the real diff."""
    import json as _json

    frozen = _json.loads(
        (REPO / "docs" / "contracts" / "frozen-baseline-df7016d.json").read_text(encoding="utf-8")
    )
    current = _json.loads(
        (REPO / "docs" / "contracts" / "current-state-manifest.json").read_text(encoding="utf-8")
    )
    assert set(frozen["knowledge_invariants"]) == set(current["knowledge_invariants"])


def test_knowledge_invariants_are_unchanged_since_the_baseline():
    _requires_git_checkout()
    result = _run("snapshot-baseline.py", "--since-baseline")
    assert result.returncode == 0
    assert "unchanged since" in result.stdout

def test_only_the_per_run_ledger_is_excluded_from_the_baseline():
    """A gate that quietly narrows what it watches is worse than no gate.

    `rhre patterns sweep` rewrites the open-findings ledger every time, so
    including it made a routine sweep fail the baseline and the only way through
    was to delete the file -- a habit that would eventually delete something
    that mattered. The exclusion is one named path, and everything durable is
    still watched: a noise rule is a judgement somebody made once, with a
    reason, and losing one is losing research.
    """
    import json

    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "contracts"
            / "current-state-manifest.json"
        ).read_text(encoding="utf-8")
    )

    excluded = manifest.get("excluded_as_regenerated")
    assert excluded == ["research_state/pattern_open.json"], excluded

    watched = manifest["authoritative_state"]
    assert not any("pattern_open" in key for key in watched)
    assert any("pattern_noise" in key for key in watched), (
        "the triage registry is durable and must stay under the gate"
    )
    assert any("claims.json" in key for key in watched)
