"""Static scans over the source tree: what must not appear, and where.

Three defects in this repository were invisible to behavioural tests because
they were about *shape* rather than outcome. A hardcoded path matched nothing
after a file moved, so every dead-end route read as novel. A status classified
by spelling promoted 14 of 21 knowledge statuses to rigorous. A deprecated enum
kept spreading into new modules faster than it could be removed.

None of those produce a failing assertion anywhere; each produces a plausible
wrong answer. So they are checked mechanically, against the AST rather than
against text -- a regex over source lines flagged this file's own prose the
first time, and a detector that cannot tell code from comment gets silenced.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "rh_research_engine"


def _modules() -> list[Path]:
    return [p for p in sorted(SRC.rglob("*.py")) if "__pycache__" not in p.parts]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _rel(path: Path) -> str:
    return str(path.relative_to(SRC)).replace("\\", "/")


# ---------------------------------------------------------------------------
# 5.2  Fail-closed durable memory: no production caller opts out
# ---------------------------------------------------------------------------


def test_no_production_call_site_allows_missing_durable_memory():
    """`allow_missing=True` is for bootstrapping fresh state, and nothing else.

    An explicit path used to bypass the resolver and return an empty list for a
    file that was not there, so a caller pointing at the wrong path got a clean
    empty knowledge base instead of an error -- and an empty base silently
    satisfies every no-go check. The flag that restores that behaviour must not
    appear on any authoritative path.
    """
    offenders: list[str] = []
    for path in _modules():
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
            if name != "KnowledgeBase":
                continue
            for keyword in node.keywords:
                if keyword.arg == "allow_missing" and not (
                    isinstance(keyword.value, ast.Constant) and keyword.value.value is False
                ):
                    offenders.append(f"{_rel(path)}:{node.lineno}")
    assert offenders == [], f"durable memory made optional at {offenders}"


def test_the_flag_still_exists_for_the_one_case_that_needs_it():
    """Guarding against a flag nobody can reach is guarding against nothing."""
    import inspect

    from rh_research_engine.core.knowledge import KnowledgeBase

    parameter = inspect.signature(KnowledgeBase).parameters["allow_missing"]
    assert parameter.default is False


# ---------------------------------------------------------------------------
# 5.4  The retired knowledge path stays retired
# ---------------------------------------------------------------------------

#: Where the retired path may still legitimately appear: the module that names
#: it as history, and the migration tooling that has to know what it was.
RETIRED_PATH_EXEMPT = {"core/knowledge.py"}


def test_no_runtime_module_hardcodes_the_retired_knowledge_path():
    """A hardcoded old path matches nothing after the move, and says nothing.

    `match_route` did exactly this. With both copies present it worked; once the
    legacy copy was deleted it matched against an empty file, so every recorded
    no-go route was reported as novel prior work.
    """
    offenders: list[str] = []
    for path in _modules():
        if _rel(path) in RETIRED_PATH_EXEMPT:
            continue
        for node in ast.walk(_parse(path)):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value == "research_state/math_knowledge.json"
            ):
                offenders.append(f"{_rel(path)}:{node.lineno}")
    assert offenders == [], f"retired path hardcoded at {offenders}"


# ---------------------------------------------------------------------------
# 6.1  No classification by spelling
# ---------------------------------------------------------------------------


def test_nothing_classifies_an_epistemic_axis_by_spelling():
    """The detector that found the original defect, run as a test.

    Reuses `tools/inventory-status.py` rather than reimplementing the scan: two
    copies of a rule is how the rule starts disagreeing with itself, and this
    one exists to prevent exactly that class of drift.
    """
    sys.path.insert(0, str(REPO / "tools"))
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "inventory_status", REPO / "tools" / "inventory-status.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    findings = module.collect_substring_classifiers()
    detail = "\n".join(
        f"  {hit['file']}:{hit['line']}  {hit['reason']}\n      {hit['source']}"
        for hit in findings
    )
    assert findings == [], f"classification by spelling:\n{detail}"


def test_the_detector_would_catch_the_defect_it_was_written_for(tmp_path):
    """A scan that reports nothing because it looks for nothing is worse than none.

    Both shapes here are real: `"equivalent" in item.status` shipped in the
    route matcher, and `match.status == "false_route"` shipped in the no-go
    audit. Neither used `.value`, which is why the first version of this
    detector -- which required it -- reported a clean tree.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "inventory_status", REPO / "tools" / "inventory-status.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    planted = tmp_path / "planted.py"
    planted.write_text(
        "def classify(item):\n"
        '    if "equivalent" in item.status:\n'
        '        return "reformulation"\n'
        '    if item.status == "false_route":\n'
        '        return "no_go"\n'
        '    if item.epistemic_status.value.startswith("exact"):\n'
        '        return "rigorous"\n'
        "    return None\n",
        encoding="utf-8",
        newline="",
    )
    original_src, original_repo = module.SRC, module.REPO
    module.SRC = module.REPO = tmp_path
    try:
        findings = module.collect_substring_classifiers()
    finally:
        module.SRC, module.REPO = original_src, original_repo
    reasons = {hit["reason"] for hit in findings}
    assert reasons == {
        "substring test against a status axis",
        "status axis compared to a string literal",
        "startswith on a status axis",
    }


# ---------------------------------------------------------------------------
# 6.3  Deprecated and legacy vocabularies stay contained
# ---------------------------------------------------------------------------

#: The deprecated lifecycle enum. Reachable only from the compatibility shim
#: that defines it and the mapping table that translates it away.
HYPOTHESIS_STATE_EXEMPT = {"supervisor/compat.py", "contracts/mappings.py"}

#: The property-graph storage vocabularies. Live reasoning runs on
#: `Confidence`/`Role`; these two remain because they are what is on disk.
PROPERTY_ENUM_EXEMPT = {"properties/models.py", "contracts/mappings.py"}


def test_the_deprecated_hypothesis_state_reaches_no_other_module():
    offenders: list[str] = []
    for path in _modules():
        if _rel(path) in HYPOTHESIS_STATE_EXEMPT:
            continue
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Name) and node.id == "HypothesisState":
                offenders.append(f"{_rel(path)}:{node.lineno}")
            elif isinstance(node, ast.Attribute) and node.attr == "HypothesisState":
                offenders.append(f"{_rel(path)}:{node.lineno}")
    assert offenders == [], f"deprecated HypothesisState used at {offenders}"


@pytest.mark.parametrize("enum_name", ["EpistemicStatus", "MathematicalRole"])
def test_no_module_branches_on_a_legacy_property_enum(enum_name):
    """Storing one is fine. Deciding an epistemic question on one is not.

    `PropertyRecord.status` is an `EpistemicStatus` because that is the on-disk
    format, and extractors set it when they build a record. What no module may
    do is *branch* on it: that would be a second copy of the rigour rule, in a
    second vocabulary, free to drift from the canonical one in
    `contracts.epistemic` that everything else consults.
    """
    offenders: list[str] = []
    for path in _modules():
        if _rel(path) in PROPERTY_ENUM_EXEMPT:
            continue
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.Compare):
                continue
            for operand in [node.left, *node.comparators]:
                if (
                    isinstance(operand, ast.Attribute)
                    and isinstance(operand.value, ast.Name)
                    and operand.value.id == enum_name
                ):
                    offenders.append(f"{_rel(path)}:{node.lineno}")
    assert offenders == [], f"{enum_name} used as a decision at {offenders}"


def test_the_legacy_rigour_set_agrees_with_the_canonical_one():
    """Two spellings of one rule, checked to be one rule.

    `properties.models.RIGOROUS_STATUSES` is the legacy-vocabulary copy. It is
    kept because tests and callers read it, and pinned here so it cannot drift
    from `contracts.epistemic.RIGOROUS`, which is the authority.
    """
    from rh_research_engine.contracts.epistemic import RIGOROUS
    from rh_research_engine.contracts.mappings import confidence_from_property_status
    from rh_research_engine.properties.models import RIGOROUS_STATUSES, EpistemicStatus

    derived = {
        status
        for status in EpistemicStatus
        if confidence_from_property_status(status) in RIGOROUS
    }
    assert set(RIGOROUS_STATUSES) == derived


def test_the_legacy_role_split_agrees_with_the_canonical_one():
    from rh_research_engine.contracts.mappings import role_from_property_role
    from rh_research_engine.contracts.roles import META_ROLES as CANONICAL_META
    from rh_research_engine.properties.models import (
        MATHEMATICAL_ROLES,
        META_ROLES,
        MathematicalRole,
    )

    derived_meta = {
        role for role in MathematicalRole if role_from_property_role(role) in CANONICAL_META
    }
    assert set(META_ROLES) == derived_meta
    assert set(MATHEMATICAL_ROLES) == set(MathematicalRole) - derived_meta


# ---------------------------------------------------------------------------
# The gate tooling itself has to run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool",
    [
        "check-line-endings.py",
        "snapshot-schemas.py",
        "inventory-status.py",
        "snapshot-baseline.py",
        "migrate-knowledge-path.py",
        "phase1-final-gate.py",
    ],
)
def test_every_gate_tool_is_invocable(tool):
    """A gate nobody can run is not a gate. Catches import-time breakage early.

    `--help` only, deliberately: this asks whether the tool loads and parses its
    arguments, not whether the invariant it guards currently holds. Several of
    these read git history, which an exported copy does not have.
    """
    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / tool), "--help"],
        cwd=REPO,
        capture_output=True,
        text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
