"""End to end: every authoritative command refuses to run without its state.

The rule these enforce is one sentence long -- *an absent research record is not
an empty one* -- and it is the rule the engine got wrong in three separate
places. A missing knowledge file returned zero records, which reads as a clean
knowledge base and silently satisfies every no-go check. A missing hypothesis
queue returned an empty plan, which reads as "nothing outstanding". A no-go
audit that could not load durable memory printed a warning and then reported
"No no-go rule violations detected" with exit code zero.

Each case below runs the real CLI in a scratch workspace with the state
removed, and asserts three things: a nonzero exit, no output artifact, and no
mutation of what state remains. The third matters because a command that fails
*after* writing half its output leaves the store in a shape no later reader can
interpret.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _run(workspace: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "rh_research_engine.cli", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        # Explicit, because the console prints box-drawing characters and the
        # Windows default codec cannot decode them -- which surfaces as a
        # decode error in the test harness rather than as a test result.
        encoding="utf-8",
        errors="replace",
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "", "PYTHONIOENCODING": "utf-8"},
    )


def _output(result: subprocess.CompletedProcess) -> str:
    """Combined output with whitespace collapsed.

    The console wraps to the terminal width, so a message can arrive split
    across lines mid-sentence and a literal substring match fails on a message
    that is actually present.
    """
    return " ".join((result.stdout + result.stderr).split())


def _tree_digest(root: Path) -> dict[str, str]:
    """Content digest of every file under `root`, for before/after comparison."""
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A research workspace with claims and a queue, but no durable memory."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "research_state").mkdir()
    result = _run(work, "init")
    assert result.returncode == 0, result.stdout + result.stderr
    return work


@pytest.fixture
def with_memory(workspace: Path) -> Path:
    """The same workspace, with durable memory and its seal copied in."""
    source = REPO / "research_state" / "authoritative" / "knowledge"
    target = workspace / "research_state" / "authoritative" / "knowledge"
    target.mkdir(parents=True)
    for name in ("math_knowledge.json", "math_knowledge.json.sha256"):
        shutil.copy2(source / name, target / name)
    return workspace


# ---------------------------------------------------------------------------
# 5.3  Commands that must fail without durable mathematical memory
# ---------------------------------------------------------------------------

#: Command line, and the artifact it would have written. `None` where the
#: command's output is a report rather than a file.
MEMORY_DEPENDENT_COMMANDS = {
    "no-go classification": (["audit-no-go"], None),
    "route matching": (["symbolic", "match-route", "boundary unitarity condition"], None),
    "novelty classification": (
        ["symbolic", "match-route", "a new positivity criterion for zeta"],
        None,
    ),
    "durable memory validation": (["knowledge", "validate"], None),
    "property build": (
        ["properties", "build", "--out", "research_state/property_graph.json"],
        "research_state/property_graph.json",
    ),
    "dre export": (
        ["dre", "export-latest", "--claim", "C005", "--out", "dre/blocked.yaml"],
        "dre/blocked.yaml",
    ),
}


@pytest.mark.parametrize(
    "label", sorted(MEMORY_DEPENDENT_COMMANDS), ids=lambda label: label.replace(" ", "-")
)
def test_an_authoritative_command_fails_closed_without_durable_memory(workspace, label):
    argv, artifact = MEMORY_DEPENDENT_COMMANDS[label]
    before = _tree_digest(workspace)

    result = _run(workspace, *argv)

    assert result.returncode != 0, f"{label} succeeded with no research record:\n{result.stdout}"
    if artifact is not None:
        assert not (workspace / artifact).exists(), (
            f"{label} wrote {artifact} despite failing"
        )
    assert _tree_digest(workspace) == before, f"{label} mutated state on the failure path"


@pytest.mark.parametrize(
    "label", sorted(MEMORY_DEPENDENT_COMMANDS), ids=lambda label: label.replace(" ", "-")
)
def test_the_same_command_succeeds_once_the_record_is_present(with_memory, label):
    """Otherwise the tests above would pass on a command that never works.

    Two commands legitimately still exit nonzero with the record present:
    `dre export-latest` has no experiment to export, and `match-route` exits 1
    by design when a route overlaps a recorded dead end. What must change is the
    *reason* -- neither may still be blocked on the research record itself.

    That both match-route queries hit a recorded no-go is itself the point: the
    durable memory shipped here contains four of them, and matching is by
    content rather than by name.
    """
    argv, _ = MEMORY_DEPENDENT_COMMANDS[label]
    result = _run(with_memory, *argv)
    combined = _output(result)
    assert "durable memory unavailable" not in combined, combined
    if "match-route" in argv:
        # Exits 1 by design when the route overlaps a recorded dead end, which
        # is the command working rather than failing.
        assert "no-go route" in combined or result.returncode == 0, combined
        return
    if label == "dre export":
        return
    assert result.returncode == 0, combined


def test_the_no_go_audit_does_not_report_clean_when_it_cannot_read_the_record(workspace):
    """The specific regression: a swallowed exception became a passing audit.

    `audit-no-go` caught every exception from route matching, printed
    "route matching unavailable", and carried on to print "No no-go rule
    violations detected" and exit zero. The one state in which the durable no-go
    records cannot be consulted was also the state that reported a clean audit.
    """
    result = _run(workspace, "audit-no-go")
    combined = _output(result)
    assert result.returncode != 0
    assert "No no-go rule violations detected" not in combined
    assert "durable memory unavailable" in combined


# ---------------------------------------------------------------------------
# 5.3  Commands that must fail without the research plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [["supervisor", "list"], ["supervisor", "next"]])
def test_supervisor_evaluation_fails_closed_without_a_queue(workspace, argv):
    """An empty plan and an unreadable one look identical on the way out.

    "No actionable frontier-relevant hypothesis is ready" is a legitimate
    answer; it must not also be what a missing plan file produces.
    """
    before = _tree_digest(workspace)
    result = _run(workspace, *argv)
    combined = _output(result)
    assert result.returncode != 0
    assert "hypothesis queue" in combined
    assert _tree_digest(workspace) == before


def test_adding_the_first_hypothesis_is_still_allowed(workspace):
    """The one command that may create the plan, or the plan can never start."""
    result = _run(
        workspace,
        "supervisor",
        "add",
        "--id",
        "H-001",
        "--statement",
        "A test hypothesis.",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (workspace / "research_state" / "hypotheses.json").exists()

    listed = _run(workspace, "supervisor", "list")
    assert listed.returncode == 0, listed.stdout + listed.stderr
    assert "H-001" in listed.stdout


# ---------------------------------------------------------------------------
# 5.1  Corrupt, unsealed, and tampered records
# ---------------------------------------------------------------------------


def test_a_broken_seal_blocks_every_reader(with_memory):
    """Re-sealing is deliberate; drifting from the seal is not."""
    memory = with_memory / "research_state" / "authoritative" / "knowledge"
    target = memory / "math_knowledge.json"
    target.write_bytes(target.read_bytes()[:-20])

    result = _run(with_memory, "knowledge", "validate")
    combined = _output(result)
    assert result.returncode != 0
    assert "does not match its seal" in combined


def test_a_missing_seal_on_the_canonical_copy_blocks_every_reader(with_memory):
    """An unsealed authoritative copy is where tampering goes undetected."""
    memory = with_memory / "research_state" / "authoritative" / "knowledge"
    (memory / "math_knowledge.json.sha256").unlink()

    result = _run(with_memory, "knowledge", "validate")
    combined = _output(result)
    assert result.returncode != 0
    assert "has no seal" in combined


def test_invalid_json_is_never_auto_repaired(with_memory):
    """A tolerant parser read the longest valid prefix and dropped the rest.

    That silently discarded every entry after the cut -- including the
    `false_route` records, which is the worst possible subset to lose.
    """
    memory = with_memory / "research_state" / "authoritative" / "knowledge"
    target = memory / "math_knowledge.json"
    target.write_bytes(target.read_bytes() + b"]}]}")
    _run(with_memory, "knowledge", "seal")

    result = _run(with_memory, "knowledge", "validate")
    combined = _output(result)
    assert result.returncode != 0
    assert "not valid JSON" in combined or "integrity" in combined
