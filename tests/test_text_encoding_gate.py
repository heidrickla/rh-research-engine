"""The encoding gate has to fail on the things it claims to catch.

`tools/check-text-encoding.py` reports `OK (160 files, ...)` on a clean tree,
and that sentence is worth exactly as much as the gate's willingness to say
something else. Nothing here had ever watched it refuse.

Each case below is a defect the gate exists for, or an exemption it must not
mistake for one. The exemptions are not padding: the gate's own comments record
that an earlier version flagged `read_text("utf-8")`, which names the encoding
perfectly well, because it only looked for the keyword form.

The subprocess cases are the reason this file was written. Ruff's `PLW1514`
covers the file-I/O half (see `pyproject.toml`), so that half is now guarded
twice; nothing covers a child process decoded with the platform default, and
ten live instances were sitting in this repository when the gate was extended
to look -- one of them in `phase1-final-gate.py`, reading `git` output to decide
Phase 1 closure.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "tools" / "check-text-encoding.py"


def _gate():
    """Import a module whose filename is not an identifier."""
    spec = importlib.util.spec_from_file_location("check_text_encoding", GATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _offences(source: str) -> list[str]:
    return [message for _, message in _gate().offences(source, Path("probe.py"))]


CAUGHT = [
    pytest.param('open("x").read()', id="open"),
    pytest.param('open("x", "w").write("y")', id="open-for-writing"),
    pytest.param('Path("x").read_text()', id="read_text"),
    pytest.param('Path("x").write_text("y")', id="write_text"),
    pytest.param('Path("x").open()', id="path-open"),
    pytest.param(
        'subprocess.run(["git"], capture_output=True, text=True)',
        id="subprocess-text",
    ),
    pytest.param(
        'subprocess.run(["git"], universal_newlines=True)',
        id="subprocess-universal-newlines",
    ),
    pytest.param('subprocess.check_output(["git"], text=True)', id="check_output"),
    pytest.param('subprocess.Popen(["git"], text=True)', id="popen"),
]


@pytest.mark.parametrize("source", CAUGHT)
def test_the_gate_refuses_a_call_that_decodes_with_the_platform_default(source):
    """Every one of these reads back as different characters under cp1252."""
    assert _offences(source), f"the gate passed {source!r}, which names no codec"


EXEMPT = [
    pytest.param('open("x", encoding="utf-8").read()', id="open-named"),
    pytest.param('Path("x").read_text(encoding="utf-8")', id="read_text-named"),
    # The gate's own regression: positional is a perfectly good way to say it.
    pytest.param('Path("x").read_text("utf-8")', id="read_text-positional"),
    pytest.param('Path("x").write_text("y", "utf-8")', id="write_text-positional"),
    pytest.param('open("x", "rb").read()', id="binary-mode"),
    pytest.param('open("x", mode="rb").read()', id="binary-mode-keyword"),
    pytest.param('Path("x").read_bytes()', id="read_bytes"),
    pytest.param(
        'subprocess.run(["git"], text=True, encoding="utf-8")',
        id="subprocess-named",
    ),
    # `check_call` returns a status and decodes nothing, so text mode on it is
    # meaningless rather than dangerous. Flagging it would be noise.
    pytest.param('subprocess.check_call(["git"], text=True)', id="check_call"),
    pytest.param('subprocess.run(["git"], capture_output=True)', id="bytes-by-default"),
]


@pytest.mark.parametrize("source", EXEMPT)
def test_the_gate_does_not_invent_an_offence(source):
    """A gate that cries wolf gets switched off, which is how the defect returns."""
    assert not _offences(source), f"the gate flagged {source!r}, which is correct code"


def test_the_gate_reports_where_and_names_the_flag_it_saw():
    """A line number and the actual keyword, or the reader has to go looking."""
    module = _gate()
    source = "import subprocess\n\nsubprocess.run(['git'], universal_newlines=True)\n"
    found = module.offences(source, Path("probe.py"))
    assert len(found) == 1
    line, message = found[0]
    assert line == 3
    assert "universal_newlines=True" in message


def test_the_checked_in_tree_names_every_codec():
    """The integration case: the gate is green on what is actually committed."""
    module = _gate()
    assert module.main() == 0


def test_unparseable_is_reported_rather_than_skipped():
    """A file the gate cannot read is not a file with no offences in it."""
    found = _offences("def broken(:\n")
    assert found and "could not parse" in found[0]
